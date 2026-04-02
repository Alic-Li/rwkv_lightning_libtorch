#include "rwkv_model.h"

#include <algorithm>
#include <regex>
#include <stdexcept>

#include "model_load/safetensors_loader.h"
#include <torch/nn/functional/linear.h>
#include <torch/nn/functional/normalization.h>

namespace F = torch::nn::functional;

void forward_seq(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor& state,
    torch::Tensor& r,
    torch::Tensor& w,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& y,
    torch::Tensor& elapsed_t);

void forward_one(
    int64_t B,
    int64_t C,
    int64_t H,
    torch::Tensor& state,
    torch::Tensor& r,
    torch::Tensor& w,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& a,
    torch::Tensor& b,
    torch::Tensor& y,
    torch::Tensor& elapsed_t);

void spmv_forward(
    int64_t D,
    int64_t C,
    torch::Tensor& vec,
    torch::Tensor& mat,
    torch::Tensor& out);

namespace {

torch::Tensor rwkv_linear(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    c10::optional<torch::Tensor> bias = c10::nullopt) {
  if (bias.has_value()) {
    return at::linear(x, weight, *bias);
  }
  return at::linear(x, weight, c10::optional<torch::Tensor>());
}

torch::Tensor normalize_last_dim(const torch::Tensor& x) {
  return F::normalize(x, F::NormalizeFuncOptions().p(2.0).dim(-1));
}

std::vector<std::string> expected_tensor_names(int64_t n_layer) {
  std::vector<std::string> names = {
      "emb.weight",
      "blocks.0.ln0.weight",
      "blocks.0.ln0.bias",
      "ln_out.weight",
      "ln_out.bias",
      "head.weight",
  };
  names.reserve(7 + static_cast<size_t>(n_layer) * 30);

  for (int64_t i = 0; i < n_layer; ++i) {
    const std::string bbb = "blocks." + std::to_string(i) + ".";
    const std::string att = bbb + "att.";
    const std::string ffn = bbb + "ffn.";
    names.insert(
        names.end(),
        {
            bbb + "ln1.weight",
            bbb + "ln1.bias",
            bbb + "ln2.weight",
            bbb + "ln2.bias",
            att + "x_r",
            att + "x_w",
            att + "x_k",
            att + "x_v",
            att + "x_a",
            att + "x_g",
            att + "w0",
            att + "w1",
            att + "w2",
            att + "a0",
            att + "a1",
            att + "a2",
            att + "g1",
            att + "g2",
            att + "k_k",
            att + "k_a",
            att + "r_k",
            att + "receptance.weight",
            att + "key.weight",
            att + "value.weight",
            att + "output.weight",
            att + "ln_x.weight",
            att + "ln_x.bias",
            ffn + "x_k",
            ffn + "key.weight",
            ffn + "value.weight",
        });
  }
  return names;
}

int64_t infer_num_layers(const SafeTensorArchive& archive) {
  const std::regex block_pattern(R"(^blocks\.(\d+)\.)");
  int64_t max_layer = -1;
  for (const auto& name : archive.tensor_names()) {
    std::smatch match;
    if (std::regex_search(name, match, block_pattern)) {
      max_layer = std::max(
          max_layer,
          static_cast<int64_t>(std::stoll(match[1].str())));
    }
  }
  TORCH_CHECK(max_layer >= 0, "failed to infer layer count from safetensors");
  return max_layer + 1;
}

torch::Tensor squeeze_tensor(torch::Tensor tensor) {
  return tensor.squeeze().contiguous();
}

torch::Tensor load_squeezed_tensor(
    const SafeTensorArchive& archive,
    const std::string& name,
    torch::Device device) {
  return squeeze_tensor(archive.load_tensor(name, device));
}

torch::Tensor load_preprocessed_embedding(
    const SafeTensorArchive& archive,
    torch::Device device) {
  // Keep this preprocessing on CPU to mirror the old pth2st path and avoid
  // backend-specific layer_norm differences during model init.
  auto emb_weight = load_squeezed_tensor(archive, "emb.weight", torch::kCPU);
  const auto ln0_weight =
      load_squeezed_tensor(archive, "blocks.0.ln0.weight", torch::kCPU);
  const auto ln0_bias =
      load_squeezed_tensor(archive, "blocks.0.ln0.bias", torch::kCPU);
  emb_weight = F::layer_norm(
      emb_weight,
      F::LayerNormFuncOptions({emb_weight.size(-1)})
          .weight(ln0_weight)
          .bias(ln0_bias));
  return emb_weight.to(device).contiguous();
}

}  // namespace

RWKVModel::RWKVModel(const std::string& weights_file, torch::Device device)
    : device_(std::move(device)) {
  SafeTensorArchive archive(weights_file);

  n_layer_ = infer_num_layers(archive);

  auto expected_names = expected_tensor_names(n_layer_);
  for (const auto& name : expected_names) {
    TORCH_CHECK(
        archive.has_tensor(name),
        "missing tensor in safetensors file: ",
        name);
  }

  emb_weight_ = load_preprocessed_embedding(archive, device_);

  ln_out_weight_ = load_squeezed_tensor(archive, "ln_out.weight", device_);
  ln_out_bias_ = load_squeezed_tensor(archive, "ln_out.bias", device_);
  head_weight_ = load_squeezed_tensor(archive, "head.weight", device_);

  vocab_size_ = emb_weight_.size(0);
  n_embd_ = emb_weight_.size(1);
  auto r_k = load_squeezed_tensor(archive, "blocks.0.att.r_k", device_);
  TORCH_CHECK(
      r_k.dim() == 2,
      "expected blocks.0.att.r_k to have shape [n_head, head_size]");
  n_head_ = r_k.size(0);
  head_size_ = r_k.size(1);

  layers_.reserve(n_layer_);
  for (int64_t i = 0; i < n_layer_; ++i) {
    const std::string bbb = "blocks." + std::to_string(i) + ".";
    const std::string att = bbb + "att.";
    const std::string ffn = bbb + "ffn.";
    RWKVLayerWeights layer;
    layer.ln1_weight = load_squeezed_tensor(archive, bbb + "ln1.weight", device_);
    layer.ln1_bias = load_squeezed_tensor(archive, bbb + "ln1.bias", device_);
    layer.ln2_weight = load_squeezed_tensor(archive, bbb + "ln2.weight", device_);
    layer.ln2_bias = load_squeezed_tensor(archive, bbb + "ln2.bias", device_);

    layer.att_x_r = load_squeezed_tensor(archive, att + "x_r", device_);
    layer.att_x_w = load_squeezed_tensor(archive, att + "x_w", device_);
    layer.att_x_k = load_squeezed_tensor(archive, att + "x_k", device_);
    layer.att_x_v = load_squeezed_tensor(archive, att + "x_v", device_);
    layer.att_x_a = load_squeezed_tensor(archive, att + "x_a", device_);
    layer.att_x_g = load_squeezed_tensor(archive, att + "x_g", device_);
    layer.att_w0 = load_squeezed_tensor(archive, att + "w0", device_);
    layer.att_w1 = load_squeezed_tensor(archive, att + "w1", device_);
    layer.att_w2 = load_squeezed_tensor(archive, att + "w2", device_);
    layer.att_a0 = load_squeezed_tensor(archive, att + "a0", device_);
    layer.att_a1 = load_squeezed_tensor(archive, att + "a1", device_);
    layer.att_a2 = load_squeezed_tensor(archive, att + "a2", device_);
    if (i == 0) {
      layer.att_v0 = layer.att_a0;
      layer.att_v1 = layer.att_a1;
      layer.att_v2 = layer.att_a2;
    } else {
      layer.att_v0 = load_squeezed_tensor(archive, att + "v0", device_);
      layer.att_v1 = load_squeezed_tensor(archive, att + "v1", device_);
      layer.att_v2 = load_squeezed_tensor(archive, att + "v2", device_);
    }
    layer.att_g1 = load_squeezed_tensor(archive, att + "g1", device_);
    layer.att_g2 = load_squeezed_tensor(archive, att + "g2", device_);
    layer.att_k_k = load_squeezed_tensor(archive, att + "k_k", device_);
    layer.att_k_a = load_squeezed_tensor(archive, att + "k_a", device_);
    layer.att_r_k =
        load_squeezed_tensor(archive, att + "r_k", device_).flatten().contiguous();
    layer.att_receptance_weight =
        load_squeezed_tensor(archive, att + "receptance.weight", device_);
    layer.att_key_weight = load_squeezed_tensor(archive, att + "key.weight", device_);
    layer.att_value_weight =
        load_squeezed_tensor(archive, att + "value.weight", device_);
    layer.att_output_weight =
        load_squeezed_tensor(archive, att + "output.weight", device_);
    layer.att_ln_x_weight = load_squeezed_tensor(archive, att + "ln_x.weight", device_);
    layer.att_ln_x_bias = load_squeezed_tensor(archive, att + "ln_x.bias", device_);

    layer.ffn_x_k = load_squeezed_tensor(archive, ffn + "x_k", device_);
    layer.ffn_key_weight = load_squeezed_tensor(archive, ffn + "key.weight", device_);
    layer.ffn_value_weight =
        load_squeezed_tensor(archive, ffn + "value.weight", device_)
            .transpose(0, 1)
            .contiguous();

    layers_.push_back(std::move(layer));
  }
}

RWKVState RWKVModel::generate_zero_state(int64_t batch_size) const {
  RWKVState state;
  auto fp16 = torch::TensorOptions().dtype(torch::kFloat16).device(device_);
  auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(device_);
  state.x_prev = torch::zeros({n_layer_, 2, batch_size, n_embd_}, fp16);
  state.att_state =
      torch::zeros({n_layer_, batch_size, n_head_, head_size_, head_size_}, fp16);
  state.elapsed_t = torch::zeros({batch_size}, i32);
  return state;
}

torch::Tensor RWKVModel::forward_prefill(
    const std::vector<std::vector<int64_t>>& token_batches,
    RWKVState& state) const {
  TORCH_CHECK(!token_batches.empty(), "token_batches must not be empty");
  const auto batch = static_cast<int64_t>(token_batches.size());
  const auto seq_len = static_cast<int64_t>(token_batches.front().size());
  TORCH_CHECK(seq_len > 0, "prompt length must be > 0");
  for (const auto& row : token_batches) {
    TORCH_CHECK(
        static_cast<int64_t>(row.size()) == seq_len,
        "all prompts must have the same length");
  }

  auto ids = torch::empty(
      {batch, seq_len},
      torch::TensorOptions().dtype(torch::kLong).device(device_));
  for (int64_t b = 0; b < batch; ++b) {
    ids.index_put_({b}, torch::tensor(token_batches[b], ids.options()));
  }
  return forward_tokens(ids, state);
}

torch::Tensor RWKVModel::forward_decode(
    const std::vector<int64_t>& token_batch,
    RWKVState& state) const {
  TORCH_CHECK(!token_batch.empty(), "token_batch must not be empty");
  if (token_batch.size() == 1) {
    return forward_one_token(token_batch[0], state).view({1, vocab_size_});
  }
  auto ids = torch::tensor(
      token_batch,
      torch::TensorOptions().dtype(torch::kLong).device(device_))
                .view({static_cast<int64_t>(token_batch.size()), 1});
  return forward_tokens(ids, state);
}

torch::Tensor RWKVModel::forward_one_token(
    int64_t token_id,
    RWKVState& state) const {
  torch::NoGradGuard guard;

  auto x = emb_weight_.select(0, token_id);
  torch::Tensor v_first;

  for (int64_t i = 0; i < n_layer_; ++i) {
    const auto& layer = layers_[i];
    auto xx = F::layer_norm(
        x,
        F::LayerNormFuncOptions({n_embd_})
            .weight(layer.ln1_weight)
            .bias(layer.ln1_bias));

    auto tmix = tmix_one(
        i,
        xx,
        state.x_prev[i].select(1, 0),
        v_first,
        state.att_state[i].select(0, 0),
        layer,
        state.elapsed_t[0]);
    x = x + tmix.first;
    v_first = tmix.second;

    xx = F::layer_norm(
        x,
        F::LayerNormFuncOptions({n_embd_})
            .weight(layer.ln2_weight)
            .bias(layer.ln2_bias));
    xx = cmix_one(xx, state.x_prev[i].select(1, 0), layer);
    x = x + xx;
  }

  x = F::layer_norm(
      x,
      F::LayerNormFuncOptions({n_embd_})
          .weight(ln_out_weight_)
          .bias(ln_out_bias_));
  x = rwkv_linear(x, head_weight_);
  state.elapsed_t[0].add_(1);
  return x;
}

torch::Tensor RWKVModel::forward_tokens(
    const torch::Tensor& token_ids,
    RWKVState& state) const {
  torch::NoGradGuard guard;

  const auto batch = token_ids.size(0);
  const auto seq_len = token_ids.size(1);
  auto x = emb_weight_.index_select(0, token_ids.reshape({-1}))
               .view({batch, seq_len, n_embd_});

  torch::Tensor v_first;
  for (int64_t i = 0; i < n_layer_; ++i) {
    const auto& layer = layers_[i];
    auto xx = F::layer_norm(
        x,
        F::LayerNormFuncOptions({n_embd_})
            .weight(layer.ln1_weight)
            .bias(layer.ln1_bias));

    auto tmix = tmix_batch(
        i,
        xx,
        state.x_prev[i],
        v_first,
        state.att_state[i],
        layer,
        state.elapsed_t);
    x = x + tmix.first;
    v_first = tmix.second;

    xx = F::layer_norm(
        x,
        F::LayerNormFuncOptions({n_embd_})
            .weight(layer.ln2_weight)
            .bias(layer.ln2_bias));
    xx = cmix_batch(xx, state.x_prev[i], layer);
    x = x + xx;
  }

  x = x.select(1, seq_len - 1);
  x = F::layer_norm(
      x,
      F::LayerNormFuncOptions({n_embd_})
          .weight(ln_out_weight_)
          .bias(ln_out_bias_));
  x = rwkv_linear(x, head_weight_);
  state.elapsed_t.add_(static_cast<int>(seq_len));
  return x;
}

std::pair<torch::Tensor, torch::Tensor> RWKVModel::tmix_batch(
    int64_t layer_id,
    const torch::Tensor& x,
    torch::Tensor x_prev,
    torch::Tensor v_first,
    torch::Tensor att_state,
    const RWKVLayerWeights& layer,
    torch::Tensor elapsed_t) const {
  const auto B = x.size(0);
  const auto T = x.size(1);
  const auto H = n_head_;
  const auto N = head_size_;

  auto prev = x_prev.select(0, 0);
  auto xx = torch::cat({prev.unsqueeze(1), x.slice(1, 0, T - 1)}, 1) - x;
  prev.copy_(x.select(1, T - 1));

  auto xr = x + xx * layer.att_x_r;
  auto xw = x + xx * layer.att_x_w;
  auto xk = x + xx * layer.att_x_k;
  auto xv = x + xx * layer.att_x_v;
  auto xa = x + xx * layer.att_x_a;
  auto xg = x + xx * layer.att_x_g;

  auto r = rwkv_linear(xr, layer.att_receptance_weight);
  auto w = rwkv_linear(
      torch::tanh(rwkv_linear(xw, layer.att_w1)), layer.att_w2, layer.att_w0);
  auto k = rwkv_linear(xk, layer.att_key_weight);
  auto v = rwkv_linear(xv, layer.att_value_weight);
  auto a = torch::sigmoid(
      rwkv_linear(rwkv_linear(xa, layer.att_a1), layer.att_a2, layer.att_a0));
  auto g =
      rwkv_linear(torch::sigmoid(rwkv_linear(xg, layer.att_g1)), layer.att_g2);

  auto kk = normalize_last_dim((k * layer.att_k_k).view({B, T, H, N}))
                .view({B, T, H * N});
  k = k * (1 + (a - 1) * layer.att_k_a);
  auto kka = kk * a;

  if (layer_id == 0) {
    v_first = v;
  } else {
        v = v +
        (v_first - v) *
            torch::sigmoid(
                rwkv_linear(
                    rwkv_linear(xv, layer.att_v1), layer.att_v2, layer.att_v0));
  }

  auto neg_kk = -kk;
  auto y = torch::empty_like(r);
  forward_seq(B, T, H * N, H, att_state, r, w, k, v, neg_kk, kka, y, elapsed_t);

  auto y2 = F::group_norm(
                y.view({B * T, H * N}),
                F::GroupNormFuncOptions(H)
                    .weight(layer.att_ln_x_weight)
                    .bias(layer.att_ln_x_bias)
                    .eps(64e-5))
                .view({B, T, H * N});
  y2 = y2 +
       ((r * k * layer.att_r_k).view({B, T, H, N}).sum(-1, true) *
        v.view({B, T, H, N}))
           .view({B, T, H * N});

  return {rwkv_linear(y2 * g, layer.att_output_weight), v_first};
}

std::pair<torch::Tensor, torch::Tensor> RWKVModel::tmix_one(
    int64_t layer_id,
    const torch::Tensor& x,
    torch::Tensor x_prev,
    torch::Tensor v_first,
    torch::Tensor att_state,
    const RWKVLayerWeights& layer,
    torch::Tensor elapsed_t) const {
  const auto H = n_head_;
  const auto N = head_size_;

  auto prev = x_prev.select(0, 0);
  auto xx = prev - x;
  prev.copy_(x);

  auto xr = x + xx * layer.att_x_r;
  auto xw = x + xx * layer.att_x_w;
  auto xk = x + xx * layer.att_x_k;
  auto xv = x + xx * layer.att_x_v;
  auto xa = x + xx * layer.att_x_a;
  auto xg = x + xx * layer.att_x_g;

  auto r = rwkv_linear(xr, layer.att_receptance_weight);
  auto w = rwkv_linear(
      torch::tanh(rwkv_linear(xw, layer.att_w1)), layer.att_w2, layer.att_w0);
  auto k = rwkv_linear(xk, layer.att_key_weight);
  auto v = rwkv_linear(xv, layer.att_value_weight);
  auto a = torch::sigmoid(
      rwkv_linear(rwkv_linear(xa, layer.att_a1), layer.att_a2, layer.att_a0));
  auto g =
      rwkv_linear(torch::sigmoid(rwkv_linear(xg, layer.att_g1)), layer.att_g2);

  auto kk = normalize_last_dim((k * layer.att_k_k).view({H, N})).view({H * N});
  k = k * (1 + (a - 1) * layer.att_k_a);
  auto kka = kk * a;

  if (layer_id == 0) {
    v_first = v;
  } else {
    v = v +
        (v_first - v) *
            torch::sigmoid(
                rwkv_linear(
                    rwkv_linear(xv, layer.att_v1), layer.att_v2, layer.att_v0));
  }

  auto neg_kk = -kk;
  auto y = torch::empty_like(r);
  forward_one(1, H * N, H, att_state, r, w, k, v, neg_kk, kka, y, elapsed_t);

  auto y2 = F::group_norm(
                y.view({1, H * N}),
                F::GroupNormFuncOptions(H)
                    .weight(layer.att_ln_x_weight)
                    .bias(layer.att_ln_x_bias)
                    .eps(64e-5))
                .view({H * N});
  y2 = y2 +
       ((r * k * layer.att_r_k).view({H, N}).sum(-1, true) * v.view({H, N}))
           .view({H * N});

  return {rwkv_linear(y2 * g, layer.att_output_weight), v_first};
}

torch::Tensor RWKVModel::cmix_batch(
    const torch::Tensor& x,
    torch::Tensor x_prev,
    const RWKVLayerWeights& layer) const {
  const auto T = x.size(1);
  auto prev = x_prev.select(0, 1);
  auto xx = torch::cat({prev.unsqueeze(1), x.slice(1, 0, T - 1)}, 1) - x;
  prev.copy_(x.select(1, T - 1));
  auto k = x + xx * layer.ffn_x_k;
  k = torch::relu(rwkv_linear(k, layer.ffn_key_weight)).pow(2);
  return torch::matmul(k, layer.ffn_value_weight);
}

torch::Tensor RWKVModel::cmix_one(
    const torch::Tensor& x,
    torch::Tensor x_prev,
    const RWKVLayerWeights& layer) const {
  auto prev = x_prev.select(0, 1);
  auto xx = prev - x;
  prev.copy_(x);
  auto k = x + xx * layer.ffn_x_k;
  k = torch::relu(rwkv_linear(k, layer.ffn_key_weight)).pow(2);
  auto out = torch::zeros(
      {layer.ffn_value_weight.size(1)},
      torch::TensorOptions().dtype(torch::kFloat16).device(device_));
  auto kk = k.contiguous();
  auto vv = layer.ffn_value_weight.contiguous();
  spmv_forward(vv.size(0), vv.size(1), kk, vv, out);
  return out;
}
