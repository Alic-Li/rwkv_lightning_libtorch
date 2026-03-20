#include "rwkv_model.h"

#include <fstream>
#include <iterator>
#include <stdexcept>

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

}  // namespace

RWKVModel::RWKVModel(const std::string& weights_file, torch::Device device)
    : device_(std::move(device)) {
  std::ifstream in(weights_file, std::ios::binary);
  TORCH_CHECK(in.good(), "failed to open weights file: ", weights_file);
  std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  TORCH_CHECK(!bytes.empty(), "empty weights file: ", weights_file);

  auto iv = torch::pickle_load(bytes);
  TORCH_CHECK(iv.isTensorList(), "weights pickle must contain a tensor list");
  std::vector<torch::Tensor> tensors = iv.toTensorVector();
  TORCH_CHECK(!tensors.empty(), "empty tensor list in weights file: ", weights_file);

  size_t idx = 0;
  auto meta = tensors[idx++].to(torch::kCPU, torch::kInt64);
  TORCH_CHECK(meta.numel() == 5, "invalid meta tensor in ", weights_file);
  n_layer_ = meta[0].item<int64_t>();
  n_head_ = meta[1].item<int64_t>();
  head_size_ = meta[2].item<int64_t>();
  n_embd_ = meta[3].item<int64_t>();
  vocab_size_ = meta[4].item<int64_t>();

  emb_weight_ = take_tensor(tensors, idx);
  ln_out_weight_ = take_tensor(tensors, idx);
  ln_out_bias_ = take_tensor(tensors, idx);
  head_weight_ = take_tensor(tensors, idx);

  layers_.reserve(n_layer_);
  for (int64_t i = 0; i < n_layer_; ++i) {
    RWKVLayerWeights layer;
    layer.ln1_weight = take_tensor(tensors, idx);
    layer.ln1_bias = take_tensor(tensors, idx);
    layer.ln2_weight = take_tensor(tensors, idx);
    layer.ln2_bias = take_tensor(tensors, idx);

    layer.att_x_r = take_tensor(tensors, idx);
    layer.att_x_w = take_tensor(tensors, idx);
    layer.att_x_k = take_tensor(tensors, idx);
    layer.att_x_v = take_tensor(tensors, idx);
    layer.att_x_a = take_tensor(tensors, idx);
    layer.att_x_g = take_tensor(tensors, idx);
    layer.att_w0 = take_tensor(tensors, idx);
    layer.att_w1 = take_tensor(tensors, idx);
    layer.att_w2 = take_tensor(tensors, idx);
    layer.att_a0 = take_tensor(tensors, idx);
    layer.att_a1 = take_tensor(tensors, idx);
    layer.att_a2 = take_tensor(tensors, idx);
    layer.att_v0 = take_tensor(tensors, idx);
    layer.att_v1 = take_tensor(tensors, idx);
    layer.att_v2 = take_tensor(tensors, idx);
    layer.att_g1 = take_tensor(tensors, idx);
    layer.att_g2 = take_tensor(tensors, idx);
    layer.att_k_k = take_tensor(tensors, idx);
    layer.att_k_a = take_tensor(tensors, idx);
    layer.att_r_k = take_tensor(tensors, idx);
    layer.att_receptance_weight = take_tensor(tensors, idx);
    layer.att_key_weight = take_tensor(tensors, idx);
    layer.att_value_weight = take_tensor(tensors, idx);
    layer.att_output_weight = take_tensor(tensors, idx);
    layer.att_ln_x_weight = take_tensor(tensors, idx);
    layer.att_ln_x_bias = take_tensor(tensors, idx);

    layer.ffn_x_k = take_tensor(tensors, idx);
    layer.ffn_key_weight = take_tensor(tensors, idx);
    layer.ffn_value_weight = take_tensor(tensors, idx);

    layers_.push_back(std::move(layer));
  }

  TORCH_CHECK(idx == tensors.size(), "unexpected tensor count in ", weights_file);
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

torch::Tensor RWKVModel::take_tensor(
    const std::vector<torch::Tensor>& tensors,
    size_t& index) const {
  TORCH_CHECK(index < tensors.size(), "weights tensor index overflow");
  return tensors[index++].to(device_, torch::kFloat16).contiguous();
}
