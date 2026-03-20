#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <torch/serialize.h>
#include <torch/torch.h>

struct RWKVLayerWeights {
  torch::Tensor ln1_weight;
  torch::Tensor ln1_bias;
  torch::Tensor ln2_weight;
  torch::Tensor ln2_bias;

  torch::Tensor att_x_r;
  torch::Tensor att_x_w;
  torch::Tensor att_x_k;
  torch::Tensor att_x_v;
  torch::Tensor att_x_a;
  torch::Tensor att_x_g;

  torch::Tensor att_w0;
  torch::Tensor att_w1;
  torch::Tensor att_w2;
  torch::Tensor att_a0;
  torch::Tensor att_a1;
  torch::Tensor att_a2;
  torch::Tensor att_v0;
  torch::Tensor att_v1;
  torch::Tensor att_v2;
  torch::Tensor att_g1;
  torch::Tensor att_g2;
  torch::Tensor att_k_k;
  torch::Tensor att_k_a;
  torch::Tensor att_r_k;
  torch::Tensor att_receptance_weight;
  torch::Tensor att_key_weight;
  torch::Tensor att_value_weight;
  torch::Tensor att_output_weight;
  torch::Tensor att_ln_x_weight;
  torch::Tensor att_ln_x_bias;

  torch::Tensor ffn_x_k;
  torch::Tensor ffn_key_weight;
  torch::Tensor ffn_value_weight;
};

struct RWKVState {
  torch::Tensor x_prev;
  torch::Tensor att_state;
  torch::Tensor elapsed_t;
};

class RWKVModel {
 public:
  RWKVModel(const std::string& weights_file, torch::Device device);

  RWKVState generate_zero_state(int64_t batch_size) const;
  torch::Tensor forward_prefill(
      const std::vector<std::vector<int64_t>>& token_batches,
      RWKVState& state) const;
  torch::Tensor forward_decode(
      const std::vector<int64_t>& token_batch,
      RWKVState& state) const;

  int64_t vocab_size() const { return vocab_size_; }
  int64_t n_layer() const { return n_layer_; }
  int64_t n_head() const { return n_head_; }
  int64_t head_size() const { return head_size_; }
  int64_t n_embd() const { return n_embd_; }

 private:
  torch::Tensor forward_tokens(const torch::Tensor& token_ids, RWKVState& state) const;
  torch::Tensor forward_one_token(int64_t token_id, RWKVState& state) const;

  std::pair<torch::Tensor, torch::Tensor> tmix_batch(
      int64_t layer_id,
      const torch::Tensor& x,
      torch::Tensor x_prev,
      torch::Tensor v_first,
      torch::Tensor att_state,
      const RWKVLayerWeights& layer,
      torch::Tensor elapsed_t) const;

  std::pair<torch::Tensor, torch::Tensor> tmix_one(
      int64_t layer_id,
      const torch::Tensor& x,
      torch::Tensor x_prev,
      torch::Tensor v_first,
      torch::Tensor att_state,
      const RWKVLayerWeights& layer,
      torch::Tensor elapsed_t) const;

  torch::Tensor cmix_batch(
      const torch::Tensor& x,
      torch::Tensor x_prev,
      const RWKVLayerWeights& layer) const;

  torch::Tensor cmix_one(
      const torch::Tensor& x,
      torch::Tensor x_prev,
      const RWKVLayerWeights& layer) const;

  torch::Tensor take_tensor(
      const std::vector<torch::Tensor>& tensors,
      size_t& index) const;

  torch::Device device_;
  int64_t n_layer_ = 0;
  int64_t n_head_ = 0;
  int64_t head_size_ = 0;
  int64_t n_embd_ = 0;
  int64_t vocab_size_ = 0;

  torch::Tensor emb_weight_;
  torch::Tensor ln_out_weight_;
  torch::Tensor ln_out_bias_;
  torch::Tensor head_weight_;

  std::vector<RWKVLayerWeights> layers_;
};
