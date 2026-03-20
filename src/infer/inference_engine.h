#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "rwkv_model.h"
#include "utils/tokenizer.h"

struct GenerateOptions {
  int max_tokens = 512;
  std::vector<int64_t> stop_tokens{0, 261, 24281};
  double temperature = 1.0;
  int top_k = 50;
  double top_p = 0.6;
  double alpha_presence = 1.0;
  double alpha_frequency = 0.1;
  double alpha_decay = 0.996;
  bool pad_zero = true;
};

class InferenceEngine {
 public:
  using StreamCallback = std::function<bool(const std::string&)>;

  InferenceEngine(
      std::shared_ptr<RWKVModel> model,
      std::shared_ptr<trie_tokenizer> tokenizer,
      std::string model_name);

  void shutdown();

  std::vector<std::string> batch_generate(
      const std::vector<std::string>& prompts,
      const GenerateOptions& options) const;

  std::vector<std::string> continuous_batching(
      const std::vector<std::string>& prompts,
      const GenerateOptions& options) const;

  std::vector<std::string> graph_generate(
      const std::vector<std::string>& prompts,
      const GenerateOptions& options) const;

  std::vector<std::string> batch_generate_state(
      const std::vector<std::string>& prompts,
      RWKVState& state,
      const GenerateOptions& options) const;

  void batch_generate_stream(
      const std::vector<std::string>& prompts,
      const GenerateOptions& options,
      int chunk_size,
      const StreamCallback& emit) const;

  void continuous_batching_stream(
      const std::vector<std::string>& prompts,
      const GenerateOptions& options,
      int chunk_size,
      const StreamCallback& emit) const;

  void graph_generate_stream(
      const std::vector<std::string>& prompts,
      const GenerateOptions& options,
      int chunk_size,
      const StreamCallback& emit) const;

  void batch_generate_state_stream(
      const std::vector<std::string>& prompts,
      RWKVState& state,
      const GenerateOptions& options,
      int chunk_size,
      const StreamCallback& emit) const;

  void big_batch_stream(
      const std::vector<std::string>& prompts,
      int max_tokens,
      double temperature,
      const std::vector<int64_t>& stop_tokens,
      int chunk_size,
      const StreamCallback& emit) const;

  std::string format_openai_prompt(
      const std::string& system,
      const std::vector<std::pair<std::string, std::string>>& messages,
      bool enable_think) const;

  int count_tokens(const std::string& text) const;

  std::shared_ptr<RWKVModel> model() const { return model_; }
  std::shared_ptr<trie_tokenizer> tokenizer() const { return tokenizer_; }
  const std::string& model_name() const { return model_name_; }

 private:
  struct ActiveBatch {
    RWKVState state;
    torch::Tensor penalties;
    std::vector<int> indices;
  };

  std::vector<int64_t> encode_prompt(const std::string& prompt, bool pad_zero) const;
  torch::Tensor forward_variable_batch(
      const std::vector<std::vector<int64_t>>& token_batches,
      RWKVState& state) const;
  ActiveBatch make_active_batch(
      RWKVState&& state,
      torch::Tensor&& penalties,
      std::vector<int>&& indices) const;
  ActiveBatch shrink_active_batch(
      const ActiveBatch& batch,
      const std::vector<int64_t>& keep_slots) const;
  torch::Tensor decode_active(
      const std::vector<int64_t>& token_batch,
      ActiveBatch& batch) const;
  torch::Tensor sample_tokens_with_penalties(
      torch::Tensor logits,
      torch::Tensor& penalties,
      const GenerateOptions& options) const;
  torch::Tensor sample_tokens_gumbel(
      torch::Tensor logits,
      double temperature,
      double eps = 6.2e-5) const;
  bool emit_sse_chunk(
      const std::vector<std::vector<int>>& token_buffers,
      const std::vector<int>& active_indices,
      std::vector<std::vector<int>>& mutable_buffers,
      const StreamCallback& emit) const;
  std::vector<std::string> decode_all(
      const std::vector<std::vector<int>>& generated) const;

  std::shared_ptr<RWKVModel> model_;
  std::shared_ptr<trie_tokenizer> tokenizer_;
  std::string model_name_;
  mutable std::mutex model_mutex_;
};
