#include "inference_engine.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

#include <json/json.h>

#include "sampling_api.h"

namespace {

int64_t make_seed() {
  static thread_local std::mt19937_64 rng(
      static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
  return static_cast<int64_t>(rng());
}

std::string make_sse_payload(const std::vector<std::pair<int, std::string>>& choices) {
  Json::Value chunk;
  chunk["object"] = "chat.completion.chunk";
  chunk["choices"] = Json::arrayValue;
  for (const auto& [index, content] : choices) {
    if (content.empty()) {
      continue;
    }
    Json::Value choice;
    choice["index"] = index;
    choice["delta"]["content"] = content;
    chunk["choices"].append(choice);
  }
  Json::StreamWriterBuilder builder;
  builder["emitUTF8"] = true;
  builder["indentation"] = "";
  return "data: " + Json::writeString(builder, chunk) + "\n\n";
}

}  // namespace

InferenceEngine::InferenceEngine(
    std::shared_ptr<RWKVModel> model,
    std::shared_ptr<trie_tokenizer> tokenizer,
    std::string model_name)
    : model_(std::move(model)),
      tokenizer_(std::move(tokenizer)),
      model_name_(std::move(model_name)) {}

void InferenceEngine::shutdown() {}

std::vector<int64_t> InferenceEngine::encode_prompt(const std::string& prompt, bool pad_zero) const {
  auto ids = tokenizer_->encode(prompt);
  std::vector<int64_t> out(ids.begin(), ids.end());
  if (pad_zero) {
    out.insert(out.begin(), 0);
  }
  if (out.empty()) {
    out.push_back(0);
  }
  return out;
}

torch::Tensor InferenceEngine::forward_variable_batch(
    const std::vector<std::vector<int64_t>>& token_batches,
    RWKVState& state) const {
  if (token_batches.empty()) {
    throw std::runtime_error("token_batches must not be empty");
  }

  const auto batch_size = static_cast<int64_t>(token_batches.size());
  std::vector<int64_t> lengths;
  lengths.reserve(token_batches.size());
  for (const auto& row : token_batches) {
    lengths.push_back(static_cast<int64_t>(row.size()));
  }

  const bool same_length =
      std::all_of(lengths.begin(), lengths.end(), [&](int64_t len) { return len == lengths.front(); });
  if (same_length) {
    return model_->forward_prefill(token_batches, state);
  }

  const auto device = state.x_prev.device();
  auto out = torch::empty(
      {batch_size, model_->vocab_size()},
      torch::TensorOptions().dtype(torch::kFloat16).device(device));

  std::vector<int64_t> pos(token_batches.size(), 0);
  while (true) {
    std::vector<int64_t> active_slots;
    active_slots.reserve(token_batches.size());
    int64_t step = std::numeric_limits<int64_t>::max();
    for (size_t i = 0; i < token_batches.size(); ++i) {
      const auto remain = lengths[i] - pos[i];
      if (remain <= 0) {
        continue;
      }
      active_slots.push_back(static_cast<int64_t>(i));
      step = std::min(step, remain);
    }
    if (active_slots.empty()) {
      break;
    }

    std::vector<std::vector<int64_t>> active_tokens;
    active_tokens.reserve(active_slots.size());
    for (auto slot : active_slots) {
      const auto begin = token_batches[slot].begin() + pos[slot];
      active_tokens.emplace_back(begin, begin + step);
    }

    auto index = torch::tensor(active_slots, torch::TensorOptions().dtype(torch::kLong).device(device));
    RWKVState substate{
        state.x_prev.index_select(2, index).contiguous(),
        state.att_state.index_select(1, index).contiguous(),
        state.elapsed_t.index_select(0, index).contiguous()};
    auto subout = model_->forward_prefill(active_tokens, substate);

    out.index_copy_(0, index, subout);
    state.x_prev.index_copy_(2, index, substate.x_prev);
    state.att_state.index_copy_(1, index, substate.att_state);
    state.elapsed_t.index_copy_(0, index, substate.elapsed_t);

    for (auto slot : active_slots) {
      pos[slot] += step;
    }
  }

  return out;
}

InferenceEngine::ActiveBatch InferenceEngine::make_active_batch(
    RWKVState&& state,
    torch::Tensor&& penalties,
    std::vector<int>&& indices) const {
  return ActiveBatch{std::move(state), std::move(penalties), std::move(indices)};
}

InferenceEngine::ActiveBatch InferenceEngine::shrink_active_batch(
    const ActiveBatch& batch,
    const std::vector<int64_t>& keep_slots) const {
  const auto device = batch.state.x_prev.device();
  auto index = torch::tensor(keep_slots, torch::TensorOptions().dtype(torch::kLong).device(device));
  RWKVState next_state{
      batch.state.x_prev.index_select(2, index).contiguous(),
      batch.state.att_state.index_select(1, index).contiguous(),
      batch.state.elapsed_t.index_select(0, index).contiguous()};
  auto next_penalties = batch.penalties.index_select(0, index).contiguous();
  std::vector<int> next_indices;
  next_indices.reserve(keep_slots.size());
  for (auto slot : keep_slots) {
    next_indices.push_back(batch.indices[static_cast<size_t>(slot)]);
  }
  return make_active_batch(std::move(next_state), std::move(next_penalties), std::move(next_indices));
}

torch::Tensor InferenceEngine::decode_active(
    const std::vector<int64_t>& token_batch,
    ActiveBatch& batch) const {
  return model_->forward_decode(token_batch, batch.state);
}

torch::Tensor InferenceEngine::sample_tokens_with_penalties(
    torch::Tensor logits,
    torch::Tensor& penalties,
    const GenerateOptions& options) const {
  logits = logits.to(torch::kFloat32).contiguous();
  if (options.temperature <= 0.0) {
    return std::get<1>(logits.max(-1, false));
  }

  const bool can_use_kernel =
      logits.is_cuda() && logits.size(-1) % 4 == 0 && logits.size(-1) <= 1048576;
  if (can_use_kernel) {
    auto rand_states = setup_rand(make_seed(), logits.size(0));
    return batch_sampling_repetition_temperature_topk_topp(
               logits,
               penalties,
               rand_states,
               options.alpha_presence,
               options.alpha_frequency,
               options.alpha_decay,
               options.temperature,
               options.top_k,
               options.top_p)
        .to(torch::kLong);
  }

  auto adjusted = logits.clone();
  if (options.alpha_presence != 0.0 || options.alpha_frequency != 0.0) {
    auto mask = penalties > 0;
    adjusted = adjusted - mask.to(adjusted.dtype()) * options.alpha_presence;
    adjusted = adjusted - penalties * options.alpha_frequency;
    penalties.mul_(options.alpha_decay);
  }
  if (options.temperature != 1.0) {
    adjusted = adjusted / options.temperature;
  }
  if (options.top_k > 0) {
    const auto k = std::min<int64_t>(options.top_k, adjusted.size(-1));
    auto kth = std::get<0>(torch::topk(adjusted, k, -1)).select(-1, k - 1).unsqueeze(-1);
    adjusted = adjusted.masked_fill(adjusted < kth, -std::numeric_limits<float>::infinity());
  }
  if (options.top_p > 0.0 && options.top_p < 1.0) {
    auto sorted = torch::sort(adjusted, -1, true);
    auto sorted_logits = std::get<0>(sorted);
    auto sorted_indices = std::get<1>(sorted);
    auto cumulative = torch::cumsum(torch::softmax(sorted_logits, -1), -1);
    auto remove = cumulative > options.top_p;
    remove.index_put_({"...", 0}, false);
    auto scatter = torch::zeros_like(remove).scatter(-1, sorted_indices, remove);
    adjusted = adjusted.masked_fill(scatter, -std::numeric_limits<float>::infinity());
  }
  return torch::multinomial(torch::softmax(adjusted, -1), 1).squeeze(-1);
}

torch::Tensor InferenceEngine::sample_tokens_gumbel(
    torch::Tensor logits,
    double temperature,
    double eps) const {
  logits = logits.to(torch::kFloat32).contiguous();
  if (temperature > 0.0) {
    if (temperature != 1.0) {
      logits.mul_(1.0 / temperature);
    }
    auto noise = torch::empty_like(logits).exponential_().clamp_min_(eps).log_().neg_();
    logits.add_(noise);
  }
  return std::get<1>(logits.max(-1, false));
}

bool InferenceEngine::emit_sse_chunk(
    const std::vector<std::vector<int>>& token_buffers,
    const std::vector<int>& active_indices,
    std::vector<std::vector<int>>& mutable_buffers,
    const StreamCallback& emit) const {
  std::vector<std::pair<int, std::string>> choices;
  for (size_t slot = 0; slot < token_buffers.size(); ++slot) {
    if (token_buffers[slot].empty()) {
      continue;
    }
    choices.emplace_back(
        active_indices[slot],
        tokenizer_->decode(mutable_buffers[slot]));
    mutable_buffers[slot].clear();
  }
  if (choices.empty()) {
    return true;
  }
  return emit(make_sse_payload(choices));
}

std::vector<std::string> InferenceEngine::decode_all(
    const std::vector<std::vector<int>>& generated) const {
  std::vector<std::string> decoded;
  decoded.reserve(generated.size());
  for (const auto& ids : generated) {
    decoded.push_back(tokenizer_->decode(ids));
  }
  return decoded;
}

std::vector<std::string> InferenceEngine::batch_generate(
    const std::vector<std::string>& prompts,
    const GenerateOptions& options) const {
  if (prompts.empty()) {
    return {};
  }

  std::lock_guard<std::mutex> lock(model_mutex_);
  auto state = model_->generate_zero_state(static_cast<int64_t>(prompts.size()));
  std::vector<std::vector<int64_t>> encoded;
  encoded.reserve(prompts.size());
  for (const auto& prompt : prompts) {
    encoded.push_back(encode_prompt(prompt, options.pad_zero));
  }

  auto logits = forward_variable_batch(encoded, state);
  auto penalties =
      torch::zeros({static_cast<int64_t>(prompts.size()), model_->vocab_size()},
                   torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
  ActiveBatch active = make_active_batch(
      std::move(state),
      std::move(penalties),
      [&]() {
        std::vector<int> idx(prompts.size());
        std::iota(idx.begin(), idx.end(), 0);
        return idx;
      }());

  std::vector<std::vector<int>> generated(prompts.size());
  std::set<int64_t> stop_set(options.stop_tokens.begin(), options.stop_tokens.end());

  for (int step = 0; step < options.max_tokens && !active.indices.empty(); ++step) {
    auto sampled = sample_tokens_with_penalties(logits, active.penalties, options).to(torch::kCPU);
    std::vector<int64_t> next_tokens;
    std::vector<int64_t> keep_slots;
    next_tokens.reserve(active.indices.size());
    keep_slots.reserve(active.indices.size());

    for (int64_t slot = 0; slot < sampled.size(0); ++slot) {
      const auto token = sampled[slot].item<int64_t>();
      const auto original_index = active.indices[static_cast<size_t>(slot)];
      if (!stop_set.count(token)) {
        generated[static_cast<size_t>(original_index)].push_back(static_cast<int>(token));
        next_tokens.push_back(token);
        keep_slots.push_back(slot);
        active.penalties.index_put_({slot, token}, active.penalties.index({slot, token}) + 1.0f);
      }
    }

    if (keep_slots.empty()) {
      break;
    }

    if (static_cast<size_t>(keep_slots.size()) != active.indices.size()) {
      active = shrink_active_batch(active, keep_slots);
    }
    logits = decode_active(next_tokens, active);
  }

  return decode_all(generated);
}

std::vector<std::string> InferenceEngine::continuous_batching(
    const std::vector<std::string>& prompts,
    const GenerateOptions& options) const {
  return batch_generate(prompts, options);
}

std::vector<std::string> InferenceEngine::graph_generate(
    const std::vector<std::string>& prompts,
    const GenerateOptions& options) const {
  return batch_generate(prompts, options);
}

std::vector<std::string> InferenceEngine::batch_generate_state(
    const std::vector<std::string>& prompts,
    RWKVState& state,
    const GenerateOptions& options) const {
  if (prompts.empty()) {
    return {};
  }

  std::lock_guard<std::mutex> lock(model_mutex_);
  auto logits = forward_variable_batch({encode_prompt(prompts.front(), false)}, state);
  auto penalties =
      torch::zeros({1, model_->vocab_size()},
                   torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
  std::set<int64_t> stop_set(options.stop_tokens.begin(), options.stop_tokens.end());
  std::vector<int> generated;

  for (int step = 0; step < options.max_tokens; ++step) {
    auto next = sample_tokens_with_penalties(logits, penalties, options).to(torch::kCPU)[0].item<int64_t>();
    if (stop_set.count(next)) {
      break;
    }
    generated.push_back(static_cast<int>(next));
    penalties.index_put_({0, next}, penalties.index({0, next}) + 1.0f);
    logits = model_->forward_decode({next}, state);
  }

  return {tokenizer_->decode(generated)};
}

void InferenceEngine::batch_generate_stream(
    const std::vector<std::string>& prompts,
    const GenerateOptions& options,
    int chunk_size,
    const StreamCallback& emit) const {
  if (prompts.empty()) {
    emit("data: [DONE]\n\n");
    return;
  }

  std::lock_guard<std::mutex> lock(model_mutex_);
  auto state = model_->generate_zero_state(static_cast<int64_t>(prompts.size()));
  std::vector<std::vector<int64_t>> encoded;
  encoded.reserve(prompts.size());
  for (const auto& prompt : prompts) {
    encoded.push_back(encode_prompt(prompt, options.pad_zero));
  }

  auto logits = forward_variable_batch(encoded, state);
  auto penalties =
      torch::zeros({static_cast<int64_t>(prompts.size()), model_->vocab_size()},
                   torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
  ActiveBatch active = make_active_batch(
      std::move(state),
      std::move(penalties),
      [&]() {
        std::vector<int> idx(prompts.size());
        std::iota(idx.begin(), idx.end(), 0);
        return idx;
      }());

  std::set<int64_t> stop_set(options.stop_tokens.begin(), options.stop_tokens.end());
  std::vector<std::vector<int>> generated(prompts.size());
  std::vector<std::vector<int>> token_buffers(prompts.size());

  for (int step = 0; step < options.max_tokens && !active.indices.empty(); ++step) {
    auto sampled = sample_tokens_with_penalties(logits, active.penalties, options).to(torch::kCPU);
    std::vector<int64_t> next_tokens;
    std::vector<int64_t> keep_slots;
    next_tokens.reserve(active.indices.size());
    keep_slots.reserve(active.indices.size());
    std::vector<std::pair<int, std::string>> choices;

    for (int64_t slot = 0; slot < sampled.size(0); ++slot) {
      const auto token = sampled[slot].item<int64_t>();
      const auto original_index = active.indices[static_cast<size_t>(slot)];
      auto& buffer = token_buffers[static_cast<size_t>(original_index)];
      if (stop_set.count(token)) {
        if (!buffer.empty()) {
          choices.emplace_back(original_index, tokenizer_->decode(buffer));
          buffer.clear();
        }
        continue;
      }

      generated[static_cast<size_t>(original_index)].push_back(static_cast<int>(token));
      buffer.push_back(static_cast<int>(token));
      active.penalties.index_put_({slot, token}, active.penalties.index({slot, token}) + 1.0f);
      next_tokens.push_back(token);
      keep_slots.push_back(slot);

      if (static_cast<int>(buffer.size()) >= chunk_size) {
        choices.emplace_back(original_index, tokenizer_->decode(buffer));
        buffer.clear();
      }
    }

    if (!choices.empty() && !emit(make_sse_payload(choices))) {
      return;
    }
    if (keep_slots.empty()) {
      break;
    }
    if (static_cast<size_t>(keep_slots.size()) != active.indices.size()) {
      active = shrink_active_batch(active, keep_slots);
    }
    logits = decode_active(next_tokens, active);
  }

  std::vector<std::pair<int, std::string>> tail;
  for (size_t i = 0; i < token_buffers.size(); ++i) {
    if (!token_buffers[i].empty()) {
      tail.emplace_back(static_cast<int>(i), tokenizer_->decode(token_buffers[i]));
    }
  }
  if (!tail.empty() && !emit(make_sse_payload(tail))) {
    return;
  }
  emit("data: [DONE]\n\n");
}

void InferenceEngine::continuous_batching_stream(
    const std::vector<std::string>& prompts,
    const GenerateOptions& options,
    int chunk_size,
    const StreamCallback& emit) const {
  batch_generate_stream(prompts, options, chunk_size, emit);
}

void InferenceEngine::graph_generate_stream(
    const std::vector<std::string>& prompts,
    const GenerateOptions& options,
    int chunk_size,
    const StreamCallback& emit) const {
  batch_generate_stream(prompts, options, chunk_size, emit);
}

void InferenceEngine::batch_generate_state_stream(
    const std::vector<std::string>& prompts,
    RWKVState& state,
    const GenerateOptions& options,
    int chunk_size,
    const StreamCallback& emit) const {
  if (prompts.empty()) {
    emit("data: [DONE]\n\n");
    return;
  }

  std::lock_guard<std::mutex> lock(model_mutex_);
  auto logits = forward_variable_batch({encode_prompt(prompts.front(), false)}, state);
  auto penalties =
      torch::zeros({1, model_->vocab_size()},
                   torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
  std::set<int64_t> stop_set(options.stop_tokens.begin(), options.stop_tokens.end());
  std::vector<int> buffer;

  for (int step = 0; step < options.max_tokens; ++step) {
    const auto token = sample_tokens_with_penalties(logits, penalties, options).to(torch::kCPU)[0].item<int64_t>();
    if (stop_set.count(token)) {
      break;
    }
    buffer.push_back(static_cast<int>(token));
    penalties.index_put_({0, token}, penalties.index({0, token}) + 1.0f);
    if (static_cast<int>(buffer.size()) >= chunk_size) {
      if (!emit(make_sse_payload({{0, tokenizer_->decode(buffer)}}))) {
        return;
      }
      buffer.clear();
    }
    logits = model_->forward_decode({token}, state);
  }

  if (!buffer.empty() && !emit(make_sse_payload({{0, tokenizer_->decode(buffer)}}))) {
    return;
  }
  emit("data: [DONE]\n\n");
}

void InferenceEngine::big_batch_stream(
    const std::vector<std::string>& prompts,
    int max_tokens,
    double temperature,
    const std::vector<int64_t>& stop_tokens,
    int chunk_size,
    const StreamCallback& emit) const {
  if (prompts.empty()) {
    emit("data: [DONE]\n\n");
    return;
  }

  std::lock_guard<std::mutex> lock(model_mutex_);
  auto state = model_->generate_zero_state(static_cast<int64_t>(prompts.size()));
  std::vector<std::vector<int64_t>> encoded;
  encoded.reserve(prompts.size());
  for (const auto& prompt : prompts) {
    encoded.push_back(encode_prompt(prompt, true));
  }

  auto logits = forward_variable_batch(encoded, state);
  std::set<int64_t> stop_set(stop_tokens.begin(), stop_tokens.end());
  std::vector<std::vector<int>> token_buffers(prompts.size());
  std::vector<bool> finished(prompts.size(), false);

  while (max_tokens-- > 0 && std::any_of(finished.begin(), finished.end(), [](bool v) { return !v; })) {
    auto sampled = sample_tokens_gumbel(logits, temperature).to(torch::kCPU);
    std::vector<int64_t> next_tokens;
    next_tokens.reserve(prompts.size());
    std::vector<std::pair<int, std::string>> choices;

    for (size_t i = 0; i < prompts.size(); ++i) {
      const auto token = sampled[static_cast<int64_t>(i)].item<int64_t>();
      next_tokens.push_back(token);
      if (finished[i]) {
        continue;
      }
      if (stop_set.count(token)) {
        finished[i] = true;
        if (!token_buffers[i].empty()) {
          choices.emplace_back(static_cast<int>(i), tokenizer_->decode(token_buffers[i]));
          token_buffers[i].clear();
        }
        continue;
      }
      token_buffers[i].push_back(static_cast<int>(token));
      if (static_cast<int>(token_buffers[i].size()) >= chunk_size) {
        choices.emplace_back(static_cast<int>(i), tokenizer_->decode(token_buffers[i]));
        token_buffers[i].clear();
      }
    }

    if (!choices.empty() && !emit(make_sse_payload(choices))) {
      return;
    }
    logits = model_->forward_decode(next_tokens, state);
  }

  std::vector<std::pair<int, std::string>> tail;
  for (size_t i = 0; i < token_buffers.size(); ++i) {
    if (!token_buffers[i].empty()) {
      tail.emplace_back(static_cast<int>(i), tokenizer_->decode(token_buffers[i]));
    }
  }
  if (!tail.empty() && !emit(make_sse_payload(tail))) {
    return;
  }
  emit("data: [DONE]\n\n");
}

std::string InferenceEngine::format_openai_prompt(
    const std::string& system,
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool enable_think) const {
  std::string prompt;
  if (!system.empty()) {
    prompt += "System: " + system + "\n\n";
  }
  for (const auto& [role, content] : messages) {
    if (content.empty()) {
      continue;
    }
    prompt += role + ": " + content + "\n\n";
  }
  prompt += enable_think ? "Assistant: <think" : "Assistant: <think>\n</think>";
  return prompt;
}

int InferenceEngine::count_tokens(const std::string& text) const {
  return static_cast<int>(tokenizer_->encode(text).size());
}
