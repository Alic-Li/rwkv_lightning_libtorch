#include "inference_engine.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>

#include <ATen/DeviceAccelerator.h>
#include <json/json.h>

#include "sampling_api.h"
#include "state_manager/state_pool.h"

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

bool is_utf8_continuation(unsigned char byte) {
  return (byte & 0xC0u) == 0x80u;
}

int utf8_sequence_length(unsigned char lead) {
  if (lead <= 0x7Fu) {
    return 1;
  }
  if (lead >= 0xC2u && lead <= 0xDFu) {
    return 2;
  }
  if (lead >= 0xE0u && lead <= 0xEFu) {
    return 3;
  }
  if (lead >= 0xF0u && lead <= 0xF4u) {
    return 4;
  }
  return 0;
}

bool is_valid_utf8_sequence(const std::string& text, size_t pos, int len) {
  const auto b0 = static_cast<unsigned char>(text[pos]);
  switch (len) {
    case 1:
      return true;
    case 2: {
      const auto b1 = static_cast<unsigned char>(text[pos + 1]);
      return is_utf8_continuation(b1);
    }
    case 3: {
      const auto b1 = static_cast<unsigned char>(text[pos + 1]);
      const auto b2 = static_cast<unsigned char>(text[pos + 2]);
      if (!is_utf8_continuation(b1) || !is_utf8_continuation(b2)) {
        return false;
      }
      if (b0 == 0xE0u && b1 < 0xA0u) {
        return false;
      }
      if (b0 == 0xEDu && b1 >= 0xA0u) {
        return false;
      }
      return true;
    }
    case 4: {
      const auto b1 = static_cast<unsigned char>(text[pos + 1]);
      const auto b2 = static_cast<unsigned char>(text[pos + 2]);
      const auto b3 = static_cast<unsigned char>(text[pos + 3]);
      if (!is_utf8_continuation(b1) || !is_utf8_continuation(b2) || !is_utf8_continuation(b3)) {
        return false;
      }
      if (b0 == 0xF0u && b1 < 0x90u) {
        return false;
      }
      if (b0 == 0xF4u && b1 > 0x8Fu) {
        return false;
      }
      return true;
    }
    default:
      return false;
  }
}

std::string take_complete_utf8(std::string& pending, bool flush_all) {
  std::string out;
  size_t i = 0;
  size_t last_copy = 0;
  while (i < pending.size()) {
    const auto lead = static_cast<unsigned char>(pending[i]);
    const int len = utf8_sequence_length(lead);
    if (len == 0) {
      out.append(pending, last_copy, i - last_copy);
      out += "\xEF\xBF\xBD";
      ++i;
      last_copy = i;
      continue;
    }
    if (i + static_cast<size_t>(len) > pending.size()) {
      break;
    }
    if (!is_valid_utf8_sequence(pending, i, len)) {
      out.append(pending, last_copy, i - last_copy);
      out += "\xEF\xBF\xBD";
      ++i;
      last_copy = i;
      continue;
    }
    i += static_cast<size_t>(len);
  }
  out.append(pending, last_copy, i - last_copy);
  pending.erase(0, i);
  if (flush_all && !pending.empty()) {
    for (size_t j = 0; j < pending.size(); ++j) {
      out += "\xEF\xBF\xBD";
    }
    pending.clear();
  }
  return out;
}

std::string decode_complete_utf8(
    const trie_tokenizer& tokenizer,
    std::vector<int>& token_buffer,
    std::string& utf8_pending,
    bool flush_all) {
  if (!token_buffer.empty()) {
    utf8_pending += tokenizer.decode(token_buffer);
    token_buffer.clear();
  }
  return take_complete_utf8(utf8_pending, flush_all);
}

void cleanup_device_cache(const c10::Device& device) {
  if (!device.is_cuda() && device.type() != c10::DeviceType::HIP) {
    return;
  }
  try {
    at::accelerator::synchronizeDevice(device.index());
  } catch (...) {
  }
  try {
    at::accelerator::emptyCache();
  } catch (...) {
  }
}

void cleanup_request_tensors(
    std::initializer_list<torch::Tensor*> tensors,
    const c10::Device& device) {
  for (auto* tensor : tensors) {
    if (tensor != nullptr) {
      *tensor = torch::Tensor();
    }
  }
  cleanup_device_cache(device);
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

InferenceEngine::PrefixPrefillResult InferenceEngine::prefill_prompt_with_prefix_cache(
    const std::string& prompt) const {
  auto encoded = encode_prompt(prompt, false);
  if (encoded.empty()) {
    throw std::runtime_error("prompt must not be empty");
  }

  RWKVState state;
  torch::Tensor logits;
  int matched_tokens = 0;

  if (auto match = StateCacheManager::instance().match_prefix_state(encoded); match.has_value()) {
    state = std::move(match->state);
    if (match->logits.has_value()) {
      logits = std::move(*match->logits);
    }
    matched_tokens = match->matched_tokens;
  } else {
    state = model_->generate_zero_state(1);
  }

  int cursor = matched_tokens;
  for (int bucket : {1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192}) {
    if (cursor >= bucket || static_cast<int>(encoded.size()) < bucket) {
      continue;
    }
    std::vector<int64_t> segment(encoded.begin() + cursor, encoded.begin() + bucket);
    if (!segment.empty()) {
      logits = forward_variable_batch({segment}, state);
      StateCacheManager::instance().put_prefix_state(
          std::vector<int64_t>(encoded.begin(), encoded.begin() + bucket),
          state,
          logits);
      cursor = bucket;
    }
  }

  if (cursor < static_cast<int>(encoded.size())) {
    std::vector<int64_t> remaining(encoded.begin() + cursor, encoded.end());
    logits = forward_variable_batch({remaining}, state);
  } else if (!logits.defined()) {
    state = model_->generate_zero_state(1);
    logits = forward_variable_batch({encoded}, state);
  }

  return PrefixPrefillResult{std::move(encoded), std::move(state), std::move(logits)};
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
  const auto device = logits.device();

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

  auto decoded = decode_all(generated);
  active.state = RWKVState();
  active.penalties = torch::Tensor();
  cleanup_request_tensors({&logits}, device);
  return decoded;
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

  auto logits = forward_variable_batch({encode_prompt(prompts.front(), false)}, state);
  auto penalties =
      torch::zeros({1, model_->vocab_size()},
                   torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
  std::set<int64_t> stop_set(options.stop_tokens.begin(), options.stop_tokens.end());
  std::vector<int> generated;
  const auto device = logits.device();

  for (int step = 0; step < options.max_tokens; ++step) {
    auto next = sample_tokens_with_penalties(logits, penalties, options).to(torch::kCPU)[0].item<int64_t>();
    if (stop_set.count(next)) {
      break;
    }
    generated.push_back(static_cast<int>(next));
    penalties.index_put_({0, next}, penalties.index({0, next}) + 1.0f);
    logits = model_->forward_decode({next}, state);
  }

  auto decoded = std::vector<std::string>{tokenizer_->decode(generated)};
  cleanup_request_tensors({&logits, &penalties}, device);
  return decoded;
}

std::string InferenceEngine::single_generate_with_prefix_cache(
    const std::string& prompt,
    const GenerateOptions& options) const {
  auto prefill = prefill_prompt_with_prefix_cache(prompt);
  auto penalties =
      torch::zeros({1, model_->vocab_size()},
                   torch::TensorOptions().dtype(torch::kFloat32).device(prefill.logits.device()));
  std::set<int64_t> stop_set(options.stop_tokens.begin(), options.stop_tokens.end());
  std::vector<int> generated;
  const auto device = prefill.logits.device();

  for (int step = 0; step < options.max_tokens; ++step) {
    auto next =
        sample_tokens_with_penalties(prefill.logits, penalties, options).to(torch::kCPU)[0].item<int64_t>();
    if (stop_set.count(next)) {
      break;
    }
    generated.push_back(static_cast<int>(next));
    penalties.index_put_({0, next}, penalties.index({0, next}) + 1.0f);
    prefill.logits = model_->forward_decode({next}, prefill.state);
  }

  const auto decoded = tokenizer_->decode(generated);
  cleanup_request_tensors({&prefill.logits, &penalties}, device);
  return decoded;
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
  std::vector<std::string> utf8_pending(prompts.size());
  const auto device = logits.device();

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
      auto& pending = utf8_pending[static_cast<size_t>(original_index)];
      if (stop_set.count(token)) {
        const auto text =
            decode_complete_utf8(*tokenizer_, buffer, pending, true);
        if (!text.empty()) {
          choices.emplace_back(original_index, text);
        }
        continue;
      }

      generated[static_cast<size_t>(original_index)].push_back(static_cast<int>(token));
      buffer.push_back(static_cast<int>(token));
      active.penalties.index_put_({slot, token}, active.penalties.index({slot, token}) + 1.0f);
      next_tokens.push_back(token);
      keep_slots.push_back(slot);

      if (static_cast<int>(buffer.size()) >= chunk_size) {
        const auto text =
            decode_complete_utf8(*tokenizer_, buffer, pending, false);
        if (!text.empty()) {
          choices.emplace_back(original_index, text);
        }
      }
    }

    if (!choices.empty() && !emit(make_sse_payload(choices))) {
      active.state = RWKVState();
      active.penalties = torch::Tensor();
      cleanup_request_tensors({&logits}, device);
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
    const auto text =
        decode_complete_utf8(*tokenizer_, token_buffers[i], utf8_pending[i], true);
    if (!text.empty()) {
      tail.emplace_back(static_cast<int>(i), text);
    }
  }
  if (!tail.empty() && !emit(make_sse_payload(tail))) {
    active.state = RWKVState();
    active.penalties = torch::Tensor();
    cleanup_request_tensors({&logits}, device);
    return;
  }
  active.state = RWKVState();
  active.penalties = torch::Tensor();
  cleanup_request_tensors({&logits}, device);
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

  auto logits = forward_variable_batch({encode_prompt(prompts.front(), false)}, state);
  auto penalties =
      torch::zeros({1, model_->vocab_size()},
                   torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
  std::set<int64_t> stop_set(options.stop_tokens.begin(), options.stop_tokens.end());
  std::vector<int> buffer;
  std::string utf8_pending;
  const auto device = logits.device();

  for (int step = 0; step < options.max_tokens; ++step) {
    const auto token = sample_tokens_with_penalties(logits, penalties, options).to(torch::kCPU)[0].item<int64_t>();
    if (stop_set.count(token)) {
      break;
    }
    buffer.push_back(static_cast<int>(token));
    penalties.index_put_({0, token}, penalties.index({0, token}) + 1.0f);
    if (static_cast<int>(buffer.size()) >= chunk_size) {
      const auto text =
          decode_complete_utf8(*tokenizer_, buffer, utf8_pending, false);
      if (!text.empty() && !emit(make_sse_payload({{0, text}}))) {
        cleanup_request_tensors({&logits, &penalties}, device);
        return;
      }
    }
    logits = model_->forward_decode({token}, state);
  }

  const auto tail = decode_complete_utf8(*tokenizer_, buffer, utf8_pending, true);
  if (!tail.empty() && !emit(make_sse_payload({{0, tail}}))) {
    cleanup_request_tensors({&logits, &penalties}, device);
    return;
  }
  cleanup_request_tensors({&logits, &penalties}, device);
  emit("data: [DONE]\n\n");
}

void InferenceEngine::single_generate_stream_with_prefix_cache(
    const std::string& prompt,
    const GenerateOptions& options,
    int chunk_size,
    const StreamCallback& emit) const {
  auto prefill = prefill_prompt_with_prefix_cache(prompt);
  auto penalties =
      torch::zeros({1, model_->vocab_size()},
                   torch::TensorOptions().dtype(torch::kFloat32).device(prefill.logits.device()));
  std::set<int64_t> stop_set(options.stop_tokens.begin(), options.stop_tokens.end());
  std::vector<int> buffer;
  std::string utf8_pending;
  const auto device = prefill.logits.device();

  for (int step = 0; step < options.max_tokens; ++step) {
    const auto token =
        sample_tokens_with_penalties(prefill.logits, penalties, options).to(torch::kCPU)[0].item<int64_t>();
    if (stop_set.count(token)) {
      break;
    }
    buffer.push_back(static_cast<int>(token));
    penalties.index_put_({0, token}, penalties.index({0, token}) + 1.0f);
    if (static_cast<int>(buffer.size()) >= chunk_size) {
      const auto text = decode_complete_utf8(*tokenizer_, buffer, utf8_pending, false);
      if (!text.empty() && !emit(make_sse_payload({{0, text}}))) {
        cleanup_request_tensors({&prefill.logits, &penalties}, device);
        return;
      }
    }
    prefill.logits = model_->forward_decode({token}, prefill.state);
  }

  const auto tail = decode_complete_utf8(*tokenizer_, buffer, utf8_pending, true);
  if (!tail.empty() && !emit(make_sse_payload({{0, tail}}))) {
    cleanup_request_tensors({&prefill.logits, &penalties}, device);
    return;
  }
  cleanup_request_tensors({&prefill.logits, &penalties}, device);
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

  auto state = model_->generate_zero_state(static_cast<int64_t>(prompts.size()));
  std::vector<std::vector<int64_t>> encoded;
  encoded.reserve(prompts.size());
  for (const auto& prompt : prompts) {
    encoded.push_back(encode_prompt(prompt, true));
  }

  auto logits = forward_variable_batch(encoded, state);
  std::set<int64_t> stop_set(stop_tokens.begin(), stop_tokens.end());
  std::vector<std::vector<int>> token_buffers(prompts.size());
  std::vector<std::string> utf8_pending(prompts.size());
  std::vector<bool> finished(prompts.size(), false);
  const auto device = logits.device();
  int step_count = 0;

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
        const auto text =
            decode_complete_utf8(*tokenizer_, token_buffers[i], utf8_pending[i], true);
        if (!text.empty()) {
          choices.emplace_back(static_cast<int>(i), text);
        }
        continue;
      }
      token_buffers[i].push_back(static_cast<int>(token));
      if (static_cast<int>(token_buffers[i].size()) >= chunk_size) {
        const auto text =
            decode_complete_utf8(*tokenizer_, token_buffers[i], utf8_pending[i], false);
        if (!text.empty()) {
          choices.emplace_back(static_cast<int>(i), text);
        }
      }
    }

    if (!choices.empty() && !emit(make_sse_payload(choices))) {
      state = RWKVState();
      cleanup_request_tensors({&logits}, device);
      return;
    }
    logits = model_->forward_decode(next_tokens, state);
    ++step_count;
    if (step_count % 100 == 0) {
      cleanup_device_cache(device);
    }
  }

  std::vector<std::pair<int, std::string>> tail;
  for (size_t i = 0; i < token_buffers.size(); ++i) {
    const auto text =
        decode_complete_utf8(*tokenizer_, token_buffers[i], utf8_pending[i], true);
    if (!text.empty()) {
      tail.emplace_back(static_cast<int>(i), text);
    }
  }
  if (!tail.empty() && !emit(make_sse_payload(tail))) {
    state = RWKVState();
    cleanup_request_tensors({&logits}, device);
    return;
  }
  state = RWKVState();
  cleanup_request_tensors({&logits}, device);
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
