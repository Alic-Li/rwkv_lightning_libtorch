#include "rwkv_model.h"
#include "utils/tokenizer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <torch/cuda.h>

namespace {

void xprint(const std::string& s) {
  const int c0 = 3;
  const int c1 = std::max(0, 80 - static_cast<int>(s.size()) - 3);
  std::cout << "\n" << std::string(c0, '#') << " " << s << " "
            << std::string(c1, '#') << "\n\n";
}

std::vector<int64_t> parse_ids(const std::string& text) {
  std::vector<int64_t> ids;
  if (text.empty()) {
    return ids;
  }
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      ids.push_back(std::stoll(item));
    }
  }
  return ids;
}

std::vector<std::vector<int64_t>> parse_batch_prompts(
    const std::string& text,
    int64_t batch_size) {
  std::vector<std::vector<int64_t>> prompts;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ';')) {
    if (!item.empty()) {
      prompts.push_back(parse_ids(item));
    }
  }
  if (prompts.empty()) {
    return prompts;
  }
  if (prompts.size() == 1 && batch_size > 1) {
    prompts.resize(batch_size, prompts.front());
  }
  return prompts;
}

int64_t greedy_sample(const torch::Tensor& logits_row) {
  return logits_row.argmax(-1).item<int64_t>();
}

torch::Tensor sampler_gumbel_batch(
    torch::Tensor logits,
    double temp = 1.0,
    double eps = 6.2e-5) {
  torch::NoGradGuard guard;
  if (temp > 0.0) {
    if (temp != 1.0) {
      logits.mul_(1.0 / temp);
    }
    auto noise = torch::empty_like(logits)
                     .exponential_()
                     .clamp_min_(eps)
                     .log_()
                     .neg_();
    logits.add_(noise);
  }
  return std::get<1>(torch::max(logits, -1, true));
}

std::vector<std::string> split_strings(const std::string& text, char delim) {
  std::vector<std::string> out;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, delim)) {
    out.push_back(item);
  }
  return out;
}

std::vector<int64_t> to_i64(const std::vector<int>& ids) {
  return std::vector<int64_t>(ids.begin(), ids.end());
}

void require_arg(bool cond, const std::string& msg) {
  if (!cond) {
    throw std::runtime_error(msg);
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string weights_file;
  std::string vocab_file;
  std::string decode_prompt_ids;
  std::string decode_prompt_text;
  int64_t decode_steps = 32;
  double decode_temp = 1.0;
  std::string batch_prompts_arg;
  std::string batch_prompts_text;
  int64_t batch_steps = 32;
  double batch_temp = 1.2;
  int64_t batch_size = 0;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next_value = [&](const std::string& name) -> std::string {
      require_arg(i + 1 < argc, "missing value for " + name);
      return argv[++i];
    };

    if (arg == "--weights") {
      weights_file = next_value(arg);
    } else if (arg == "--vocab") {
      vocab_file = next_value(arg);
    } else if (arg == "--decode-prompt-ids") {
      decode_prompt_ids = next_value(arg);
    } else if (arg == "--decode-prompt") {
      decode_prompt_text = next_value(arg);
    } else if (arg == "--decode-steps") {
      decode_steps = std::stoll(next_value(arg));
    } else if (arg == "--decode-temp") {
      decode_temp = std::stod(next_value(arg));
    } else if (arg == "--batch-prompts") {
      batch_prompts_arg = next_value(arg);
    } else if (arg == "--batch-prompts-text") {
      batch_prompts_text = next_value(arg);
    } else if (arg == "--batch-steps") {
      batch_steps = std::stoll(next_value(arg));
    } else if (arg == "--batch-temp") {
      batch_temp = std::stod(next_value(arg));
    } else if (arg == "--batch-size") {
      batch_size = std::stoll(next_value(arg));
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  require_arg(!weights_file.empty(), "--weights is required");
  require_arg(!vocab_file.empty(), "--vocab is required");
  require_arg(torch::cuda::is_available(), "CUDA is required");

  RWKVModel model(weights_file, torch::Device(torch::kCUDA, 0));
  trie_tokenizer tokenizer;
  require_arg(
      tokenizer.load(vocab_file) == RWKV_SUCCESS,
      "failed to load vocab: " + vocab_file);

  if (!decode_prompt_ids.empty() || !decode_prompt_text.empty()) {
    xprint("Decode");
    require_arg(
        decode_prompt_ids.empty() || decode_prompt_text.empty(),
        "use only one of --decode-prompt-ids or --decode-prompt");
    auto prompt = !decode_prompt_text.empty() ? to_i64(tokenizer.encode(decode_prompt_text))
                                              : parse_ids(decode_prompt_ids);
    require_arg(!prompt.empty(), "decode prompt must not be empty");

    if (!decode_prompt_text.empty()) {
      std::cout << decode_prompt_text;
    } else {
      std::cout << "prompt token ids:";
      for (auto id : prompt) {
        std::cout << " " << id;
      }
      std::cout << "\n";
    }

    auto state = model.generate_zero_state(1);
    auto logits = model.forward_prefill({prompt}, state);

    std::vector<int64_t> generated;
    generated.reserve(decode_steps);
    std::string decoded;
    std::vector<double> times;
    std::vector<double> all_times;
    times.reserve(decode_steps);
    all_times.reserve(decode_steps);

    auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < decode_steps; ++i) {
      auto t00 = std::chrono::steady_clock::now();
      auto next = sampler_gumbel_batch(logits, decode_temp)[0][0].item<int64_t>();
      generated.push_back(next);
      auto piece = tokenizer.decode(static_cast<int>(next));
      decoded += piece;
      std::cout << piece << std::flush;
      torch::cuda::synchronize();
      auto tf0 = std::chrono::steady_clock::now();
      logits = model.forward_decode({next}, state);
      torch::cuda::synchronize();
      auto tf1 = std::chrono::steady_clock::now();
      times.push_back(
          std::chrono::duration_cast<std::chrono::duration<double>>(tf1 - tf0)
              .count());
      all_times.push_back(
          std::chrono::duration_cast<std::chrono::duration<double>>(tf1 - t00)
              .count());
    }
    torch::cuda::synchronize();
    auto t1 = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
            .count();
    std::sort(times.begin(), times.end());
    std::sort(all_times.begin(), all_times.end());
    const double forward_p50 = times[times.size() / 2];
    const double full_p50 = all_times[all_times.size() / 2];

    // std::cout << "generated token ids:";
    // for (auto id : generated) {
    //   std::cout << " " << id;
    // }
    std::cout << "\n";
    std::cout << "Token/s = " << (1.0 / forward_p50)
              << " (forward), " << (1.0 / full_p50)
              << " (full) || total " << elapsed << "s\n";
  }

  if (!batch_prompts_arg.empty() || !batch_prompts_text.empty() || batch_size > 0) {
    xprint("Decode (batch)");
    require_arg(
        batch_prompts_arg.empty() || batch_prompts_text.empty(),
        "use only one of --batch-prompts or --batch-prompts-text");
    std::vector<std::vector<int64_t>> prompts;
    if (!batch_prompts_text.empty()) {
      for (const auto& s : split_strings(batch_prompts_text, ';')) {
        prompts.push_back(to_i64(tokenizer.encode(s)));
      }
      if (prompts.size() == 1 && batch_size > 1) {
        prompts.resize(batch_size, prompts.front());
      }
    } else {
      prompts = parse_batch_prompts(batch_prompts_arg, batch_size);
    }
    require_arg(!prompts.empty(), "batch prompts must not be empty");
    if (batch_size == 0) {
      batch_size = static_cast<int64_t>(prompts.size());
    }
    require_arg(
        static_cast<int64_t>(prompts.size()) == batch_size,
        "batch prompt count must match --batch-size");

    auto state = model.generate_zero_state(batch_size);
    auto logits = model.forward_prefill(prompts, state);
    std::vector<std::vector<int64_t>> generated(batch_size);
    std::vector<double> times;
    std::vector<double> all_times;
    times.reserve(batch_steps);
    all_times.reserve(batch_steps);

    auto t0 = std::chrono::steady_clock::now();
    for (int64_t step = 0; step < batch_steps; ++step) {
      auto t00 = std::chrono::steady_clock::now();
      auto next_tensor = sampler_gumbel_batch(logits, batch_temp)
                             .to(torch::kCPU)
                             .view({batch_size});
      std::vector<int64_t> next_tokens(batch_size);
      for (int64_t b = 0; b < batch_size; ++b) {
        next_tokens[b] = next_tensor[b].item<int64_t>();
        generated[b].push_back(next_tokens[b]);
      }
      torch::cuda::synchronize();
      auto tf0 = std::chrono::steady_clock::now();
      logits = model.forward_decode(next_tokens, state);
      torch::cuda::synchronize();
      auto tf1 = std::chrono::steady_clock::now();
      times.push_back(
          std::chrono::duration_cast<std::chrono::duration<double>>(tf1 - tf0)
              .count());
      all_times.push_back(
          std::chrono::duration_cast<std::chrono::duration<double>>(tf1 - t00)
              .count());
    }
    torch::cuda::synchronize();
    auto t1 = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
            .count();
    std::sort(times.begin(), times.end());
    std::sort(all_times.begin(), all_times.end());
    const double forward_p50 = times[times.size() / 2];
    const double full_p50 = all_times[all_times.size() / 2];

    std::cout << "batch_size=" << batch_size
              << " batch_steps=" << batch_steps
              << " || Token/s = " << (batch_size / forward_p50)
              << " (forward), " << (batch_size / full_p50)
              << " (full) || total " << elapsed << "s\n";

    for (int64_t b = 0; b < std::min<int64_t>(batch_size, 4); ++b) {
      std::vector<int> ids;
      ids.reserve(generated[b].size());
      for (auto id : generated[b]) {
        ids.push_back(static_cast<int>(id));
      }
      std::cout << "sample[" << b << "] text: " << tokenizer.decode(ids) << "\n";
    }
  }

  return 0;
}
