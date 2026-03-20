#include <atomic>
#include <csignal>
#include <iostream>
#include <optional>
#include <string>

#include <drogon/drogon.h>

#include "API_servers/api_service.h"
#include "infer/inference_engine.h"
#include "model_load/model_loader.h"
#include "state_manager/state_pool.h"

namespace {
std::atomic<bool> g_shutdown{false};

void handle_signal(int) {
  g_shutdown = true;
  try {
    StateCacheManager::instance().shutdown();
  } catch (...) {
  }
  drogon::app().quit();
}
}  // namespace

int main(int argc, char* argv[]) {
  std::string model_path;
  std::string vocab_path;
  uint16_t port = 8000;
  std::optional<std::string> password;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };
    if (arg == "--model-path") {
      model_path = require_value(arg);
    } else if (arg == "--vocab-path") {
      vocab_path = require_value(arg);
    } else if (arg == "--port") {
      port = static_cast<uint16_t>(std::stoi(require_value(arg)));
    } else if (arg == "--password") {
      password = require_value(arg);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (model_path.empty()) {
    throw std::runtime_error("--model-path is required");
  }
  if (vocab_path.empty()) {
    throw std::runtime_error("--vocab-path is required");
  }

  auto ctx = load_model_and_tokenizer(model_path, vocab_path);
  InferenceEngine engine(ctx.model, ctx.tokenizer, ctx.model_name);
  StateCacheManager::instance().initialize(torch::Device(torch::kCUDA, 0));

  std::signal(SIGINT, handle_signal);
  std::signal(SIGTERM, handle_signal);

  register_api_routes(engine, password);

  drogon::app()
      .addListener("0.0.0.0", port)
      .setThreadNum(4)
      .run();

  StateCacheManager::instance().shutdown();
  return 0;
}
