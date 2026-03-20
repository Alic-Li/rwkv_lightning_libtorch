#include "model_loader.h"

#include <filesystem>
#include <iostream>
#include <regex>
#include <stdexcept>

LoadedModelContext load_model_and_tokenizer(const std::string& model_path) {
  namespace fs = std::filesystem;

  std::cout << "\n[INFO] Loading RWKV-7 model from " << model_path << "\n\n";

  LoadedModelContext ctx;
  ctx.model_path = model_path;
  ctx.model_name = fs::path(model_path).filename().string();
  if (fs::path(model_path).extension() == ".pth") {
    ctx.model_name = fs::path(model_path).stem().string();
  }

  auto device = torch::Device(torch::kCUDA, 0);
  ctx.model = std::make_shared<RWKVModel>(model_path, device);

  auto tokenizer = std::make_shared<trie_tokenizer>();
  if (tokenizer->load("src/infer/rwkv_vocab_v20230424.txt") != RWKV_SUCCESS) {
    throw std::runtime_error("failed to load tokenizer vocab: src/infer/rwkv_vocab_v20230424.txt");
  }
  ctx.tokenizer = std::move(tokenizer);
  ctx.rocm_flag = false;

  std::cout << "[INFO] Model loaded successfully.\n\n";
  return ctx;
}
