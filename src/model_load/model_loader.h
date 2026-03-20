#pragma once

#include <memory>
#include <string>

#include "infer/rwkv_model.h"
#include "infer/utils/tokenizer.h"

struct LoadedModelContext {
  std::shared_ptr<RWKVModel> model;
  std::shared_ptr<trie_tokenizer> tokenizer;
  std::string model_path;
  std::string model_name;
  bool rocm_flag = false;
};

LoadedModelContext load_model_and_tokenizer(const std::string& model_path);
