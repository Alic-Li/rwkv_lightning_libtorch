#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

struct SafeTensorEntry {
  std::string dtype;
  std::vector<int64_t> shape;
  size_t start = 0;
  size_t end = 0;
};

class SafeTensorArchive {
 public:
  explicit SafeTensorArchive(const std::string& path);

  bool has_tensor(const std::string& name) const;

  torch::Tensor load_tensor(
      const std::string& name,
      torch::Device device) const;

  size_t tensor_count() const { return entries_.size(); }
  std::vector<std::string> tensor_names() const;

 private:
  std::vector<char> bytes_;
  size_t data_base_offset_ = 0;
  std::unordered_map<std::string, SafeTensorEntry> entries_;
};
