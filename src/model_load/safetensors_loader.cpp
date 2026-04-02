#include "safetensors_loader.h"

#include <cctype>
#include <fstream>
#include <iterator>
#include <string_view>

namespace {

class SafeTensorHeaderParser {
 public:
  explicit SafeTensorHeaderParser(std::string_view input) : input_(input) {}

  std::unordered_map<std::string, SafeTensorEntry> parse() {
    std::unordered_map<std::string, SafeTensorEntry> entries;
    skip_ws();
    expect('{');
    skip_ws();
    if (consume('}')) {
      return entries;
    }

    while (true) {
      std::string key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      if (key == "__metadata__") {
        skip_value();
      } else {
        entries.emplace(std::move(key), parse_tensor_entry());
      }
      skip_ws();
      if (consume('}')) {
        break;
      }
      expect(',');
      skip_ws();
    }
    skip_ws();
    TORCH_CHECK(
        pos_ == input_.size(),
        "unexpected trailing characters in safetensors header");
    return entries;
  }

 private:
  char peek() const {
    TORCH_CHECK(pos_ < input_.size(), "unexpected end of safetensors header");
    return input_[pos_];
  }

  void skip_ws() {
    while (pos_ < input_.size() &&
           std::isspace(static_cast<unsigned char>(input_[pos_]))) {
      ++pos_;
    }
  }

  bool consume(char c) {
    if (pos_ < input_.size() && input_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  void expect(char c) {
    TORCH_CHECK(peek() == c, "invalid safetensors header near byte ", pos_);
    ++pos_;
  }

  std::string parse_string() {
    expect('"');
    std::string out;
    while (true) {
      TORCH_CHECK(
          pos_ < input_.size(),
          "unterminated string in safetensors header");
      char c = input_[pos_++];
      if (c == '"') {
        return out;
      }
      if (c == '\\') {
        TORCH_CHECK(
            pos_ < input_.size(),
            "unterminated escape in safetensors header");
        char esc = input_[pos_++];
        switch (esc) {
          case '"':
          case '\\':
          case '/':
            out.push_back(esc);
            break;
          case 'b':
            out.push_back('\b');
            break;
          case 'f':
            out.push_back('\f');
            break;
          case 'n':
            out.push_back('\n');
            break;
          case 'r':
            out.push_back('\r');
            break;
          case 't':
            out.push_back('\t');
            break;
          default:
            TORCH_CHECK(
                false,
                "unsupported escape sequence in safetensors header");
        }
      } else {
        out.push_back(c);
      }
    }
  }

  int64_t parse_int() {
    TORCH_CHECK(
        pos_ < input_.size(),
        "unexpected end of integer in safetensors header");
    bool negative = false;
    if (input_[pos_] == '-') {
      negative = true;
      ++pos_;
    }
    TORCH_CHECK(
        pos_ < input_.size() &&
            std::isdigit(static_cast<unsigned char>(input_[pos_])),
        "invalid integer in safetensors header");
    int64_t value = 0;
    while (pos_ < input_.size() &&
           std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
      value = value * 10 + (input_[pos_] - '0');
      ++pos_;
    }
    return negative ? -value : value;
  }

  std::vector<int64_t> parse_int_array() {
    std::vector<int64_t> values;
    expect('[');
    skip_ws();
    if (consume(']')) {
      return values;
    }
    while (true) {
      values.push_back(parse_int());
      skip_ws();
      if (consume(']')) {
        break;
      }
      expect(',');
      skip_ws();
    }
    return values;
  }

  SafeTensorEntry parse_tensor_entry() {
    SafeTensorEntry entry;
    expect('{');
    skip_ws();
    if (consume('}')) {
      return entry;
    }

    while (true) {
      std::string key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      if (key == "dtype") {
        entry.dtype = parse_string();
      } else if (key == "shape") {
        entry.shape = parse_int_array();
      } else if (key == "data_offsets") {
        auto offsets = parse_int_array();
        TORCH_CHECK(
            offsets.size() == 2,
            "data_offsets must contain exactly 2 values");
        entry.start = static_cast<size_t>(offsets[0]);
        entry.end = static_cast<size_t>(offsets[1]);
      } else {
        skip_value();
      }
      skip_ws();
      if (consume('}')) {
        break;
      }
      expect(',');
      skip_ws();
    }

    TORCH_CHECK(!entry.dtype.empty(), "safetensors entry missing dtype");
    TORCH_CHECK(
        entry.end >= entry.start,
        "safetensors entry has invalid data_offsets");
    return entry;
  }

  void skip_value() {
    skip_ws();
    const char c = peek();
    if (c == '{') {
      ++pos_;
      skip_ws();
      if (consume('}')) {
        return;
      }
      while (true) {
        parse_string();
        skip_ws();
        expect(':');
        skip_ws();
        skip_value();
        skip_ws();
        if (consume('}')) {
          return;
        }
        expect(',');
        skip_ws();
      }
    }
    if (c == '[') {
      ++pos_;
      skip_ws();
      if (consume(']')) {
        return;
      }
      while (true) {
        skip_value();
        skip_ws();
        if (consume(']')) {
          return;
        }
        expect(',');
        skip_ws();
      }
    }
    if (c == '"') {
      parse_string();
      return;
    }
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '-') {
      parse_int();
      return;
    }
    if (input_.substr(pos_, 4) == "null") {
      pos_ += 4;
      return;
    }
    if (input_.substr(pos_, 4) == "true") {
      pos_ += 4;
      return;
    }
    if (input_.substr(pos_, 5) == "false") {
      pos_ += 5;
      return;
    }
    TORCH_CHECK(
        false,
        "unsupported JSON value in safetensors header near byte ",
        pos_);
  }

  std::string_view input_;
  size_t pos_ = 0;
};

uint64_t read_u64_le(const char* data) {
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value |= static_cast<uint64_t>(static_cast<unsigned char>(data[i]))
             << (8 * i);
  }
  return value;
}

torch::ScalarType scalar_type_for_dtype(const std::string& dtype) {
  if (dtype == "F16") {
    return torch::kFloat16;
  }
  if (dtype == "F32") {
    return torch::kFloat32;
  }
  if (dtype == "BF16") {
    return torch::kBFloat16;
  }
  if (dtype == "I64") {
    return torch::kInt64;
  }
  if (dtype == "I32") {
    return torch::kInt32;
  }
  if (dtype == "I16") {
    return torch::kInt16;
  }
  if (dtype == "I8") {
    return torch::kInt8;
  }
  if (dtype == "U8") {
    return torch::kUInt8;
  }
  TORCH_CHECK(false, "unsupported safetensors dtype: ", dtype);
}

size_t element_size_for_dtype(const std::string& dtype) {
  if (dtype == "F16" || dtype == "BF16" || dtype == "I16") {
    return 2;
  }
  if (dtype == "F32" || dtype == "I32") {
    return 4;
  }
  if (dtype == "I64") {
    return 8;
  }
  if (dtype == "I8" || dtype == "U8") {
    return 1;
  }
  TORCH_CHECK(false, "unsupported safetensors dtype: ", dtype);
}

}  // namespace

SafeTensorArchive::SafeTensorArchive(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  TORCH_CHECK(in.good(), "failed to open weights file: ", path);
  bytes_ = std::vector<char>(
      std::istreambuf_iterator<char>(in),
      std::istreambuf_iterator<char>());
  TORCH_CHECK(bytes_.size() >= 8, "invalid safetensors file: ", path);

  const uint64_t header_size = read_u64_le(bytes_.data());
  TORCH_CHECK(
      header_size <= bytes_.size() - 8,
      "invalid safetensors header length in ",
      path);
  data_base_offset_ = 8 + static_cast<size_t>(header_size);
  const std::string_view header(
      bytes_.data() + 8,
      static_cast<size_t>(header_size));
  entries_ = SafeTensorHeaderParser(header).parse();
  TORCH_CHECK(!entries_.empty(), "empty safetensors header in ", path);
}

bool SafeTensorArchive::has_tensor(const std::string& name) const {
  return entries_.find(name) != entries_.end();
}

std::vector<std::string> SafeTensorArchive::tensor_names() const {
  std::vector<std::string> names;
  names.reserve(entries_.size());
  for (const auto& [name, _] : entries_) {
    names.push_back(name);
  }
  return names;
}

torch::Tensor SafeTensorArchive::load_tensor(
    const std::string& name,
    torch::Device device) const {
  const auto it = entries_.find(name);
  TORCH_CHECK(it != entries_.end(), "missing tensor in safetensors file: ", name);

  const auto& entry = it->second;
  const auto dtype = scalar_type_for_dtype(entry.dtype);
  const auto element_size = element_size_for_dtype(entry.dtype);

  int64_t expected_bytes = 1;
  for (int64_t dim : entry.shape) {
    TORCH_CHECK(dim >= 0, "negative tensor dimension for ", name);
    expected_bytes *= dim;
  }
  expected_bytes *= static_cast<int64_t>(element_size);

  TORCH_CHECK(
      entry.end - entry.start == static_cast<size_t>(expected_bytes),
      "tensor byte size mismatch for ",
      name);
  TORCH_CHECK(
      data_base_offset_ + entry.end <= bytes_.size(),
      "tensor data offset out of range for ",
      name);

  const void* data_ptr = bytes_.data() + data_base_offset_ + entry.start;
  auto cpu = torch::from_blob(
                 const_cast<void*>(data_ptr),
                 entry.shape,
                 torch::TensorOptions().dtype(dtype).device(torch::kCPU))
                 .clone();
  return cpu.to(device, dtype).contiguous();
}
