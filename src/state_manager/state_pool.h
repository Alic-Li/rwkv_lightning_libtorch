#pragma once

#include <array>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <sqlite3.h>

#include "infer/rwkv_model.h"

class StateCacheManager {
 public:
  static StateCacheManager& instance();

  void initialize(torch::Device device, int l1_capacity = 16, int l2_capacity = 32, const std::string& db_path = "rwkv_sessions.db");
  void shutdown();

  void put_state(const std::string& session_id, const RWKVState& state);
  std::optional<RWKVState> get_state(const std::string& session_id);
  bool delete_state_from_any_level(const std::string& session_id);
  bool put_prefix_state(
      const std::vector<int64_t>& prefix_tokens,
      const RWKVState& state,
      const std::optional<torch::Tensor>& logits = std::nullopt);

  struct PrefixMatch {
    std::string state_id;
    int matched_tokens = 0;
    int bucket_len = 0;
    RWKVState state;
    std::optional<torch::Tensor> logits;
    std::string cache_source;
  };

  std::optional<PrefixMatch> match_prefix_state(
      const std::vector<int64_t>& prompt_tokens);

  struct StateSummary {
    std::vector<std::string> l1_cache;
    std::vector<std::string> l2_cache;
    std::vector<std::string> database;
    std::unordered_map<int, int> prefix_l2_counts;
    std::unordered_map<int, std::vector<std::string>> prefix_l2_cache;
    int prefix_database_count = 0;
  };

  StateSummary list_all_states();
  std::optional<double> get_db_timestamp(const std::string& session_id);

 private:
  StateCacheManager() = default;
  ~StateCacheManager() = default;
  StateCacheManager(const StateCacheManager&) = delete;
  StateCacheManager& operator=(const StateCacheManager&) = delete;

  struct CacheEntry {
    RWKVState state;
    std::list<std::string>::iterator it;
  };

  struct PrefixCacheEntry {
    std::string state_id;
    int bucket_len = 0;
    int token_count = 0;
    std::vector<int64_t> prefix_tokens;
    RWKVState state_cpu;
    std::optional<torch::Tensor> logits_cpu;
    double last_updated = 0.0;
    std::list<std::string>::iterator it;
  };

  static const std::array<int, 8>& prefix_cache_buckets();
  std::vector<torch::Tensor> state_to_tensors(const RWKVState& state) const;
  RWKVState tensors_to_state(const std::vector<torch::Tensor>& tensors, torch::Device device) const;
  RWKVState clone_state(const RWKVState& state) const;
  RWKVState clone_state_to_cpu(const RWKVState& state) const;
  std::optional<torch::Tensor> clone_optional_tensor(
      const std::optional<torch::Tensor>& tensor,
      torch::Device device) const;
  std::string serialize_state(const RWKVState& state) const;
  RWKVState deserialize_state(const std::string& blob) const;
  std::string serialize_tensor(const torch::Tensor& tensor) const;
  torch::Tensor deserialize_tensor(const std::string& blob, torch::Device device) const;

  void touch_entry(
      std::unordered_map<std::string, CacheEntry>& cache,
      std::list<std::string>& order,
      const std::string& key);
  void erase_entry(
      std::unordered_map<std::string, CacheEntry>& cache,
      std::list<std::string>& order,
      const std::string& key);
  void persist_state(const std::string& session_id, const RWKVState& state);
  void touch_prefix_entry(int bucket_len, const std::string& key);
  void erase_prefix_entry(int bucket_len, const std::string& key);
  void persist_prefix_entry(const PrefixCacheEntry& entry);
  std::optional<PrefixCacheEntry> load_prefix_entry_from_db_locked(
      const std::vector<int64_t>& prefix_tokens,
      int bucket_len);
  void init_db();

  bool initialized_ = false;
  torch::Device device_ = torch::Device(torch::kCUDA, 0);
  int l1_capacity_ = 16;
  int l2_capacity_ = 32;
  int prefix_bucket_capacity_ = 16;
  std::string db_path_ = "rwkv_sessions.db";

  mutable std::mutex mutex_;
  sqlite3* db_ = nullptr;

  std::unordered_map<std::string, CacheEntry> l1_cache_;
  std::unordered_map<std::string, CacheEntry> l2_cache_;
  std::list<std::string> l1_order_;
  std::list<std::string> l2_order_;
  std::unordered_map<int, std::unordered_map<std::string, PrefixCacheEntry>> prefix_l2_cache_;
  std::unordered_map<int, std::list<std::string>> prefix_l2_order_;
};
