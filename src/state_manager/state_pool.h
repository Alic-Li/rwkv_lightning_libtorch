#pragma once

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

  struct StateSummary {
    std::vector<std::string> l1_cache;
    std::vector<std::string> l2_cache;
    std::vector<std::string> database;
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

  std::vector<torch::Tensor> state_to_tensors(const RWKVState& state) const;
  RWKVState tensors_to_state(const std::vector<torch::Tensor>& tensors, torch::Device device) const;
  RWKVState clone_state(const RWKVState& state) const;
  std::string serialize_state(const RWKVState& state) const;
  RWKVState deserialize_state(const std::string& blob) const;

  void touch_entry(
      std::unordered_map<std::string, CacheEntry>& cache,
      std::list<std::string>& order,
      const std::string& key);
  void erase_entry(
      std::unordered_map<std::string, CacheEntry>& cache,
      std::list<std::string>& order,
      const std::string& key);
  void persist_state(const std::string& session_id, const RWKVState& state);
  void init_db();

  bool initialized_ = false;
  torch::Device device_ = torch::Device(torch::kCUDA, 0);
  int l1_capacity_ = 16;
  int l2_capacity_ = 32;
  std::string db_path_ = "rwkv_sessions.db";

  mutable std::mutex mutex_;
  sqlite3* db_ = nullptr;

  std::unordered_map<std::string, CacheEntry> l1_cache_;
  std::unordered_map<std::string, CacheEntry> l2_cache_;
  std::list<std::string> l1_order_;
  std::list<std::string> l2_order_;
};
