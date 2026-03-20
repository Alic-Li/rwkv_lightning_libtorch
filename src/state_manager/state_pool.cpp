#include "state_pool.h"

#include <ctime>
#include <sstream>
#include <stdexcept>

StateCacheManager& StateCacheManager::instance() {
  static StateCacheManager mgr;
  return mgr;
}

void StateCacheManager::initialize(torch::Device device, int l1_capacity, int l2_capacity, const std::string& db_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    return;
  }
  device_ = device;
  l1_capacity_ = l1_capacity;
  l2_capacity_ = l2_capacity;
  db_path_ = db_path;
  init_db();
  initialized_ = true;
}

void StateCacheManager::shutdown() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    return;
  }

  for (const auto& [sid, entry] : l1_cache_) {
    persist_state(sid, entry.state);
  }
  for (const auto& [sid, entry] : l2_cache_) {
    persist_state(sid, entry.state);
  }
  l1_cache_.clear();
  l2_cache_.clear();
  l1_order_.clear();
  l2_order_.clear();

  if (db_) {
    sqlite3_close(db_);
    db_ = nullptr;
  }
  initialized_ = false;
}

void StateCacheManager::init_db() {
  if (sqlite3_open(db_path_.c_str(), &db_) != SQLITE_OK) {
    throw std::runtime_error("failed to open sqlite db: " + db_path_);
  }
  const char* sql =
      "CREATE TABLE IF NOT EXISTS sessions ("
      "session_id TEXT PRIMARY KEY,"
      "state_blob BLOB,"
      "last_updated REAL"
      ")";
  char* err = nullptr;
  if (sqlite3_exec(db_, sql, nullptr, nullptr, &err) != SQLITE_OK) {
    std::string msg = err ? err : "unknown sqlite error";
    sqlite3_free(err);
    throw std::runtime_error("failed to init sqlite db: " + msg);
  }
}

std::vector<torch::Tensor> StateCacheManager::state_to_tensors(const RWKVState& state) const {
  return {state.x_prev, state.att_state, state.elapsed_t};
}

RWKVState StateCacheManager::tensors_to_state(const std::vector<torch::Tensor>& tensors, torch::Device device) const {
  if (tensors.size() != 3) {
    throw std::runtime_error("invalid serialized state tensor count");
  }
  RWKVState state;
  state.x_prev = tensors[0].to(device).contiguous();
  state.att_state = tensors[1].to(device).contiguous();
  state.elapsed_t = tensors[2].to(device).contiguous();
  return state;
}

RWKVState StateCacheManager::clone_state(const RWKVState& state) const {
  return {state.x_prev.clone(), state.att_state.clone(), state.elapsed_t.clone()};
}

std::string StateCacheManager::serialize_state(const RWKVState& state) const {
  std::stringstream ss;
  auto tensors = state_to_tensors({state.x_prev.to(torch::kCPU), state.att_state.to(torch::kCPU), state.elapsed_t.to(torch::kCPU)});
  torch::save(tensors, ss);
  return ss.str();
}

RWKVState StateCacheManager::deserialize_state(const std::string& blob) const {
  std::stringstream ss(blob);
  std::vector<torch::Tensor> tensors;
  torch::load(tensors, ss);
  return tensors_to_state(tensors, device_);
}

void StateCacheManager::touch_entry(
    std::unordered_map<std::string, CacheEntry>& cache,
    std::list<std::string>& order,
    const std::string& key) {
  auto it = cache.find(key);
  if (it == cache.end()) {
    return;
  }
  order.erase(it->second.it);
  order.push_back(key);
  it->second.it = std::prev(order.end());
}

void StateCacheManager::erase_entry(
    std::unordered_map<std::string, CacheEntry>& cache,
    std::list<std::string>& order,
    const std::string& key) {
  auto it = cache.find(key);
  if (it == cache.end()) {
    return;
  }
  order.erase(it->second.it);
  cache.erase(it);
}

void StateCacheManager::persist_state(const std::string& session_id, const RWKVState& state) {
  const auto blob = serialize_state(state);
  sqlite3_stmt* stmt = nullptr;
  const char* sql = "INSERT OR REPLACE INTO sessions (session_id, state_blob, last_updated) VALUES (?, ?, ?)";
  sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_blob(stmt, 2, blob.data(), static_cast<int>(blob.size()), SQLITE_TRANSIENT);
  sqlite3_bind_double(stmt, 3, static_cast<double>(time(nullptr)));
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

void StateCacheManager::put_state(const std::string& session_id, const RWKVState& state) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || session_id.empty()) {
    return;
  }

  erase_entry(l1_cache_, l1_order_, session_id);
  erase_entry(l2_cache_, l2_order_, session_id);

  l1_order_.push_back(session_id);
  l1_cache_[session_id] = {clone_state(state), std::prev(l1_order_.end())};

  while (static_cast<int>(l1_cache_.size()) > l1_capacity_) {
    auto evicted_id = l1_order_.front();
    auto evicted_state = clone_state(l1_cache_.at(evicted_id).state);
    erase_entry(l1_cache_, l1_order_, evicted_id);
    l2_order_.push_back(evicted_id);
    l2_cache_[evicted_id] = {RWKVState{evicted_state.x_prev.to(torch::kCPU), evicted_state.att_state.to(torch::kCPU), evicted_state.elapsed_t.to(torch::kCPU)}, std::prev(l2_order_.end())};
  }

  while (static_cast<int>(l2_cache_.size()) > l2_capacity_) {
    auto evicted_id = l2_order_.front();
    auto evicted_state = l2_cache_.at(evicted_id).state;
    erase_entry(l2_cache_, l2_order_, evicted_id);
    persist_state(evicted_id, evicted_state);
  }
}

std::optional<RWKVState> StateCacheManager::get_state(const std::string& session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || session_id.empty()) {
    return std::nullopt;
  }

  auto l1_it = l1_cache_.find(session_id);
  if (l1_it != l1_cache_.end()) {
    touch_entry(l1_cache_, l1_order_, session_id);
    return clone_state(l1_it->second.state);
  }

  auto l2_it = l2_cache_.find(session_id);
  if (l2_it != l2_cache_.end()) {
    auto restored = tensors_to_state(state_to_tensors(l2_it->second.state), device_);
    erase_entry(l2_cache_, l2_order_, session_id);
    l1_order_.push_back(session_id);
    l1_cache_[session_id] = {clone_state(restored), std::prev(l1_order_.end())};
    return clone_state(restored);
  }

  sqlite3_stmt* stmt = nullptr;
  const char* sql = "SELECT state_blob FROM sessions WHERE session_id = ?";
  sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
  std::optional<RWKVState> result;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    const auto* data = static_cast<const char*>(sqlite3_column_blob(stmt, 0));
    const auto size = sqlite3_column_bytes(stmt, 0);
    result = deserialize_state(std::string(data, data + size));
    l1_order_.push_back(session_id);
    l1_cache_[session_id] = {clone_state(*result), std::prev(l1_order_.end())};
  }
  sqlite3_finalize(stmt);
  return result;
}

bool StateCacheManager::delete_state_from_any_level(const std::string& session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool removed = false;
  if (l1_cache_.count(session_id)) {
    erase_entry(l1_cache_, l1_order_, session_id);
    removed = true;
  }
  if (l2_cache_.count(session_id)) {
    erase_entry(l2_cache_, l2_order_, session_id);
    removed = true;
  }
  sqlite3_stmt* stmt = nullptr;
  sqlite3_prepare_v2(db_, "DELETE FROM sessions WHERE session_id = ?", -1, &stmt, nullptr);
  sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_step(stmt);
  if (sqlite3_changes(db_) > 0) {
    removed = true;
  }
  sqlite3_finalize(stmt);
  return removed;
}

StateCacheManager::StateSummary StateCacheManager::list_all_states() {
  std::lock_guard<std::mutex> lock(mutex_);
  StateSummary out;
  out.l1_cache.assign(l1_order_.begin(), l1_order_.end());
  out.l2_cache.assign(l2_order_.begin(), l2_order_.end());

  sqlite3_stmt* stmt = nullptr;
  sqlite3_prepare_v2(db_, "SELECT session_id FROM sessions ORDER BY last_updated DESC", -1, &stmt, nullptr);
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    out.database.emplace_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
  }
  sqlite3_finalize(stmt);
  return out;
}

std::optional<double> StateCacheManager::get_db_timestamp(const std::string& session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3_stmt* stmt = nullptr;
  sqlite3_prepare_v2(db_, "SELECT last_updated FROM sessions WHERE session_id = ?", -1, &stmt, nullptr);
  sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
  std::optional<double> ts;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    ts = sqlite3_column_double(stmt, 0);
  }
  sqlite3_finalize(stmt);
  return ts;
}
