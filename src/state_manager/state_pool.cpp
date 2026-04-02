#include "state_pool.h"

#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

constexpr std::array<int, 8> kPrefixCacheBuckets{
    1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192};

std::string serialize_token_ids(const std::vector<int64_t>& tokens, int limit = -1) {
  std::ostringstream oss;
  const int count = limit >= 0 ? std::min<int>(limit, static_cast<int>(tokens.size()))
                               : static_cast<int>(tokens.size());
  for (int i = 0; i < count; ++i) {
    if (i != 0) {
      oss << ' ';
    }
    oss << tokens[static_cast<size_t>(i)];
  }
  return oss.str();
}

std::string hash_token_ids(const std::vector<int64_t>& tokens, int limit = -1) {
  constexpr uint64_t kOffset = 14695981039346656037ull;
  constexpr uint64_t kPrime = 1099511628211ull;
  const auto payload = serialize_token_ids(tokens, limit);
  uint64_t hash = kOffset;
  for (unsigned char c : payload) {
    hash ^= static_cast<uint64_t>(c);
    hash *= kPrime;
  }
  std::ostringstream oss;
  oss << std::hex << std::setfill('0') << std::setw(16) << hash;
  return oss.str();
}

std::vector<int64_t> prefix_slice(
    const std::vector<int64_t>& tokens,
    int bucket_len) {
  return {
      tokens.begin(),
      tokens.begin() + std::min<int>(bucket_len, static_cast<int>(tokens.size()))};
}

std::unordered_map<int, std::string> build_prefix_hashes(
    const std::vector<int64_t>& tokens) {
  std::unordered_map<int, std::string> hashes;
  for (int bucket : kPrefixCacheBuckets) {
    if (static_cast<int>(tokens.size()) >= bucket) {
      hashes.emplace(bucket, hash_token_ids(tokens, bucket));
    }
  }
  return hashes;
}

bool is_valid_prefix_bucket(int bucket_len) {
  return std::find(
             kPrefixCacheBuckets.begin(),
             kPrefixCacheBuckets.end(),
             bucket_len) != kPrefixCacheBuckets.end();
}

}  // namespace

StateCacheManager& StateCacheManager::instance() {
  static StateCacheManager mgr;
  return mgr;
}

const std::array<int, 8>& StateCacheManager::prefix_cache_buckets() {
  return kPrefixCacheBuckets;
}

void StateCacheManager::initialize(
    torch::Device device,
    int l1_capacity,
    int l2_capacity,
    const std::string& db_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_) {
    return;
  }
  device_ = device;
  l1_capacity_ = l1_capacity;
  l2_capacity_ = l2_capacity;
  db_path_ = db_path;
  for (int bucket : kPrefixCacheBuckets) {
    prefix_l2_cache_[bucket] = {};
    prefix_l2_order_[bucket] = {};
  }
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
  for (const auto& [bucket, cache] : prefix_l2_cache_) {
    for (const auto& [key, entry] : cache) {
      persist_prefix_entry(entry);
    }
  }

  l1_cache_.clear();
  l2_cache_.clear();
  l1_order_.clear();
  l2_order_.clear();
  prefix_l2_cache_.clear();
  prefix_l2_order_.clear();

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
  const char* session_sql =
      "CREATE TABLE IF NOT EXISTS sessions ("
      "session_id TEXT PRIMARY KEY,"
      "state_blob BLOB,"
      "last_updated REAL"
      ")";
  const char* prefix_sql =
      "CREATE TABLE IF NOT EXISTS prefix_cache ("
      "state_id TEXT PRIMARY KEY,"
      "bucket_len INTEGER NOT NULL,"
      "token_count INTEGER NOT NULL,"
      "prefix_hash_1024 TEXT,"
      "prefix_hash_2048 TEXT,"
      "prefix_hash_3072 TEXT,"
      "prefix_hash_4096 TEXT,"
      "prefix_hash_5120 TEXT,"
      "prefix_hash_6144 TEXT,"
      "prefix_hash_7168 TEXT,"
      "prefix_hash_8192 TEXT,"
      "state_blob BLOB NOT NULL,"
      "logits_blob BLOB,"
      "last_updated REAL"
      ")";

  char* err = nullptr;
  if (sqlite3_exec(db_, session_sql, nullptr, nullptr, &err) != SQLITE_OK) {
    std::string msg = err ? err : "unknown sqlite error";
    sqlite3_free(err);
    throw std::runtime_error("failed to init sqlite sessions db: " + msg);
  }
  if (sqlite3_exec(db_, prefix_sql, nullptr, nullptr, &err) != SQLITE_OK) {
    std::string msg = err ? err : "unknown sqlite error";
    sqlite3_free(err);
    throw std::runtime_error("failed to init sqlite prefix db: " + msg);
  }

  for (int bucket : kPrefixCacheBuckets) {
    const auto sql =
        "CREATE INDEX IF NOT EXISTS idx_prefix_cache_" + std::to_string(bucket) +
        " ON prefix_cache (bucket_len, prefix_hash_" + std::to_string(bucket) +
        ", last_updated)";
    if (sqlite3_exec(db_, sql.c_str(), nullptr, nullptr, &err) != SQLITE_OK) {
      std::string msg = err ? err : "unknown sqlite error";
      sqlite3_free(err);
      throw std::runtime_error("failed to init sqlite prefix index: " + msg);
    }
  }
}

std::vector<torch::Tensor> StateCacheManager::state_to_tensors(const RWKVState& state) const {
  return {state.x_prev, state.att_state, state.elapsed_t};
}

RWKVState StateCacheManager::tensors_to_state(
    const std::vector<torch::Tensor>& tensors,
    torch::Device device) const {
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

RWKVState StateCacheManager::clone_state_to_cpu(const RWKVState& state) const {
  return {
      state.x_prev.detach().to(torch::kCPU).clone(),
      state.att_state.detach().to(torch::kCPU).clone(),
      state.elapsed_t.detach().to(torch::kCPU).clone()};
}

std::optional<torch::Tensor> StateCacheManager::clone_optional_tensor(
    const std::optional<torch::Tensor>& tensor,
    torch::Device device) const {
  if (!tensor.has_value()) {
    return std::nullopt;
  }
  return tensor->detach().to(device).clone();
}

std::string StateCacheManager::serialize_state(const RWKVState& state) const {
  std::stringstream ss;
  auto tensors = state_to_tensors(clone_state_to_cpu(state));
  torch::save(tensors, ss);
  return ss.str();
}

RWKVState StateCacheManager::deserialize_state(const std::string& blob) const {
  std::stringstream ss(blob);
  std::vector<torch::Tensor> tensors;
  torch::load(tensors, ss);
  return tensors_to_state(tensors, device_);
}

std::string StateCacheManager::serialize_tensor(const torch::Tensor& tensor) const {
  std::stringstream ss;
  torch::save(tensor.detach().to(torch::kCPU).clone(), ss);
  return ss.str();
}

torch::Tensor StateCacheManager::deserialize_tensor(
    const std::string& blob,
    torch::Device device) const {
  std::stringstream ss(blob);
  torch::Tensor tensor;
  torch::load(tensor, ss);
  return tensor.to(device).contiguous();
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

void StateCacheManager::touch_prefix_entry(int bucket_len, const std::string& key) {
  auto cache_it = prefix_l2_cache_.find(bucket_len);
  if (cache_it == prefix_l2_cache_.end()) {
    return;
  }
  auto order_it = prefix_l2_order_.find(bucket_len);
  auto entry_it = cache_it->second.find(key);
  if (order_it == prefix_l2_order_.end() || entry_it == cache_it->second.end()) {
    return;
  }
  order_it->second.erase(entry_it->second.it);
  order_it->second.push_back(key);
  entry_it->second.it = std::prev(order_it->second.end());
}

void StateCacheManager::erase_prefix_entry(int bucket_len, const std::string& key) {
  auto cache_it = prefix_l2_cache_.find(bucket_len);
  if (cache_it == prefix_l2_cache_.end()) {
    return;
  }
  auto entry_it = cache_it->second.find(key);
  if (entry_it == cache_it->second.end()) {
    return;
  }
  prefix_l2_order_[bucket_len].erase(entry_it->second.it);
  cache_it->second.erase(entry_it);
}

void StateCacheManager::persist_state(const std::string& session_id, const RWKVState& state) {
  const auto blob = serialize_state(state);
  sqlite3_stmt* stmt = nullptr;
  const char* sql =
      "INSERT OR REPLACE INTO sessions (session_id, state_blob, last_updated) VALUES (?, ?, ?)";
  sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_blob(stmt, 2, blob.data(), static_cast<int>(blob.size()), SQLITE_TRANSIENT);
  sqlite3_bind_double(stmt, 3, static_cast<double>(time(nullptr)));
  sqlite3_step(stmt);
  sqlite3_finalize(stmt);
}

void StateCacheManager::persist_prefix_entry(const PrefixCacheEntry& entry) {
  const auto state_blob = serialize_state(entry.state_cpu);
  const auto prefix_hashes = build_prefix_hashes(entry.prefix_tokens);
  sqlite3_stmt* stmt = nullptr;
  const char* sql =
      "INSERT OR REPLACE INTO prefix_cache ("
      "state_id, bucket_len, token_count, "
      "prefix_hash_1024, prefix_hash_2048, prefix_hash_3072, prefix_hash_4096, "
      "prefix_hash_5120, prefix_hash_6144, prefix_hash_7168, prefix_hash_8192, "
      "state_blob, logits_blob, last_updated) "
      "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)";
  sqlite3_prepare_v2(db_, sql, -1, &stmt, nullptr);
  sqlite3_bind_text(stmt, 1, entry.state_id.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int(stmt, 2, entry.bucket_len);
  sqlite3_bind_int(stmt, 3, entry.token_count);

  int bind_index = 4;
  for (int bucket : kPrefixCacheBuckets) {
    const auto it = prefix_hashes.find(bucket);
    if (it != prefix_hashes.end()) {
      sqlite3_bind_text(stmt, bind_index, it->second.c_str(), -1, SQLITE_TRANSIENT);
    } else {
      sqlite3_bind_null(stmt, bind_index);
    }
    ++bind_index;
  }

  sqlite3_bind_blob(
      stmt, bind_index++, state_blob.data(), static_cast<int>(state_blob.size()), SQLITE_TRANSIENT);
  if (entry.logits_cpu.has_value()) {
    const auto logits_blob = serialize_tensor(*entry.logits_cpu);
    sqlite3_bind_blob(
        stmt,
        bind_index++,
        logits_blob.data(),
        static_cast<int>(logits_blob.size()),
        SQLITE_TRANSIENT);
  } else {
    sqlite3_bind_null(stmt, bind_index++);
  }
  sqlite3_bind_double(stmt, bind_index, entry.last_updated);
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
    const auto evicted_id = l1_order_.front();
    auto evicted_state = clone_state_to_cpu(l1_cache_.at(evicted_id).state);
    erase_entry(l1_cache_, l1_order_, evicted_id);
    l2_order_.push_back(evicted_id);
    l2_cache_[evicted_id] = {std::move(evicted_state), std::prev(l2_order_.end())};
  }

  while (static_cast<int>(l2_cache_.size()) > l2_capacity_) {
    const auto evicted_id = l2_order_.front();
    const auto evicted_state = l2_cache_.at(evicted_id).state;
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

bool StateCacheManager::put_prefix_state(
    const std::vector<int64_t>& prefix_tokens,
    const RWKVState& state,
    const std::optional<torch::Tensor>& logits) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) {
    return false;
  }
  const int bucket_len = static_cast<int>(prefix_tokens.size());
  if (!is_valid_prefix_bucket(bucket_len)) {
    return false;
  }

  auto& bucket_cache = prefix_l2_cache_[bucket_len];
  auto& bucket_order = prefix_l2_order_[bucket_len];
  const auto state_id = serialize_token_ids(prefix_tokens);

  erase_prefix_entry(bucket_len, state_id);
  bucket_order.push_back(state_id);
  bucket_cache[state_id] = PrefixCacheEntry{
      state_id,
      bucket_len,
      bucket_len,
      prefix_tokens,
      clone_state_to_cpu(state),
      clone_optional_tensor(logits, torch::kCPU),
      static_cast<double>(time(nullptr)),
      std::prev(bucket_order.end())};

  persist_prefix_entry(bucket_cache.at(state_id));

  while (static_cast<int>(bucket_cache.size()) > prefix_bucket_capacity_) {
    const auto evicted_id = bucket_order.front();
    const auto evicted_entry = bucket_cache.at(evicted_id);
    erase_prefix_entry(bucket_len, evicted_id);
    persist_prefix_entry(evicted_entry);
  }
  return true;
}

std::optional<StateCacheManager::PrefixCacheEntry>
StateCacheManager::load_prefix_entry_from_db_locked(
    const std::vector<int64_t>& prefix_tokens,
    int bucket_len) {
  const auto state_id = serialize_token_ids(prefix_tokens, bucket_len);
  const auto hash_value = hash_token_ids(prefix_tokens, bucket_len);
  sqlite3_stmt* stmt = nullptr;
  const auto sql =
      "SELECT state_blob, logits_blob, last_updated FROM prefix_cache WHERE state_id = ? "
      "AND bucket_len = ? AND prefix_hash_" +
      std::to_string(bucket_len) + " = ? LIMIT 1";
  sqlite3_prepare_v2(db_, sql.c_str(), -1, &stmt, nullptr);
  sqlite3_bind_text(stmt, 1, state_id.c_str(), -1, SQLITE_TRANSIENT);
  sqlite3_bind_int(stmt, 2, bucket_len);
  sqlite3_bind_text(stmt, 3, hash_value.c_str(), -1, SQLITE_TRANSIENT);

  std::optional<PrefixCacheEntry> result;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    const auto* state_blob = static_cast<const char*>(sqlite3_column_blob(stmt, 0));
    const auto state_size = sqlite3_column_bytes(stmt, 0);
    std::optional<torch::Tensor> logits_cpu = std::nullopt;
    if (sqlite3_column_type(stmt, 1) != SQLITE_NULL) {
      const auto* logits_blob = static_cast<const char*>(sqlite3_column_blob(stmt, 1));
      const auto logits_size = sqlite3_column_bytes(stmt, 1);
      logits_cpu = deserialize_tensor(
          std::string(logits_blob, logits_blob + logits_size), torch::kCPU);
    }

    auto state_cpu = tensors_to_state(
        state_to_tensors(deserialize_state(std::string(state_blob, state_blob + state_size))),
        torch::kCPU);

    auto& bucket_order = prefix_l2_order_[bucket_len];
    bucket_order.push_back(state_id);
    result = PrefixCacheEntry{
        state_id,
        bucket_len,
        bucket_len,
        prefix_slice(prefix_tokens, bucket_len),
        std::move(state_cpu),
        std::move(logits_cpu),
        sqlite3_column_double(stmt, 2),
        std::prev(bucket_order.end())};
    prefix_l2_cache_[bucket_len][state_id] = *result;
  }
  sqlite3_finalize(stmt);
  return result;
}

std::optional<StateCacheManager::PrefixMatch> StateCacheManager::match_prefix_state(
    const std::vector<int64_t>& prompt_tokens) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_ || prompt_tokens.empty()) {
    return std::nullopt;
  }

  for (auto it = kPrefixCacheBuckets.rbegin(); it != kPrefixCacheBuckets.rend(); ++it) {
    const int bucket = *it;
    if (static_cast<int>(prompt_tokens.size()) < bucket) {
      continue;
    }

    const auto state_id = serialize_token_ids(prompt_tokens, bucket);
    auto cache_it = prefix_l2_cache_.find(bucket);
    if (cache_it != prefix_l2_cache_.end()) {
      auto entry_it = cache_it->second.find(state_id);
      if (entry_it != cache_it->second.end()) {
        touch_prefix_entry(bucket, state_id);
        return PrefixMatch{
            entry_it->second.state_id,
            bucket,
            bucket,
            tensors_to_state(state_to_tensors(entry_it->second.state_cpu), device_),
            clone_optional_tensor(entry_it->second.logits_cpu, device_),
            "l2_ram"};
      }
    }

    auto loaded = load_prefix_entry_from_db_locked(prompt_tokens, bucket);
    if (loaded.has_value()) {
      return PrefixMatch{
          loaded->state_id,
          bucket,
          bucket,
          tensors_to_state(state_to_tensors(loaded->state_cpu), device_),
          clone_optional_tensor(loaded->logits_cpu, device_),
          "disk"};
    }
  }

  return std::nullopt;
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
  for (int bucket : kPrefixCacheBuckets) {
    const auto cache_it = prefix_l2_cache_.find(bucket);
    const int count =
        cache_it == prefix_l2_cache_.end() ? 0 : static_cast<int>(cache_it->second.size());
    out.prefix_l2_counts[bucket] = count;

    std::vector<std::string> keys;
    const auto order_it = prefix_l2_order_.find(bucket);
    if (order_it != prefix_l2_order_.end()) {
      keys.assign(order_it->second.begin(), order_it->second.end());
    }
    out.prefix_l2_cache[bucket] = std::move(keys);
  }

  sqlite3_stmt* stmt = nullptr;
  sqlite3_prepare_v2(
      db_, "SELECT session_id FROM sessions ORDER BY last_updated DESC", -1, &stmt, nullptr);
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    out.database.emplace_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)));
  }
  sqlite3_finalize(stmt);

  sqlite3_prepare_v2(db_, "SELECT COUNT(*) FROM prefix_cache", -1, &stmt, nullptr);
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    out.prefix_database_count = sqlite3_column_int(stmt, 0);
  }
  sqlite3_finalize(stmt);

  return out;
}

std::optional<double> StateCacheManager::get_db_timestamp(const std::string& session_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  sqlite3_stmt* stmt = nullptr;
  sqlite3_prepare_v2(
      db_, "SELECT last_updated FROM sessions WHERE session_id = ?", -1, &stmt, nullptr);
  sqlite3_bind_text(stmt, 1, session_id.c_str(), -1, SQLITE_TRANSIENT);
  std::optional<double> ts;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    ts = sqlite3_column_double(stmt, 0);
  }
  sqlite3_finalize(stmt);
  return ts;
}
