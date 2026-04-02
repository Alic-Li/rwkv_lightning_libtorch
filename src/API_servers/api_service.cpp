#include "api_service.h"

#include <chrono>
#include <cctype>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <drogon/drogon.h>

#include "infer/inference_engine.h"
#include "state_manager/state_pool.h"

using namespace drogon;

namespace {

std::mutex g_dialogue_mutex;
std::unordered_map<std::string, int> g_dialogue_counters;

Json::Value make_error(const std::string& message) {
  Json::Value out;
  out["error"] = message;
  return out;
}

HttpResponsePtr json_response(Json::Value payload, HttpStatusCode code = k200OK) {
  Json::StreamWriterBuilder builder;
  builder["emitUTF8"] = true;
  builder["indentation"] = "";
  auto resp = HttpResponse::newHttpResponse();
  resp->setContentTypeCode(CT_APPLICATION_JSON);
  resp->setBody(Json::writeString(builder, payload));
  resp->setStatusCode(code);
  return resp;
}

std::optional<std::string> bearer_token(const HttpRequestPtr& req) {
  auto auth = req->getHeader("authorization");
  if (auth.empty()) {
    auth = req->getHeader("Authorization");
  }
  if (auth.rfind("Bearer ", 0) != 0) {
    return std::nullopt;
  }
  return auth.substr(7);
}

bool check_password(
    const HttpRequestPtr& req,
    const Json::Value& body,
    const std::optional<std::string>& password,
    HttpResponsePtr& out) {
  if (!password.has_value()) {
    return true;
  }
  const auto body_pw = body.get("password", "").asString();
  const auto token = bearer_token(req);
  if (body_pw == *password || (token.has_value() && *token == *password)) {
    return true;
  }
  out = json_response(make_error("Unauthorized: invalid or missing password"), k401Unauthorized);
  return false;
}

GenerateOptions parse_options(const Json::Value& body) {
  GenerateOptions options;
  options.max_tokens = body.get("max_tokens", 8192).asInt();
  options.temperature = body.get("temperature", 1.0).asDouble();
  options.top_k = body.get("top_k", 20).asInt();
  options.top_p = body.get("top_p", 0.6).asDouble();
  options.alpha_presence = body.get("alpha_presence", 1.0).asDouble();
  options.alpha_frequency = body.get("alpha_frequency", 0.1).asDouble();
  options.alpha_decay = body.get("alpha_decay", 0.996).asDouble();
  options.pad_zero = body.get("pad_zero", true).asBool();
  if (body.isMember("stop_tokens") && body["stop_tokens"].isArray()) {
    options.stop_tokens.clear();
    for (const auto& item : body["stop_tokens"]) {
      options.stop_tokens.push_back(item.asInt64());
    }
  }
  return options;
}

std::string normalize_content(const Json::Value& content) {
  if (content.isString()) {
    return content.asString();
  }
  if (content.isArray()) {
    std::string text;
    for (const auto& item : content) {
      if (item.isObject() && item.get("type", "").asString() == "text") {
        text += item.get("text", "").asString();
      } else if (item.isString()) {
        text += item.asString();
      }
    }
    return text;
  }
  if (content.isNull()) {
    return "";
  }
  return content.asString();
}

std::vector<std::string> parse_contents(const Json::Value& body) {
  std::vector<std::string> prompts;
  if (body.isMember("contents") && body["contents"].isArray()) {
    for (const auto& item : body["contents"]) {
      prompts.push_back(item.asString());
    }
  }
  return prompts;
}

std::string create_translation_prompt(
    const std::string& source_lang,
    const std::string& target_lang,
    const std::string& text) {
  return source_lang + ": " + text + "\n\n" + target_lang + ":";
}

int allocate_next_dialogue_idx(StateCacheManager& manager, const std::string& session_index) {
  std::lock_guard<std::mutex> lock(g_dialogue_mutex);
  auto it = g_dialogue_counters.find(session_index);
  if (it != g_dialogue_counters.end()) {
    return it->second++;
  }

  int max_idx = 0;
  const auto states = manager.list_all_states();
  const auto prefix = session_index + ":";
  auto scan = [&](const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
      if (key.rfind(prefix, 0) != 0) {
        continue;
      }
      const auto tail = key.substr(prefix.size());
      if (!tail.empty() && std::all_of(tail.begin(), tail.end(), ::isdigit)) {
        max_idx = std::max(max_idx, std::stoi(tail));
      }
    }
  };
  scan(states.l1_cache);
  scan(states.l2_cache);
  scan(states.database);
  g_dialogue_counters[session_index] = max_idx + 2;
  return max_idx + 1;
}

Json::Value build_choices(const std::vector<std::string>& texts) {
  Json::Value choices = Json::arrayValue;
  for (size_t i = 0; i < texts.size(); ++i) {
    Json::Value choice;
    choice["index"] = static_cast<int>(i);
    choice["message"]["role"] = "assistant";
    choice["message"]["content"] = texts[i];
    choice["finish_reason"] = "stop";
    choices.append(choice);
  }
  return choices;
}

HttpResponsePtr make_sse_response(const std::function<void(ResponseStreamPtr)>& producer) {
  auto resp = HttpResponse::newAsyncStreamResponse(
      [producer](ResponseStreamPtr stream) { producer(std::move(stream)); },
      true);
  resp->setContentTypeString("text/event-stream; charset=utf-8");
  resp->addHeader("Cache-Control", "no-cache");
  resp->addHeader("X-Accel-Buffering", "no");
  return resp;
}

void release_request_state(RWKVState& state) {
  state = RWKVState();
}

void start_streaming_task(
    ResponseStreamPtr stream,
    const std::function<void(const InferenceEngine::StreamCallback&)>& task) {
  std::thread([stream = std::move(stream), task]() mutable {
    auto emit = [&stream](const std::string& chunk) -> bool { return stream->send(chunk); };
    try {
      task(emit);
    } catch (const std::exception& e) {
      Json::Value err;
      err["error"] = e.what();
      Json::StreamWriterBuilder builder;
      builder["indentation"] = "";
      stream->send("data: " + Json::writeString(builder, err) + "\n\n");
      stream->send("data: [DONE]\n\n");
    }
    stream->close();
  }).detach();
}

std::string extract_openai_prompt(const Json::Value& body) {
  const auto contents = parse_contents(body);
  if (!contents.empty()) {
    return contents.front();
  }
  if (body.isMember("messages") && body["messages"].isArray() && !body["messages"].empty()) {
    return normalize_content(body["messages"][body["messages"].size() - 1]["content"]);
  }
  return "";
}

std::string format_openai_prompt(const Json::Value& body, const InferenceEngine& engine) {
  std::string current_prompt;
  const auto contents = parse_contents(body);
  if (!contents.empty()) {
    current_prompt = contents.front();
  }

  std::vector<std::pair<std::string, std::string>> history_messages;
  std::vector<std::string> system_parts;
  const auto system_field = body.get("system", "").asString();
  if (!system_field.empty()) {
    system_parts.push_back(system_field);
  }

  if (body.isMember("messages") && body["messages"].isArray()) {
    for (const auto& msg : body["messages"]) {
      auto role = msg.get("role", "user").asString();
      auto content = normalize_content(msg["content"]);
      if (content.empty()) {
        continue;
      }
      if (role == "system") {
        system_parts.push_back(content);
        continue;
      }
      if (!role.empty()) {
        role[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(role[0])));
      }
      history_messages.emplace_back(role.empty() ? "User" : role, content);
    }
  }

  if (current_prompt.empty() && !history_messages.empty()) {
    current_prompt = history_messages.back().second;
  }

  if (!current_prompt.empty() && !history_messages.empty() &&
      history_messages.back().second == current_prompt) {
    history_messages.pop_back();
  }

  std::vector<std::pair<std::string, std::string>> messages;
  for (const auto& system : system_parts) {
    if (!system.empty()) {
      messages.emplace_back("System", system);
    }
  }
  for (const auto& [role, content] : history_messages) {
    messages.emplace_back(role, content);
  }
  if (!current_prompt.empty()) {
    messages.emplace_back("User", current_prompt);
  }
  if (messages.empty()) {
    messages.emplace_back("User", extract_openai_prompt(body));
  }

  std::string system;
  std::vector<std::pair<std::string, std::string>> dialogue_messages;
  for (const auto& [role, content] : messages) {
    if (role == "System") {
      if (!system.empty()) {
        system += "\n\n";
      }
      system += content;
    } else {
      dialogue_messages.emplace_back(role, content);
    }
  }
  return engine.format_openai_prompt(system, dialogue_messages, body.get("enable_think", false).asBool());
}

Json::Value build_openai_response(
    const InferenceEngine& engine,
    const Json::Value& body,
    const std::string& prompt,
    const std::string& completion) {
  Json::Value resp;
  resp["id"] = "chatcmpl-rwkv-lightning";
  resp["object"] = "chat.completion";
  resp["created"] =
      static_cast<Json::Int64>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
  resp["model"] = body.get("model", engine.model_name()).asString();
  resp["choices"] = Json::arrayValue;
  Json::Value choice;
  choice["index"] = 0;
  choice["message"]["role"] = "assistant";
  choice["message"]["content"] = completion;
  choice["finish_reason"] = "stop";
  resp["choices"].append(choice);
  resp["usage"]["prompt_tokens"] = engine.count_tokens(prompt);
  resp["usage"]["completion_tokens"] = engine.count_tokens(completion);
  resp["usage"]["total_tokens"] =
      resp["usage"]["prompt_tokens"].asInt() + resp["usage"]["completion_tokens"].asInt();
  return resp;
}

std::string make_openai_response_id() {
  static thread_local std::mt19937_64 rng(
      static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
  std::ostringstream oss;
  oss << "chatcmpl-" << std::hex << rng();
  return oss.str();
}

std::string json_to_sse(const Json::Value& payload) {
  Json::StreamWriterBuilder builder;
  builder["emitUTF8"] = true;
  builder["indentation"] = "";
  return "data: " + Json::writeString(builder, payload) + "\n\n";
}

Json::Value build_openai_stream_chunk_base(
    const std::string& response_id,
    Json::Int64 created,
    const std::string& model) {
  Json::Value chunk;
  chunk["id"] = response_id;
  chunk["object"] = "chat.completion.chunk";
  chunk["created"] = created;
  chunk["model"] = model;
  chunk["choices"] = Json::arrayValue;
  return chunk;
}

bool stream_openai_from_internal(
    const InferenceEngine::StreamCallback& emit,
    const std::function<void(const InferenceEngine::StreamCallback&)>& producer,
    const std::string& response_id,
    Json::Int64 created,
    const std::string& model) {
  Json::Value start = build_openai_stream_chunk_base(response_id, created, model);
  Json::Value start_choice;
  start_choice["index"] = 0;
  start_choice["delta"]["role"] = "assistant";
  start_choice["finish_reason"] = Json::Value::null;
  start["choices"].append(start_choice);
  if (!emit(json_to_sse(start))) {
    return false;
  }

  bool sent_finish = false;
  auto wrapped_emit = [&](const std::string& raw) -> bool {
    if (raw == "data: [DONE]\n\n") {
      if (!sent_finish) {
        Json::Value finish = build_openai_stream_chunk_base(response_id, created, model);
        Json::Value finish_choice;
        finish_choice["index"] = 0;
        finish_choice["delta"] = Json::Value(Json::objectValue);
        finish_choice["finish_reason"] = "stop";
        finish["choices"].append(finish_choice);
        if (!emit(json_to_sse(finish))) {
          return false;
        }
        sent_finish = true;
      }
      return emit("data: [DONE]\n\n");
    }

    constexpr std::string_view prefix = "data: ";
    if (raw.rfind(prefix.data(), 0) != 0) {
      return true;
    }
    const auto payload = raw.substr(prefix.size(), raw.size() - prefix.size() - 2);
    Json::CharReaderBuilder reader_builder;
    std::unique_ptr<Json::CharReader> reader(reader_builder.newCharReader());
    Json::Value parsed;
    std::string errs;
    if (!reader->parse(payload.data(), payload.data() + payload.size(), &parsed, &errs)) {
      return true;
    }

    const auto& choices = parsed["choices"];
    if (!choices.isArray() || choices.empty()) {
      return true;
    }

    const auto content = choices[0]["delta"].get("content", "").asString();
    const auto finish_reason = choices[0]["finish_reason"];
    if (!content.empty()) {
      Json::Value chunk = build_openai_stream_chunk_base(response_id, created, model);
      Json::Value choice;
      choice["index"] = 0;
      choice["delta"]["content"] = content;
      choice["finish_reason"] = Json::Value::null;
      chunk["choices"].append(choice);
      if (!emit(json_to_sse(chunk))) {
        return false;
      }
    }
    if (!finish_reason.isNull()) {
      Json::Value chunk = build_openai_stream_chunk_base(response_id, created, model);
      Json::Value choice;
      choice["index"] = 0;
      choice["delta"] = Json::Value(Json::objectValue);
      choice["finish_reason"] = finish_reason;
      chunk["choices"].append(choice);
      sent_finish = true;
      if (!emit(json_to_sse(chunk))) {
        return false;
      }
    }
    return true;
  };

  producer(wrapped_emit);
  if (!sent_finish) {
    Json::Value finish = build_openai_stream_chunk_base(response_id, created, model);
    Json::Value finish_choice;
    finish_choice["index"] = 0;
    finish_choice["delta"] = Json::Value(Json::objectValue);
    finish_choice["finish_reason"] = "stop";
    finish["choices"].append(finish_choice);
    if (!emit(json_to_sse(finish))) {
      return false;
    }
    sent_finish = true;
  }
  return emit("data: [DONE]\n\n");
}

}  // namespace

void register_api_routes(
    InferenceEngine& engine,
    const std::optional<std::string>& password) {
  auto& app = drogon::app();
  app.registerPostHandlingAdvice([](const HttpRequestPtr&, const HttpResponsePtr& resp) {
    resp->addHeader("Access-Control-Allow-Origin", "*");
    resp->addHeader("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
    resp->addHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  });

  auto handle_options = [](const HttpRequestPtr&, std::function<void(const HttpResponsePtr&)>&& cb) {
    auto resp = HttpResponse::newHttpResponse();
    resp->setStatusCode(k204NoContent);
    cb(resp);
  };

  for (const auto& path : {
           "/",
           "/v1/models",
           "/v1/chat/completions",
           "/v2/chat/completions",
           "/translate/v1/batch-translate",
           "/FIM/v1/batch-FIM",
           "/state/chat/completions",
           "/multi_state/chat/completions",
           "/state/status",
           "/state/delete",
           "/big_batch/completions",
           "/openai/v1/chat/completions"}) {
    app.registerHandler(path, handle_options, {Options});
  }

  app.registerHandler(
      "/v1/models",
      [&engine](const HttpRequestPtr&, std::function<void(const HttpResponsePtr&)>&& cb) {
        Json::Value resp;
        resp["object"] = "list";
        resp["data"] = Json::arrayValue;
        Json::Value model;
        model["id"] = engine.model_name();
        model["object"] = "model";
        model["owned_by"] = "rwkv_lightning";
        resp["data"].append(model);
        cb(json_response(std::move(resp)));
      },
      {Get});

  app.registerHandler(
      "/v1/chat/completions",
      [&engine, password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        if (!json) {
          cb(json_response(make_error("Invalid JSON"), k400BadRequest));
          return;
        }
        HttpResponsePtr auth_resp;
        if (!check_password(req, *json, password, auth_resp)) {
          cb(auth_resp);
          return;
        }
        auto prompts = parse_contents(*json);
        if (prompts.empty()) {
          cb(json_response(make_error("Empty prompts list"), k400BadRequest));
          return;
        }
        auto options = parse_options(*json);
        if ((*json).get("stream", false).asBool()) {
          cb(make_sse_response([&engine, prompts, options, chunk_size = (*json).get("chunk_size", 4).asInt()](
                                   ResponseStreamPtr stream) {
            start_streaming_task(
                std::move(stream),
                [&, prompts, options, chunk_size](const InferenceEngine::StreamCallback& emit) {
                  engine.batch_generate_stream(prompts, options, chunk_size, emit);
                });
          }));
          return;
        }

        Json::Value resp;
        resp["id"] = "rwkv7-batch";
        resp["object"] = "chat.completion";
        resp["model"] = (*json).get("model", "rwkv7").asString();
        resp["choices"] = build_choices(engine.batch_generate(prompts, options));
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/v2/chat/completions",
      [&engine, password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        if (!json) {
          cb(json_response(make_error("Invalid JSON"), k400BadRequest));
          return;
        }
        HttpResponsePtr auth_resp;
        if (!check_password(req, *json, password, auth_resp)) {
          cb(auth_resp);
          return;
        }
        auto prompts = parse_contents(*json);
        if (prompts.empty()) {
          cb(json_response(make_error("Empty prompts list"), k400BadRequest));
          return;
        }
        auto options = parse_options(*json);
        if ((*json).get("stream", false).asBool()) {
          cb(make_sse_response([&engine, prompts, options, chunk_size = (*json).get("chunk_size", 4).asInt()](
                                   ResponseStreamPtr stream) {
            start_streaming_task(
                std::move(stream),
                [&, prompts, options, chunk_size](const InferenceEngine::StreamCallback& emit) {
                  engine.continuous_batching_stream(prompts, options, chunk_size, emit);
                });
          }));
          return;
        }
        Json::Value resp;
        resp["id"] = "rwkv7-batch";
        resp["object"] = "chat.completion";
        resp["model"] = (*json).get("model", "rwkv7").asString();
        resp["choices"] = build_choices(engine.continuous_batching(prompts, options));
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/translate/v1/batch-translate",
      [&engine](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        if (!json) {
          cb(json_response(make_error("Invalid JSON"), k400BadRequest));
          return;
        }
        const auto source_lang = (*json).get("source_lang", "auto").asString();
        const auto target_lang = (*json).get("target_lang", "").asString();
        if (target_lang.empty()) {
          cb(json_response(make_error("Missing target_lang"), k400BadRequest));
          return;
        }
        std::vector<std::string> prompts;
        if ((*json).isMember("text_list") && (*json)["text_list"].isArray()) {
          for (const auto& item : (*json)["text_list"]) {
            prompts.push_back(create_translation_prompt(source_lang, target_lang, item.asString()));
          }
        }
        GenerateOptions options;
        options.max_tokens = 2048;
        options.temperature = 1.0;
        options.top_k = 1;
        options.top_p = 0.0;
        options.alpha_presence = 0.0;
        options.alpha_frequency = 0.0;
        options.stop_tokens = {0};
        const auto results = engine.batch_generate(prompts, options);
        Json::Value resp;
        resp["translations"] = Json::arrayValue;
        for (const auto& text : results) {
          Json::Value item;
          item["detected_source_lang"] = source_lang == "auto" ? "en" : source_lang;
          item["text"] = text;
          resp["translations"].append(item);
        }
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/FIM/v1/batch-FIM",
      [&engine, password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        if (!json) {
          cb(json_response(make_error("Invalid JSON"), k400BadRequest));
          return;
        }
        HttpResponsePtr auth_resp;
        if (!check_password(req, *json, password, auth_resp)) {
          cb(auth_resp);
          return;
        }

        std::vector<std::string> prompts;
        if ((*json).isMember("prefix") && (*json)["prefix"].isArray() &&
            (*json).isMember("suffix") && (*json)["suffix"].isArray()) {
          const auto count = std::min((*json)["prefix"].size(), (*json)["suffix"].size());
          for (Json::ArrayIndex i = 0; i < count; ++i) {
            prompts.push_back(
                "✿prefix✿✿suffix✿" + (*json)["suffix"][i].asString() + "✿middle✿" +
                (*json)["prefix"][i].asString());
          }
        }
        if (prompts.empty()) {
          cb(json_response(make_error("Empty FIM prompts"), k400BadRequest));
          return;
        }

        auto options = parse_options(*json);
        const bool stream = (*json).get("stream", false).asBool();
        if (prompts.size() == 1) {
          if (stream) {
            cb(make_sse_response([&engine, prompts, options, chunk_size = (*json).get("chunk_size", 4).asInt()](
                                     ResponseStreamPtr stream_ptr) {
              start_streaming_task(
                  std::move(stream_ptr),
                  [&, prompts, options, chunk_size](const InferenceEngine::StreamCallback& emit) {
                    engine.graph_generate_stream(prompts, options, chunk_size, emit);
                  });
            }));
            return;
          }
          Json::Value resp;
          resp["id"] = "rwkv7-batch-v3";
          resp["object"] = "chat.completion";
          resp["model"] = (*json).get("model", "rwkv7").asString();
          resp["choices"] = build_choices(engine.graph_generate(prompts, options));
          cb(json_response(std::move(resp)));
          return;
        }

        if (stream) {
          cb(make_sse_response([&engine, prompts, options, chunk_size = (*json).get("chunk_size", 4).asInt()](
                                   ResponseStreamPtr stream_ptr) {
            start_streaming_task(
                std::move(stream_ptr),
                [&, prompts, options, chunk_size](const InferenceEngine::StreamCallback& emit) {
                  engine.batch_generate_stream(prompts, options, chunk_size, emit);
                });
          }));
          return;
        }

        Json::Value resp;
        resp["id"] = "rwkv7-batch";
        resp["object"] = "FIM.completion";
        resp["model"] = (*json).get("model", "rwkv7").asString();
        resp["choices"] = build_choices(engine.batch_generate(prompts, options));
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/state/chat/completions",
      [&engine, password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        if (!json) {
          cb(json_response(make_error("Invalid JSON"), k400BadRequest));
          return;
        }
        HttpResponsePtr auth_resp;
        if (!check_password(req, *json, password, auth_resp)) {
          cb(auth_resp);
          return;
        }
        const auto session_id = (*json).get("session_id", "").asString();
        auto prompts = parse_contents(*json);
        if (session_id.empty()) {
          cb(json_response(make_error("Missing session_id"), k400BadRequest));
          return;
        }
        if (prompts.size() != 1) {
          cb(json_response(make_error("Request must be single prompt"), k400BadRequest));
          return;
        }

        auto& manager = StateCacheManager::instance();
        auto state = manager.get_state(session_id).value_or(engine.model()->generate_zero_state(1));
        auto options = parse_options(*json);
        if ((*json).get("stream", false).asBool()) {
          cb(make_sse_response([&engine, &manager, session_id, state = std::move(state), prompts, options,
                                chunk_size = (*json).get("chunk_size", 4).asInt()](ResponseStreamPtr stream) mutable {
            start_streaming_task(
                std::move(stream),
                [&, session_id, state = std::move(state), prompts, options, chunk_size](
                    const InferenceEngine::StreamCallback& emit) mutable {
                  engine.batch_generate_state_stream(prompts, state, options, chunk_size, emit);
                  manager.put_state(session_id, state);
                  release_request_state(state);
                });
          }));
          return;
        }
        auto texts = engine.batch_generate_state(prompts, state, options);
        manager.put_state(session_id, state);
        release_request_state(state);
        Json::Value resp;
        resp["id"] = "rwkv7-batch";
        resp["object"] = "chat.completion";
        resp["model"] = (*json).get("model", "rwkv7").asString();
        resp["choices"] = build_choices(texts);
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/multi_state/chat/completions",
      [&engine, password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        if (!json) {
          cb(json_response(make_error("Invalid JSON"), k400BadRequest));
          return;
        }
        HttpResponsePtr auth_resp;
        if (!check_password(req, *json, password, auth_resp)) {
          cb(auth_resp);
          return;
        }
        if (!json->isMember("dialogue_idx")) {
          cb(json_response(make_error("Missing dialogue_idx parameter"), k400BadRequest));
          return;
        }
        const auto session_index = (*json).get("session_id", "").asString();
        if (session_index.empty()) {
          cb(json_response(make_error("Missing session_id parameter"), k400BadRequest));
          return;
        }
        auto prompts = parse_contents(*json);
        if (prompts.size() != 1) {
          cb(json_response(make_error("Request must be single prompt"), k400BadRequest));
          return;
        }
        const auto dialogue_idx = (*json).get("dialogue_idx", 0).asInt();
        const auto state_key = session_index + ":" + std::to_string(dialogue_idx);
        auto& manager = StateCacheManager::instance();
        auto state = manager.get_state(state_key);
        if (!state.has_value()) {
          if (dialogue_idx != 0) {
            cb(json_response(make_error("State not found for dialogue_idx"), k404NotFound));
            return;
          }
          state = engine.model()->generate_zero_state(1);
        }

        auto options = parse_options(*json);
        if ((*json).get("stream", false).asBool()) {
          cb(make_sse_response([&engine, &manager, session_index, state = std::move(*state), prompts, options,
                                chunk_size = (*json).get("chunk_size", 4).asInt()](ResponseStreamPtr stream) mutable {
            start_streaming_task(
                std::move(stream),
                [&, session_index, state = std::move(state), prompts, options, chunk_size](
                    const InferenceEngine::StreamCallback& emit) mutable {
                  engine.batch_generate_state_stream(prompts, state, options, chunk_size, emit);
                  const auto next_idx = allocate_next_dialogue_idx(manager, session_index);
                  const auto new_session_id = session_index + ":" + std::to_string(next_idx);
                  manager.put_state(new_session_id, state);
                  release_request_state(state);
                  Json::Value meta;
                  meta["object"] = "multi_state.dialogue_idx";
                  meta["session_id"] = new_session_id;
                  meta["dialogue_idx"] = next_idx;
                  Json::StreamWriterBuilder builder;
                  builder["indentation"] = "";
                  emit("data: " + Json::writeString(builder, meta) + "\n\n");
                });
          }));
          return;
        }

        auto texts = engine.batch_generate_state(prompts, *state, options);
        const auto next_idx = allocate_next_dialogue_idx(manager, session_index);
        manager.put_state(session_index + ":" + std::to_string(next_idx), *state);
        release_request_state(*state);
        Json::Value resp;
        resp["id"] = "rwkv7-multi-state";
        resp["object"] = "chat.completion";
        resp["model"] = (*json).get("model", "rwkv7").asString();
        resp["choices"] = build_choices(texts);
        resp["dialogue_idx"] = next_idx;
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/state/status",
      [&password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        Json::Value body = json ? *json : Json::Value(Json::objectValue);
        HttpResponsePtr auth_resp;
        if (!check_password(req, body, password, auth_resp)) {
          cb(auth_resp);
          return;
        }
        auto& manager = StateCacheManager::instance();
        const auto states = manager.list_all_states();
        Json::Value resp;
        resp["status"] = "success";
        resp["l1_cache_count"] = static_cast<int>(states.l1_cache.size());
        resp["l2_cache_count"] = static_cast<int>(states.l2_cache.size());
        resp["database_count"] = static_cast<int>(states.database.size());
        resp["total_sessions"] =
            static_cast<int>(states.l1_cache.size() + states.l2_cache.size() + states.database.size());
        resp["sessions"] = Json::arrayValue;
        auto append_sessions = [&](const std::vector<std::string>& ids, const std::string& level) {
          for (const auto& id : ids) {
            Json::Value item;
            item["session_id"] = id;
            item["cache_level"] = level;
            if (level == "Database (Disk)") {
              if (auto ts = manager.get_db_timestamp(id); ts.has_value()) {
                item["timestamp"] = *ts;
              }
            }
            resp["sessions"].append(item);
          }
        };
        append_sessions(states.l1_cache, "L1 (VRAM)");
        append_sessions(states.l2_cache, "L2 (RAM)");
        append_sessions(states.database, "Database (Disk)");
        cb(json_response(std::move(resp)));
      },
      {Post});

  app.registerHandler(
      "/state/delete",
      [&password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        if (!json) {
          cb(json_response(make_error("Invalid JSON"), k400BadRequest));
          return;
        }
        HttpResponsePtr auth_resp;
        if (!check_password(req, *json, password, auth_resp)) {
          cb(auth_resp);
          return;
        }
        const auto session_id = (*json).get("session_id", "").asString();
        if (session_id.empty()) {
          cb(json_response(make_error("Missing session_id"), k400BadRequest));
          return;
        }
        const bool ok = StateCacheManager::instance().delete_state_from_any_level(session_id);
        Json::Value resp;
        resp["status"] = ok ? "success" : "not_found";
        resp["message"] = ok ? ("Session " + session_id + " deleted successfully")
                             : ("Session " + session_id + " not found in database");
        cb(json_response(std::move(resp), ok ? k200OK : k404NotFound));
      },
      {Post});

  app.registerHandler(
      "/big_batch/completions",
      [&engine, password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        if (!json) {
          cb(json_response(make_error("Invalid JSON"), k400BadRequest));
          return;
        }
        HttpResponsePtr auth_resp;
        if (!check_password(req, *json, password, auth_resp)) {
          cb(auth_resp);
          return;
        }
        auto prompts = parse_contents(*json);
        if (prompts.empty()) {
          cb(json_response(make_error("Empty prompts list"), k400BadRequest));
          return;
        }
        cb(make_sse_response([&engine, prompts, max_tokens = (*json).get("max_tokens", 512).asInt(),
                              temperature = (*json).get("temperature", 1.0).asDouble(),
                              chunk_size = (*json).get("chunk_size", 4).asInt(),
                              stop_tokens = parse_options(*json).stop_tokens](ResponseStreamPtr stream) {
          start_streaming_task(
              std::move(stream),
              [&, prompts, max_tokens, temperature, chunk_size, stop_tokens](
                  const InferenceEngine::StreamCallback& emit) {
                engine.big_batch_stream(prompts, max_tokens, temperature, stop_tokens, chunk_size, emit);
              });
        }));
      },
      {Post});

  app.registerHandler(
      "/openai/v1/chat/completions",
      [&engine, password](const HttpRequestPtr& req, std::function<void(const HttpResponsePtr&)>&& cb) {
        auto json = req->getJsonObject();
        if (!json) {
          cb(json_response(make_error("Invalid JSON"), k400BadRequest));
          return;
        }
        HttpResponsePtr auth_resp;
        if (!check_password(req, *json, password, auth_resp)) {
          cb(auth_resp);
          return;
        }
        const auto prompt = format_openai_prompt(*json, engine);
        const auto options = parse_options(*json);
        const bool use_prefix_cache = (*json).get("use_prefix_cache", true).asBool();
        const auto response_id = make_openai_response_id();
        const auto created = static_cast<Json::Int64>(
            std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
        const auto model_name = (*json).get("model", engine.model_name()).asString();
        if ((*json).get("stream", false).asBool()) {
          cb(make_sse_response([&engine, prompt, options, use_prefix_cache, response_id, created, model_name,
                                chunk_size = (*json).get("chunk_size", 4).asInt()](
                                   ResponseStreamPtr stream) {
            start_streaming_task(
                std::move(stream),
                [&, prompt, options, use_prefix_cache, chunk_size, response_id, created, model_name](
                    const InferenceEngine::StreamCallback& emit) {
                  stream_openai_from_internal(
                      emit,
                      [&](const InferenceEngine::StreamCallback& inner_emit) {
                        if (use_prefix_cache) {
                          engine.single_generate_stream_with_prefix_cache(prompt, options, chunk_size, inner_emit);
                          return;
                        }
                        engine.batch_generate_stream({prompt}, options, chunk_size, inner_emit);
                      },
                      response_id,
                      created,
                      model_name);
                });
          }));
          return;
        }
        const auto texts = use_prefix_cache
                               ? std::vector<std::string>{engine.single_generate_with_prefix_cache(prompt, options)}
                               : engine.batch_generate({prompt}, options);
        auto resp = build_openai_response(engine, *json, prompt, texts.empty() ? std::string{} : texts.front());
        resp["id"] = response_id;
        resp["created"] = created;
        resp["model"] = model_name;
        cb(json_response(std::move(resp)));
      },
      {Post});
}
