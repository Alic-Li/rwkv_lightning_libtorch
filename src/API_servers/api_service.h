#pragma once

#include <memory>
#include <optional>
#include <string>

namespace drogon {
class HttpAppFramework;
}

class InferenceEngine;

void register_api_routes(
    InferenceEngine& engine,
    const std::optional<std::string>& password);
