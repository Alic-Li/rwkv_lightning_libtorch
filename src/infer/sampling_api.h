#pragma once

#include <cstdint>

#include <torch/torch.h>

torch::Tensor setup_rand(int64_t seed, int64_t batch_size);

torch::Tensor batch_sampling_repetition_temperature_topk_topp(
    torch::Tensor& logits,
    torch::Tensor& penalties,
    torch::Tensor& states,
    double presence_penalty,
    double repetition_penalty,
    double penalty_decay,
    double temperature,
    int64_t top_k,
    double top_p);

torch::Tensor batch_sampling_temperature_topk_topp(
    torch::Tensor& logits,
    torch::Tensor& states,
    double temperature,
    int64_t top_k,
    double top_p);
