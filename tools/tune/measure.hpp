#pragma once

#include <array>
#include <cstddef>
#include <cuda/devices>

namespace culbm::tune
{
    inline constexpr int sample_count = 101;
    inline constexpr int threads = 128;
    inline constexpr int blocks_per_sm = 8;
    inline constexpr std::size_t timed_useful_bytes = 1ULL << 34;
    inline constexpr std::size_t warp_units = 1ULL << 23;
    inline constexpr int partial_offset = 7;

    struct CopySamples
    {
        std::size_t bytes;
        int passes;
        float median_gbps;
        std::array<float, sample_count> elapsed_ms, bandwidth_gbps;
    };
    struct Lambda
    {
        float l2, dram, value;
        CopySamples l2_samples, dram_samples;
    };
    struct Gamma
    {
        float full, partial, value;
        std::array<float, sample_count> full_ms, partial_ms;
    };

    Lambda measure_lambda(cuda::device_ref device);
    Gamma measure_gamma(cuda::device_ref device);
}
