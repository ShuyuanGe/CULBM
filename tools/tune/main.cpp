#include <algorithm>
#include <cmath>
#include <print>
#include <stdexcept>
#include <string_view>
#include <cuda_runtime.h>
#include "CLI11.hpp"
#include "gpu/error.hpp"
#include "report.hpp"
#include "selector.hpp"

int main(int argc, char** argv) try
{
    CLI::App app{"Measure a GPU and select a tile and temporal depth"};
    int device = 0, q = 0, i_min = 2, i_max = 20, b_max = 160;
    double lambda = 0, gamma = 0, capacity = 0;
    app.add_option("--device", device, "CUDA-visible device index")->default_val(0)->check(CLI::NonNegativeNumber);
    app.add_option("--q", q, "Velocity count (FP32)")->required()->check(CLI::IsMember({15, 19, 27}));
    auto* lambda_opt = app.add_option("--lambda", lambda, "Measured bandwidth ratio");
    auto* gamma_opt = app.add_option("--gamma", gamma, "Measured partial-sector penalty");
    auto* capacity_opt = app.add_option("--capacity", capacity, "Available capacity in MiB");
    app.add_option("--i-min", i_min, "Minimum depth")->default_val(2);
    app.add_option("--i-max", i_max, "Maximum depth")->default_val(20);
    app.add_option("--b-max", b_max, "Maximum tile edge")->default_val(160);
    try { app.parse(argc, argv); }
    catch (const CLI::ParseError& e) { return app.exit(e); }

    const int supplied = (lambda_opt->count() > 0) + (gamma_opt->count() > 0) + (capacity_opt->count() > 0);
    if (supplied != 0 and supplied != 3) throw std::invalid_argument("Provide --lambda, --gamma and --capacity together.");
    if (i_min < 2 or i_max < i_min or b_max < 32) throw std::invalid_argument("Require 2 <= i-min <= i-max and b-max >= 32.");
    if (supplied == 0)
    {
        culbm::gpu::check(cudaSetDevice(device));
        const cuda::device_ref gpu{device};
        const auto name = gpu.name();
        const auto persisting = gpu.attribute(cuda::device_attributes::max_persisting_l2_cache_size);
        const auto window = gpu.attribute(cuda::device_attributes::max_access_policy_window_size);
        const auto capacity_bytes = std::min(persisting, window);
        if (capacity_bytes <= 0) throw std::runtime_error("No persisting L2 capacity or access-policy window is available.");
        capacity = capacity_bytes / 1048576.0;
        std::println("Device {}: {}", device, std::string_view{name.data(), name.size()});
        std::println("capacity: {} MiB ({} bytes)", capacity, capacity_bytes);
        std::println("Measuring lambda...");
        const auto lambda_result = culbm::tune::measure_lambda(gpu);
        culbm::tune::report(lambda_result);
        lambda = lambda_result.value;
        std::println("Measuring gamma...");
        const auto gamma_result = culbm::tune::measure_gamma(gpu);
        culbm::tune::report(gamma_result);
        gamma = gamma_result.value;
    }
    else std::println("Inputs: lambda={}, gamma={}, capacity={} MiB", lambda, gamma, capacity);
    if (not std::isfinite(lambda) or lambda < 0 or not std::isfinite(gamma) or gamma < 0 or not std::isfinite(capacity) or capacity <= 0)
        throw std::invalid_argument("lambda and gamma must be finite and nonnegative; capacity must be positive and finite.");
    const auto result = culbm::tune::select(q, capacity * 1048576.0, lambda, gamma, i_min, i_max, b_max);
    std::println("B*: [{},{},{}]", result.bx, result.by, result.bz);
    std::println("I*: {}", result.i);
    std::println("--block '[{},1,1]' --grid '[1,{},{}]' --inner-steps {}", result.bx, result.by, result.bz, result.i);
}
catch (const std::exception& error)
{
    std::println(stderr, "{}", error.what());
    return 1;
}
