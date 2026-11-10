#include <algorithm>
#include <cmath>
#include <optional>
#include <print>
#include <stdexcept>
#include <string_view>
#include <cuda_runtime.h>
#include "CLI11.hpp"
#include "gpu/error.hpp"
#include "report.hpp"
#include "selector.hpp"

namespace
{
    void validate(std::optional<double> lambda, std::optional<double> gamma, std::optional<double> capacity)
    {
        if ((lambda and (not std::isfinite(*lambda) or *lambda < 0)) or
            (gamma and (not std::isfinite(*gamma) or *gamma < 0)) or
            (capacity and (not std::isfinite(*capacity) or *capacity <= 0)))
            throw std::invalid_argument("lambda and gamma must be finite and nonnegative; capacity must be positive and finite.");
    }

}

int main(int argc, char** argv) try
{
    CLI::App app{"Measure a GPU and select a tile and temporal depth"};
    int device = 0, q = 0, i_min = 2, i_max = 20, b_max = 160;
    std::optional<double> lambda, gamma, capacity;
    app.add_option("--device", device, "CUDA-visible device index")->default_val(0)->check(CLI::NonNegativeNumber);
    app.add_option("--q", q, "Velocity count (FP32)")->required()->check(CLI::IsMember({15, 19, 27}));
    app.add_option_function<double>("--lambda", [&](double value) { lambda = value; }, "Measured bandwidth ratio");
    app.add_option_function<double>("--gamma", [&](double value) { gamma = value; }, "Measured partial-sector penalty");
    app.add_option_function<double>("--capacity", [&](double value) { capacity = value; }, "Available capacity in MiB");
    app.add_option("--i-min", i_min, "Minimum depth")->default_val(2);
    app.add_option("--i-max", i_max, "Maximum depth")->default_val(20);
    app.add_option("--b-max", b_max, "Maximum tile edge")->default_val(160);
    try { app.parse(argc, argv); }
    catch (const CLI::ParseError& e) { return app.exit(e); }

    if (i_min < 2 or i_max < i_min or b_max < 32)
        throw std::invalid_argument("Require 2 <= i-min <= i-max and b-max >= 32.");
    validate(lambda, gamma, capacity);
    const bool needs_gpu = not (lambda and gamma and capacity);
    if (needs_gpu)
    {
        culbm::gpu::check(cudaSetDevice(device));
        const cuda::device_ref gpu{device};
        const auto name = gpu.name();
        std::println("[Device]");
        std::println("Device {}: {}", device, std::string_view{name.data(), name.size()});
        if (not capacity)
        {
            const auto persisting = gpu.attribute(cuda::device_attributes::max_persisting_l2_cache_size);
            const auto window = gpu.attribute(cuda::device_attributes::max_access_policy_window_size);
            const auto capacity_bytes = std::min(persisting, window);
            if (capacity_bytes <= 0) throw std::runtime_error("No persisting L2 capacity or access-policy window is available.");
            capacity = capacity_bytes / 1048576.0;
            std::println("capacity: {:.6g} MiB ({} bytes)", *capacity, capacity_bytes);
        }
        if (not lambda)
        {
            std::println("\n[Lambda]");
            const auto lambda_result = culbm::tune::measure_lambda(gpu);
            culbm::tune::report(lambda_result);
            lambda = lambda_result.value;
        }
        if (not gamma)
        {
            std::println("\n[Gamma]");
            const auto gamma_result = culbm::tune::measure_gamma(gpu);
            culbm::tune::report(gamma_result);
            gamma = gamma_result.value;
        }
    }
    validate(lambda, gamma, capacity);
    std::println("{}[Inputs]", needs_gpu ? "\n" : "");
    std::println("lambda: {:.5g}\ngamma: {:.4g}\ncapacity: {:.6g} MiB", *lambda, *gamma, *capacity);
    const auto result = culbm::tune::select(q, *capacity * 1048576.0, *lambda, *gamma, i_min, i_max, b_max);
    std::println("\n[Selection]");
    std::println("B*: [{},{},{}]", result.bx, result.by, result.bz);
    std::println("I*: {}", result.i);
    std::println("\n[Launch options]");
    std::println("--block '[{},1,1]' --grid '[1,{},{}]' --inner-steps {}", result.bx, result.by, result.bz, result.i);
}
catch (const std::exception& error)
{
    std::println(stderr, "{}", error.what());
    return 1;
}
