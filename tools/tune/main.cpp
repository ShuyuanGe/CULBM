#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <print>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include "CLI11.hpp"
#include "gpu/error.hpp"
#include "report.hpp"
#include "selector.hpp"

namespace
{
    struct Options
    {
        int device = 0, q = 0, i_min = 2, i_max = 20;
        std::optional<int> b_max;
        std::optional<double> lambda, gamma, capacity;
        std::string report_json;
    };

    struct Device
    {
        int index, driver_version, runtime_version;
        cudaDeviceProp properties;
    };

    void validate(const Options& options)
    {
        for (auto value : {options.lambda, options.gamma})
            if (value and (not std::isfinite(*value) or *value < 0))
                throw std::invalid_argument("lambda and gamma must be finite and nonnegative.");
        if (options.capacity and (not std::isfinite(*options.capacity) or *options.capacity <= 0))
            throw std::invalid_argument("Persisting capacity must be finite and positive.");
        if (options.i_min < 2 or options.i_max < options.i_min or options.i_max > 512 or
            (options.b_max and *options.b_max < 32))
            throw std::invalid_argument("Require 2 <= i-min <= i-max <= 512 and b-max >= 32.");
    }

    template <typename T, std::size_t N>
    void write_array(std::ostream& out, const std::array<T, N>& values)
    {
        const char* separator = "";
        out << '[';
        for (const auto value : values)
        {
            out << separator << value;
            separator = ",";
        }
        out << ']';
    }

    void write_copy(std::ostream& out, const culbm::tune::CopySamples& samples)
    {
        out << "{\"bytes\":" << samples.bytes << ",\"passes\":" << samples.passes
            << ",\"median_gbps\":" << samples.median_gbps << ",\"elapsed_ms\":";
        write_array(out, samples.elapsed_ms);
        out << ",\"bandwidth_gbps\":";
        write_array(out, samples.bandwidth_gbps);
        out << '}';
    }

    void write_json(const Options& options, const std::array<bool, 3>& provided, const culbm::tune::Selection& result,
                    const std::optional<Device>& device, const std::optional<culbm::tune::Lambda>& lambda,
                    const std::optional<culbm::tune::Gamma>& gamma)
    {
        using namespace culbm::tune;
        std::ofstream out;
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out.open(options.report_json);
        out << std::setprecision(std::numeric_limits<double>::max_digits10);
        out << "{\n\"schema_version\":2,\n\"model_id\":\"" << model_id << "\",\n\"device\":";
        if (device)
        {
            const auto& properties = device->properties;
            out << "{\"index\":" << device->index << ",\"name\":" << std::quoted(properties.name)
                << ",\"driver_version\":" << device->driver_version << ",\"runtime_version\":" << device->runtime_version
                << ",\"compute_capability\":[" << properties.major << ',' << properties.minor
                << "],\"multiprocessors\":" << properties.multiProcessorCount
                << ",\"l2_capacity_bytes\":" << properties.l2CacheSize << '}';
        }
        else out << "null";
        out << ",\n\"inputs\":{\"q\":" << options.q << ",\"lambda\":" << *options.lambda
            << ",\"gamma\":" << *options.gamma << ",\"capacity_mib\":" << *options.capacity
            << ",\"capacity_bytes\":" << *options.capacity * 1048576.0 << "},\n\"sources\":{"
            << "\"lambda\":\"" << (provided[0] ? "provided" : "measured")
            << "\",\"gamma\":\"" << (provided[1] ? "provided" : "measured")
            << "\",\"capacity\":\"" << (provided[2] ? "provided" : "queried") << "\"},\n"
            << "\"search\":{\"i_min\":" << options.i_min << ",\"i_max\":" << options.i_max
            << ",\"bx_max\":1024,\"b_max\":";
        if (options.b_max) out << *options.b_max;
        else out << "null";
        out << ",\"candidate_count\":" << result.candidate_count << "},\n\"selection\":{\"B\":["
            << result.bx << ',' << result.by << ',' << result.bz << "],\"I\":" << result.i
            << ",\"score\":" << result.score << ",\"block\":[" << result.bx << ",1,1],\"grid\":[1,"
            << result.by << ',' << result.bz << "]},\n\"calibration\":{\"lambda\":";
        if (lambda)
        {
            out << "{\"method\":\"two_size_copy\",\"samples\":" << sample_count
                << ",\"threads\":" << threads << ",\"blocks_per_sm\":" << blocks_per_sm
                << ",\"timed_useful_bytes\":" << timed_useful_bytes << ",\"warmup_passes\":2,\"l2\":";
            write_copy(out, lambda->l2_samples);
            out << ",\"dram\":";
            write_copy(out, lambda->dram_samples);
            out << '}';
        }
        else out << "null";
        out << ",\"gamma\":";
        if (gamma)
        {
            out << "{\"method\":\"matched_pure_store\",\"samples\":" << sample_count
                << ",\"threads\":" << threads << ",\"blocks_per_sm\":" << blocks_per_sm
                << ",\"warp_units\":" << warp_units << ",\"partial_offset\":" << partial_offset
                << ",\"warmup_launches_per_control\":1,\"buffer_bytes\":" << warp_units * 128
                << ",\"full_median_ms\":" << gamma->full << ",\"partial_median_ms\":" << gamma->partial
                << ",\"full_ms\":";
            write_array(out, gamma->full_ms);
            out << ",\"partial_ms\":";
            write_array(out, gamma->partial_ms);
            out << '}';
        }
        else out << "null";
        out << "}}\n";
        out.close();
    }
}

int main(int argc, char** argv) try
{
    CLI::App app{"Measure a GPU and select a tile and temporal depth"};
    Options options;
    app.add_option("--device", options.device, "CUDA-visible device index")->default_val(0)->check(CLI::NonNegativeNumber);
    app.add_option("--q", options.q, "Velocity count (FP32)")->required()->check(CLI::IsMember({15, 19, 27}));
    app.add_option_function<double>("--lambda", [&](double value) { options.lambda = value; }, "Measured bandwidth ratio");
    app.add_option_function<double>("--gamma", [&](double value) { options.gamma = value; }, "Measured partial-sector penalty");
    app.add_option_function<double>("--capacity", [&](double value) { options.capacity = value; }, "Persisting-L2 capacity in MiB");
    app.add_option("--i-min", options.i_min, "Minimum depth")->default_val(2);
    app.add_option("--i-max", options.i_max, "Maximum depth")->default_val(20);
    app.add_option_function<int>("--b-max", [&](int value) { options.b_max = value; }, "Optional tile-edge bound for restricted exploration");
    app.add_option("--report-json", options.report_json, "Write inputs, search, selection, and measured raw samples as JSON");
    try { app.parse(argc, argv); }
    catch (const CLI::ParseError& e) { return app.exit(e); }

    validate(options);
    const std::array<bool, 3> provided{options.lambda.has_value(), options.gamma.has_value(),
                                     options.capacity.has_value()};
    const bool needs_gpu = not (provided[0] and provided[1] and provided[2]);
    std::optional<Device> device;
    std::optional<culbm::tune::Lambda> lambda;
    std::optional<culbm::tune::Gamma> gamma;
    if (needs_gpu)
    {
        culbm::gpu::check(cudaSetDevice(options.device));
        const cuda::device_ref gpu{options.device};
        device.emplace();
        device->index = options.device;
        culbm::gpu::check(cudaGetDeviceProperties(&device->properties, options.device));
        culbm::gpu::check(cudaDriverGetVersion(&device->driver_version));
        culbm::gpu::check(cudaRuntimeGetVersion(&device->runtime_version));
        std::println("[Device]\nDevice {}: {}", options.device, device->properties.name);
        if (not options.capacity)
        {
            const auto persisting = gpu.attribute(cuda::device_attributes::max_persisting_l2_cache_size);
            const auto window = gpu.attribute(cuda::device_attributes::max_access_policy_window_size);
            const auto capacity_bytes = std::min(persisting, window);
            if (capacity_bytes <= 0) throw std::runtime_error("No persisting L2 capacity or access-policy window is available.");
            options.capacity = capacity_bytes / 1048576.0;
        }
        if (not options.lambda)
        {
            std::println("\n[Lambda]");
            lambda = culbm::tune::measure_lambda(gpu);
            culbm::tune::report(*lambda);
            options.lambda = lambda->value;
        }
        if (not options.gamma)
        {
            std::println("\n[Gamma]");
            gamma = culbm::tune::measure_gamma(gpu);
            culbm::tune::report(*gamma);
            options.gamma = gamma->value;
        }
    }
    validate(options);
    std::println("{}[Inputs]", needs_gpu ? "\n" : "");
    std::println("lambda: {:.5g}\ngamma: {:.4g}\ncapacity: {:.6g} MiB",
                 *options.lambda, *options.gamma, *options.capacity);
    const auto result = culbm::tune::select(options.q, *options.capacity * 1048576.0,
                                           *options.lambda, *options.gamma, options.i_min, options.i_max, options.b_max);
    std::println("\n[Selection]\nB*: [{},{},{}]\nI*: {}\nscore: {:.12g}\ncandidates: {}",
                 result.bx, result.by, result.bz, result.i, result.score, result.candidate_count);
    std::println("\n[Launch options]\n--block '[{},1,1]' --grid '[1,{},{}]' --inner-steps {}",
                 result.bx, result.by, result.bz, result.i);
    if (not options.report_json.empty()) write_json(options, provided, result, device, lambda, gamma);
}
catch (const std::exception& error)
{
    std::println(stderr, "{}", error.what());
    return 1;
}
