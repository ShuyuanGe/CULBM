#include "measure.hpp"
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <algorithm>
#include <array>
#include <cstddef>

namespace
{
    using namespace culbm::tune;

    __global__ void copy_kernel(float* destination, const float* source, std::size_t count, int passes) {
        const std::size_t stride = gridDim.x * blockDim.x;
        for (int pass = 0; pass < passes; ++pass)
            for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += stride)
                __stcg(destination + index, __ldcg(source + index));
    }

    CopySamples measure(float* source, float* destination, std::size_t bytes, int blocks, cuda::stream_ref stream, cuda::timed_event& start, cuda::timed_event& stop) {
        const std::size_t count = bytes / sizeof(float);
        const int passes = cuda::ceil_div(timed_useful_bytes, 2 * bytes);
        const auto launch = cuda::make_config(cuda::grid_dims(blocks), cuda::block_dims<threads>());
        CopySamples result{bytes, passes, 0.0F, {}, {}};
        for (int sample = 0; sample < sample_count; ++sample) {
            cuda::launch(stream, launch, copy_kernel, destination, source, count, 2);
            start.record(stream);
            cuda::launch(stream, launch, copy_kernel, destination, source, count, passes);
            stop.record(stream);
            stop.sync();
            const float elapsed_ms = (stop - start).count() * 1e-6f;
            result.elapsed_ms[sample] = elapsed_ms;
            result.bandwidth_gbps[sample] = 2.0f * bytes * passes / (elapsed_ms * 1.0e6f);
        }
        auto bandwidths = result.bandwidth_gbps;
        std::sort(bandwidths.begin(), bandwidths.end());
        result.median_gbps = bandwidths[sample_count / 2];
        return result;
    }

}

culbm::tune::Lambda culbm::tune::measure_lambda(cuda::device_ref device) {
    const auto l2_capacity_value = device.attribute(cuda::device_attributes::l2_cache_size);
    const std::size_t l2_capacity = l2_capacity_value;
    const std::size_t l2_bytes = cuda::round_down(l2_capacity / 4, 256);
    const std::size_t dram_bytes = cuda::round_down(l2_capacity * 3, 256);

    cuda::stream stream{device};
    cuda::device_memory_pool pool{device};
    cuda::timed_event start{device}, stop{device};
    cuda::device_buffer<float> source{stream, pool.as_ref(), dram_bytes / sizeof(float), cuda::no_init};
    cuda::device_buffer<float> destination{stream, pool.as_ref(), dram_bytes / sizeof(float), cuda::no_init};

    const auto multiprocessors = device.attribute(cuda::device_attributes::multiprocessor_count);
    const int blocks = multiprocessors * blocks_per_sm;
    const auto dram = measure(source.data(), destination.data(), dram_bytes, blocks, stream, start, stop);
    const auto l2 = measure(source.data(), destination.data(), l2_bytes, blocks, stream, start, stop);
    return {l2.median_gbps, dram.median_gbps, dram.median_gbps / l2.median_gbps, l2, dram};
}
