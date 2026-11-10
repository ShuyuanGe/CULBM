#include "measure.hpp"
#include <cuda/buffer>
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

    template <bool partial>
    __global__ void store_kernel(float* destination) {
        const std::size_t lane = threadIdx.x % warpSize;
        const std::size_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
        const std::size_t warp_stride = gridDim.x * blockDim.x / warpSize;

        for (std::size_t unit = warp; unit < warp_units; unit += warp_stride) {
            if (not partial || lane < warpSize - partial_offset) {
                const std::size_t offset = partial ? lane + partial_offset : lane;
                __stcs(destination + unit * warpSize + offset, 0.0F);
            }
        }
    }

    template <bool partial>
    float time_store(float* destination, int blocks, cuda::stream_ref stream, cuda::timed_event& start, cuda::timed_event& stop) {
        const auto launch = cuda::make_config(cuda::grid_dims(blocks), cuda::block_dims<threads>());
        cuda::launch(stream, launch, store_kernel<partial>, destination);
        start.record(stream);
        cuda::launch(stream, launch, store_kernel<partial>, destination);
        stop.record(stream);
        stop.sync();
        return (stop - start).count() * 1e-6f;
    }

}

culbm::tune::Gamma culbm::tune::measure_gamma(cuda::device_ref device) {
    const auto multiprocessors = device.attribute(cuda::device_attributes::multiprocessor_count);
    const int blocks = multiprocessors * blocks_per_sm;

    cuda::stream stream{device};
    cuda::device_memory_pool pool{device};
    cuda::timed_event start{device}, stop{device};
    cuda::device_buffer<float> destination{stream, pool.as_ref(), warp_units * 128 / sizeof(float), cuda::no_init};

    Gamma result{0.0F, 0.0F, 0.0F, {}, {}};
    for (int sample = 0; sample < sample_count; ++sample) {
        result.full_ms[sample] = time_store<false>(destination.data(), blocks, stream, start, stop);
        result.partial_ms[sample] = time_store<true>(destination.data(), blocks, stream, start, stop);
    }

    auto full_times = result.full_ms;
    auto partial_times = result.partial_ms;
    std::sort(full_times.begin(), full_times.end());
    std::sort(partial_times.begin(), partial_times.end());
    result.full = full_times[sample_count / 2];
    result.partial = partial_times[sample_count / 2];
    result.value = 4.0F * result.partial / result.full - 3.0F;
    return result;
}
