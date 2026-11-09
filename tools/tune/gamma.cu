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
constexpr std::size_t kWarpUnits = 1ULL << 23;
constexpr int kThreads = 128;
constexpr int kBlocksPerSm = 8;
constexpr int kSamples = 101;
constexpr int kPartialOffset = 7;

template <bool Partial>
__global__ void store_kernel(float* destination) {
    const std::size_t lane = threadIdx.x % warpSize;
    const std::size_t warp = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const std::size_t warp_stride = gridDim.x * blockDim.x / warpSize;

    for (std::size_t unit = warp; unit < kWarpUnits; unit += warp_stride) {
        if (not Partial || lane < warpSize - kPartialOffset) {
            const std::size_t offset = Partial ? lane + kPartialOffset : lane;
            __stcs(destination + unit * warpSize + offset, 0.0F);
        }
    }
}

template <bool Partial>
float time_store(float* destination, int blocks, cuda::stream_ref stream, cuda::timed_event& start, cuda::timed_event& stop) {
    const auto launch = cuda::make_config(cuda::grid_dims(blocks), cuda::block_dims<kThreads>());
    cuda::launch(stream, launch, store_kernel<Partial>, destination);
    start.record(stream);
    cuda::launch(stream, launch, store_kernel<Partial>, destination);
    stop.record(stream);
    stop.sync();
    return (stop - start).count() * 1e-6f;
}

}

culbm::tune::Gamma culbm::tune::measureGamma(cuda::device_ref device) {
    const auto multiprocessors = device.attribute(cuda::device_attributes::multiprocessor_count);
    const int blocks = multiprocessors * kBlocksPerSm;

    cuda::stream stream{device};
    cuda::device_memory_pool pool{device};
    cuda::timed_event start{device}, stop{device};
    cuda::device_buffer<float> destination{stream, pool.as_ref(), kWarpUnits * 128 / sizeof(float), cuda::no_init};

    std::array<float, kSamples> full_times{};
    std::array<float, kSamples> partial_times{};
    for (int sample = 0; sample < kSamples; ++sample) {
        full_times[sample] = time_store<false>(destination.data(), blocks, stream, start, stop);
        partial_times[sample] = time_store<true>(destination.data(), blocks, stream, start, stop);
    }

    std::sort(full_times.begin(), full_times.end());
    std::sort(partial_times.begin(), partial_times.end());
    const float full_median = full_times[kSamples / 2];
    const float partial_median = partial_times[kSamples / 2];
    return {full_median, partial_median, 4.0F * partial_median / full_median - 3.0F};
}
