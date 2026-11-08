#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <algorithm>
#include <array>
#include <cstddef>
#include <print>

constexpr std::size_t kTimedUsefulBytes = 1ULL << 34;
constexpr int kThreads = 128;
constexpr int kBlocksPerSm = 8;
constexpr int kSamples = 101;

std::size_t align_down(std::size_t bytes) { return bytes / 256 * 256; }

__global__ void copy_kernel(float* destination, const float* source, std::size_t count, int passes) {
    const std::size_t stride = gridDim.x * blockDim.x;
    for (int pass = 0; pass < passes; ++pass)
        for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += stride)
            __stcg(destination + index, __ldcg(source + index));
}

float measure(float* source, float* destination, std::size_t bytes, int blocks, cuda::stream_ref stream, cuda::timed_event& start, cuda::timed_event& stop) {
    const std::size_t count = bytes / sizeof(float);
    const int passes = (kTimedUsefulBytes + 2 * bytes - 1) / (2 * bytes);
    const auto launch = cuda::make_config(cuda::grid_dims(blocks), cuda::block_dims<kThreads>());
    std::array<float, kSamples> bandwidths{};
    for (float& bandwidth : bandwidths) {
        cuda::launch(stream, launch, copy_kernel, destination, source, count, 2);
        start.record(stream);
        cuda::launch(stream, launch, copy_kernel, destination, source, count, passes);
        stop.record(stream);
        stop.sync();
        const float elapsed_ms = (stop - start).count() * 1e-6f;
        bandwidth = 2.0f * bytes * passes / (elapsed_ms * 1.0e6f);
    }
    std::sort(bandwidths.begin(), bandwidths.end());
    return bandwidths[kSamples / 2];
}

int main() {
    const cuda::device_ref device{0};
    const auto l2_capacity_value = device.attribute(cuda::device_attributes::l2_cache_size);
    const std::size_t l2_capacity = l2_capacity_value;
    const std::size_t l2_bytes = align_down(l2_capacity / 4);
    const std::size_t dram_bytes = align_down(l2_capacity * 3);

    cuda::stream stream{device};
    cuda::device_memory_pool pool{device};
    cuda::timed_event start{device}, stop{device};
    cuda::device_buffer<float> source{stream, pool.as_ref(), dram_bytes / sizeof(float), cuda::no_init};
    cuda::device_buffer<float> destination{stream, pool.as_ref(), dram_bytes / sizeof(float), cuda::no_init};

    const auto multiprocessors = device.attribute(cuda::device_attributes::multiprocessor_count);
    const int blocks = multiprocessors * kBlocksPerSm;
    const float dram = measure(source.data(), destination.data(), dram_bytes, blocks, stream, start, stop);
    const float l2 = measure(source.data(), destination.data(), l2_bytes, blocks, stream, start, stop);
    const float lambda = dram / l2;

    std::println("L2 median: {:.4g} GB/s", l2);
    std::println("DRAM median: {:.4g} GB/s", dram);
    std::println("lambda: {:.5f}", lambda);
}
