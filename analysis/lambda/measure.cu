#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>

constexpr int kTimedPasses = 128;
constexpr int kSamples = 101;

std::size_t align_down(std::size_t bytes) { return bytes / 256 * 256; }

__global__ void copy_kernel(uint4* destination, const uint4* source, std::size_t count, int repeats) {
    const std::size_t stride = gridDim.x * blockDim.x;
    for (int repeat = 0; repeat < repeats; ++repeat)
        for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += stride)
            destination[index] = __ldcg(source + index);
}

float measure(uint4* source, uint4* destination, std::size_t bytes, int blocks, int threads, cudaEvent_t start, cudaEvent_t stop) {
    const std::size_t count = bytes / sizeof(uint4);
    std::array<float, kSamples> bandwidths{};
    for (float& bandwidth : bandwidths) {
        copy_kernel<<<blocks, threads>>>(destination, source, count, 1);
        cudaEventRecord(start);
        copy_kernel<<<blocks, threads>>>(destination, source, count, kTimedPasses);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_ms = 0.0F;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        bandwidth = 2.0f * bytes * kTimedPasses / (elapsed_ms * 1.0e6f);
    }
    std::sort(bandwidths.begin(), bandwidths.end());
    return bandwidths[kSamples / 2];
}

int main() {
    int l2_capacity;
    cudaDeviceGetAttribute(&l2_capacity, cudaDevAttrL2CacheSize, 0);
    const std::size_t l2_bytes = align_down(l2_capacity / 16 * 7);
    const std::size_t dram_bytes = align_down(l2_capacity);

    cudaEvent_t start = nullptr, stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    uint4* source = nullptr;
    uint4* destination = nullptr;
    cudaMalloc(&source, dram_bytes);
    cudaMalloc(&destination, dram_bytes);

    int blocks, threads;
    cudaOccupancyMaxPotentialBlockSize(&blocks, &threads, copy_kernel);
    const float dram = measure(source, destination, dram_bytes, blocks, threads, start, stop);
    const float l2 = measure(source, destination, l2_bytes, blocks, threads, start, stop);
    const float lambda = dram / l2;

    std::printf("L2 median: %.4g GB/s\n", l2);
    std::printf("DRAM median: %.4g GB/s\n", dram);
    std::printf("lambda: %.4f\n", lambda);
}
