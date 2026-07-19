#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>

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

float measure(float* source, float* destination, std::size_t bytes, int blocks, cudaEvent_t start, cudaEvent_t stop) {
    const std::size_t count = bytes / sizeof(float);
    const int passes = (kTimedUsefulBytes + 2 * bytes - 1) / (2 * bytes);
    std::array<float, kSamples> bandwidths{};
    for (float& bandwidth : bandwidths) {
        copy_kernel<<<blocks, kThreads>>>(destination, source, count, 2);
        cudaEventRecord(start);
        copy_kernel<<<blocks, kThreads>>>(destination, source, count, passes);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        bandwidth = 2.0f * bytes * passes / (elapsed_ms * 1.0e6f);
    }
    std::sort(bandwidths.begin(), bandwidths.end());
    return bandwidths[kSamples / 2];
}

int main() {
    int l2_capacity_value;
    cudaDeviceGetAttribute(&l2_capacity_value, cudaDevAttrL2CacheSize, 0);
    const std::size_t l2_capacity = l2_capacity_value;
    const std::size_t l2_bytes = align_down(l2_capacity / 4);
    const std::size_t dram_bytes = align_down(l2_capacity * 3);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float *source, *destination;
    cudaMalloc(&source, dram_bytes);
    cudaMalloc(&destination, dram_bytes);

    int multiprocessors = 0;
    cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0);
    const int blocks = multiprocessors * kBlocksPerSm;
    const float dram = measure(source, destination, dram_bytes, blocks, start, stop);
    const float l2 = measure(source, destination, l2_bytes, blocks, start, stop);
    const float lambda = dram / l2;

    std::printf("L2 median: %.4g GB/s\n", l2);
    std::printf("DRAM median: %.4g GB/s\n", dram);
    std::printf("lambda: %.5f\n", lambda);
}
