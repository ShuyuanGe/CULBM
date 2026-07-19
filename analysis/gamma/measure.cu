#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>

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
        if (!Partial || lane < warpSize - kPartialOffset) {
            const std::size_t offset = Partial ? lane + kPartialOffset : lane;
            __stcs(destination + unit * warpSize + offset, 0.0F);
        }
    }
}

template <bool Partial>
float time_store(float* destination, int blocks, cudaEvent_t start, cudaEvent_t stop) {
    store_kernel<Partial><<<blocks, kThreads>>>(destination);
    cudaEventRecord(start);
    store_kernel<Partial><<<blocks, kThreads>>>(destination);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    return elapsed_ms;
}

int main() {
    int multiprocessors = 0;
    cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0);
    const int blocks = multiprocessors * kBlocksPerSm;

    float* destination;
    cudaMalloc(&destination, kWarpUnits * 128);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::array<float, kSamples> full_times{};
    std::array<float, kSamples> partial_times{};
    for (int sample = 0; sample < kSamples; ++sample) {
        full_times[sample] = time_store<false>(destination, blocks, start, stop);
        partial_times[sample] = time_store<true>(destination, blocks, start, stop);
    }

    std::sort(full_times.begin(), full_times.end());
    std::sort(partial_times.begin(), partial_times.end());
    const float full_median = full_times[kSamples / 2];
    const float partial_median = partial_times[kSamples / 2];
    const float gamma = 4.0F * partial_median / full_median - 3.0F;

    std::printf("4F median: %.5f ms\n", full_median);
    std::printf("3F+1P median: %.5f ms\n", partial_median);
    std::printf("gamma: %.4f\n", gamma);
}
