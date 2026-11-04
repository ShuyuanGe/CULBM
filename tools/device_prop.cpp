#include <cuda_runtime.h>
#include <exception>
#include <print>
#include "exception.cuh"

int main()
{
    try
    {
        int count, driver, runtime;
        culbm::basic::error::check(cudaGetDeviceCount(&count));
        culbm::basic::error::check(cudaDriverGetVersion(&driver));
        culbm::basic::error::check(cudaRuntimeGetVersion(&runtime));
        std::println("CUDA driver: {}.{}, runtime: {}.{}, devices: {}", driver / 1000, driver % 1000 / 10, runtime / 1000, runtime % 1000 / 10, count);

        for (int device = 0; device < count; ++device)
        {
            cudaDeviceProp prop{};
            culbm::basic::error::check(cudaGetDeviceProperties(&prop, device));
            std::println("\nDevice {}: {} (compute {}.{})", device, prop.name, prop.major, prop.minor);
            std::println("  SMs: {}, warp size: {}", prop.multiProcessorCount, prop.warpSize);
            std::println("  Global memory: {:.1f} MiB, memory bus: {} bits", prop.totalGlobalMem / 1048576.0, prop.memoryBusWidth);
            std::println("  L2 cache: {:.1f} MiB, persisting L2 limit: {:.1f} MiB, access window limit: {:.1f} MiB", prop.l2CacheSize / 1048576.0, prop.persistingL2CacheMaxSize / 1048576.0, prop.accessPolicyMaxWindowSize / 1048576.0);
            std::println("  Registers per SM / block: {} / {}", prop.regsPerMultiprocessor, prop.regsPerBlock);
            std::println("  Shared memory per SM / block / opt-in block: {:.1f} / {:.1f} / {:.1f} KiB", prop.sharedMemPerMultiprocessor / 1024.0, prop.sharedMemPerBlock / 1024.0, prop.sharedMemPerBlockOptin / 1024.0);
            std::println("  Maximum threads per SM / block: {} / {}, blocks per SM: {}", prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerBlock, prop.maxBlocksPerMultiProcessor);
            std::println("  Maximum block dimensions: [{}, {}, {}]", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
            std::println("  Maximum grid dimensions: [{}, {}, {}]", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
            std::println("  Concurrent kernels: {}, asynchronous copy engines: {}, unified addressing: {}", prop.concurrentKernels != 0, prop.asyncEngineCount, prop.unifiedAddressing != 0);
        }
    }
    catch (const std::exception& error)
    {
        std::println(stderr, "{}", error.what());
        return 1;
    }
}
