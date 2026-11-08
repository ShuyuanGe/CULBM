#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include <print>
#include <string_view>
#include <cuda/devices>
#include "exception.cuh"

int main()
{
    try
    {
        int driver, runtime;
        culbm::basic::error::check(cudaDriverGetVersion(&driver));
        culbm::basic::error::check(cudaRuntimeGetVersion(&runtime));
        std::println("CUDA driver: {}.{}, runtime: {}.{}, devices: {}", driver / 1000, driver % 1000 / 10, runtime / 1000, runtime % 1000 / 10, cuda::devices.size());

        for (const auto device : cuda::devices)
        {
            using namespace cuda::device_attributes;
            const auto attr = [device](auto attribute) { return device.attribute(attribute); };
            const auto name = device.name();
            cudaDeviceProp prop{};
            culbm::basic::error::check(cudaGetDeviceProperties(&prop, device.get()));
            std::println("\nDevice {}: {} (compute {}.{})", device.get(), std::string_view{name.data(), name.size()}, attr(compute_capability_major), attr(compute_capability_minor));
            std::println("  SMs: {}, warp size: {}", attr(multiprocessor_count), attr(warp_size));
            std::println("  Global memory: {:.1f} MiB, memory bus: {} bits", prop.totalGlobalMem / 1048576.0, attr(global_memory_bus_width));
            std::println("  L2 cache: {:.1f} MiB, persisting L2 limit: {:.1f} MiB, access window limit: {:.1f} MiB", attr(l2_cache_size) / 1048576.0, attr(max_persisting_l2_cache_size) / 1048576.0, attr(max_access_policy_window_size) / 1048576.0);
            std::println("  Registers per SM / block: {} / {}", attr(max_registers_per_multiprocessor), attr(max_registers_per_block));
            std::println("  Shared memory per SM / block / opt-in block: {:.1f} / {:.1f} / {:.1f} KiB", attr(max_shared_memory_per_multiprocessor) / 1024.0, attr(max_shared_memory_per_block) / 1024.0, attr(max_shared_memory_per_block_optin) / 1024.0);
            std::println("  Maximum threads per SM / block: {} / {}, blocks per SM: {}", attr(max_threads_per_multiprocessor), attr(max_threads_per_block), attr(max_blocks_per_multiprocessor));
            std::println("  Maximum block dimensions: [{}, {}, {}]", attr(max_block_dim_x), attr(max_block_dim_y), attr(max_block_dim_z));
            std::println("  Maximum grid dimensions: [{}, {}, {}]", attr(max_grid_dim_x), attr(max_grid_dim_y), attr(max_grid_dim_z));
            std::println("  Concurrent kernels: {}, asynchronous copy engines: {}, unified addressing: {}", attr(concurrent_kernels), device.attribute<cudaDevAttrAsyncEngineCount>(), attr(unified_addressing));
        }
    }
    catch (const std::exception& error)
    {
        std::println(std::cerr, "{}", error.what());
        return 1;
    }
}
