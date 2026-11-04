#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include "exception.cuh"

namespace culbm::basic::cuda
{
    struct DeviceDeleter
    {
        void operator()(void* ptr) const noexcept { cudaFree(ptr); }
    };

    template <typename T>
    using DeviceBuffer = std::unique_ptr<T, DeviceDeleter>;

    template <typename T>
    DeviceBuffer<T> allocate(std::size_t count)
    {
        T* ptr;
        culbm::basic::error::check(cudaMalloc(&ptr, count * sizeof(T)));
        return DeviceBuffer<T>(ptr);
    }

    template <typename T>
    void fill(T* data, std::size_t size, T value, cudaStream_t stream)
    {
        static_assert(sizeof(T) == sizeof(std::uint32_t));
        culbm::basic::error::check(cuMemsetD32Async(
            reinterpret_cast<CUdeviceptr>(data), std::bit_cast<std::uint32_t>(value), size, stream));
    }

    struct Stream
    {
        cudaStream_t value = nullptr;
        Stream() = default;
        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;
        ~Stream() { if (value) cudaStreamDestroy(value); }
        void create() { culbm::basic::error::check(cudaStreamCreate(&value)); }
        operator cudaStream_t() const { return value; }
    };

    struct Event
    {
        cudaEvent_t value = nullptr;
        Event() = default;
        Event(const Event&) = delete;
        Event& operator=(const Event&) = delete;
        ~Event() { if (value) cudaEventDestroy(value); }
        void create() { culbm::basic::error::check(cudaEventCreate(&value)); }
        operator cudaEvent_t() const { return value; }
    };
}
