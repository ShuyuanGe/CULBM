#pragma once

#include <cuda_runtime.h>

namespace culbm::gpu::io
{
    struct Default
    {
        template <typename T>
        static constexpr T load(const T* ptr) { return *ptr; }
        template <typename T>
        static constexpr void store(T* ptr, T value) { *ptr = value; }
    };

    struct Streaming
    {
        template <typename T>
        static constexpr T load(const T* ptr) { return __ldcs(ptr); }
        template <typename T>
        static constexpr void store(T* ptr, T value) { __stcs(ptr, value); }
    };

    struct Cached
    {
        template <typename T>
        static constexpr T load(const T* ptr) { return __ldca(ptr); }
        template <typename T>
        static constexpr void store(T* ptr, T value) { __stwb(ptr, value); }
    };
}
