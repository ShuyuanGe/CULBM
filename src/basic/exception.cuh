#pragma once

#include <stdexcept>
#include <cuda.h>
#include <driver_types.h>
#include <source_location>

namespace culbm::basic::error
{
    class CudaRuntimeError : public std::runtime_error
    {
        public:
            CudaRuntimeError(cudaError_t err, const std::source_location& loc = std::source_location::current());

    };

    class CudaDriverError : public std::runtime_error
    {
        public:
            CudaDriverError(CUresult err, const std::source_location& loc = std::source_location::current());
    };

    void check(cudaError_t err, const std::source_location& loc = std::source_location::current());
    void check(CUresult err, const std::source_location& loc = std::source_location::current());
}
