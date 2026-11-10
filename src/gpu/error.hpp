#pragma once

#include <stdexcept>
#include <cuda.h>
#include <driver_types.h>
#include <source_location>

namespace culbm::gpu
{
    class RuntimeError : public std::runtime_error
    {
        public:
            RuntimeError(cudaError_t err, const std::source_location& loc = std::source_location::current());

    };

    class DriverError : public std::runtime_error
    {
        public:
            DriverError(CUresult err, const std::source_location& loc = std::source_location::current());
    };

    void check(cudaError_t err, const std::source_location& loc = std::source_location::current());
    void check(CUresult err, const std::source_location& loc = std::source_location::current());
}
