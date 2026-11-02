#pragma once

#include <stdexcept>
#include <driver_types.h>
#include <source_location>

namespace culbm::basic::error
{
    class CudaRuntimeError : public std::runtime_error
    {
        public:
            CudaRuntimeError(cudaError_t err, const std::source_location& loc = std::source_location::current());

            const char* what() const noexcept;
    };

    void check(cudaError_t err, const std::source_location& loc = std::source_location::current());
}
