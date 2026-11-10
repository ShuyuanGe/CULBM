#include <format>
#include "error.hpp"
#include "cuda_runtime_api.h"

namespace culbm::gpu
{
    namespace
    {
        std::string format_error(const char* api, const char* description, int code, const std::source_location& loc)
        {
            return std::format("CUDA {} error: {} (code {}) at {}:{}",
                               api, description ? description : "unknown error", code, loc.file_name(), loc.line());
        }

        const char* driver_description(CUresult err)
        {
            const char* description = nullptr;
            cuGetErrorString(err, &description);
            return description;
        }
    }

    RuntimeError::RuntimeError(cudaError_t err, const std::source_location& loc)
        : std::runtime_error(format_error("runtime", cudaGetErrorString(err), err, loc)) {}

    DriverError::DriverError(CUresult err, const std::source_location& loc)
        : std::runtime_error(format_error("driver", driver_description(err), err, loc)) {}

    void check(cudaError_t err, const std::source_location& loc)
    {
        if (err != cudaSuccess) throw RuntimeError(err, loc);
    }

    void check(CUresult err, const std::source_location& loc)
    {
        if (err != CUDA_SUCCESS) throw DriverError(err, loc);
    }
}
