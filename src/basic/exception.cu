#include <format>
#include "exception.cuh"
#include "cuda_runtime_api.h"

namespace culbm::basic::error
{
    namespace
    {
        std::string formatError(const char* api, const char* description, int code, const std::source_location& loc)
        {
            return std::format("CUDA {} error: {} (code {}) at {}:{}",
                               api, description ? description : "unknown error", code, loc.file_name(), loc.line());
        }

        const char* driverDescription(CUresult err)
        {
            const char* description = nullptr;
            cuGetErrorString(err, &description);
            return description;
        }
    }

    CudaRuntimeError::CudaRuntimeError(cudaError_t err, const std::source_location& loc)
        : std::runtime_error(formatError("runtime", cudaGetErrorString(err), err, loc)) {}

    CudaDriverError::CudaDriverError(CUresult err, const std::source_location& loc)
        : std::runtime_error(formatError("driver", driverDescription(err), err, loc)) {}

    void check(cudaError_t err, const std::source_location& loc)
    {
        if (err != cudaSuccess) throw CudaRuntimeError(err, loc);
    }

    void check(CUresult err, const std::source_location& loc)
    {
        if (err != CUDA_SUCCESS) throw CudaDriverError(err, loc);
    }
}
