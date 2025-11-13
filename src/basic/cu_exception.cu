#include <format>
#include "cu_exception.cuh"
#include "cuda_runtime_api.h"

namespace gf::basic::cu
{
    CudaRuntimeError::CudaRuntimeError(cudaError_t err, std::source_location loc)
        : std::runtime_error(
            std::format(
                "====CUDA RUNTIME ERROR====\n\tdescription:{}\n\tlocation: {}:{}\n", 
                cudaGetErrorString(err), 
                loc.file_name(),
                loc.column()
            )
        ) {}

    const char* CudaRuntimeError::what() const noexcept { return std::runtime_error::what(); }

    void check(cudaError_t err, std::source_location loc)
    {
        if(err != cudaSuccess) throw CudaRuntimeError(err, loc);
    }
}