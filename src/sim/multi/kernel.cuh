#pragma once

#include <array>
#include <string_view>
#include <cuda_runtime.h>
#include "core/types.hpp"
#include "flags.hpp"

namespace culbm::sim::multi
{
    struct KernelParam
    {
        real_t invTau;
        flag_t *flagBuf = nullptr;
        real_t *rhoBuf  = nullptr;
        real_t *vxBuf   = nullptr;
        real_t *vyBuf   = nullptr;
        real_t *vzBuf   = nullptr;
        std::array<ddf_t*, 27> srcDdf {nullptr};
        ddf_t  *dstDdf = nullptr;
    };

    struct InitKernelParam
    {
        dim3 devDim{};
        uint3 devIdx{};
        flag_t *flagBuf = nullptr;
        real_t *rhoBuf  = nullptr;
        real_t *vxBuf   = nullptr;
        real_t *vyBuf   = nullptr;
        real_t *vzBuf   = nullptr;
        ddf_t  *srcDdf = nullptr;
        ddf_t  *dstDdf = nullptr;
    };

    struct Kernels { void (*init)(InitKernelParam); void (*step)(KernelParam); };

    Kernels makeKernels(int q, std::string_view kind);
}
