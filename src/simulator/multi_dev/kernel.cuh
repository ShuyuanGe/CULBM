#pragma once

#include <array>
#include <cuda_runtime.h>
#include "types.hpp"
#include "flags.hpp"
#include "method/kind.hpp"

namespace culbm::simulator::multi_dev
{
    struct KernelParam
    {
        real_t invTau;
        flag_t *flagBuf = nullptr;
        real_t *rhoBuf  = nullptr;
        real_t *vxBuf   = nullptr;
        real_t *vyBuf   = nullptr;
        real_t *vzBuf   = nullptr;
        std::array<ddf_t*, 27> srcDDFBuf {nullptr};
        ddf_t  *dstDDFBuf = nullptr;
    };

    struct InitKernelParam
    {
        uint3 devDim{};
        uint3 devIdx{};
        flag_t *flagBuf = nullptr;
        real_t *rhoBuf  = nullptr;
        real_t *vxBuf   = nullptr;
        real_t *vyBuf   = nullptr;
        real_t *vzBuf   = nullptr;
        ddf_t  *srcDDFBuf = nullptr;
        ddf_t  *dstDDFBuf = nullptr;
    };

    struct Kernels { const void* init; const void* step; };

    template <int NDir, culbm::lbm_core::method::Kind Kind>
    Kernels kernels();
}
