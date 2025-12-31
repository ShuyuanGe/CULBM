#pragma once

#include <array>
#include "vec3.hpp"
#include "config.hpp"

namespace culbm::simulator::multi_dev
{
    template<std::uint32_t NDIR>
    struct KernelParam
    {
        real_t invTau;
        flag_t *flagBuf = nullptr;
        real_t *rhoBuf  = nullptr;
        real_t *vxBuf   = nullptr;
        real_t *vyBuf   = nullptr;
        real_t *vzBuf   = nullptr;
        std::array<ddf_t*, NDIR> srcDDFBuf {nullptr};
        ddf_t  *dstDDFBuf = nullptr;
    };

    __global__ void D3Q27BGKKernel(const KernelParam<27> __grid_constant__ param);

    template<std::uint32_t NDIR>
    struct InitKernelParam
    {
        culbm::basic::Vec3<std::uint32_t> devDim;
        culbm::basic::Vec3<std::uint32_t> devIdx;
        flag_t *flagBuf = nullptr;
        real_t *rhoBuf  = nullptr;
        real_t *vxBuf   = nullptr;
        real_t *vyBuf   = nullptr;
        real_t *vzBuf   = nullptr;
        ddf_t  *srcDDFBuf = nullptr;
        ddf_t  *dstDDFBuf = nullptr;
    };

    __global__ void D3Q27BGKInitKernel(const InitKernelParam<27> __grid_constant__ param);
}