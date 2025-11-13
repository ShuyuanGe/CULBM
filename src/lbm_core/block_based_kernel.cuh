#pragma once

#include "config.hpp"
#include "defines.hpp"
#include <cuda_runtime.h>
#include "device_function.hpp"

namespace gf::lbm_core
{
    namespace detail
    {
        DefOptionalMember(DomX, domX);
        DefOptionalMember(DomY, domY);
        DefOptionalMember(DomZ, domZ);
        DefOptionalMember(InvTau, invTau);
    }

    struct BlockBasedKernelParam final :
        public detail::OptionalMemberDomX<not USE_STATIC_CONFIG, i32>,
        public detail::OptionalMemberDomY<not USE_STATIC_CONFIG, i32>, 
        public detail::OptionalMemberDomZ<not USE_STATIC_CONFIG, i32>, 
        public detail::OptionalMemberInvTau<not USE_STATIC_CONFIG, i32>
    {
        i32 xOff = 0, yOff = 0, zOff = 0;
        ddf_t *srcBuf = nullptr, *dstBuf = nullptr;
        flag_t *flagBuf = nullptr;
        real_t *rhoBuf = nullptr;
        real_t *vxBuf = nullptr, *vyBuf = nullptr, *vzBuf = nullptr;
        ddf_t *blkBuf0 = nullptr, *blkBuf1 = nullptr;
    };

    __global__ void staticBlockBasedKernel(const BlockBasedKernelParam __grid_constant__ param);

    __global__ void dynamicBlockBasedKernelFirstIter(const BlockBasedKernelParam __grid_constant__ param);
    __global__ void dynamicBlockBasedKernelLastIter(const BlockBasedKernelParam __grid_constant__ param);
    __global__ void dynamicBlockBasedKernelInnerIter(const BlockBasedKernelParam __grid_constant__ param);

    __global__ void dynamicBlockBasedKernelSingleIter(const BlockBasedKernelParam __grid_constant__ param);
}