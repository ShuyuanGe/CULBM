#pragma once

#include <cuda_runtime.h>
#include "local_config.hpp"

namespace gf::simulator::single_dev_expt
{
    struct HaloBlockingL1L2Param
    {
        real_t invTau = 0;
        idx_t nloop = 0;
        idx_t offx = 0, offy = 0, offz = 0;
        idx_t glbnx = 0, glbny = 0, glbnz = 0;
        const flag_t* blkFlagBuf = nullptr;
        real_t* glbRhoBuf = nullptr;
        real_t* glbVxBuf = nullptr;
        real_t* glbVyBuf = nullptr;
        real_t* glbVzBuf = nullptr;
        ddf_t* glbSrcDDFBuf = nullptr;
        ddf_t* glbDstDDFBuf = nullptr;
        real_t* glbSwapDDFBuf = nullptr;
    };

    __global__ void HaloBlockingL1L2D3Q27PullKernel(const HaloBlockingL1L2Param __grid_constant__ param);
    __global__ void HaloBlockingL1L2D3Q27DumpKernel(const HaloBlockingL1L2Param __grid_constant__ param);

    struct HaloBlockingStaticL2Param
    {
        real_t invTau = 0;
        idx_t nloop = 0;
        idx_t offx = 0, offy = 0, offz = 0;
        idx_t glbnx = 0, glbny = 0, glbnz = 0;
        const flag_t* blkFlagBuf = nullptr;
        real_t* glbRhoBuf = nullptr;
        real_t* glbVxBuf = nullptr;
        real_t* glbVyBuf = nullptr;
        real_t* glbVzBuf = nullptr;
        ddf_t* glbSrcDDFBuf = nullptr;
        ddf_t* glbDstDDFBuf = nullptr;
        real_t* blkDDFBuf0 = nullptr;
        real_t* blkDDFBuf1 = nullptr;
    };

    __global__ void HaloBlockingStaticL2D3Q27PullKernel(const HaloBlockingStaticL2Param __grid_constant__ param);
    __global__ void HaloBlockingStaticL2D3Q27DumpKernel(const HaloBlockingStaticL2Param __grid_constant__ param);

    struct HaloBlockingDynamicL2InplaceParam
    {
        real_t invTau = 0;
        idx_t offx = 0, offy = 0, offz = 0;
        idx_t glbnx = 0, glbny = 0, glbnz = 0;
        const flag_t* blkFlagBuf = nullptr;
        real_t* glbRhoBuf = nullptr;
        real_t* glbVxBuf = nullptr;
        real_t* glbVyBuf = nullptr;
        real_t* glbVzBuf = nullptr;
        ddf_t* glbSrcDDFBuf = nullptr;
        ddf_t* glbDstDDFBuf = nullptr;
        real_t* blkDDFBuf = nullptr;
    };

    template<bool OneIter>
    __global__ void HaloBlockingDynamicL2D3Q27InplaceFirstKernel(const HaloBlockingDynamicL2InplaceParam __grid_constant__ param);
    template<bool EvenIter>
    __global__ void HaloBlockingDynamicL2D3Q27InplaceMiddleKernel(const HaloBlockingDynamicL2InplaceParam __grid_constant__ param);
    template<bool EvenIter>
    __global__ void HaloBlockingDynamicL2D3Q27InplaceLastKernel(const HaloBlockingDynamicL2InplaceParam __grid_constant__ param);
    __global__ void HaloBlockingDynamicL2D3Q27InplaceDumpKernel(const HaloBlockingDynamicL2InplaceParam __grid_constant__ param);

    struct HaloBlockingStaticL2InplaceParam
    {
        real_t invTau = 0;
        idx_t nloop = 0;
        idx_t offx = 0, offy = 0, offz = 0;
        idx_t glbnx = 0, glbny = 0, glbnz = 0;
        const flag_t* blkFlagBuf = nullptr;
        real_t* glbRhoBuf = nullptr;
        real_t* glbVxBuf = nullptr;
        real_t* glbVyBuf = nullptr;
        real_t* glbVzBuf = nullptr;
        ddf_t* glbSrcDDFBuf = nullptr;
        ddf_t* glbDstDDFBuf = nullptr;
        real_t* blkDDFBuf = nullptr;
    };

    __global__ void HaloBlockingStaticL2D3Q27InplaceKernel(const HaloBlockingStaticL2InplaceParam __grid_constant__ param);
    __global__ void HaloBlockingStaticL2D3Q27InplaceDumpKernel(const HaloBlockingStaticL2InplaceParam __grid_constant__ param);

    struct StaticPullParam
    {
        real_t invTau = 0;
        const flag_t* glbFlagBuf = nullptr;
        real_t* glbRhoBuf = nullptr;
        real_t* glbVxBuf = nullptr;
        real_t* glbVyBuf = nullptr;
        real_t* glbVzBuf = nullptr;
        ddf_t* glbSrcDDFBuf = nullptr;
        ddf_t* glbDstDDFBuf = nullptr;
    };

    __global__ void StaticD3Q27PullKernel(const StaticPullParam __grid_constant__ param);
    __global__ void StaticD3Q27PullDumpKernel(const StaticPullParam __grid_constant__ param);
}