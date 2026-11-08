#pragma once

#include <cstdint>
#include <string_view>
#include <cuda_runtime.h>
#include "types.hpp"
#include "flags.hpp"
#include "execution.hpp"

namespace culbm::simulator::single_dev
{
    struct BlockingParam
    {
        static constexpr bool dumpCorrectOnly = true;
        real_t invTau = 0;
        idx_t offx = 0, offy = 0, offz = 0;
        idx_t glbnx = 0, glbny = 0, glbnz = 0;
        const flag_t* flagBuf = nullptr;
        real_t* rhoBuf = nullptr;
        real_t* vxBuf = nullptr;
        real_t* vyBuf = nullptr;
        real_t* vzBuf = nullptr;
        ddf_t* glbSrcDDFBuf = nullptr;
        ddf_t* glbDstDDFBuf = nullptr;
        ddf_t* tileDDFBuf = nullptr;
        idx_t glbDDFRingNz = 0;
        idx_t glbSrcDDFOffz = 0;
        idx_t glbDstDDFOffz = 0;
    };

    struct PlainParam
    {
        static constexpr bool dumpCorrectOnly = false;
        real_t invTau = 0;
        const flag_t* flagBuf = nullptr;
        real_t* rhoBuf = nullptr;
        real_t* vxBuf = nullptr;
        real_t* vyBuf = nullptr;
        real_t* vzBuf = nullptr;
        ddf_t* glbSrcDDFBuf = nullptr;
        ddf_t* glbDstDDFBuf = nullptr;
    };

    using Equilibrium = void (*)(real_t, real_t, real_t, real_t, ddf_t*);

    using PlainKernel = void (*)(PlainParam);
    using BlockingKernel = void (*)(BlockingParam);

    struct PlainPullKernels
    {
        static constexpr PlainScheme scheme = PlainScheme::TwoFieldPull;
        static PlainPullKernels make(int q, std::string_view kind);
        Equilibrium equilibrium;
        PlainKernel step, dump;
    };

    struct PlainEsoKernels
    {
        static constexpr PlainScheme scheme = PlainScheme::EsoTwist;
        static PlainEsoKernels make(int q, std::string_view kind);
        Equilibrium equilibrium;
        PlainKernel stepEven, stepOdd, dumpEven, dumpOdd;
    };

    struct BlockingKernels
    {
        static BlockingKernels make(int q, std::string_view kind, BlockingGlobalBuffer buffer, bool hints);
        Equilibrium equilibrium;
        BlockingKernel single, first, middleEven, middleOdd, lastEven, lastOdd, dump;
    };
}
