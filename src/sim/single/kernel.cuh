#pragma once

#include <string_view>
#include <cuda_runtime.h>
#include "core/types.hpp"
#include "flags.hpp"
#include "mode.hpp"

namespace culbm::sim::single
{
    struct BlockingParam
    {
        static constexpr bool dumpOwnedOnly = true;
        real_t invTau = 0;
        idx_t originX = 0, originY = 0, originZ = 0;
        idx_t globalNx = 0, globalNy = 0, globalNz = 0;
        const flag_t* flagBuf = nullptr;
        real_t* rhoBuf = nullptr;
        real_t* vxBuf = nullptr;
        real_t* vyBuf = nullptr;
        real_t* vzBuf = nullptr;
        ddf_t* srcDdf = nullptr;
        ddf_t* dstDdf = nullptr;
        ddf_t* tileDdf = nullptr;
        idx_t ringNz = 0;
        idx_t srcZOffset = 0;
        idx_t dstZOffset = 0;
    };

    struct PlainParam
    {
        static constexpr bool dumpOwnedOnly = false;
        real_t invTau = 0;
        const flag_t* flagBuf = nullptr;
        real_t* rhoBuf = nullptr;
        real_t* vxBuf = nullptr;
        real_t* vyBuf = nullptr;
        real_t* vzBuf = nullptr;
        ddf_t* srcDdf = nullptr;
        ddf_t* dstDdf = nullptr;
    };

    using Equilibrium = void (*)(real_t, real_t, real_t, real_t, ddf_t*);

    using PlainKernel = void (*)(PlainParam);
    using BlockingKernel = void (*)(BlockingParam);

    struct PullKernels
    {
        static constexpr Mode mode = Mode::Pull;
        static PullKernels make(int q, std::string_view kind);
        Equilibrium equilibrium;
        PlainKernel step, dump;
    };

    struct TwistKernels
    {
        static constexpr Mode mode = Mode::Twist;
        static TwistKernels make(int q, std::string_view kind);
        Equilibrium equilibrium;
        PlainKernel stepEven, stepOdd, dumpEven, dumpOdd;
    };

    struct BlockingKernels
    {
        static BlockingKernels make(int q, std::string_view kind, Mode buffer, bool hints);
        Equilibrium equilibrium;
        BlockingKernel single, first, middleEven, middleOdd, lastEven, lastOdd, dump;
    };
}
