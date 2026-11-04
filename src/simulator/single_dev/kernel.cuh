#pragma once

#include <cstdint>
#include <string_view>
#include <cuda_runtime.h>
#include "types.hpp"
#include "flags.hpp"
#include "execution.hpp"

namespace culbm::simulator::single_dev
{
    using idx_t = std::int32_t;
    using glb_idx_t = std::int64_t;

    struct BlockingParam
    {
        static constexpr bool dumpCorrectOnly = true;
        real_t invTau = 0;
        idx_t offx = 0, offy = 0, offz = 0;
        idx_t glbnx = 0, glbny = 0, glbnz = 0;
        const flag_t* flagBuf = nullptr;
        real_t* glbRhoBuf = nullptr;
        real_t* glbVxBuf = nullptr;
        real_t* glbVyBuf = nullptr;
        real_t* glbVzBuf = nullptr;
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
        real_t* glbRhoBuf = nullptr;
        real_t* glbVxBuf = nullptr;
        real_t* glbVyBuf = nullptr;
        real_t* glbVzBuf = nullptr;
        ddf_t* glbSrcDDFBuf = nullptr;
        ddf_t* glbDstDDFBuf = nullptr;
    };

    using Equilibrium = void (*)(real_t, real_t, real_t, real_t, ddf_t*);

    struct PlainPullKernels
    {
        static constexpr PlainScheme scheme = PlainScheme::TwoFieldPull;
        Equilibrium equilibrium;
        const void *step, *dump;
    };

    struct PlainEsoKernels
    {
        static constexpr PlainScheme scheme = PlainScheme::EsoTwist;
        Equilibrium equilibrium;
        const void *stepEven, *stepOdd, *dumpEven, *dumpOdd;
    };

    struct BlockingKernels
    {
        Equilibrium equilibrium;
        const void *single, *first, *middleEven, *middleOdd, *lastEven, *lastOdd, *dump;
    };

    using KernelSelection = std::variant<PlainPullKernels, PlainEsoKernels, BlockingKernels>;

    KernelSelection makeKernels(int q, std::string_view kind, const StrategySelection& strategy, bool hints);
}
