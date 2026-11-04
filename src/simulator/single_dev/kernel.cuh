#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include "types.hpp"
#include "flags.hpp"
#include "method/kind.hpp"

namespace culbm::simulator::single_dev
{
    using idx_t = std::int32_t;
    using glb_idx_t = std::int64_t;

    struct BlockingParam
    {
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
        ddf_t* blkDDFBuf = nullptr;
        idx_t glbDDFRingNz = 0;
        idx_t glbSrcDDFOffz = 0;
        idx_t glbDstDDFOffz = 0;
    };

    struct PlainParam
    {
        real_t invTau = 0;
        const flag_t* flagBuf = nullptr;
        real_t* glbRhoBuf = nullptr;
        real_t* glbVxBuf = nullptr;
        real_t* glbVyBuf = nullptr;
        real_t* glbVzBuf = nullptr;
        ddf_t* glbSrcDDFBuf = nullptr;
        ddf_t* glbDstDDFBuf = nullptr;
    };

    struct Kernels
    {
        void (*equilibrium)(real_t, real_t, real_t, real_t, ddf_t*);
        struct { const void *step, *dump; } plain;
        struct { const void *stepEven, *stepOdd, *dumpEven, *dumpOdd; } plainEso;
        struct { const void *single, *first, *middleEven, *middleOdd, *lastEven, *lastOdd, *dump; } blocking;
    };

    template <int NDir, culbm::lbm_core::method::Kind Kind>
    Kernels kernels(bool cacheHints, bool ring);
}
