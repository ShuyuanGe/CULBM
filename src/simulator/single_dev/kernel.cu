#include "kernel.cuh"
#include "method/method.hpp"
#include "macroscopic.hpp"
#include "mapping.cuh"
#include "accessor.hpp"
#include "ddf.hpp"
#include "mem.cuh"
#include <algorithm>
#include <type_traits>

namespace culbm::simulator::single_dev
{
    namespace mem      = culbm::basic::mem;
    namespace accessor = culbm::lbm_core::accessor;
    namespace ddf      = culbm::lbm_core::ddf;
    using culbm::lbm_core::method::Method;
    using culbm::lbm_core::method::Kind;
    using culbm::lbm_core::Stencil;

    enum class Mode { Plain, Single, First, MiddleOdd, MiddleEven, LastOdd, LastEven, EsoEven, EsoOdd };

    template <Mode mode>
    constexpr bool isPlain = mode == Mode::Plain || mode == Mode::EsoEven || mode == Mode::EsoOdd;

    template <int NDir, Kind Kind, Mode mode, bool CacheHints, bool UseRing>
    __global__ void StreamCollideKernel(const std::conditional_t<isPlain<mode>, PlainParam, BlockingParam> __grid_constant__ param)
    {
        using GlobalIO = std::conditional_t<CacheHints, mem::Streaming, mem::Default>;
        using TileIO   = std::conditional_t<CacheHints, mem::Cached,    mem::Default>;
        using GlobalAccessor = accessor::PullTwoFields;
        using EvenAccessor   = accessor::EsoTwistEven;
        using OddAccessor    = accessor::EsoTwistOdd;
        const auto pos = mapping::makePosition(param);
        const auto flag = param.flagBuf[pos.blki];
        const auto makeGlobal = [&] { return mapping::makeGlobalView<UseRing>(param, pos); };
        const auto makeTile =
            [&, param, /* capture value to consume more regs and generate less instructions;
                        * capture reference instead to help resue reg when it spills */
             offsets = mapping::TileOffsets{pos} /* reuse offset across load and store */
            ]<typename Param = decltype(param)> { return mapping::makeTileView<Param>(param, pos, offsets); };

        ddf_t f[NDir];
        if (flag & LOAD_DDF_BIT)
        {
            if      constexpr (mode == Mode::Plain || mode == Mode::Single || mode == Mode::First)
                ddf::load<NDir, GlobalAccessor, GlobalIO>(f, makeGlobal());
            else if constexpr (mode == Mode::EsoEven)
                ddf::load<NDir, EvenAccessor, GlobalIO>(f, makeGlobal());
            else if constexpr (mode == Mode::EsoOdd)
                ddf::load<NDir, OddAccessor, GlobalIO>(f, makeGlobal());
            else if constexpr (mode == Mode::MiddleOdd || mode == Mode::LastOdd)
                ddf::load<NDir, EvenAccessor, TileIO>(f, makeTile());
            else if constexpr (mode == Mode::MiddleEven || mode == Mode::LastEven)
                ddf::load<NDir, OddAccessor, TileIO>(f, makeTile());
        }

        if (flag & EQU_DDF_BIT)
        {
            const auto rho = param.glbRhoBuf[pos.glbi];
            const auto vx = param.glbVxBuf[pos.glbi];
            const auto vy = param.glbVyBuf[pos.glbi];
            const auto vz = param.glbVzBuf[pos.glbi];
            Method<Kind, NDir>::equilibrium(rho, vx, vy, vz, f);
        }

        if (flag & BOUNCE_BACK_BIT)
        {
            #pragma unroll
            for (int i = 0; i < NDir / 2; ++i)
                std::swap(f[i], f[Stencil<NDir>::opposite(i)]);
        }

        if (flag & COLLIDE_BIT)
            Method<Kind, NDir>::collide(param.invTau, f);

        if (flag & STORE_DDF_BIT)
        {
            if      constexpr (mode == Mode::Plain || mode == Mode::Single)
                ddf::store<NDir, GlobalAccessor, GlobalIO>(f, makeGlobal());
            else if constexpr (mode == Mode::EsoEven)
                ddf::store<NDir, EvenAccessor, GlobalIO>(f, makeGlobal());
            else if constexpr (mode == Mode::EsoOdd)
                ddf::store<NDir, OddAccessor, GlobalIO>(f, makeGlobal());
            else if constexpr (mode == Mode::First || mode == Mode::MiddleEven)
                ddf::store<NDir, OddAccessor, TileIO>(f, makeTile());
            else if constexpr (mode == Mode::MiddleOdd)
                ddf::store<NDir, EvenAccessor, TileIO>(f, makeTile());
            else if constexpr (mode == Mode::LastOdd || mode == Mode::LastEven)
                if (flag & CORRECT_BIT)
                    ddf::store<NDir, GlobalAccessor, GlobalIO>(f, makeGlobal());
        }
    }

    template <int NDir, bool Plain, bool UseRing, bool Eso, bool Even>
    __global__ void DumpKernel(const std::conditional_t<Plain, PlainParam, BlockingParam> __grid_constant__ param)
    {
        const auto pos = mapping::makePosition(param);
        const auto flag = param.flagBuf[pos.blki];
        if constexpr (not Plain)
            if (not (flag & CORRECT_BIT)) return;

        ddf_t f[NDir];
        const auto view = mapping::makeGlobalView<UseRing>(param, pos);
        if      constexpr (not Eso)
            ddf::load<NDir, accessor::PullTwoFields, mem::Default>(f, view);
        else if constexpr (Even)
            ddf::load<NDir, accessor::EsoTwistEven, mem::Default>(f, view);
        else if constexpr (not Even)
            ddf::load<NDir, accessor::EsoTwistOdd, mem::Default>(f, view);
        real_t rho, vx, vy, vz;
        culbm::lbm_core::macroscopic::compute<NDir>(rho, vx, vy, vz, f);

        if (flag & DUMP_RHO_BIT) param.glbRhoBuf[pos.glbi] = rho;
        if (flag & DUMP_VX_BIT)  param.glbVxBuf[pos.glbi] = vx;
        if (flag & DUMP_VY_BIT)  param.glbVyBuf[pos.glbi] = vy;
        if (flag & DUMP_VZ_BIT)  param.glbVzBuf[pos.glbi] = vz;
    }

    template <int NDir, Kind Kind, bool UseRing, bool CacheHints>
    Kernels makeKernels()
    {
        return {
            .equilibrium  = &Method<Kind, NDir>::equilibrium,

            .plain = {
                .step = (const void*)&StreamCollideKernel<NDir, Kind, Mode::Plain, false, false>,
                .dump = (const void*)&DumpKernel<NDir, true, false, false, false>
            },
            .plainEso = {
                .stepEven = (const void*)&StreamCollideKernel<NDir, Kind, Mode::EsoEven, false, false>,
                .stepOdd  = (const void*)&StreamCollideKernel<NDir, Kind, Mode::EsoOdd,  false, false>,
                .dumpEven = (const void*)&DumpKernel<NDir, true, false, true, true>,
                .dumpOdd  = (const void*)&DumpKernel<NDir, true, false, true, false>
            },
            .blocking = {
                .single     = (const void*)&StreamCollideKernel<NDir, Kind, Mode::Single,     CacheHints, UseRing>,
                .first      = (const void*)&StreamCollideKernel<NDir, Kind, Mode::First,      CacheHints, UseRing>,
                .middleEven = (const void*)&StreamCollideKernel<NDir, Kind, Mode::MiddleEven, CacheHints, false>,
                .middleOdd  = (const void*)&StreamCollideKernel<NDir, Kind, Mode::MiddleOdd,  CacheHints, false>,
                .lastEven   = (const void*)&StreamCollideKernel<NDir, Kind, Mode::LastEven,   CacheHints, UseRing>,
                .lastOdd    = (const void*)&StreamCollideKernel<NDir, Kind, Mode::LastOdd,    CacheHints, UseRing>,
                .dump       = (const void*)&DumpKernel<NDir, false, UseRing, false, false>
            }
        };
    }

    template <int NDir, Kind Kind, bool UseRing>
    Kernels selectKernels(bool hints)
    {
        if (hints) return makeKernels<NDir, Kind, UseRing, true>();
        else       return makeKernels<NDir, Kind, UseRing, false>();
    }

    template <int NDir, Kind Kind>
    Kernels kernels(bool hints, bool ring)
    {
        if (ring) return selectKernels<NDir, Kind, true>(hints);
        else      return selectKernels<NDir, Kind, false>(hints);
    }

    template Kernels kernels<15, Kind::Bgk>(bool, bool);
    template Kernels kernels<15, Kind::Mrt>(bool, bool);
    template Kernels kernels<19, Kind::Bgk>(bool, bool);
    template Kernels kernels<19, Kind::Mrt>(bool, bool);
    template Kernels kernels<19, Kind::Cumulant>(bool, bool);
    template Kernels kernels<27, Kind::Bgk>(bool, bool);
    template Kernels kernels<27, Kind::Mrt>(bool, bool);
    template Kernels kernels<27, Kind::Cumulant>(bool, bool);
}
