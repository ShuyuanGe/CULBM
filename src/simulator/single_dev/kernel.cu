#include "kernel.cuh"
#include "method/method.hpp"
#include "method/dispatch.hpp"
#include "macroscopic.hpp"
#include "mapping.cuh"
#include "accessor.hpp"
#include "ddf.hpp"
#include "mem.cuh"
#include <algorithm>
#include <type_traits>
#include <utility>

namespace culbm::simulator::single_dev
{
    namespace mem      = culbm::basic::mem;
    namespace accessor = culbm::core::accessor;
    namespace ddf      = culbm::core::ddf;
    using culbm::core::method::Method;
    using culbm::core::method::Kind;
    using culbm::core::Stencil;

    template <typename AccessorType, typename ViewType>
    struct Access
    {
        using Accessor = AccessorType;
        using View = ViewType;
    };

    enum class StoreScope { Flagged, Correct };

    template <int NDir, Kind Kind, typename Param, typename Read, typename Write, bool CacheHints, StoreScope Scope>
    __global__ void StreamCollideKernel(const Param param)
    {
        using GlobalIO = std::conditional_t<CacheHints, mem::Streaming, mem::Default>;
        using TileIO   = std::conditional_t<CacheHints, mem::Cached,    mem::Default>;
        using ReadIO   = std::conditional_t<Read::View::space  == mapping::ViewSpace::Tile, TileIO, GlobalIO>;
        using WriteIO  = std::conditional_t<Write::View::space == mapping::ViewSpace::Tile, TileIO, GlobalIO>;
        const auto pos = mapping::makePosition(param);
        const auto flag = param.flagBuf[pos.localIndex];

        ddf_t f[NDir];
        if (flag & LOAD_DDF_BIT)
            ddf::load<NDir, typename Read::Accessor, ReadIO>(f, Read::View::make(param, pos));

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
            if constexpr (Scope == StoreScope::Flagged)
                ddf::store<NDir, typename Write::Accessor, WriteIO>(f, Write::View::make(param, pos));
            else if constexpr (Scope == StoreScope::Correct)
                if (flag & CORRECT_BIT)
                    ddf::store<NDir, typename Write::Accessor, WriteIO>(f, Write::View::make(param, pos));
        }
    }

    template <int NDir, typename Param, typename Read>
    __global__ void DumpKernel(const Param param)
    {
        const auto pos = mapping::makePosition(param);
        const auto flag = param.flagBuf[pos.localIndex];
        if constexpr (Param::dumpCorrectOnly)
            if (not (flag & CORRECT_BIT)) return;

        ddf_t f[NDir];
        const auto view = Read::View::make(param, pos);
        ddf::load<NDir, typename Read::Accessor, mem::Default>(f, view);
        real_t rho, vx, vy, vz;
        culbm::core::macroscopic::compute<NDir>(rho, vx, vy, vz, f);

        if (flag & DUMP_RHO_BIT) param.glbRhoBuf[pos.glbi] = rho;
        if (flag & DUMP_VX_BIT)  param.glbVxBuf[pos.glbi] = vx;
        if (flag & DUMP_VY_BIT)  param.glbVyBuf[pos.glbi] = vy;
        if (flag & DUMP_VZ_BIT)  param.glbVzBuf[pos.glbi] = vz;
    }

    template <int NDir, Kind Kind>
    PlainPullKernels plainPullKernels()
    {
        using Pull = Access<accessor::PullTwoFields, mapping::GlobalView>;
        return {
            .equilibrium = &Method<Kind, NDir>::equilibrium,
            .step = (const void*)&StreamCollideKernel<NDir, Kind, PlainParam, Pull, Pull, false, StoreScope::Flagged>,
            .dump = (const void*)&DumpKernel<NDir, PlainParam, Pull>
        };
    }

    template <int NDir, Kind Kind>
    PlainEsoKernels plainEsoKernels()
    {
        using Even = Access<accessor::EsoTwistEven, mapping::GlobalView>;
        using Odd  = Access<accessor::EsoTwistOdd, mapping::GlobalView>;
        return {
            .equilibrium = &Method<Kind, NDir>::equilibrium,
            .stepEven = (const void*)&StreamCollideKernel<NDir, Kind, PlainParam, Even, Even, false, StoreScope::Flagged>,
            .stepOdd  = (const void*)&StreamCollideKernel<NDir, Kind, PlainParam, Odd,  Odd,  false, StoreScope::Flagged>,
            .dumpEven = (const void*)&DumpKernel<NDir, PlainParam, Even>,
            .dumpOdd  = (const void*)&DumpKernel<NDir, PlainParam, Odd>
        };
    }

    template <int NDir, Kind Kind, typename GlobalView, bool CacheHints>
    BlockingKernels makeBlockingKernels()
    {
        using GlobalPull = Access<accessor::PullTwoFields, GlobalView>;
        using TileEven = Access<accessor::EsoTwistEven, mapping::TileView>;
        using TileOdd  = Access<accessor::EsoTwistOdd,  mapping::TileView>;
        return {
            .equilibrium = &Method<Kind, NDir>::equilibrium,
            .single     = (const void*)&StreamCollideKernel<NDir, Kind, BlockingParam, GlobalPull, GlobalPull, CacheHints, StoreScope::Flagged>,
            .first      = (const void*)&StreamCollideKernel<NDir, Kind, BlockingParam, GlobalPull, TileOdd,    CacheHints, StoreScope::Flagged>,
            .middleEven = (const void*)&StreamCollideKernel<NDir, Kind, BlockingParam, TileOdd,    TileOdd,    CacheHints, StoreScope::Flagged>,
            .middleOdd  = (const void*)&StreamCollideKernel<NDir, Kind, BlockingParam, TileEven,   TileEven,   CacheHints, StoreScope::Flagged>,
            .lastEven   = (const void*)&StreamCollideKernel<NDir, Kind, BlockingParam, TileOdd,    GlobalPull, CacheHints, StoreScope::Correct>,
            .lastOdd    = (const void*)&StreamCollideKernel<NDir, Kind, BlockingParam, TileEven,   GlobalPull, CacheHints, StoreScope::Correct>,
            .dump       = (const void*)&DumpKernel<NDir, BlockingParam, GlobalPull>
        };
    }

    template <int NDir, Kind Kind, typename GlobalView>
    BlockingKernels selectBlockingKernels(bool hints)
    {
        if (hints) return makeBlockingKernels<NDir, Kind, GlobalView, true>();
        else       return makeBlockingKernels<NDir, Kind, GlobalView, false>();
    }

    template <int NDir, Kind Kind>
    BlockingKernels blockingKernels(bool hints, BlockingGlobalBuffer buffer)
    {
        switch (buffer)
        {
            case BlockingGlobalBuffer::Double: return selectBlockingKernels<NDir, Kind, mapping::GlobalView>(hints);
            case BlockingGlobalBuffer::Ring:   return selectBlockingKernels<NDir, Kind, mapping::RingView>(hints);
        }
        std::unreachable();
    }

    PlainPullKernels makePlainPullKernels(int q, std::string_view kind)
    {
        return culbm::core::method::dispatch(q, kind, []<int NDir, Kind Kind> { return plainPullKernels<NDir, Kind>();
        });
    }

    PlainEsoKernels makePlainEsoKernels(int q, std::string_view kind)
    {
        return culbm::core::method::dispatch(q, kind, []<int NDir, Kind Kind> { return plainEsoKernels<NDir, Kind>();
        });
    }

    BlockingKernels makeBlockingKernels(int q, std::string_view kind, BlockingGlobalBuffer buffer, bool hints)
    {
        return culbm::core::method::dispatch(q, kind, [&]<int NDir, Kind Kind> { return blockingKernels<NDir, Kind>(hints, buffer);
        });
    }
}
