#include "kernel.cuh"
#include "core/collision/model.hpp"
#include "core/collision/dispatch.hpp"
#include "core/macroscopic.hpp"
#include "mapping.cuh"
#include "core/accessor.hpp"
#include "core/ddf.hpp"
#include "gpu/io.cuh"
#include <cuda/std/utility>
#include <type_traits>
#include <utility>

namespace culbm::sim::single
{
    namespace io       = culbm::gpu::io;
    namespace accessor = culbm::core::accessor;
    namespace ddf      = culbm::core::ddf;
    using culbm::core::collision::Model;
    using culbm::core::collision::Kind;
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
        using GlobalIO = std::conditional_t<CacheHints, io::Streaming, io::Default>;
        using TileIO   = std::conditional_t<CacheHints, io::Cached,    io::Default>;
        using ReadIO   = std::conditional_t<Read::View::space  == mapping::ViewSpace::Tile, TileIO, GlobalIO>;
        using WriteIO  = std::conditional_t<Write::View::space == mapping::ViewSpace::Tile, TileIO, GlobalIO>;
        const auto pos = mapping::makePosition(param);
        const auto flag = param.flagBuf[pos.localIndex];

        ddf_t f[NDir];
        if (flag & LOAD_DDF_BIT)
            ddf::load<NDir, typename Read::Accessor, ReadIO>(f, Read::View::make(param, pos));

        if (flag & EQU_DDF_BIT)
        {
            const auto rho = param.rhoBuf[pos.globalIndex];
            const auto vx = param.vxBuf[pos.globalIndex];
            const auto vy = param.vyBuf[pos.globalIndex];
            const auto vz = param.vzBuf[pos.globalIndex];
            Model<Kind, NDir>::equilibrium(rho, vx, vy, vz, f);
        }

        if (flag & BOUNCE_BACK_BIT)
        {
            #pragma unroll
            for (int i = 0; i < NDir / 2; ++i)
                std::swap(f[i], f[Stencil<NDir>::opposite(i)]);
        }

        if (flag & COLLIDE_BIT)
            Model<Kind, NDir>::collide(param.invTau, f);

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
        ddf::load<NDir, typename Read::Accessor, io::Default>(f, view);
        real_t rho, vx, vy, vz;
        culbm::core::macroscopic::compute<NDir>(rho, vx, vy, vz, f);

        if (flag & DUMP_RHO_BIT) param.rhoBuf[pos.globalIndex] = rho;
        if (flag & DUMP_VX_BIT)  param.vxBuf[pos.globalIndex] = vx;
        if (flag & DUMP_VY_BIT)  param.vyBuf[pos.globalIndex] = vy;
        if (flag & DUMP_VZ_BIT)  param.vzBuf[pos.globalIndex] = vz;
    }

    PullKernels PullKernels::make(int q, std::string_view kind)
    {
        return culbm::core::collision::dispatch(q, kind, [&]<int NDir, Kind Kind> {
            using Pull = Access<accessor::PullTwoFields, mapping::GlobalView>;
            return PullKernels{
                .equilibrium = &Model<Kind, NDir>::equilibrium,
                .step = &StreamCollideKernel<NDir, Kind, PlainParam, Pull, Pull, false, StoreScope::Flagged>,
                .dump = &DumpKernel<NDir, PlainParam, Pull>
            };
        });
    }

    TwistKernels TwistKernels::make(int q, std::string_view kind)
    {
        return culbm::core::collision::dispatch(q, kind, [&]<int NDir, Kind Kind> {
            using Even = Access<accessor::EsoTwistEven, mapping::GlobalView>;
            using Odd  = Access<accessor::EsoTwistOdd, mapping::GlobalView>;
            return TwistKernels{
                .equilibrium = &Model<Kind, NDir>::equilibrium,
                .stepEven = &StreamCollideKernel<NDir, Kind, PlainParam, Even, Even, false, StoreScope::Flagged>,
                .stepOdd  = &StreamCollideKernel<NDir, Kind, PlainParam, Odd,  Odd,  false, StoreScope::Flagged>,
                .dumpEven = &DumpKernel<NDir, PlainParam, Even>,
                .dumpOdd  = &DumpKernel<NDir, PlainParam, Odd>
            };
        });
    }

    BlockingKernels BlockingKernels::make(int q, std::string_view kind, Mode buffer, bool hints)
    {
        return culbm::core::collision::dispatch(q, kind, [&]<int NDir, Kind Kind> {
            const auto make = [&]<typename View, bool CacheHints> {
                using GlobalPull = Access<accessor::PullTwoFields, View>;
                using TileEven = Access<accessor::EsoTwistEven, mapping::TileView>;
                using TileOdd  = Access<accessor::EsoTwistOdd,  mapping::TileView>;
                return BlockingKernels{
                    .equilibrium = &Model<Kind, NDir>::equilibrium,
                    .single     = &StreamCollideKernel<NDir, Kind, BlockingParam, GlobalPull, GlobalPull, CacheHints, StoreScope::Flagged>,
                    .first      = &StreamCollideKernel<NDir, Kind, BlockingParam, GlobalPull, TileOdd,    CacheHints, StoreScope::Flagged>,
                    .middleEven = &StreamCollideKernel<NDir, Kind, BlockingParam, TileOdd,    TileOdd,    CacheHints, StoreScope::Flagged>,
                    .middleOdd  = &StreamCollideKernel<NDir, Kind, BlockingParam, TileEven,   TileEven,   CacheHints, StoreScope::Flagged>,
                    .lastEven   = &StreamCollideKernel<NDir, Kind, BlockingParam, TileOdd,    GlobalPull, CacheHints, StoreScope::Correct>,
                    .lastOdd    = &StreamCollideKernel<NDir, Kind, BlockingParam, TileEven,   GlobalPull, CacheHints, StoreScope::Correct>,
                    .dump       = &DumpKernel<NDir, BlockingParam, GlobalPull>
                };
            };
            const auto select = [hints, &make]<typename View> {
                if (hints) return make.template operator()<View, true>();
                else       return make.template operator()<View, false>();
            };
            switch (buffer)
            {
                case Mode::BlockDouble: return select.template operator()<mapping::GlobalView>();
                case Mode::BlockRing:   return select.template operator()<mapping::RingView>();
                case Mode::Pull:
                case Mode::Twist: break;
            }
            ::cuda::std::unreachable();
        });
    }
}
