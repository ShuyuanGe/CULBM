#include "kernel.cuh"
#include "core/collision/model.hpp"
#include "core/collision/dispatch.hpp"
#include "core/macroscopic.hpp"
#include "mapping.cuh"
#include "core/accessor.hpp"
#include "core/ddf.hpp"
#include "gpu/io.cuh"
#include <algorithm>
#include <cuda/std/utility>
#include <type_traits>

namespace culbm::sim::single
{
    namespace io       = culbm::gpu::io;
    namespace accessor = culbm::core::accessor;
    namespace ddf      = culbm::core::ddf;
    using culbm::core::collision::Model;
    using culbm::core::collision::Kind;

    template <int n_dir, Kind op>
    void host_equilibrium(real_t rho, real_t vx, real_t vy, real_t vz, ddf_t* dst)
    {
        ddf_t f[n_dir];
        Model<op, n_dir>::equilibrium(rho, vx, vy, vz, f);
        std::ranges::copy(f, dst);
    }

    template <typename AccessorType, ddf::View ViewType>
    struct Access
    {
        using Accessor = AccessorType;
        using View = ViewType;
    };

    enum class StoreScope { flagged, owned };

    template <int n_dir, Kind collision_kind, typename Param, typename Read, typename Write, bool cache_hints, StoreScope scope>
    __global__ void stream_collide_kernel(const Param param)
    {
        using GlobalIO = std::conditional_t<cache_hints, io::Streaming, io::Default>;
        using TileIO   = std::conditional_t<cache_hints, io::Cached,    io::Default>;
        using ReadIO   = std::conditional_t<Read::View::space  == mapping::ViewSpace::tile, TileIO, GlobalIO>;
        using WriteIO  = std::conditional_t<Write::View::space == mapping::ViewSpace::tile, TileIO, GlobalIO>;
        const auto pos = mapping::make_position(param);
        const auto flag = param.flag_buf[pos.local_index];

        ddf_t f[n_dir];
        if (flag & LOAD_DDF_BIT)
            ddf::load<n_dir, typename Read::Accessor, ReadIO>(f, Read::View::make(param, pos));

        if (flag & EQU_DDF_BIT)
        {
            const auto rho = param.rho_buf[pos.global_index];
            const auto vx = param.vx_buf[pos.global_index];
            const auto vy = param.vy_buf[pos.global_index];
            const auto vz = param.vz_buf[pos.global_index];
            Model<collision_kind, n_dir>::equilibrium(rho, vx, vy, vz, f);
        }

        if (flag & BOUNCE_BACK_BIT)
            Model<collision_kind, n_dir>::bounce_back(f);

        if (flag & COLLIDE_BIT)
            Model<collision_kind, n_dir>::collide(param.inv_tau, f);

        if (flag & STORE_DDF_BIT)
        {
            if constexpr (scope == StoreScope::flagged)
                ddf::store<n_dir, typename Write::Accessor, WriteIO>(f, Write::View::make(param, pos));
            else if constexpr (scope == StoreScope::owned)
                if (flag & OWNED_BIT)
                    ddf::store<n_dir, typename Write::Accessor, WriteIO>(f, Write::View::make(param, pos));
        }
    }

    template <int n_dir, typename Param, typename Read>
    __global__ void dump_macro_kernel(const Param param)
    {
        const auto pos = mapping::make_position(param);
        const auto flag = param.flag_buf[pos.local_index];
        if constexpr (Param::dump_owned_only)
            if (not (flag & OWNED_BIT)) return;

        ddf_t f[n_dir];
        const auto view = Read::View::make(param, pos);
        ddf::load<n_dir, typename Read::Accessor, io::Default>(f, view);
        real_t rho, vx, vy, vz;
        culbm::core::macroscopic::compute<n_dir>(rho, vx, vy, vz, f);

        if (flag & DUMP_RHO_BIT) param.rho_buf[pos.global_index] = rho;
        if (flag & DUMP_VX_BIT)  param.vx_buf[pos.global_index] = vx;
        if (flag & DUMP_VY_BIT)  param.vy_buf[pos.global_index] = vy;
        if (flag & DUMP_VZ_BIT)  param.vz_buf[pos.global_index] = vz;
    }

    template <int n_dir, typename Param, typename Write>
    __global__ void dump_ddf_kernel(const Param param, ddf_t* output)
    {
        const auto pos = mapping::make_position(param);
        if constexpr (Param::dump_owned_only)
            if (not (param.flag_buf[pos.local_index] & OWNED_BIT)) return;

        const auto view = Write::View::make(param, pos);
        [&]<std::size_t... Q>(std::index_sequence<Q...>) {
            ([&] {
                constexpr auto a = Write::Accessor::template write<n_dir, Q>();
                output[static_cast<ddf_offset_t>(Q) * pos.global_size + pos.global_index] =
                    *view.template read_address<a.direction, a.x, a.y, a.z>();
            }(), ...);
        }(std::make_index_sequence<n_dir>{});
    }

    PullKernels PullKernels::make(int q, std::string_view kind)
    {
        return culbm::core::collision::dispatch(q, kind, [&]<int n_dir, Kind collision_kind> {
            using Pull = Access<accessor::PullTwoFields, mapping::GlobalView>;
            return PullKernels{
                .equilibrium = &host_equilibrium<n_dir, collision_kind>,
                .step        = &stream_collide_kernel<n_dir, collision_kind, PlainParam, Pull, Pull, false, StoreScope::flagged>,
                .dump_macro  = &dump_macro_kernel<n_dir, PlainParam, Pull>,
                .dump_ddf    = &dump_ddf_kernel<n_dir, PlainParam, Pull>
            };
        });
    }

    TwistKernels TwistKernels::make(int q, std::string_view kind)
    {
        return culbm::core::collision::dispatch(q, kind, [&]<int n_dir, Kind collision_kind> {
            using Even = Access<accessor::EsoTwistEven, mapping::GlobalView>;
            using Odd  = Access<accessor::EsoTwistOdd, mapping::GlobalView>;
            return TwistKernels{
                .equilibrium     = &host_equilibrium<n_dir, collision_kind>,
                .step_even       = &stream_collide_kernel<n_dir, collision_kind, PlainParam, Even, Even, false, StoreScope::flagged>,
                .step_odd        = &stream_collide_kernel<n_dir, collision_kind, PlainParam, Odd,  Odd,  false, StoreScope::flagged>,
                .dump_macro_even = &dump_macro_kernel<n_dir, PlainParam, Even>,
                .dump_macro_odd  = &dump_macro_kernel<n_dir, PlainParam, Odd>,
                .dump_ddf_even   = &dump_ddf_kernel<n_dir, PlainParam, Even>,
                .dump_ddf_odd    = &dump_ddf_kernel<n_dir, PlainParam, Odd>
            };
        });
    }

    BlockingKernels BlockingKernels::make(int q, std::string_view kind, Mode buffer, bool hints)
    {
        return culbm::core::collision::dispatch(q, kind, [&]<int n_dir, Kind collision_kind> {
            const auto make = [&]<ddf::View View, bool cache_hints> {
                using GlobalPull = Access<accessor::PullTwoFields, View>;
                using TileEven   = Access<accessor::EsoTwistEven, mapping::TileView>;
                using TileOdd    = Access<accessor::EsoTwistOdd,  mapping::TileView>;
                return BlockingKernels{
                    .equilibrium = &host_equilibrium<n_dir, collision_kind>,
                    .single      = &stream_collide_kernel<n_dir, collision_kind, BlockingParam, GlobalPull, GlobalPull, cache_hints, StoreScope::flagged>,
                    .first       = &stream_collide_kernel<n_dir, collision_kind, BlockingParam, GlobalPull, TileOdd,    cache_hints, StoreScope::flagged>,
                    .middle_even = &stream_collide_kernel<n_dir, collision_kind, BlockingParam, TileOdd,    TileOdd,    cache_hints, StoreScope::flagged>,
                    .middle_odd  = &stream_collide_kernel<n_dir, collision_kind, BlockingParam, TileEven,   TileEven,   cache_hints, StoreScope::flagged>,
                    .last_even   = &stream_collide_kernel<n_dir, collision_kind, BlockingParam, TileOdd,    GlobalPull, cache_hints, StoreScope::owned>,
                    .last_odd    = &stream_collide_kernel<n_dir, collision_kind, BlockingParam, TileEven,   GlobalPull, cache_hints, StoreScope::owned>,
                    .dump_macro  = &dump_macro_kernel<n_dir, BlockingParam, GlobalPull>,
                    .dump_ddf    = &dump_ddf_kernel<n_dir, BlockingParam, GlobalPull>
                };
            };
            const auto select = [hints, &make]<ddf::View View> {
                if (hints) return make.template operator()<View, true>();
                else       return make.template operator()<View, false>();
            };
            switch (buffer)
            {
                case Mode::block_double: return select.template operator()<mapping::GlobalView>();
                case Mode::block_ring:   return select.template operator()<mapping::RingView>();
                case Mode::pull:
                case Mode::twist: break;
            }
            ::cuda::std::unreachable();
        });
    }
}
