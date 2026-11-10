#pragma once

#include <cassert>
#include <ranges>
#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/std/span>
#include "core/blocking.hpp"
#include "gpu/error.hpp"
#include "physical/resource.hpp"
#include "init.hpp"

namespace culbm::sim::single::strategy
{
    template <Mode buffer>
    class Blocking
    {
        State& state;
        BlockingKernels kernels;
        ::cuda::device_buffer<flag_t> flags;
        ::cuda::device_buffer<ddf_t> tile, src, dst;
        BlockingParam param;
        dim3 tile_dim, tile_counts;
        std::uint32_t tile_count;
        idx_t ring_shift_z;

    public:
        Blocking(State& state) :
            state(state), kernels(BlockingKernels::make(state.config.q, state.config.collision, buffer, state.config.cache_hint == 1)),
            flags(state.stream, state.pool.as_ref()),
            tile(state.stream, ::cuda::mr::synchronous_resource_adapter{physical::Resource{state.device}},
                 static_cast<std::size_t>(state.config.q) * state.config.tile_size(), ::cuda::no_init),
            src(state.stream, state.pool.as_ref()), dst(state.stream, state.pool.as_ref()),
            tile_dim(state.config.launch_dim()), tile_counts(state.config.tile_counts()),
            tile_count(tile_counts.x * tile_counts.y * tile_counts.z),
            ring_shift_z(tile_dim.z + 1)
        {
            const auto& config = state.config;
            init_flags();
            auto global_size = config.global_size();
            if constexpr (buffer == Mode::block_ring)
            {
                assert(tile_dim.z >= config.inner_steps);
                global_size = static_cast<std::size_t>(config.global_dim.x) * config.global_dim.y * (config.global_dim.z + ring_shift_z);
            }
            detail::init_ddf(state, kernels.equilibrium, global_size, buffer == Mode::block_double, src, dst);
            configure_cache();
            param = BlockingParam{
                .inv_tau = config.inv_tau,
                .global_nx = static_cast<idx_t>(config.global_dim.x),
                .global_ny = static_cast<idx_t>(config.global_dim.y),
                .global_nz = static_cast<idx_t>(config.global_dim.z),
                .rho_buf = state.rho_buf.data(),
                .vx_buf = state.vx_buf.data(),
                .vy_buf = state.vy_buf.data(),
                .vz_buf = state.vz_buf.data(),
                .src_ddf = src.data(),
                .dst_ddf = dst.data(),
                .tile_ddf = tile.data()
            };
            if constexpr (buffer == Mode::block_ring)
            {
                param.dst_ddf = src.data();
                param.dst_zoffset = config.global_dim.z;
                param.ring_nz = param.dst_zoffset + ring_shift_z;
            }
        }

        ~Blocking()
        {
            cudaStreamAttrValue attr{};
            cudaStreamSetAttribute(state.stream.get(), cudaStreamAttributeAccessPolicyWindow, &attr);
        }

        std::uint32_t advance()
        {
            const auto& config = state.config;
            const auto launch = ::cuda::make_config(::cuda::grid_dims(config.grid_dim), ::cuda::block_dims(config.block_dim));
            std::uint32_t step = 0;
            for (; step < config.steps_per_batch; step += config.inner_steps)
            {
                for (std::uint32_t index = 0; index < tile_count; ++index)
                {
                    const auto tile = coordinates(index);
                    if (config.interior_tiles_only and
                        (tile.x == 0 or tile.x == tile_counts.x - 1 or
                         tile.y == 0 or tile.y == tile_counts.y - 1 or
                         tile.z == 0 or tile.z == tile_counts.z - 1))
                        continue;
                    set_tile(index, tile);
                    for (std::int32_t inner_step = 0; inner_step < config.inner_steps; ++inner_step)
                    {
                        BlockingKernel kernel;
                        if (inner_step == 0)
                        {
                            if (config.inner_steps == 1) kernel = kernels.single;
                            else                         kernel = kernels.first;
                        }
                        else if (inner_step < config.inner_steps - 1)
                        {
                            if (inner_step % 2 == 0) kernel = kernels.middle_even;
                            else                     kernel = kernels.middle_odd;
                        }
                        else
                        {
                            if (inner_step % 2 == 0) kernel = kernels.last_even;
                            else                     kernel = kernels.last_odd;
                        }
                        ::cuda::launch(state.stream, launch, kernel, param);
                    }
                }
                if constexpr (buffer == Mode::block_ring)
                {
                    param.src_zoffset = param.dst_zoffset;
                    param.dst_zoffset -= ring_shift_z;
                    if (param.dst_zoffset < 0) param.dst_zoffset += param.ring_nz;
                }
                else if constexpr (buffer == Mode::block_double)
                    std::swap(param.src_ddf, param.dst_ddf);
            }
            return step;
        }

        void dump()
        {
            const auto& config = state.config;
            const auto launch = ::cuda::make_config(::cuda::grid_dims(config.grid_dim), ::cuda::block_dims(config.block_dim));
            for (std::uint32_t index = 0; index < tile_count; ++index)
            {
                set_tile(index, coordinates(index));
                ::cuda::launch(state.stream, launch, kernels.dump_macro, param);
                if (config.dump_ddf)
                    ::cuda::launch(state.stream, launch, kernels.dump_ddf, param, state.ddf_buf.data());
            }
        }

        std::size_t cells_per_step() const
        {
            const auto& config = state.config;
            if (config.interior_tiles_only)
                return static_cast<std::size_t>(tile_counts.x - 2) * (tile_counts.y - 2) * (tile_counts.z - 2) *
                    (tile_dim.x - 2 * (config.inner_steps - 1)) *
                    (tile_dim.y - 2 * (config.inner_steps - 1)) *
                    (tile_dim.z - 2 * (config.inner_steps - 1));
            return config.global_size();
        }

    private:
        void init_flags()
        {
            const auto& config = state.config;
            const std::uint32_t tile_size = tile_dim.x * tile_dim.y * tile_dim.z;
            flags = ::cuda::device_buffer<flag_t>(state.stream, state.pool.as_ref(), tile_count * tile_size, ::cuda::no_init);
            const auto domain_flags = detail::load_flags(config);

            std::vector<flag_t> tile_flags(tile_count * tile_size);
            for (std::uint32_t index = 0; index < tile_count; ++index)
            {
                const auto tile = coordinates(index);
                const auto bounds_x = culbm::core::blocking::tile_bounds<idx_t>(tile.x, tile_dim.x, tile_counts.x, config.inner_steps, config.global_dim.x);
                const auto bounds_y = culbm::core::blocking::tile_bounds<idx_t>(tile.y, tile_dim.y, tile_counts.y, config.inner_steps, config.global_dim.y);
                const auto bounds_z = culbm::core::blocking::tile_bounds<idx_t>(tile.z, tile_dim.z, tile_counts.z, config.inner_steps, config.global_dim.z);
                const auto tile_base = index * tile_size;

                for (auto [global_z, global_y, global_x] : std::views::cartesian_product(
                         std::views::iota(bounds_z.load_begin, bounds_z.load_end),
                         std::views::iota(bounds_y.load_begin, bounds_y.load_end),
                         std::views::iota(bounds_x.load_begin, bounds_x.load_end)))
                {
                    const idx_t local_x = global_x - bounds_x.load_begin;
                    const idx_t local_y = global_y - bounds_y.load_begin;
                    const idx_t local_z = global_z - bounds_z.load_begin;
                    const idx_t domain_index = global_x + config.global_dim.x * (global_y + config.global_dim.y * global_z);
                    const idx_t tile_index = local_x + tile_dim.x * (local_y + tile_dim.y * local_z);
                    const bool owned =
                        bounds_z.store_begin <= global_z and global_z < bounds_z.store_end and
                        bounds_y.store_begin <= global_y and global_y < bounds_y.store_end and
                        bounds_x.store_begin <= global_x and global_x < bounds_x.store_end;
                    tile_flags[tile_base + tile_index] = domain_flags[domain_index] | (owned ? OWNED_BIT : 0);
                }
            }

            ::cuda::copy_bytes(state.stream, ::cuda::std::span{tile_flags.data(), tile_flags.size()}, flags);
            state.stream.sync();
            if (not config.init_folder.empty())
                std::println("Init flag successfully, from file: {}/flag.dat.", config.init_folder);
            else
                std::println("Init flag successfully, using default flags.");
        }

        void configure_cache()
        {
            const auto& config = state.config;
            if (config.disable_persistent_l2)
            {
                std::println("Persisting L2 configuration skipped.");
            }
            else
            {
                auto num_bytes = tile.size() * sizeof(ddf_t);
                const auto max_size = state.device.attribute(::cuda::device_attributes::max_persisting_l2_cache_size);
                if (num_bytes > max_size) {
                    std::println(
                        "Warning: Requested persisting L2 cache size {} exceeds the maximum allowed size {}. Capping to maximum.",
                        num_bytes, max_size
                    );
                    num_bytes = max_size;
                }
                gpu::check(cudaDeviceSetLimit(cudaLimit::cudaLimitPersistingL2CacheSize, num_bytes));
                cudaStreamAttrValue stream_attr;
                stream_attr.accessPolicyWindow.base_ptr = tile.data();
                stream_attr.accessPolicyWindow.num_bytes = num_bytes;
                stream_attr.accessPolicyWindow.hitRatio = 1.0;
                stream_attr.accessPolicyWindow.hitProp = cudaAccessProperty::cudaAccessPropertyPersisting;
                stream_attr.accessPolicyWindow.missProp = cudaAccessProperty::cudaAccessPropertyStreaming;
                gpu::check(cudaStreamSetAttribute(state.stream.get(), cudaLaunchAttributeID::cudaStreamAttributeAccessPolicyWindow, &stream_attr));
            }
        }

        uint3 coordinates(std::uint32_t index) const
        {
            return { index % tile_counts.x,
                     (index / tile_counts.x) % tile_counts.y,
                     index / (tile_counts.x * tile_counts.y) };
        }

        void set_tile(std::uint32_t index, uint3 tile)
        {
            const auto& config = state.config;
            using culbm::core::blocking::tile_bounds;
            param.origin_x = tile_bounds<idx_t>(tile.x, tile_dim.x, tile_counts.x, config.inner_steps, config.global_dim.x).load_begin;
            param.origin_y = tile_bounds<idx_t>(tile.y, tile_dim.y, tile_counts.y, config.inner_steps, config.global_dim.y).load_begin;
            param.origin_z = tile_bounds<idx_t>(tile.z, tile_dim.z, tile_counts.z, config.inner_steps, config.global_dim.z).load_begin;
            param.flag_buf = flags.data() + index * tile_dim.x * tile_dim.y * tile_dim.z;
        }

    };
}
