#pragma once

#include <cuda/algorithm>
#include <cuda/launch>
#include <cuda/std/span>
#include <utility>
#include "init.hpp"

namespace culbm::sim::single::strategy
{
    template <typename Kernels>
    class Plain
    {
        State& state;
        Kernels kernels;
        ::cuda::device_buffer<flag_t> flags;
        ::cuda::device_buffer<ddf_t> src, dst;
        PlainParam param;

    public:
        Plain(State& state) :
            state(state), kernels(Kernels::make(state.config.q, state.config.collision)),
            flags(state.stream, state.pool.as_ref()), src(state.stream, state.pool.as_ref()), dst(state.stream, state.pool.as_ref())
        {
            constexpr auto mode = Kernels::mode;
            const auto& config = state.config;
            const auto domain_flags = detail::load_flags(config);
            if (mode == Mode::twist and
                std::ranges::any_of(domain_flags, [](flag_t f) { return not (f & STORE_DDF_BIT); }))
                throw std::invalid_argument("--ddf-buffer single requires STORE_DDF_BIT for every domain cell.");
            flags = ::cuda::device_buffer<flag_t>(state.stream, state.pool.as_ref(), domain_flags.size(), ::cuda::no_init);
            if (not config.init_folder.empty())
                std::println("Init flag successfully, from file: {}/flag.dat.", config.init_folder);
            else
                std::println("Init flag successfully, using default flags.");
            ::cuda::copy_bytes(state.stream, ::cuda::std::span{domain_flags.data(), domain_flags.size()}, flags);
            state.stream.sync();
            detail::init_ddf(state, kernels.equilibrium, config.global_size(), mode == Mode::pull, src, dst);
            param = PlainParam{
                .inv_tau = config.inv_tau,
                .flag_buf = flags.data(),
                .rho_buf = state.rho_buf.data(),
                .vx_buf = state.vx_buf.data(),
                .vy_buf = state.vy_buf.data(),
                .vz_buf = state.vz_buf.data(),
                .src_ddf = src.data(),
                .dst_ddf = dst.data()
            };
            if (mode == Mode::twist) param.dst_ddf = src.data();
            const auto launch_dim = config.launch_dim();
            const auto global_dim = config.global_dim;
            if (launch_dim.x != global_dim.x or launch_dim.y != global_dim.y or launch_dim.z != global_dim.z)
                throw std::runtime_error("Dimension of domain doesn't match computation grid!");
        }

        std::uint32_t advance()
        {
            const auto& config = state.config;
            const auto launch = ::cuda::make_config(::cuda::grid_dims(config.grid_dim), ::cuda::block_dims(config.block_dim));
            for (std::uint32_t step = 0; step < config.steps_per_batch; ++step)
            {
                if constexpr (Kernels::mode == Mode::twist)
                {
                    const bool even = ((state.step + step) % 2) == 0;
                    ::cuda::launch(state.stream, launch, even ? kernels.step_even : kernels.step_odd, param);
                }
                else if constexpr (Kernels::mode == Mode::pull)
                {
                    ::cuda::launch(state.stream, launch, kernels.step, param);
                    std::swap(param.src_ddf, param.dst_ddf);
                }
            }
            return config.steps_per_batch;
        }

        void dump()
        {
            const auto& config = state.config;
            const auto launch = ::cuda::make_config(::cuda::grid_dims(config.grid_dim), ::cuda::block_dims(config.block_dim));
            if constexpr (Kernels::mode == Mode::twist)
            {
                const bool even = (state.step % 2) == 0;
                ::cuda::launch(state.stream, launch, even ? kernels.dump_even : kernels.dump_odd, param);
            }
            else if constexpr (Kernels::mode == Mode::pull)
                ::cuda::launch(state.stream, launch, kernels.dump, param);
        }

        std::size_t cells_per_step() const
        {
            return state.config.global_size();
        }
    };
}
