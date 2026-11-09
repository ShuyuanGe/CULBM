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
            const auto domainFlags = detail::loadFlags(config);
            if (mode == Mode::Twist and
                std::ranges::any_of(domainFlags, [](flag_t f) { return not (f & STORE_DDF_BIT); }))
                throw std::invalid_argument("--ddfBuffer esotwist requires STORE_DDF_BIT for every domain cell.");
            flags = ::cuda::device_buffer<flag_t>(state.stream, state.pool.as_ref(), domainFlags.size(), ::cuda::no_init);
            if (not config.initFolder.empty())
                std::println("Init flag successfully, from file: {}/flag.dat.", config.initFolder);
            ::cuda::copy_bytes(state.stream, ::cuda::std::span{domainFlags.data(), domainFlags.size()}, flags);
            state.stream.sync();
            detail::initDdf(state, kernels.equilibrium, config.globalSize(), mode == Mode::Pull, src, dst);
            param = PlainParam{
                .invTau = config.invTau,
                .flagBuf = flags.data(),
                .rhoBuf = state.rhoBuf.data(),
                .vxBuf = state.vxBuf.data(),
                .vyBuf = state.vyBuf.data(),
                .vzBuf = state.vzBuf.data(),
                .srcDdf = src.data(),
                .dstDdf = dst.data()
            };
            if (mode == Mode::Twist) param.dstDdf = src.data();
            const auto launchDim = config.launchDim();
            const auto globalDim = config.globalDim;
            if (launchDim.x != globalDim.x or launchDim.y != globalDim.y or launchDim.z != globalDim.z)
                throw std::runtime_error("Dimension of domain doesn't match computation grid!");
        }

        std::uint32_t advance()
        {
            const auto& config = state.config;
            const auto launch = ::cuda::make_config(::cuda::grid_dims(config.gridDim), ::cuda::block_dims(config.blockDim));
            for (std::uint32_t step = 0; step < config.stepsPerBatch; ++step)
            {
                if constexpr (Kernels::mode == Mode::Twist)
                {
                    const bool even = ((state.step + step) % 2) == 0;
                    ::cuda::launch(state.stream, launch, even ? kernels.stepEven : kernels.stepOdd, param);
                }
                else if constexpr (Kernels::mode == Mode::Pull)
                {
                    ::cuda::launch(state.stream, launch, kernels.step, param);
                    std::swap(param.srcDdf, param.dstDdf);
                }
            }
            return config.stepsPerBatch;
        }

        void dump()
        {
            const auto& config = state.config;
            const auto launch = ::cuda::make_config(::cuda::grid_dims(config.gridDim), ::cuda::block_dims(config.blockDim));
            if constexpr (Kernels::mode == Mode::Twist)
            {
                const bool even = (state.step % 2) == 0;
                ::cuda::launch(state.stream, launch, even ? kernels.dumpEven : kernels.dumpOdd, param);
            }
            else if constexpr (Kernels::mode == Mode::Pull)
                ::cuda::launch(state.stream, launch, kernels.dump, param);
        }

        std::size_t cellsPerStep() const
        {
            return state.config.globalSize();
        }
    };
}
