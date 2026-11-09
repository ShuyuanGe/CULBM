#pragma once

#include <deque>
#include <optional>
#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>
#include "config.hpp"
#include "kernel.cuh"
#include "simulator.hpp"

namespace culbm::sim::multi
{
    struct Simulator::State
    {
        struct Device
        {
            ::cuda::stream stream;
            ::cuda::device_memory_pool pool;
            ::cuda::timed_event start, end;
            ::cuda::device_buffer<flag_t> flagBuf;
            ::cuda::device_buffer<real_t> rhoBuf, vxBuf, vyBuf, vzBuf;
            ::cuda::device_buffer<ddf_t> ddf0, ddf1;

            Device(int id, std::size_t size, int q) :
                stream(::cuda::device_ref{id}), pool(::cuda::device_ref{id}),
                start(::cuda::device_ref{id}), end(::cuda::device_ref{id}),
                flagBuf(stream, pool.as_ref(), size, ::cuda::no_init),
                rhoBuf(stream, pool.as_ref(), size, ::cuda::no_init),
                vxBuf(stream, pool.as_ref(), size, ::cuda::no_init),
                vyBuf(stream, pool.as_ref(), size, ::cuda::no_init),
                vzBuf(stream, pool.as_ref(), size, ::cuda::no_init),
                ddf0(stream, pool.as_ref(), q * size, ::cuda::no_init),
                ddf1(stream, pool.as_ref(), q * size, ::cuda::no_init) {}
        };

        Config config;
        Kernels kernels;
        std::uint32_t step = 0;
        std::deque<std::optional<Device>> devices;

        explicit State(Config config);
        void init(std::uint32_t id, util::Logger& logger, std::barrier<>& barrier);
        void advance(std::uint32_t id, std::uint32_t endStep, util::Logger& logger, std::barrier<>& barrier);
        void write(std::uint32_t id, std::uint32_t step, uint3 position) const;
        void clear(std::uint32_t id, util::Logger& logger);
    };
}
