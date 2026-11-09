#pragma once

#include <deque>
#include <barrier>
#include <optional>
#include <cuda/buffer>
#include <cuda/memory_pool>
#include <cuda/stream>
#include "config.hpp"
#include "kernel.cuh"
#include "util/logger.hpp"

namespace culbm::sim::multi
{
    struct State
    {
        struct Device
        {
            ::cuda::stream stream;
            ::cuda::device_memory_pool pool;
            ::cuda::timed_event startEvent, endEvent;
            ::cuda::device_buffer<flag_t> flagBuf;
            ::cuda::device_buffer<real_t> rhoBuf, vxBuf, vyBuf, vzBuf;
            ::cuda::device_buffer<ddf_t> ddf0, ddf1;

            Device(int id, std::size_t size, int q);
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
