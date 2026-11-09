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
            ::cuda::timed_event start_event, end_event;
            ::cuda::device_buffer<flag_t> flag_buf;
            ::cuda::device_buffer<real_t> rho_buf, vx_buf, vy_buf, vz_buf;
            ::cuda::device_buffer<ddf_t> ddf0, ddf1;

            Device(int id, std::size_t size, int q);
        };

        Config config;
        std::deque<std::optional<Device>> devices;

        explicit State(Config config);
        void init(std::uint32_t id, void (*init_kernel)(InitKernelParam), util::Logger& logger, std::barrier<>& barrier);
        void write(std::uint32_t id, std::uint32_t step, uint3 position) const;
        void clear(std::uint32_t id, util::Logger& logger);
    };
}
