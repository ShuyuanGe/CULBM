#pragma once

#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>
#include "config.hpp"

namespace culbm::sim::single
{
    struct State
    {
        const Config& config;
        std::uint32_t step = 0;
        ::cuda::device_ref device;
        ::cuda::stream stream;
        ::cuda::device_memory_pool pool;
        ::cuda::timed_event start_event, end_event;
        ::cuda::device_buffer<real_t> rho_buf, vx_buf, vy_buf, vz_buf;
        ::cuda::device_buffer<ddf_t> ddf_buf;

        explicit State(const Config& config);
        void write() const;
    };
}
