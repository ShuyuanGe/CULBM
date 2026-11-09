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
        ::cuda::timed_event startEvent, endEvent;
        ::cuda::device_buffer<real_t> rhoBuf, vxBuf, vyBuf, vzBuf;

        explicit State(const Config& config);
        void write() const;
    };
}
