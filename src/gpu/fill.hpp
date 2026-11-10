#pragma once

#include <cstddef>
#include <cuda/stream>

namespace culbm::gpu
{
    void fill(float* data, std::size_t size, float value, ::cuda::stream_ref stream);
}
