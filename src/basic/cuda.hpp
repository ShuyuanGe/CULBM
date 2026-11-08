#pragma once

#include <cstddef>
#include <cuda/stream>

namespace culbm::basic::cuda
{
    void fill(float* data, std::size_t size, float value, ::cuda::stream_ref stream);
}
