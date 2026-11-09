#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/system/cuda/execution_policy.h>
#include "fill.hpp"

namespace culbm::gpu
{
    void fill(float* data, std::size_t size, float value, ::cuda::stream_ref stream)
    {
        thrust::fill_n(thrust::cuda::par_nosync.on(stream.get()), thrust::device_pointer_cast(data), size, value);
    }
}
