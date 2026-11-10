#pragma once

#include <cuda/devices>

namespace culbm::tune
{
    struct Lambda { float l2, dram, value; };
    struct Gamma { float full, partial, value; };

    Lambda measure_lambda(cuda::device_ref device);
    Gamma measure_gamma(cuda::device_ref device);
}
