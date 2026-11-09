#pragma once

#include <cuda/devices>

namespace culbm::tune
{
    struct Lambda { float l2, dram, value; };
    struct Gamma { float full, partial, value; };

    Lambda measureLambda(cuda::device_ref device);
    Gamma measureGamma(cuda::device_ref device);
}
