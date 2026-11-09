#pragma once

#include <print>
#include "measure.hpp"

namespace culbm::tune
{
    inline void report(Lambda result)
    {
        std::println("L2 median: {:.4g} GB/s", result.l2);
        std::println("DRAM median: {:.4g} GB/s", result.dram);
        std::println("lambda: {:.5g}", result.value);
    }

    inline void report(Gamma result)
    {
        std::println("4F median: {:.4f} ms", result.full);
        std::println("3F+1P median: {:.4f} ms", result.partial);
        std::println("gamma: {:.4g}", result.value);
    }
}
