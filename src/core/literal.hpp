#pragma once

#include "core/types.hpp"

namespace culbm::literal
{
    constexpr real_t operator""_r(long double v) { return v; }
}
