#pragma once

#include "config.hpp"

namespace culbm::literal
{
    constexpr real_t operator""_r(long double v) { return static_cast<real_t>(v); }
}