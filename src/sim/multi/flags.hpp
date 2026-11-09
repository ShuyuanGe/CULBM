#pragma once

#include <cstdint>

namespace culbm::sim::multi
{
    using flag_t = std::uint32_t;

    constexpr flag_t LOAD_DDF_BIT       = 1u << 0;
    constexpr flag_t REV_LOAD_DDF_BIT   = 1u << 1;
    constexpr flag_t EQU_DDF_BIT        = 1u << 2;
    constexpr flag_t COLLIDE_BIT        = 1u << 3;
    constexpr flag_t STORE_DDF_BIT      = 1u << 4;
    constexpr flag_t CORRECT_BIT        = 1u << 5;
    constexpr flag_t DEV_BND_BIT        = 1u << 6;

    constexpr flag_t FLUID_FLAG         = LOAD_DDF_BIT | COLLIDE_BIT | STORE_DDF_BIT;
    constexpr flag_t BOUNCE_BACK_FLAG   = REV_LOAD_DDF_BIT | STORE_DDF_BIT;
}
