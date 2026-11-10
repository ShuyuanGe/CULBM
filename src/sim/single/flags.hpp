#pragma once

#include <cstdint>

namespace culbm::sim::single
{
    using flag_t = std::uint32_t;

    constexpr flag_t LOAD_DDF_BIT    = 1u << 0;
    constexpr flag_t EQU_DDF_BIT     = 1u << 1;
    constexpr flag_t COLLIDE_BIT     = 1u << 2;
    constexpr flag_t BOUNCE_BACK_BIT = 1u << 3;
    constexpr flag_t STORE_DDF_BIT   = 1u << 4;

    constexpr flag_t DUMP_RHO_BIT    = 1u << 27;
    constexpr flag_t DUMP_VX_BIT     = 1u << 28;
    constexpr flag_t DUMP_VY_BIT     = 1u << 29;
    constexpr flag_t DUMP_VZ_BIT     = 1u << 30;
    constexpr flag_t OWNED_BIT       = 1u << 31;
}
