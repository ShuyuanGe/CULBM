#pragma once

#include <cstdint>

namespace culbm
{
    using dir_t = std::uint8_t;         // Discrete velocity direction index.
    using nbr_t = std::int8_t;          // Neighbor slot index or signed offset.
    using idx_t = std::int32_t;         // Spatial index, size, or signed offset.
    using ddf_offset_t = std::int64_t;  // DDF array offset including direction.

    using real_t = float;               // Macroscopic physical quantity.
    using ddf_t = float;                // Discrete distribution function value.
}
