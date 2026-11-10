#pragma once

#include <algorithm>
#include <concepts>
#include <cuda/cmath>

namespace culbm::core::blocking
{
    template <std::integral T>
    constexpr bool valid_axis(T dom_len, T tile_len, T inner_steps)
    {
        if (dom_len <= tile_len) return true;
        if (tile_len + 1 <= inner_steps) return false;
        if (2 * (tile_len + 1 - inner_steps) < dom_len and tile_len + 2 <= 2 * inner_steps) return false;
        return true;
    }

    template <std::integral T>
    constexpr T tile_count(T dom_len, T tile_len, T inner_steps)
    {
        if (dom_len <= tile_len)
            return 1;
        if (dom_len <= 2 * (tile_len + 1 - inner_steps))
            return 2;
        return 2 + ::cuda::ceil_div(dom_len - 2 * (tile_len + 1 - inner_steps), tile_len + 2 - 2 * inner_steps);
    }

    template <std::integral T>
    constexpr T boundary(T tile_idx, T tile_len, T tile_count, T inner_steps, T dom_len)
    {
        if (tile_idx == tile_count)
            return dom_len;
        if (tile_idx == 0)
            return 0;
        return (tile_len + 1 - inner_steps) + (tile_idx - 1) * (tile_len + 2 - 2 * inner_steps);
    }
    template <std::signed_integral T>
    struct TileBounds
    {
        T store_begin, store_end, load_begin, load_end;
    };

    template <std::signed_integral T>
    constexpr TileBounds<T> tile_bounds(T tile_idx, T tile_len, T tile_count, T inner_steps, T dom_len)
    {
        const T store_begin = boundary(tile_idx, tile_len, tile_count, inner_steps, dom_len);
        const T store_end = boundary<T>(tile_idx + 1, tile_len, tile_count, inner_steps, dom_len);
        return {store_begin, store_end,
                std::max<T>(store_begin - (inner_steps - 1), 0),
                std::min<T>(store_end + (inner_steps - 1), dom_len)};
    }
}
