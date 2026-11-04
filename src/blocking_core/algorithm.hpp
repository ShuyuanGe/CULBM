#pragma once

#include "math.hpp"

namespace culbm::blocking_core
{
    template <std::integral T>
    constexpr bool validTileAxisConfig(T domLen, T tileLen, T tileIter)
    {
        if (tileLen < domLen and tileLen + 1 <= tileIter) return false;
        if (2 * (tileLen + 1 - tileIter) < domLen and tileLen + 2 <= tileIter) return false;
        return true;
    }

    template <std::integral T>
    constexpr T calcTileCount(T domLen, T tileLen, T tileIter)
    {
        if (domLen <= tileLen)
            return 1;
        if (domLen <= 2 * (tileLen + 1 - tileIter))
            return 2;
        return 2 + culbm::basic::divCeil<T>(domLen - 2 * (tileLen + 1 - tileIter), tileLen + 2 - 2 * tileIter);
    }

    template <std::integral T>
    constexpr T calcValidPrev(T tileIdx, T tileLen, T tileCount, T tileIter, T domLen)
    {
        if (tileIdx == tileCount)
            return domLen;
        if (tileIdx == 0)
            return 0;
        return (tileLen + 1 - tileIter) + (tileIdx - 1) * (tileLen + 2 - 2 * tileIter);
    }
}