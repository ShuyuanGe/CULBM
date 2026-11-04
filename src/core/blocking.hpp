#pragma once

#include <algorithm>
#include "math.hpp"

namespace culbm::core::blocking
{
    template <std::integral T>
    constexpr bool validAxis(T domLen, T tileLen, T tileIter)
    {
        if (tileLen < domLen and tileLen + 1 <= tileIter) return false;
        if (2 * (tileLen + 1 - tileIter) < domLen and tileLen + 2 <= tileIter) return false;
        return true;
    }

    template <std::integral T>
    constexpr T tileCount(T domLen, T tileLen, T tileIter)
    {
        if (domLen <= tileLen)
            return 1;
        if (domLen <= 2 * (tileLen + 1 - tileIter))
            return 2;
        return 2 + culbm::basic::divCeil<T>(domLen - 2 * (tileLen + 1 - tileIter), tileLen + 2 - 2 * tileIter);
    }

    template <std::integral T>
    constexpr T boundary(T tileIdx, T tileLen, T tileCount, T tileIter, T domLen)
    {
        if (tileIdx == tileCount)
            return domLen;
        if (tileIdx == 0)
            return 0;
        return (tileLen + 1 - tileIter) + (tileIdx - 1) * (tileLen + 2 - 2 * tileIter);
    }
    template <std::signed_integral T>
    struct TileBounds
    {
        T storeBegin, storeEnd, loadBegin, loadEnd;
    };

    template <std::signed_integral T>
    constexpr TileBounds<T> tileBounds(T tileIdx, T tileLen, T tileCount, T tileIter, T domLen)
    {
        const T storeBegin = boundary<T>(tileIdx, tileLen, tileCount, tileIter, domLen);
        const T storeEnd = boundary<T>(tileIdx + 1, tileLen, tileCount, tileIter, domLen);
        return {storeBegin, storeEnd,
                std::max<T>(storeBegin - (tileIter - 1), 0),
                std::min<T>(storeEnd + (tileIter - 1), domLen)};
    }
}