#pragma once

#include <algorithm>
#include <concepts>
#include <cuda/cmath>

namespace culbm::core::blocking
{
    template <std::integral T>
    constexpr bool validAxis(T domLen, T tileLen, T innerSteps)
    {
        if (domLen <= tileLen) return true;
        if (tileLen + 1 <= innerSteps) return false;
        if (2 * (tileLen + 1 - innerSteps) < domLen and tileLen + 2 <= 2 * innerSteps) return false;
        return true;
    }

    template <std::integral T>
    constexpr T tileCount(T domLen, T tileLen, T innerSteps)
    {
        if (domLen <= tileLen)
            return 1;
        if (domLen <= 2 * (tileLen + 1 - innerSteps))
            return 2;
        return 2 + ::cuda::ceil_div(domLen - 2 * (tileLen + 1 - innerSteps), tileLen + 2 - 2 * innerSteps);
    }

    template <std::integral T>
    constexpr T boundary(T tileIdx, T tileLen, T tileCount, T innerSteps, T domLen)
    {
        if (tileIdx == tileCount)
            return domLen;
        if (tileIdx == 0)
            return 0;
        return (tileLen + 1 - innerSteps) + (tileIdx - 1) * (tileLen + 2 - 2 * innerSteps);
    }
    template <std::signed_integral T>
    struct TileBounds
    {
        T storeBegin, storeEnd, loadBegin, loadEnd;
    };

    template <std::signed_integral T>
    constexpr TileBounds<T> tileBounds(T tileIdx, T tileLen, T tileCount, T innerSteps, T domLen)
    {
        const T storeBegin = boundary(tileIdx, tileLen, tileCount, innerSteps, domLen);
        const T storeEnd = boundary<T>(tileIdx + 1, tileLen, tileCount, innerSteps, domLen);
        return {storeBegin, storeEnd,
                std::max<T>(storeBegin - (innerSteps - 1), 0),
                std::min<T>(storeEnd + (innerSteps - 1), domLen)};
    }
}
