#pragma once

#include <algorithm>
#include <cstddef>
#include "core/stencil.hpp"

namespace culbm::core::accessor
{
    struct Access { int direction, x, y, z; };

    // lbmpy.fieldaccess.StreamPullTwoFieldsAccessor
    struct PullTwoFields
    {
        template <int NDir, std::size_t Q>
        static consteval Access read()
        {
            constexpr auto d = Stencil<NDir>::directions[Q];
            return {Q, -d.x, -d.y, -d.z};
        }

        template <int NDir, std::size_t Q>
        static consteval Access write() { return {Q, 0, 0, 0}; }
    };

    // lbmpy.fieldaccess.EsoTwistEvenTimeStepAccessor
    struct EsoTwistEven
    {
        template <int NDir, std::size_t Q>
        static consteval Access read()
        {
            constexpr auto d = Stencil<NDir>::directions[Q];
            return {Q, std::max(-d.x, 0), std::max(-d.y, 0), std::max(-d.z, 0)};
        }

        template <int NDir, std::size_t Q>
        static consteval Access write()
        {
            constexpr auto d = Stencil<NDir>::directions[Q];
            constexpr int reverse = Stencil<NDir>::opposite(Q);
            return {reverse, std::max(d.x, 0), std::max(d.y, 0), std::max(d.z, 0)};
        }
    };

    // lbmpy.fieldaccess.EsoTwistOddTimeStepAccessor
    struct EsoTwistOdd
    {
        template <int NDir, std::size_t Q>
        static consteval Access read()
        {
            constexpr auto d = Stencil<NDir>::directions[Q];
            constexpr int reverse = Stencil<NDir>::opposite(Q);
            return {reverse, std::max(-d.x, 0), std::max(-d.y, 0), std::max(-d.z, 0)};
        }

        template <int NDir, std::size_t Q>
        static consteval Access write()
        {
            constexpr auto d = Stencil<NDir>::directions[Q];
            return {Q, std::max(d.x, 0), std::max(d.y, 0), std::max(d.z, 0)};
        }
    };

}
