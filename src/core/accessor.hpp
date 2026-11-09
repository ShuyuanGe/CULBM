#pragma once

#include <cstddef>
#include "core/stencil.hpp"

namespace culbm::core::accessor
{
    struct Access { int direction, x, y, z; };

    consteval int positive(int value) { return value > 0 ? value : 0; }

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

    // CULBM bounce-back: read the opposite population with the pull offset reversed.
    struct ReversePull
    {
        template <int NDir, std::size_t Q>
        static consteval Access read()
        {
            constexpr auto d = Stencil<NDir>::directions[Q];
            constexpr int reverse = Stencil<NDir>::opposite(Q);
            return {reverse, d.x, d.y, d.z};
        }
    };

    // lbmpy.fieldaccess.EsoTwistEvenTimeStepAccessor
    struct EsoTwistEven
    {
        template <int NDir, std::size_t Q>
        static consteval Access read()
        {
            constexpr auto d = Stencil<NDir>::directions[Q];
            return {Q, positive(-d.x), positive(-d.y), positive(-d.z)};
        }

        template <int NDir, std::size_t Q>
        static consteval Access write()
        {
            constexpr auto d = Stencil<NDir>::directions[Q];
            constexpr int reverse = Stencil<NDir>::opposite(Q);
            return {reverse, positive(d.x), positive(d.y), positive(d.z)};
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
            return {reverse, positive(-d.x), positive(-d.y), positive(-d.z)};
        }

        template <int NDir, std::size_t Q>
        static consteval Access write()
        {
            constexpr auto d = Stencil<NDir>::directions[Q];
            return {Q, positive(d.x), positive(d.y), positive(d.z)};
        }
    };

}
