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
        template <int n_dir, std::size_t Q>
        static consteval Access read()
        {
            constexpr auto d = Stencil<n_dir>::directions[Q];
            return {Q, -d.x, -d.y, -d.z};
        }

        template <int n_dir, std::size_t Q>
        static consteval Access write() { return {Q, 0, 0, 0}; }
    };

    template <bool odd>
    struct EsoTwist
    {
        template <int n_dir, std::size_t Q>
        static consteval Access read()
        {
            constexpr auto d = Stencil<n_dir>::directions[Q];
            constexpr int direction = odd ? Stencil<n_dir>::opposite(Q) : Q;
            return {direction, std::max(-d.x, 0), std::max(-d.y, 0), std::max(-d.z, 0)};
        }

        template <int n_dir, std::size_t Q>
        static consteval Access write()
        {
            constexpr auto d = Stencil<n_dir>::directions[Q];
            constexpr int direction = odd ? Q : Stencil<n_dir>::opposite(Q);
            return {direction, std::max(d.x, 0), std::max(d.y, 0), std::max(d.z, 0)};
        }
    };

    // lbmpy.fieldaccess.EsoTwistEvenTimeStepAccessor / EsoTwistOddTimeStepAccessor
    using EsoTwistEven = EsoTwist<false>;
    using EsoTwistOdd = EsoTwist<true>;

}
