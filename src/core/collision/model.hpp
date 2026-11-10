#pragma once

#include "core/types.hpp"
#include "core/stencil.hpp"
#include <utility>
#include "kind.hpp"
#include "core/generated/d3q15.cuh"
#include "core/generated/d3q19.cuh"
#include "core/generated/d3q27.cuh"

namespace culbm::core::collision
{
    template <Kind collision_kind, int n_dir>
        requires requires { typename Stencil<n_dir>; } && (n_dir != 15 || collision_kind != Kind::cumulant)
    struct Model
    {
        static constexpr void bounce_back(ddf_t (&f)[n_dir])
        {
            #pragma unroll
            for (int q = 0; q < n_dir / 2; ++q)
                std::swap(f[q], f[Stencil<n_dir>::opposite(q)]);
        }

        static constexpr void collide(real_t omega, ddf_t (&f)[n_dir])
        {
            if constexpr (n_dir == 15)
            {
                if      constexpr (collision_kind == Kind::bgk)      generated::d3q15::bgk_collide(omega, f);
                else if constexpr (collision_kind == Kind::mrt)      generated::d3q15::mrt_collide(omega, f);
            }
            else if constexpr (n_dir == 19)
            {
                if      constexpr (collision_kind == Kind::bgk)      generated::d3q19::bgk_collide(omega, f);
                else if constexpr (collision_kind == Kind::mrt)      generated::d3q19::mrt_collide(omega, f);
                else if constexpr (collision_kind == Kind::cumulant) generated::d3q19::cumulant_collide(omega, f);
            }
            else if constexpr (n_dir == 27)
            {
                if      constexpr (collision_kind == Kind::bgk)      generated::d3q27::bgk_collide(omega, f);
                else if constexpr (collision_kind == Kind::mrt)      generated::d3q27::mrt_collide(omega, f);
                else if constexpr (collision_kind == Kind::cumulant) generated::d3q27::cumulant_collide(omega, f);
            }
        }

        static constexpr void equilibrium(real_t rho, real_t ux, real_t uy, real_t uz, ddf_t (&f)[n_dir])
        {
            if constexpr (n_dir == 15)
            {
                if      constexpr (collision_kind == Kind::bgk)      generated::d3q15::bgk_equilibrium(rho, ux, uy, uz, f);
                else if constexpr (collision_kind == Kind::mrt)      generated::d3q15::mrt_equilibrium(rho, ux, uy, uz, f);
            }
            else if constexpr (n_dir == 19)
            {
                if      constexpr (collision_kind == Kind::bgk)      generated::d3q19::bgk_equilibrium(rho, ux, uy, uz, f);
                else if constexpr (collision_kind == Kind::mrt)      generated::d3q19::mrt_equilibrium(rho, ux, uy, uz, f);
                else if constexpr (collision_kind == Kind::cumulant) generated::d3q19::cumulant_equilibrium(rho, ux, uy, uz, f);
            }
            else if constexpr (n_dir == 27)
            {
                if      constexpr (collision_kind == Kind::bgk)      generated::d3q27::bgk_equilibrium(rho, ux, uy, uz, f);
                else if constexpr (collision_kind == Kind::mrt)      generated::d3q27::mrt_equilibrium(rho, ux, uy, uz, f);
                else if constexpr (collision_kind == Kind::cumulant) generated::d3q27::cumulant_equilibrium(rho, ux, uy, uz, f);
            }
        }
    };

}
