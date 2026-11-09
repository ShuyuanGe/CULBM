#pragma once

#include "core/types.hpp"
#include "kind.hpp"
#include "core/generated/d3q15.cuh"
#include "core/generated/d3q19.cuh"
#include "core/generated/d3q27.cuh"

namespace culbm::core::collision
{
    template <Kind Kind, int NDir>
        requires (NDir != 15 || Kind != Kind::Cumulant)
    struct Model
    {
        static constexpr void collide(real_t omega, ddf_t (&f)[NDir])
        {
            if constexpr (NDir == 15)
            {
                if      constexpr (Kind == Kind::Bgk)      generated::d3q15::bgk_collide(omega, f);
                else if constexpr (Kind == Kind::Mrt)      generated::d3q15::mrt_collide(omega, f);
            }
            else if constexpr (NDir == 19)
            {
                if      constexpr (Kind == Kind::Bgk)      generated::d3q19::bgk_collide(omega, f);
                else if constexpr (Kind == Kind::Mrt)      generated::d3q19::mrt_collide(omega, f);
                else if constexpr (Kind == Kind::Cumulant) generated::d3q19::cumulant_collide(omega, f);
            }
            else if constexpr (NDir == 27)
            {
                if      constexpr (Kind == Kind::Bgk)      generated::d3q27::bgk_collide(omega, f);
                else if constexpr (Kind == Kind::Mrt)      generated::d3q27::mrt_collide(omega, f);
                else if constexpr (Kind == Kind::Cumulant) generated::d3q27::cumulant_collide(omega, f);
            }
        }

        static constexpr void equilibrium(real_t rho, real_t ux, real_t uy, real_t uz, ddf_t (&f)[NDir])
        {
            if constexpr (NDir == 15)
            {
                if      constexpr (Kind == Kind::Bgk)      generated::d3q15::bgk_equilibrium(rho, ux, uy, uz, f);
                else if constexpr (Kind == Kind::Mrt)      generated::d3q15::mrt_equilibrium(rho, ux, uy, uz, f);
            }
            else if constexpr (NDir == 19)
            {
                if      constexpr (Kind == Kind::Bgk)      generated::d3q19::bgk_equilibrium(rho, ux, uy, uz, f);
                else if constexpr (Kind == Kind::Mrt)      generated::d3q19::mrt_equilibrium(rho, ux, uy, uz, f);
                else if constexpr (Kind == Kind::Cumulant) generated::d3q19::cumulant_equilibrium(rho, ux, uy, uz, f);
            }
            else if constexpr (NDir == 27)
            {
                if      constexpr (Kind == Kind::Bgk)      generated::d3q27::bgk_equilibrium(rho, ux, uy, uz, f);
                else if constexpr (Kind == Kind::Mrt)      generated::d3q27::mrt_equilibrium(rho, ux, uy, uz, f);
                else if constexpr (Kind == Kind::Cumulant) generated::d3q27::cumulant_equilibrium(rho, ux, uy, uz, f);
            }
        }
    };

}
