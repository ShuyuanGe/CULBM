#pragma once

#include "collision/generated/d3q15.cuh"
#include "collision/generated/d3q19.cuh"
#include "collision/generated/d3q27.cuh"

namespace culbm::core::macroscopic
{
    template <int NDir>
    constexpr void compute(real_t& rho, real_t& ux, real_t& uy, real_t& uz, const ddf_t* f)
    {
        if      constexpr (NDir == 15) collision::generated::d3q15::calc_rho_u(rho, ux, uy, uz, f);
        else if constexpr (NDir == 19) collision::generated::d3q19::calc_rho_u(rho, ux, uy, uz, f);
        else if constexpr (NDir == 27) collision::generated::d3q27::calc_rho_u(rho, ux, uy, uz, f);
    }
}
