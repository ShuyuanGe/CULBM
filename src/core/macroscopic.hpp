#pragma once

#include "core/generated/d3q15.cuh"
#include "core/generated/d3q19.cuh"
#include "core/generated/d3q27.cuh"

namespace culbm::core::macroscopic
{
    template <int NDir>
    constexpr void compute(real_t& rho, real_t& ux, real_t& uy, real_t& uz, const ddf_t (&f)[NDir])
    {
        if      constexpr (NDir == 15) generated::d3q15::calc_rho_u(rho, ux, uy, uz, f);
        else if constexpr (NDir == 19) generated::d3q19::calc_rho_u(rho, ux, uy, uz, f);
        else if constexpr (NDir == 27) generated::d3q27::calc_rho_u(rho, ux, uy, uz, f);
    }
}
