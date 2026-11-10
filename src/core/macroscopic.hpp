#pragma once

#include "core/stencil.hpp"
#include "core/generated/d3q15.cuh"
#include "core/generated/d3q19.cuh"
#include "core/generated/d3q27.cuh"

namespace culbm::core::macroscopic
{
    template <int n_dir>
        requires requires { typename Stencil<n_dir>; }
    constexpr void compute(real_t& rho, real_t& ux, real_t& uy, real_t& uz, const ddf_t (&f)[n_dir])
    {
        if      constexpr (n_dir == 15) generated::d3q15::calc_rho_u(rho, ux, uy, uz, f);
        else if constexpr (n_dir == 19) generated::d3q19::calc_rho_u(rho, ux, uy, uz, f);
        else if constexpr (n_dir == 27) generated::d3q27::calc_rho_u(rho, ux, uy, uz, f);
    }
}
