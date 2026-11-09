#pragma once

#include <string_view>
#include <cuda_runtime.h>
#include "core/types.hpp"
#include "flags.hpp"
#include "mode.hpp"

namespace culbm::sim::single
{
    struct BlockingParam
    {
        static constexpr bool dump_owned_only = true;
        real_t inv_tau = 0;
        idx_t origin_x = 0, origin_y = 0, origin_z = 0;
        idx_t global_nx = 0, global_ny = 0, global_nz = 0;
        const flag_t* flag_buf = nullptr;
        real_t* rho_buf = nullptr;
        real_t* vx_buf = nullptr;
        real_t* vy_buf = nullptr;
        real_t* vz_buf = nullptr;
        ddf_t* src_ddf = nullptr;
        ddf_t* dst_ddf = nullptr;
        ddf_t* tile_ddf = nullptr;
        idx_t ring_nz = 0;
        idx_t src_zoffset = 0;
        idx_t dst_zoffset = 0;
    };

    struct PlainParam
    {
        static constexpr bool dump_owned_only = false;
        real_t inv_tau = 0;
        const flag_t* flag_buf = nullptr;
        real_t* rho_buf = nullptr;
        real_t* vx_buf = nullptr;
        real_t* vy_buf = nullptr;
        real_t* vz_buf = nullptr;
        ddf_t* src_ddf = nullptr;
        ddf_t* dst_ddf = nullptr;
    };

    using Equilibrium = void (*)(real_t, real_t, real_t, real_t, ddf_t*);

    using PlainKernel = void (*)(PlainParam);
    using BlockingKernel = void (*)(BlockingParam);

    struct PullKernels
    {
        static constexpr Mode mode = Mode::pull;
        static PullKernels make(int q, std::string_view kind);
        Equilibrium equilibrium;
        PlainKernel step, dump;
    };

    struct TwistKernels
    {
        static constexpr Mode mode = Mode::twist;
        static TwistKernels make(int q, std::string_view kind);
        Equilibrium equilibrium;
        PlainKernel step_even, step_odd, dump_even, dump_odd;
    };

    struct BlockingKernels
    {
        static BlockingKernels make(int q, std::string_view kind, Mode buffer, bool hints);
        Equilibrium equilibrium;
        BlockingKernel single, first, middle_even, middle_odd, last_even, last_odd, dump;
    };
}
