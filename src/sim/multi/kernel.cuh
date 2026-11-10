#pragma once

#include <array>
#include <string_view>
#include <cuda_runtime.h>
#include "core/types.hpp"
#include "flags.hpp"

namespace culbm::sim::multi
{
    struct KernelParam
    {
        real_t inv_tau;
        flag_t *flag_buf = nullptr;
        real_t *rho_buf  = nullptr;
        real_t *vx_buf   = nullptr;
        real_t *vy_buf   = nullptr;
        real_t *vz_buf   = nullptr;
        std::array<ddf_t*, 27> src_ddf {nullptr};
        ddf_t  *dst_ddf = nullptr;
    };

    struct InitKernelParam
    {
        dim3 dev_dim{};
        uint3 dev_pos{};
        flag_t *flag_buf = nullptr;
        real_t *rho_buf  = nullptr;
        real_t *vx_buf   = nullptr;
        real_t *vy_buf   = nullptr;
        real_t *vz_buf   = nullptr;
        ddf_t  *src_ddf  = nullptr;
        ddf_t  *dst_ddf  = nullptr;
    };

    struct Kernels
    {
        static Kernels make(int q, std::string_view kind);
        void (*init)(InitKernelParam);
        void (*step)(KernelParam);
    };
}
