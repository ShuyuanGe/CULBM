#include "kernel.cuh"
#include "mapping.cuh"
#include "core/collision/model.hpp"
#include "core/collision/dispatch.hpp"
#include "core/macroscopic.hpp"
#include "core/accessor.hpp"
#include "core/ddf.hpp"
#include "gpu/io.cuh"

namespace culbm::sim::multi
{
    namespace accessor = culbm::core::accessor;
    namespace ddf      = culbm::core::ddf;
    using culbm::gpu::io::Default;
    using culbm::core::collision::Model;
    using culbm::core::collision::Kind;

    template <int n_dir, Kind collision_kind>
    __global__ void step_kernel(const KernelParam param)
    {
        const mapping::Position pos;

        const auto flag = param.flag_buf[pos.idx];
        real_t rho, vx, vy, vz;
        ddf_t f[n_dir];

        const flag_t load_flags = flag & (LOAD_DDF_BIT | REV_LOAD_DDF_BIT | DEV_BND_BIT);
        if (load_flags & (LOAD_DDF_BIT | REV_LOAD_DDF_BIT))
        {
            if (not (load_flags & DEV_BND_BIT))
            {
                const auto view = mapping::DeviceView<false>::make(param, pos);
                ddf::load<n_dir, accessor::PullTwoFields, Default>(f, view);
            }
            else
            {
                const auto view = mapping::DeviceView<true>::make(param, pos);
                ddf::load<n_dir, accessor::PullTwoFields, Default>(f, view);
            }

            if (flag & REV_LOAD_DDF_BIT)
                Model<collision_kind, n_dir>::bounce_back(f);
        }

        if (flag & EQU_DDF_BIT)
        {
            rho = param.rho_buf[pos.idx];
            vx  = param.vx_buf[pos.idx];
            vy  = param.vy_buf[pos.idx];
            vz  = param.vz_buf[pos.idx];
            Model<collision_kind, n_dir>::equilibrium(rho, vx, vy, vz, f);
        }

        if (flag & COLLIDE_BIT)
        {
            culbm::core::macroscopic::compute<n_dir>(rho, vx, vy, vz, f);
            Model<collision_kind, n_dir>::collide(param.inv_tau, f);
        }
        else if ((flag & STORE_DDF_BIT) and not (flag & EQU_DDF_BIT))
        {
            culbm::core::macroscopic::compute<n_dir>(rho, vx, vy, vz, f);
        }

        if (flag & STORE_DDF_BIT)
        {
            const auto view = mapping::DeviceView<false>::make(param, pos);
            ddf::store<n_dir, accessor::PullTwoFields, Default>(f, view);
            param.rho_buf[pos.idx] = rho;
            param.vx_buf[pos.idx]  = vx;
            param.vy_buf[pos.idx]  = vy;
            param.vz_buf[pos.idx]  = vz;
        }
    }

    constexpr struct
    {
        bool enabled = true;
        int cx = 256, cy = 256, cz = 128, r = 80;
    } obstacle;

    template <int n_dir, Kind collision_kind>
    __global__ void init_kernel(const InitKernelParam param)
    {
        const mapping::Position pos;

        flag_t flag = LOAD_DDF_BIT | COLLIDE_BIT | STORE_DDF_BIT;
        real_t rho = 1;
        real_t vx = 0;
        real_t vy = 0;
        real_t vz = 0;
        ddf_t f[n_dir];

        // spherical obstacle
        if constexpr (obstacle.enabled) {
            constexpr int cx = obstacle.cx;
            constexpr int cy = obstacle.cy;
            constexpr int cz = obstacle.cz;
            constexpr int r = obstacle.r;
            constexpr int r2 = r * r;
            const idx_t gx = param.dev_pos.x * pos.nx + pos.x;
            const idx_t gy = param.dev_pos.y * pos.ny + pos.y;
            const idx_t gz = param.dev_pos.z * pos.nz + pos.z;

            if ((gx - cx) * (gx - cx) + (gy - cy) * (gy - cy) + (gz - cz) * (gz - cz) < r2) {
                flag = BOUNCE_BACK_FLAG;
                vx = 0;
                vy = 0;
                vz = 0;
            }
        }

        if (
            (param.dev_pos.y == 0 and pos.y == 0) or                            // front face
            (param.dev_pos.y == param.dev_dim.y - 1 and pos.y == pos.ny - 1) or // back face
            (param.dev_pos.z == 0 and pos.z == 0) or                            // down face
            (param.dev_pos.z == param.dev_dim.z - 1 and pos.z == pos.nz - 1)    // up face
        )
        {
            flag = REV_LOAD_DDF_BIT | STORE_DDF_BIT;
        }
        else if (param.dev_pos.x == 0 and pos.x == 0)
        {
            flag = EQU_DDF_BIT | STORE_DDF_BIT;                // left face
            vx = 0.05;
        }
        else if (param.dev_pos.x == param.dev_dim.x - 1 and pos.x == pos.nx - 1)
        {
            flag = EQU_DDF_BIT | STORE_DDF_BIT;                // right face
        }

        if (pos.x == 0 or pos.x == pos.nx - 1 or pos.y == 0 or pos.y == pos.ny - 1 or pos.z == 0 or pos.z == pos.nz - 1)
        {
            flag |= DEV_BND_BIT;                               // device boundary
        }

        Model<collision_kind, n_dir>::equilibrium(rho, vx, vy, vz, f);

        param.flag_buf[pos.idx] = flag;
        param.rho_buf[pos.idx] = rho;
        param.vx_buf[pos.idx] = vx;
        param.vy_buf[pos.idx] = vy;
        param.vz_buf[pos.idx] = vz;

        #pragma unroll
        for (dir_t dir = 0 ; dir < n_dir ; ++dir)
        {
            param.src_ddf[dir * pos.n + pos.idx] = f[dir];
            param.dst_ddf[dir * pos.n + pos.idx] = f[dir];
        }
    }

    Kernels Kernels::make(int q, std::string_view kind)
    {
        return culbm::core::collision::dispatch(q, kind, [&]<int n_dir, Kind collision_kind> {
            return Kernels{
                .init = &init_kernel<n_dir, collision_kind>,
                .step = &step_kernel<n_dir, collision_kind>
            };
        });
    }
}
