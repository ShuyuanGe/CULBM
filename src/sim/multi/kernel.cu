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

    template <int NDir, Kind Kind>
    __global__ void StepKernel(const KernelParam param)
    {
        const mapping::Position pos;

        const auto flag = param.flagBuf[pos.idx];
        real_t rho, vx, vy, vz;
        ddf_t f[NDir];

        const flag_t loadFlags = flag & (LOAD_DDF_BIT | REV_LOAD_DDF_BIT | DEV_BND_BIT);
        if (loadFlags & (LOAD_DDF_BIT | REV_LOAD_DDF_BIT))
        {
            if (not (loadFlags & DEV_BND_BIT))
            {
                const auto view = mapping::DeviceView<false>::make(param, pos);
                ddf::load<NDir, accessor::PullTwoFields, Default>(f, view);
            }
            else
            {
                const auto view = mapping::DeviceView<true>::make(param, pos);
                ddf::load<NDir, accessor::PullTwoFields, Default>(f, view);
            }

            if (flag & REV_LOAD_DDF_BIT)
                Model<Kind, NDir>::bounceBack(f);
        }

        if (flag & EQU_DDF_BIT)
        {
            rho = param.rhoBuf[pos.idx];
            vx  = param.vxBuf[pos.idx];
            vy  = param.vyBuf[pos.idx];
            vz  = param.vzBuf[pos.idx];
            Model<Kind, NDir>::equilibrium(rho, vx, vy, vz, f);
        }

        if (flag & COLLIDE_BIT)
        {
            culbm::core::macroscopic::compute<NDir>(rho, vx, vy, vz, f);
            Model<Kind, NDir>::collide(param.invTau, f);
        }
        else if ((flag & STORE_DDF_BIT) and not (flag & EQU_DDF_BIT))
        {
            culbm::core::macroscopic::compute<NDir>(rho, vx, vy, vz, f);
        }

        if (flag & STORE_DDF_BIT)
        {
            const auto view = mapping::DeviceView<false>::make(param, pos);
            ddf::store<NDir, accessor::PullTwoFields, Default>(f, view);
            param.rhoBuf[pos.idx] = rho;
            param.vxBuf[pos.idx]  = vx;
            param.vyBuf[pos.idx]  = vy;
            param.vzBuf[pos.idx]  = vz;
        }
    }

    constexpr struct
    {
        bool enabled = true;
        int cx = 256, cy = 256, cz = 128, r = 80;
    } obstacle;

    template <int NDir, Kind Kind>
    __global__ void InitKernel(const InitKernelParam param)
    {
        const mapping::Position pos;

        flag_t flag = LOAD_DDF_BIT | COLLIDE_BIT | STORE_DDF_BIT;
        real_t rho = 1;
        real_t vx = 0;
        real_t vy = 0;
        real_t vz = 0;
        ddf_t f[NDir];

        // spherical obstacle
        if constexpr (obstacle.enabled) {
            constexpr int cx = obstacle.cx;
            constexpr int cy = obstacle.cy;
            constexpr int cz = obstacle.cz;
            constexpr int r = obstacle.r;
            constexpr int r2 = r * r;
            const idx_t gx = param.devPos.x * pos.nx + pos.x;
            const idx_t gy = param.devPos.y * pos.ny + pos.y;
            const idx_t gz = param.devPos.z * pos.nz + pos.z;

            if ((gx - cx) * (gx - cx) + (gy - cy) * (gy - cy) + (gz - cz) * (gz - cz) < r2) {
                flag = BOUNCE_BACK_FLAG;
                vx = 0;
                vy = 0;
                vz = 0;
            }
        }

        if (
            (param.devPos.y == 0 and pos.y == 0) or                             // front face
            (param.devPos.y == param.devDim.y - 1 and pos.y == pos.ny - 1) or   // back face
            (param.devPos.z == 0 and pos.z == 0) or                             // down face
            (param.devPos.z == param.devDim.z - 1 and pos.z == pos.nz - 1)       // up face
        )
        {
            flag = REV_LOAD_DDF_BIT | STORE_DDF_BIT;
        }
        else if (param.devPos.x == 0 and pos.x == 0)
        {
            flag = EQU_DDF_BIT | STORE_DDF_BIT;                // left face
            vx = 0.05;
        }
        else if (param.devPos.x == param.devDim.x - 1 and pos.x == pos.nx - 1)
        {
            flag = EQU_DDF_BIT | STORE_DDF_BIT;                // right face
        }

        if (pos.x == 0 or pos.x == pos.nx - 1 or pos.y == 0 or pos.y == pos.ny - 1 or pos.z == 0 or pos.z == pos.nz - 1)
        {
            flag |= DEV_BND_BIT;                               // device boundary
        }

        Model<Kind, NDir>::equilibrium(rho, vx, vy, vz, f);

        param.flagBuf[pos.idx] = flag;
        param.rhoBuf[pos.idx] = rho;
        param.vxBuf[pos.idx] = vx;
        param.vyBuf[pos.idx] = vy;
        param.vzBuf[pos.idx] = vz;

        #pragma unroll
        for (dir_t dir = 0 ; dir < NDir ; ++dir)
        {
            param.srcDdf[dir * pos.n + pos.idx] = f[dir];
            param.dstDdf[dir * pos.n + pos.idx] = f[dir];
        }
    }

    Kernels Kernels::make(int q, std::string_view kind)
    {
        return culbm::core::collision::dispatch(q, kind, [&]<int NDir, Kind Kind> {
            return Kernels{
                .init = &InitKernel<NDir, Kind>,
                .step = &StepKernel<NDir, Kind>
            };
        });
    }
}
