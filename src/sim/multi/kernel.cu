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
    __global__ void
    StepKernel(const KernelParam __grid_constant__ param)
    {
        const idx_t nx = gridDim.x * blockDim.x;
        const idx_t ny = gridDim.y * blockDim.y;
        const idx_t nz = gridDim.z * blockDim.z;
        const idx_t  n = nx * ny * nz;
        const idx_t  x = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t  y = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t  z = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t  idx = x + nx * (y + ny * z);

        const auto flag = param.flagBuf[idx];
        real_t rho, vx, vy, vz;
        ddf_t f[NDir];

        const flag_t loadFlags = flag & (LOAD_DDF_BIT | REV_LOAD_DDF_BIT | DEV_BND_BIT);
        if (loadFlags & (LOAD_DDF_BIT | REV_LOAD_DDF_BIT))
        {
            const idx_t pdx = (x == nx - 1) ? 1 - nx : 1;
            const idx_t pdy = (y == ny - 1) ? nx * (1 - ny) : nx;
            const idx_t pdz = (z == nz - 1) ? nx * ny * (1 - nz) : nx * ny;
            const idx_t ndx = (x == 0) ? nx - 1 : -1;
            const idx_t ndy = (y == 0) ? nx * (ny - 1) : -nx;
            const idx_t ndz = (z == 0) ? nx * ny * (nz - 1) : -nx * ny;

            if (not (loadFlags & DEV_BND_BIT))
            {
                const mapping::DeviceView<false> view{param.srcDdf.data(), param.dstDdf, idx, n, ndx, pdx, ndy, pdy, ndz, pdz};
                ddf::load<NDir, accessor::PullTwoFields, Default>(f, view);
            }
            else
            {
                const nbr_t nbrpdx = (x == nx - 1) ? 1 : 0;
                const nbr_t nbrpdy = (y == ny - 1) ? 3 : 0;
                const nbr_t nbrpdz = (z == nz - 1) ? 9 : 0;
                const nbr_t nbrndx = (x == 0) ? -1 : 0;
                const nbr_t nbrndy = (y == 0) ? -3 : 0;
                const nbr_t nbrndz = (z == 0) ? -9 : 0;

                const mapping::DeviceView<true> view{param.srcDdf.data(), param.dstDdf, idx, n, ndx, pdx, ndy, pdy, ndz, pdz, nbrndx, nbrpdx, nbrndy, nbrpdy, nbrndz, nbrpdz};
                ddf::load<NDir, accessor::PullTwoFields, Default>(f, view);
            }

            if (flag & REV_LOAD_DDF_BIT)
                Model<Kind, NDir>::bounceBack(f);
        }

        if (flag & EQU_DDF_BIT)
        {
            rho = param.rhoBuf[idx];
            vx  = param.vxBuf[idx];
            vy  = param.vyBuf[idx];
            vz  = param.vzBuf[idx];
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
            const mapping::DeviceView<false> view{param.srcDdf.data(), param.dstDdf, idx, n};
            ddf::store<NDir, accessor::PullTwoFields, Default>(f, view);
            param.rhoBuf[idx] = rho;
            param.vxBuf[idx]  = vx;
            param.vyBuf[idx]  = vy;
            param.vzBuf[idx]  = vz;
        }
    }

    constexpr struct
    {
        bool enabled = true;
        int cx = 256, cy = 256, cz = 128, r = 80;
    } obstacle;

    template <int NDir, Kind Kind>
    __global__ void
    InitKernel(const InitKernelParam __grid_constant__ param)
    {
        const idx_t nx = gridDim.x * blockDim.x;
        const idx_t ny = gridDim.y * blockDim.y;
        const idx_t nz = gridDim.z * blockDim.z;
        const idx_t x = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t y = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t z = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t n = nx * ny * nz;
        const idx_t idx = x + nx * (y + ny * z);

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
            const idx_t gx = param.devIdx.x * nx + x;
            const idx_t gy = param.devIdx.y * ny + y;
            const idx_t gz = param.devIdx.z * nz + z;

            if ((gx - cx) * (gx - cx) + (gy - cy) * (gy - cy) + (gz - cz) * (gz - cz) < r2) {
                flag = BOUNCE_BACK_FLAG;
                vx = 0;
                vy = 0;
                vz = 0;
            }
        }

        if (
            (param.devIdx.y == 0 and y == 0) or                        // front face
            (param.devIdx.y == param.devDim.y - 1 and y == ny - 1) or  // back face
            (param.devIdx.z == 0 and z == 0) or                        // down face
            (param.devIdx.z == param.devDim.z - 1 and z == nz - 1)     // up face
        )
        {
            flag = REV_LOAD_DDF_BIT | STORE_DDF_BIT;
        }
        else if (param.devIdx.x == 0 and x == 0)
        {
            flag = EQU_DDF_BIT | STORE_DDF_BIT;                // left face
            vx = 0.05;
        }
        else if (param.devIdx.x == param.devDim.x - 1 and x == nx - 1)
        {
            flag = EQU_DDF_BIT | STORE_DDF_BIT;                // right face
        }

        if (x == 0 or x == nx - 1 or y == 0 or y == ny - 1 or z == 0 or z == nz - 1)
        {
            flag |= DEV_BND_BIT;                               // device boundary
        }

        Model<Kind, NDir>::equilibrium(rho, vx, vy, vz, f);

        param.flagBuf[idx] = flag;
        param.rhoBuf[idx] = rho;
        param.vxBuf[idx] = vx;
        param.vyBuf[idx] = vy;
        param.vzBuf[idx] = vz;

        #pragma unroll
        for (dir_t dir = 0 ; dir < NDir ; ++dir)
        {
            param.srcDdf[dir * n + idx] = f[dir];
            param.dstDdf[dir * n + idx] = f[dir];
        }
    }

    Kernels makeKernels(int q, std::string_view kind)
    {
        return culbm::core::collision::dispatch(q, kind, [&]<int NDir, Kind Kind> {
            return Kernels{
                .init = &InitKernel<NDir, Kind>,
                .step = &StepKernel<NDir, Kind>
            };
        });
    }
}
