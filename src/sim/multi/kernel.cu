#include "kernel.cuh"
#include "core/collision/model.hpp"
#include "core/collision/dispatch.hpp"
#include "core/macroscopic.hpp"
#include "core/accessor.hpp"
#include "core/ddf.hpp"
#include "gpu/io.cuh"

namespace culbm::sim::multi::detail
{
    template <bool CrossDevice>
    struct DeviceView
    {
        ddf_t* const* src;
        ddf_t* dst;
        idx_t idx, n, ndx, pdx, ndy, pdy, ndz, pdz;
        nbr_t nbrndx, nbrpdx, nbrndy, nbrpdy, nbrndz, nbrpdz;

        template <int Dir, int DX, int DY, int DZ>
        constexpr const ddf_t* readAddress() const
        {
            nbr_t peer = 13;
            if constexpr (CrossDevice)
                peer += (DX < 0 ? nbrndx : DX > 0 ? nbrpdx : 0)
                      + (DY < 0 ? nbrndy : DY > 0 ? nbrpdy : 0)
                      + (DZ < 0 ? nbrndz : DZ > 0 ? nbrpdz : 0);

            const idx_t cell = idx
                + (DX < 0 ? ndx : DX > 0 ? pdx : 0)
                + (DY < 0 ? ndy : DY > 0 ? pdy : 0)
                + (DZ < 0 ? ndz : DZ > 0 ? pdz : 0);
            return src[peer] + static_cast<ddf_offset_t>(Dir) * n + cell;
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr ddf_t* writeAddress() const
        {
            return dst + (static_cast<ddf_offset_t>(Dir) * n + idx);
        }
    };
}

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

        const auto flagn = param.flagBuf[idx];
        real_t rhon, vxn, vyn, vzn;
        ddf_t fin[NDir];

        const flag_t iFlagn = flagn & (LOAD_DDF_BIT | REV_LOAD_DDF_BIT | DEV_BND_BIT);
        if (iFlagn)
        {
            const idx_t pdx = (x == nx - 1) ? 1 - nx : 1;
            const idx_t pdy = (y == ny - 1) ? nx * (1 - ny) : nx;
            const idx_t pdz = (z == nz - 1) ? nx * ny * (1 - nz) : nx * ny;
            const idx_t ndx = (x == 0) ? nx - 1 : -1;
            const idx_t ndy = (y == 0) ? nx * (ny - 1) : -nx;
            const idx_t ndz = (z == 0) ? nx * ny * (nz - 1) : -nx * ny;

            if (iFlagn == LOAD_DDF_BIT)
            {
                const detail::DeviceView<false> view{param.srcDdf.data(), param.dstDdf, idx, n, ndx, pdx, ndy, pdy, ndz, pdz};
                ddf::load<NDir, accessor::PullTwoFields, Default>(fin, view);
            }

            if (iFlagn == REV_LOAD_DDF_BIT)
            {
                const detail::DeviceView<false> view{param.srcDdf.data(), param.dstDdf, idx, n, ndx, pdx, ndy, pdy, ndz, pdz};
                ddf::load<NDir, accessor::ReversePull, Default>(fin, view);
            }

            if (iFlagn & DEV_BND_BIT)
            {
                const nbr_t nbrpdx = (x == nx - 1) ? 1 : 0;
                const nbr_t nbrpdy = (y == ny - 1) ? 3 : 0;
                const nbr_t nbrpdz = (z == nz - 1) ? 9 : 0;
                const nbr_t nbrndx = (x == 0) ? -1 : 0;
                const nbr_t nbrndy = (y == 0) ? -3 : 0;
                const nbr_t nbrndz = (z == 0) ? -9 : 0;

                if (iFlagn == (LOAD_DDF_BIT | DEV_BND_BIT))
                {
                    const detail::DeviceView<true> view{param.srcDdf.data(), param.dstDdf, idx, n, ndx, pdx, ndy, pdy, ndz, pdz, nbrndx, nbrpdx, nbrndy, nbrpdy, nbrndz, nbrpdz};
                    ddf::load<NDir, accessor::PullTwoFields, Default>(fin, view);
                }

                if (iFlagn == (REV_LOAD_DDF_BIT | DEV_BND_BIT))
                {
                    const detail::DeviceView<true> view{param.srcDdf.data(), param.dstDdf, idx, n, ndx, pdx, ndy, pdy, ndz, pdz, nbrndx, nbrpdx, nbrndy, nbrpdy, nbrndz, nbrpdz};
                    ddf::load<NDir, accessor::ReversePull, Default>(fin, view);
                }
            }
        }

        if (flagn & EQU_DDF_BIT)
        {
            rhon = param.rhoBuf[idx];
            vxn  = param.vxBuf[idx];
            vyn  = param.vyBuf[idx];
            vzn  = param.vzBuf[idx];
            Model<Kind, NDir>::equilibrium(rhon, vxn, vyn, vzn, fin);
        }

        if (flagn & COLLIDE_BIT)
        {
            culbm::core::macroscopic::compute<NDir>(rhon, vxn, vyn, vzn, fin);
            Model<Kind, NDir>::collide(param.invTau, fin);
        }
        else if ((flagn & STORE_DDF_BIT) and not (flagn & EQU_DDF_BIT))
        {
            culbm::core::macroscopic::compute<NDir>(rhon, vxn, vyn, vzn, fin);
        }

        if (flagn & STORE_DDF_BIT)
        {
            const detail::DeviceView<false> view{param.srcDdf.data(), param.dstDdf, idx, n};
            ddf::store<NDir, accessor::PullTwoFields, Default>(fin, view);
            param.rhoBuf[idx] = rhon;
            param.vxBuf[idx]  = vxn;
            param.vyBuf[idx]  = vyn;
            param.vzBuf[idx]  = vzn;
        }
    }

    constexpr auto spherical_obstacle_params = [] {
        struct SphericalObstacleParams {
            bool enabled = true;
            int cx = 256;
            int cy = 256;
            int cz = 128;
            int r = 80;
        } params;
        return params;
    }();

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

        flag_t flagn = LOAD_DDF_BIT | COLLIDE_BIT | STORE_DDF_BIT;
        real_t rhon = 1;
        real_t vxn = 0;
        real_t vyn = 0;
        real_t vzn = 0;
        ddf_t fin[NDir];

        // spherical obstacle
        if constexpr (spherical_obstacle_params.enabled) {
            constexpr int cx = spherical_obstacle_params.cx;
            constexpr int cy = spherical_obstacle_params.cy;
            constexpr int cz = spherical_obstacle_params.cz;
            constexpr int r = spherical_obstacle_params.r;
            constexpr int r2 = r * r;
            const idx_t gx = param.devIdx.x * nx + x;
            const idx_t gy = param.devIdx.y * ny + y;
            const idx_t gz = param.devIdx.z * nz + z;

            if ((gx - cx) * (gx - cx) + (gy - cy) * (gy - cy) + (gz - cz) * (gz - cz) < r2) {
                flagn = BOUNCE_BACK_FLAG;
                vxn = 0;
                vyn = 0;
                vzn = 0;
            }
        }

        if (
            (param.devIdx.y == 0 and y == 0) or                        // front face
            (param.devIdx.y == param.devDim.y - 1 and y == ny - 1) or  // back face
            (param.devIdx.z == 0 and z == 0) or                        // down face
            (param.devIdx.z == param.devDim.z - 1 and z == nz - 1)     // up face
        )
        {
            flagn = REV_LOAD_DDF_BIT | STORE_DDF_BIT;
        }
        else if (param.devIdx.x == 0 and x == 0)
        {
            flagn = EQU_DDF_BIT | STORE_DDF_BIT;                // left face
            vxn = 0.05;
        }
        else if (param.devIdx.x == param.devDim.x - 1 and x == nx - 1)
        {
            flagn = EQU_DDF_BIT | STORE_DDF_BIT;                // right face
        }

        if (x == 0 or x == nx - 1 or y == 0 or y == ny - 1 or z == 0 or z == nz - 1)
        {
            flagn |= DEV_BND_BIT;                               // device boundary
        }

        Model<Kind, NDir>::equilibrium(rhon, vxn, vyn, vzn, fin);

        param.flagBuf[idx] = flagn;
        param.rhoBuf[idx] = rhon;
        param.vxBuf[idx] = vxn;
        param.vyBuf[idx] = vyn;
        param.vzBuf[idx] = vzn;

        #pragma unroll
        for (dir_t dir = 0 ; dir < NDir ; ++dir)
        {
            param.srcDdf[dir * n + idx] = fin[dir];
            param.dstDdf[dir * n + idx] = fin[dir];
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
