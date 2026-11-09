#pragma once

#include "kernel.cuh"

namespace culbm::sim::multi::mapping
{
    struct Position
    {
        idx_t nx, ny, nz, n, x, y, z, idx;

        __device__ Position()
            : nx(gridDim.x * blockDim.x),
              ny(gridDim.y * blockDim.y),
              nz(gridDim.z * blockDim.z), n(nx * ny * nz),
              x(blockDim.x * blockIdx.x + threadIdx.x),
              y(blockDim.y * blockIdx.y + threadIdx.y),
              z(blockDim.z * blockIdx.z + threadIdx.z), idx(x + nx * (y + ny * z)) {}
    };

    template <bool CrossDevice>
    struct DeviceView
    {
        ddf_t* const* src;
        ddf_t* dst;
        idx_t idx, n, ndx, pdx, ndy, pdy, ndz, pdz;
        nbr_t nbrndx, nbrpdx, nbrndy, nbrpdy, nbrndz, nbrpdz;

        static constexpr DeviceView make(const KernelParam& param, const Position& pos)
        {
            DeviceView view{
                param.srcDdf.data(), param.dstDdf, pos.idx, pos.n,
                (pos.x == 0) ? pos.nx - 1 : -1,
                (pos.x == pos.nx - 1) ? 1 - pos.nx : 1,
                (pos.y == 0) ? pos.nx * (pos.ny - 1) : -pos.nx,
                (pos.y == pos.ny - 1) ? pos.nx * (1 - pos.ny) : pos.nx,
                (pos.z == 0) ? pos.nx * pos.ny * (pos.nz - 1) : -pos.nx * pos.ny,
                (pos.z == pos.nz - 1) ? pos.nx * pos.ny * (1 - pos.nz) : pos.nx * pos.ny
            };
            if constexpr (CrossDevice)
            {
                view.nbrndx = (pos.x == 0) ? -1 : 0;
                view.nbrpdx = (pos.x == pos.nx - 1) ? 1 : 0;
                view.nbrndy = (pos.y == 0) ? -3 : 0;
                view.nbrpdy = (pos.y == pos.ny - 1) ? 3 : 0;
                view.nbrndz = (pos.z == 0) ? -9 : 0;
                view.nbrpdz = (pos.z == pos.nz - 1) ? 9 : 0;
            }
            return view;
        }

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
