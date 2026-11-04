#pragma once

#include "kernel.cuh"

namespace culbm::simulator::single_dev::mapping
{
    struct Position
    {
        idx_t blkx, blky, blkz, blknx, blkny, blknz, blki, blkn;
        idx_t glbx, glby, glbz, glbnx, glbny, glbnz, glbi, glbn;

        constexpr Position() = default;

        __device__ Position(idx_t offx, idx_t offy, idx_t offz, idx_t nx, idx_t ny, idx_t nz)
            : blkx(blockDim.x * blockIdx.x + threadIdx.x),
              blky(blockDim.y * blockIdx.y + threadIdx.y),
              blkz(blockDim.z * blockIdx.z + threadIdx.z),
              blknx(gridDim.x * blockDim.x),
              blkny(gridDim.y * blockDim.y),
              blknz(gridDim.z * blockDim.z),
              blki((blkny * blkz + blky) * blknx + blkx), blkn(blknx * blkny * blknz),
              glbx(offx + blkx), glby(offy + blky), glbz(offz + blkz),
              glbnx(nx), glbny(ny), glbnz(nz),
              glbi((glbny * glbz + glby) * glbnx + glbx), glbn(glbnx * glbny * glbnz) {}
    };

    __device__ inline Position makePosition(const PlainParam&)
    {
        return Position(0, 0, 0, gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z);
    }

    __device__ inline Position makePosition(const BlockingParam& param)
    {
        return Position{param.offx, param.offy, param.offz, param.glbnx, param.glbny, param.glbnz};
    }

    template <int Delta>
    constexpr idx_t wrappedOffset(idx_t coord, idx_t size, idx_t stride)
    {
        static_assert(Delta >= -1 && Delta <= 1);
        if      constexpr (Delta == -1) return coord == 0 ? (size - 1) * stride : -stride;
        else if constexpr (Delta ==  1) return coord == size - 1 ? (1 - size) * stride : stride;
        else if constexpr (Delta ==  0) return 0;
    }

    template <int Delta>
    constexpr idx_t clampedOffset(idx_t coord, idx_t size, idx_t stride)
    {
        static_assert(Delta >= -1 && Delta <= 1);
        if      constexpr (Delta == -1) return coord == 0 ? 0 : -stride;
        else if constexpr (Delta ==  1) return coord == size - 1 ? 0 : stride;
        else if constexpr (Delta ==  0) return 0;
    }

    struct GlobalView
    {
        ddf_t* src;
        ddf_t* dst;
        Position pos;

        template <int DX, int DY, int DZ>
        constexpr idx_t cellOffset() const
        {
            return pos.glbi
                + wrappedOffset<DX>(pos.glbx, pos.glbnx, 1)
                + wrappedOffset<DY>(pos.glby, pos.glbny, pos.glbnx)
                + wrappedOffset<DZ>(pos.glbz, pos.glbnz, pos.glbnx * pos.glbny);
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr const ddf_t* readAddress() const
        {
            return src + static_cast<glb_idx_t>(Dir) * pos.glbn + cellOffset<DX, DY, DZ>();
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr ddf_t* writeAddress() const
        {
            return dst + static_cast<glb_idx_t>(Dir) * pos.glbn + cellOffset<DX, DY, DZ>();
        }
    };

    struct RingView
    {
        GlobalView global;
        idx_t nz, srcOffz, dstOffz;

        template <int Dir, int DX, int DY, int DZ>
        constexpr glb_idx_t offset(idx_t offz) const
        {
            const auto nx = global.pos.glbnx;
            const idx_t nxy = nx * global.pos.glbny;
            const idx_t xy = global.pos.glby * nx + global.pos.glbx
                + clampedOffset<DX>(global.pos.glbx, nx, 1)
                + clampedOffset<DY>(global.pos.glby, global.pos.glbny, nx);
            const idx_t z = global.pos.glbz + clampedOffset<DZ>(global.pos.glbz, global.pos.glbnz, 1);
            idx_t ringZ = z + offz;
            if (ringZ >= nz) ringZ -= nz;
            return static_cast<glb_idx_t>(Dir) * nxy * nz
                + static_cast<glb_idx_t>(ringZ) * nxy + xy;
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr const ddf_t* readAddress() const
        {
            return global.src + offset<Dir, DX, DY, DZ>(srcOffz);
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr ddf_t* writeAddress() const
        {
            return global.src + offset<Dir, DX, DY, DZ>(dstOffz);
        }
    };

    template <bool UseRing>
    constexpr auto makeGlobalView(const auto& param, const Position& pos)
    {
        if constexpr (UseRing)
            return RingView{{param.glbSrcDDFBuf, param.glbDstDDFBuf, pos}, param.glbDDFRingNz, param.glbSrcDDFOffz, param.glbDstDDFOffz};
        else
            return GlobalView{param.glbSrcDDFBuf, param.glbDstDDFBuf, pos};
    }

    struct TileOffsets
    {
        const idx_t pdx, pdy, pdz;

        constexpr TileOffsets(const Position& pos)
            : pdx(wrappedOffset<1>(pos.blkx, pos.blknx, 1)),
              pdy(wrappedOffset<1>(pos.blky, pos.blkny, pos.blknx)),
              pdz(wrappedOffset<1>(pos.blkz, pos.blknz, pos.blknx * pos.blkny)) {}
    };

    struct TileView
    {
        ddf_t* data;
        Position pos;
        TileOffsets offsets;

        template <int Dir, int DX, int DY, int DZ>
        constexpr ddf_t* address() const
        {
            const idx_t offset = Dir * pos.blkn + pos.blki
                + DX * offsets.pdx + DY * offsets.pdy + DZ * offsets.pdz;
            return data + offset;
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr const ddf_t* readAddress() const { return address<Dir, DX, DY, DZ>(); }

        template <int Dir, int DX, int DY, int DZ>
        constexpr ddf_t* writeAddress() const { return address<Dir, DX, DY, DZ>(); }
    };

    constexpr TileView makeTileView(const auto& param, const Position& pos, const TileOffsets& offsets)
    {
        return TileView{param.blkDDFBuf, pos, offsets};
    }
}
