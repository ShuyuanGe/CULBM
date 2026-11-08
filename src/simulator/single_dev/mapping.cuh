#pragma once

#include "kernel.cuh"

namespace culbm::simulator::single_dev::mapping
{
    struct Position
    {
        idx_t localX, localY, localZ, localNx, localNy, localNz, localIndex, localSize;
        idx_t glbx, glby, glbz, glbnx, glbny, glbnz, glbi, glbn;

        constexpr Position() = default;

        __device__ Position(idx_t offx, idx_t offy, idx_t offz, idx_t nx, idx_t ny, idx_t nz)
            : localX(blockDim.x * blockIdx.x + threadIdx.x),
              localY(blockDim.y * blockIdx.y + threadIdx.y),
              localZ(blockDim.z * blockIdx.z + threadIdx.z),
              localNx(gridDim.x * blockDim.x),
              localNy(gridDim.y * blockDim.y),
              localNz(gridDim.z * blockDim.z),
              localIndex((localNy * localZ + localY) * localNx + localX), localSize(localNx * localNy * localNz),
              glbx(offx + localX), glby(offy + localY), glbz(offz + localZ),
              glbnx(nx), glbny(ny), glbnz(nz),
              glbi((glbny * glbz + glby) * glbnx + glbx), glbn(glbnx * glbny * glbnz) {}
    };

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

    enum class ViewSpace { Global, Tile };

    struct GlobalView
    {
        static constexpr ViewSpace space = ViewSpace::Global;
        ddf_t* src;
        ddf_t* dst;
        Position pos;

        static constexpr GlobalView make(const auto& param, const Position& pos)
        {
            return {param.glbSrcDDFBuf, param.glbDstDDFBuf, pos};
        }

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
            return src + static_cast<ddf_offset_t>(Dir) * pos.glbn + cellOffset<DX, DY, DZ>();
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr ddf_t* writeAddress() const
        {
            return dst + static_cast<ddf_offset_t>(Dir) * pos.glbn + cellOffset<DX, DY, DZ>();
        }
    };

    struct RingView
    {
        static constexpr ViewSpace space = ViewSpace::Global;
        GlobalView global;
        idx_t nz, srcOffz, dstOffz;

        static constexpr RingView make(const BlockingParam& param, const Position& pos)
        {
            return {{param.glbSrcDDFBuf, param.glbDstDDFBuf, pos}, param.glbDDFRingNz, param.glbSrcDDFOffz, param.glbDstDDFOffz};
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr ddf_offset_t offset(idx_t offz) const
        {
            const auto nx = global.pos.glbnx;
            const idx_t nxy = nx * global.pos.glbny;
            const idx_t xy = global.pos.glby * nx + global.pos.glbx
                + clampedOffset<DX>(global.pos.glbx, nx, 1)
                + clampedOffset<DY>(global.pos.glby, global.pos.glbny, nx);
            const idx_t z = global.pos.glbz + clampedOffset<DZ>(global.pos.glbz, global.pos.glbnz, 1);
            idx_t ringZ = z + offz;
            if (ringZ >= nz) ringZ -= nz;
            return static_cast<ddf_offset_t>(Dir) * nxy * nz
                + static_cast<ddf_offset_t>(ringZ) * nxy + xy;
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

    struct TileOffsets
    {
        const idx_t pdx, pdy, pdz;

        constexpr TileOffsets(const Position& pos)
            : pdx(wrappedOffset<1>(pos.localX, pos.localNx, 1)),
              pdy(wrappedOffset<1>(pos.localY, pos.localNy, pos.localNx)),
              pdz(wrappedOffset<1>(pos.localZ, pos.localNz, pos.localNx * pos.localNy)) {}
    };

    struct TilePosition : Position
    {
        // Precalculate offsets once for load and store.
        TileOffsets offsets;

        constexpr explicit TilePosition(const Position& pos) : Position(pos), offsets(pos) {}
    };

    __device__ inline Position makePosition(const PlainParam&)
    {
        return Position(0, 0, 0, gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z);
    }

    __device__ inline TilePosition makePosition(const BlockingParam& param)
    {
        return TilePosition(Position(param.offx, param.offy, param.offz, param.glbnx, param.glbny, param.glbnz));
    }

    struct TileView
    {
        static constexpr ViewSpace space = ViewSpace::Tile;
        ddf_t* data;
        TilePosition pos;

        static constexpr TileView make(const BlockingParam& param, const TilePosition& pos)
        {
            return {param.tileDDFBuf, pos};
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr ddf_t* address() const
        {
            const idx_t offset = Dir * pos.localSize + pos.localIndex
                + DX * pos.offsets.pdx + DY * pos.offsets.pdy + DZ * pos.offsets.pdz;
            return data + offset;
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr const ddf_t* readAddress() const { return address<Dir, DX, DY, DZ>(); }

        template <int Dir, int DX, int DY, int DZ>
        constexpr ddf_t* writeAddress() const { return address<Dir, DX, DY, DZ>(); }
    };

}
