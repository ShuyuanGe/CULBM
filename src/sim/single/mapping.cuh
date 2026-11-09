#pragma once

#include "kernel.cuh"

namespace culbm::sim::single::mapping
{
    struct Position
    {
        idx_t localX, localY, localZ, localNx, localNy, localNz, localIndex, localSize;
        idx_t globalX, globalY, globalZ, globalNx, globalNy, globalNz, globalIndex, globalSize;

        constexpr Position() = default;

        __device__ Position(idx_t originX, idx_t originY, idx_t originZ, idx_t nx, idx_t ny, idx_t nz)
            : localX(blockDim.x * blockIdx.x + threadIdx.x),
              localY(blockDim.y * blockIdx.y + threadIdx.y),
              localZ(blockDim.z * blockIdx.z + threadIdx.z),
              localNx(gridDim.x * blockDim.x),
              localNy(gridDim.y * blockDim.y),
              localNz(gridDim.z * blockDim.z),
              localIndex((localNy * localZ + localY) * localNx + localX), localSize(localNx * localNy * localNz),
              globalX(originX + localX), globalY(originY + localY), globalZ(originZ + localZ),
              globalNx(nx), globalNy(ny), globalNz(nz),
              globalIndex((globalNy * globalZ + globalY) * globalNx + globalX), globalSize(globalNx * globalNy * globalNz) {}
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
            return {param.srcDdf, param.dstDdf, pos};
        }

        template <int DX, int DY, int DZ>
        constexpr idx_t cellOffset() const
        {
            return pos.globalIndex
                + wrappedOffset<DX>(pos.globalX, pos.globalNx, 1)
                + wrappedOffset<DY>(pos.globalY, pos.globalNy, pos.globalNx)
                + wrappedOffset<DZ>(pos.globalZ, pos.globalNz, pos.globalNx * pos.globalNy);
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr const ddf_t* readAddress() const
        {
            return src + static_cast<ddf_offset_t>(Dir) * pos.globalSize + cellOffset<DX, DY, DZ>();
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr ddf_t* writeAddress() const
        {
            return dst + static_cast<ddf_offset_t>(Dir) * pos.globalSize + cellOffset<DX, DY, DZ>();
        }
    };

    struct RingView
    {
        static constexpr ViewSpace space = ViewSpace::Global;
        GlobalView global;
        idx_t nz, srcZOffset, dstZOffset;

        static constexpr RingView make(const BlockingParam& param, const Position& pos)
        {
            return {{param.srcDdf, param.dstDdf, pos}, param.ringNz, param.srcZOffset, param.dstZOffset};
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr ddf_offset_t offset(idx_t originZ) const
        {
            const auto nx = global.pos.globalNx;
            const idx_t nxy = nx * global.pos.globalNy;
            const idx_t xy = global.pos.globalY * nx + global.pos.globalX
                + clampedOffset<DX>(global.pos.globalX, nx, 1)
                + clampedOffset<DY>(global.pos.globalY, global.pos.globalNy, nx);
            const idx_t z = global.pos.globalZ + clampedOffset<DZ>(global.pos.globalZ, global.pos.globalNz, 1);
            idx_t ringZ = z + originZ;
            if (ringZ >= nz) ringZ -= nz;
            return static_cast<ddf_offset_t>(Dir) * nxy * nz
                + static_cast<ddf_offset_t>(ringZ) * nxy + xy;
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr const ddf_t* readAddress() const
        {
            return global.src + offset<Dir, DX, DY, DZ>(srcZOffset);
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr ddf_t* writeAddress() const
        {
            return global.src + offset<Dir, DX, DY, DZ>(dstZOffset);
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
        return TilePosition(Position(param.originX, param.originY, param.originZ, param.globalNx, param.globalNy, param.globalNz));
    }

    struct TileView
    {
        static constexpr ViewSpace space = ViewSpace::Tile;
        ddf_t* data;
        TilePosition pos;

        static constexpr TileView make(const BlockingParam& param, const TilePosition& pos)
        {
            return {param.tileDdf, pos};
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
