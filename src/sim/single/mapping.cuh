#pragma once

#include "kernel.cuh"

namespace culbm::sim::single::mapping
{
    struct Position
    {
        idx_t local_x, local_y, local_z, local_nx, local_ny, local_nz, local_index, local_size;
        idx_t global_x, global_y, global_z, global_nx, global_ny, global_nz, global_index, global_size;

        constexpr Position() = default;

        __device__ Position(idx_t origin_x, idx_t origin_y, idx_t origin_z, idx_t nx, idx_t ny, idx_t nz)
            : local_x(blockDim.x * blockIdx.x + threadIdx.x),
              local_y(blockDim.y * blockIdx.y + threadIdx.y),
              local_z(blockDim.z * blockIdx.z + threadIdx.z),
              local_nx(gridDim.x * blockDim.x),
              local_ny(gridDim.y * blockDim.y),
              local_nz(gridDim.z * blockDim.z),
              local_index((local_ny * local_z + local_y) * local_nx + local_x), local_size(local_nx * local_ny * local_nz),
              global_x(origin_x + local_x), global_y(origin_y + local_y), global_z(origin_z + local_z),
              global_nx(nx), global_ny(ny), global_nz(nz),
              global_index((global_ny * global_z + global_y) * global_nx + global_x), global_size(global_nx * global_ny * global_nz) {}
    };

    template <int delta>
    constexpr idx_t wrapped_offset(idx_t coord, idx_t size, idx_t stride)
    {
        static_assert(delta >= -1 && delta <= 1);
        if      constexpr (delta == -1) return coord == 0 ? (size - 1) * stride : -stride;
        else if constexpr (delta ==  1) return coord == size - 1 ? (1 - size) * stride : stride;
        else if constexpr (delta ==  0) return 0;
    }

    template <int delta>
    constexpr idx_t clamped_offset(idx_t coord, idx_t size, idx_t stride)
    {
        static_assert(delta >= -1 && delta <= 1);
        if      constexpr (delta == -1) return coord == 0 ? 0 : -stride;
        else if constexpr (delta ==  1) return coord == size - 1 ? 0 : stride;
        else if constexpr (delta ==  0) return 0;
    }

    enum class ViewSpace { global, tile };

    struct GlobalView
    {
        static constexpr ViewSpace space = ViewSpace::global;
        ddf_t* src;
        ddf_t* dst;
        Position pos;

        static constexpr GlobalView make(const auto& param, const Position& pos)
        {
            return {param.src_ddf, param.dst_ddf, pos};
        }

        template <int DX, int DY, int DZ>
        constexpr idx_t cell_offset() const
        {
            return pos.global_index
                + wrapped_offset<DX>(pos.global_x, pos.global_nx, 1)
                + wrapped_offset<DY>(pos.global_y, pos.global_ny, pos.global_nx)
                + wrapped_offset<DZ>(pos.global_z, pos.global_nz, pos.global_nx * pos.global_ny);
        }

        template <int dir, int DX, int DY, int DZ>
        constexpr const ddf_t* read_address() const
        {
            return src + static_cast<ddf_offset_t>(dir) * pos.global_size + cell_offset<DX, DY, DZ>();
        }

        template <int dir, int DX, int DY, int DZ>
        constexpr ddf_t* write_address() const
        {
            return dst + static_cast<ddf_offset_t>(dir) * pos.global_size + cell_offset<DX, DY, DZ>();
        }
    };

    struct RingView
    {
        static constexpr ViewSpace space = ViewSpace::global;
        GlobalView global;
        idx_t ring_nz, src_zoffset, dst_zoffset;

        static constexpr RingView make(const BlockingParam& param, const Position& pos)
        {
            return {{param.src_ddf, param.dst_ddf, pos}, param.ring_nz, param.src_zoffset, param.dst_zoffset};
        }

        template <int dir, int DX, int DY, int DZ>
        constexpr ddf_offset_t offset(idx_t z_offset) const
        {
            const auto nx = global.pos.global_nx;
            const idx_t nxy = nx * global.pos.global_ny;
            const idx_t xy = global.pos.global_y * nx + global.pos.global_x
                + clamped_offset<DX>(global.pos.global_x, nx, 1)
                + clamped_offset<DY>(global.pos.global_y, global.pos.global_ny, nx);
            const idx_t z = global.pos.global_z + clamped_offset<DZ>(global.pos.global_z, global.pos.global_nz, 1);
            idx_t ring_z = z + z_offset;
            if (ring_z >= ring_nz) ring_z -= ring_nz;
            return static_cast<ddf_offset_t>(dir) * nxy * ring_nz
                + static_cast<ddf_offset_t>(ring_z) * nxy + xy;
        }

        template <int dir, int DX, int DY, int DZ>
        constexpr const ddf_t* read_address() const
        {
            return global.src + offset<dir, DX, DY, DZ>(src_zoffset);
        }

        template <int dir, int DX, int DY, int DZ>
        constexpr ddf_t* write_address() const
        {
            return global.src + offset<dir, DX, DY, DZ>(dst_zoffset);
        }

    };

    struct TileOffsets
    {
        const idx_t pdx, pdy, pdz;

        constexpr TileOffsets(const Position& pos)
            : pdx(wrapped_offset<1>(pos.local_x, pos.local_nx, 1)),
              pdy(wrapped_offset<1>(pos.local_y, pos.local_ny, pos.local_nx)),
              pdz(wrapped_offset<1>(pos.local_z, pos.local_nz, pos.local_nx * pos.local_ny)) {}
    };

    struct TilePosition : Position
    {
        // Precalculate offsets once for load and store.
        TileOffsets offsets;

        constexpr explicit TilePosition(const Position& pos) : Position(pos), offsets(pos) {}
    };

    __device__ inline Position make_position(const PlainParam&)
    {
        return Position(0, 0, 0, gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z);
    }

    __device__ inline TilePosition make_position(const BlockingParam& param)
    {
        return TilePosition(Position(param.origin_x, param.origin_y, param.origin_z, param.global_nx, param.global_ny, param.global_nz));
    }

    struct TileView
    {
        static constexpr ViewSpace space = ViewSpace::tile;
        ddf_t* data;
        TilePosition pos;

        static constexpr TileView make(const BlockingParam& param, const TilePosition& pos)
        {
            return {param.tile_ddf, pos};
        }

        template <int dir, int DX, int DY, int DZ>
        constexpr ddf_t* address() const
        {
            const idx_t offset = dir * pos.local_size + pos.local_index
                + DX * pos.offsets.pdx + DY * pos.offsets.pdy + DZ * pos.offsets.pdz;
            return data + offset;
        }

        template <int dir, int DX, int DY, int DZ>
        constexpr const ddf_t* read_address() const { return address<dir, DX, DY, DZ>(); }

        template <int dir, int DX, int DY, int DZ>
        constexpr ddf_t* write_address() const { return address<dir, DX, DY, DZ>(); }
    };

}
