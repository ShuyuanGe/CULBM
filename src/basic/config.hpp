#pragma once

#include <numeric>
#include <concepts>
#include "types.hpp"

using idx_t = i64;
using flag_t = u32;
using ddf_t = f32;
using real_t = f32;

#define USE_STATIC_CONFIG true

constexpr u32 NDIR = 27;
constexpr i32 DOM_NX = 512;
constexpr i32 DOM_NY = 256;
constexpr i32 DOM_NZ = 256;
constexpr real_t TAU = 0.8;

constexpr real_t INV_TAU = 1. / TAU;

enum struct DeviceModule
{
    NVIDIA_GeForce_RTX_4090_D, 
    L2_Cache_72MB
};

namespace block_based_config
{
    namespace detail
    {
        template<DeviceModule MODULE>
        struct BlockBasedParam
        {
            static constexpr Vec3<i32> getGridDim();
            static constexpr Vec3<i32> getBlockDim();
            static constexpr i32       getIter();
        };

        template<>
        struct BlockBasedParam<DeviceModule::NVIDIA_GeForce_RTX_4090_D>
        {
            static constexpr Vec3<i32>  getGridDim()        { return Vec3<i32>{ 2,  3, 19}; }
            static constexpr Vec3<i32>  getBlockDim()       { return Vec3<i32>{32, 16,  2}; }
            static constexpr i32        getIter()           { return 4; }                 
        };

        template<>
        struct BlockBasedParam<DeviceModule::L2_Cache_72MB>
        {
            static constexpr Vec3<i32>  getGridDim()        { return Vec3<i32>{ 2,  3, 38}; }
            static constexpr Vec3<i32>  getBlockDim()       { return Vec3<i32>{32, 16,  2}; }
            static constexpr i32        getIter()           { return 7; }
        };
    }

    constexpr DeviceModule MODULE = DeviceModule::L2_Cache_72MB;
    constexpr Vec3<i32> GRID_DIM = detail::BlockBasedParam<MODULE>::getGridDim();
    constexpr Vec3<i32> BLOCK_DIM = detail::BlockBasedParam<MODULE>::getBlockDim();
    constexpr Vec3<i32> BLOCKING_DIM = {GRID_DIM.x*BLOCK_DIM.x, GRID_DIM.y*BLOCK_DIM.y, GRID_DIM.z*BLOCK_DIM.z};
    constexpr i32 BLOCKING_ITER = detail::BlockBasedParam<MODULE>::getIter();
}

constexpr flag_t LOAD_DDF_BIT       = static_cast<flag_t>(1) << 0;
constexpr flag_t REV_LOAD_DDF_BIT   = static_cast<flag_t>(1) << 1;
constexpr flag_t EQU_DDF_BIT        = static_cast<flag_t>(1) << 2;
constexpr flag_t COLLIDE_BIT        = static_cast<flag_t>(1) << 3;
constexpr flag_t STORE_DDF_BIT      = static_cast<flag_t>(1) << 4;
constexpr flag_t CORRECT_BIT        = static_cast<flag_t>(1) << 5;

constexpr flag_t FLUID_FLAG         = LOAD_DDF_BIT | COLLIDE_BIT | STORE_DDF_BIT;
constexpr flag_t BOUNCE_BACK_FLAG   = REV_LOAD_DDF_BIT | STORE_DDF_BIT;