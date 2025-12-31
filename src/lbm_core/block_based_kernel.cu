#include "config.hpp"
#include "device_function.hpp"
#include <cooperative_groups.h>
#include "block_based_kernel.cuh"

using namespace cooperative_groups;

namespace culbm::lbm_core
{
    using namespace block_based_config;

    __global__ __launch_bounds__(BLOCK_DIM.x * BLOCK_DIM.y * BLOCK_DIM.z)
    void staticBlockBasedKernel(const BlockBasedKernelParam __grid_constant__ param)
    {
        const i32 locX = BLOCK_DIM.x * blockIdx.x + threadIdx.x;
        const i32 locY = BLOCK_DIM.y * blockIdx.y + threadIdx.y;
        const i32 locZ = BLOCK_DIM.z * blockIdx.z + threadIdx.z;
        const i32 locIdx = locX+BLOCKING_DIM.x*(locY+BLOCKING_DIM.y*locZ);
        const i32 glbX = param.xOff + locX;
        const i32 glbY = param.yOff + locY;
        const i32 glbZ = param.zOff + locZ;

#if USE_STATIC_CONFIG
        i32 ndx = (glbX==0) ? 0 : -1;
        i32 pdx = (glbX==DOM_NX-1) ? 0 : 1;
        i32 ndy = (glbY==0) ? 0 : -DOM_NX;
        i32 pdy = (glbY==DOM_NY-1) ? 0 : DOM_NX;
        i32 ndz = (glbZ==0) ? 0 : -DOM_NX*DOM_NY;
        i32 pdz = (glbZ==DOM_NZ-1) ? 0 : DOM_NX*DOM_NY;
        const i32 glbIdx = glbX + DOM_NX * (glbY + DOM_NY * glbZ);
        constexpr i32 nCell = DOM_NX * DOM_NY * DOM_NZ;
        constexpr real_t invTau = INV_TAU;
#else
        i32 ndx = (glbX==0) ? 0 : -1;
        i32 pdx = (glbX==param.domX-1) ? 0 : 1;
        i32 ndy = (glbY==0) ? 0 : -param.domX;
        i32 pdy = (glbY==param.domY-1) ? 0 : param.domX;
        i32 ndz = (glbZ==0) ? 0 : -param.domX * param.domY;
        i32 pdz = (glbZ==param.domZ-1) ? 0 : param.domX * param.domY;
        const i32 glbIdx = glbX + param.domX * (glbY + param.domY * glbZ);
        const i32 nCell = param.domX * param.domY * param.domZ;
        const real_t invTau = param.invTau;
#endif  // USE_STATIC_CONFIG
        real_t rhon, vxn, vyn, vzn;
        real_t fn[NDIR];
        const flag_t flagn = param.flagBuf[locIdx];
        
        if((flagn&LOAD_DDF_BIT)==LOAD_DDF_BIT)
        {
            pullLoad<NDIR>(
                false,
                glbIdx, ndx, pdx, ndy, pdy, ndz, pdz,
                nCell,
                std::begin(fn), 
                param.srcBuf
            );
        }

        if((flagn&REV_LOAD_DDF_BIT)==REV_LOAD_DDF_BIT)
        {
            pullLoad<NDIR>(
                true, 
                glbIdx, ndx, pdx, ndy, pdy, ndz, pdz,
                nCell,
                std::begin(fn), 
                param.srcBuf
            );            
        }

        if((flagn&EQU_DDF_BIT)!=0)
        {
            rhon    = param.rhoBuf[glbIdx];
            vxn     = param.vxBuf[glbIdx];
            vyn     = param.vyBuf[glbIdx];
            vzn     = param.vzBuf[glbIdx];
            bgk::calcEqu<NDIR>(rhon, vxn, vyn, vzn, std::begin(fn));
        }

        if((flagn&COLLIDE_BIT)!=0)
        {
            bgk::collision<NDIR>(
                invTau, rhon, vxn, vyn, vzn, std::begin(fn)
            );
        }

        if constexpr (BLOCKING_ITER > 1)
        {
            auto thisGrid = this_grid();
            ndx = (locX == 0) ? 0 : -1;
            pdx = (locX == BLOCKING_DIM.x-1) ? 0 : 1;
            ndy = (locY == 0) ? 0 : -BLOCKING_DIM.x;
            pdy = (locY == BLOCKING_DIM.y-1) ? 0 : BLOCKING_DIM.x;
            ndz = (locZ == 0) ? 0 : -BLOCKING_DIM.x * BLOCKING_DIM.y;
            pdz = (locZ == BLOCKING_DIM.z-1) ? 0 : BLOCKING_DIM.x * BLOCKING_DIM.y;
            
            #pragma unroll
            for(u32 iter=1 ; iter<BLOCKING_ITER ; ++iter)
            {
                if((flagn&STORE_DDF_BIT)!=0)
                {
                    store<NDIR>(
                        false,
                        locIdx, BLOCKING_DIM.x * BLOCKING_DIM.y * BLOCKING_DIM.z, 
                        std::cbegin(fn), 
                        (iter&1)!=0 ? param.blkBuf0 : param.blkBuf1
                    );
                }

                thisGrid.sync();

                if((flagn&LOAD_DDF_BIT)==LOAD_DDF_BIT)
                {
                    pullLoad<NDIR>(
                        false,
                        locIdx, ndx, pdx, ndy, pdy, ndz, pdz, BLOCKING_DIM.x * BLOCKING_DIM.y * BLOCKING_DIM.z, 
                        std::begin(fn),
                        (iter&1)!=0 ? param.blkBuf0 : param.blkBuf1
                    );
                }

                if((flagn&REV_LOAD_DDF_BIT)==REV_LOAD_DDF_BIT)
                {
                    pullLoad<NDIR>(
                        true, 
                        locIdx, ndx, pdx, ndy, pdy, ndz, pdz, BLOCKING_DIM.x * BLOCKING_DIM.y * BLOCKING_DIM.z, 
                        std::begin(fn),
                        (iter&1)!=0 ? param.blkBuf0 : param.blkBuf1
                    );
                }

                if((flagn&COLLIDE_BIT)!=0)
                {
                    bgk::collision<NDIR>(
                        invTau, rhon, vxn, vyn, vzn, std::begin(fn)
                    );
                }
            }
        }

        if((flagn&(CORRECT_BIT|STORE_DDF_BIT))==(CORRECT_BIT|STORE_DDF_BIT))
        {
            store<NDIR>(
                false, 
                glbIdx, 
                nCell,
                std::begin(fn), 
                param.dstBuf
            );
            param.rhoBuf[glbIdx] = rhon;
            param.vxBuf[glbIdx] = vxn;
            param.vyBuf[glbIdx] = vyn;
            param.vzBuf[glbIdx] = vzn;
        }
    }

    __global__ __launch_bounds__(BLOCK_DIM.x * BLOCK_DIM.y * BLOCK_DIM.z)
    void dynamicBlockBasedKernelFirstIter(const BlockBasedKernelParam __grid_constant__ param)
    {
        const i32 locX = BLOCK_DIM.x * blockIdx.x + threadIdx.x;
        const i32 locY = BLOCK_DIM.y * blockIdx.y + threadIdx.y;
        const i32 locZ = BLOCK_DIM.z * blockIdx.z + threadIdx.z;
        const i32 locIdx = locX + BLOCKING_DIM.x * (locY + BLOCKING_DIM.y * locZ);
        const i32 glbX = param.xOff + locX;
        const i32 glbY = param.yOff + locY;
        const i32 glbZ = param.zOff + locZ;

#if USE_STATIC_CONFIG
        i32 ndx = (glbX==0) ? 0 : -1;
        i32 pdx = (glbX==DOM_NX-1) ? 0 : 1;
        i32 ndy = (glbY==0) ? 0 : -DOM_NX;
        i32 pdy = (glbY==DOM_NY-1) ? 0 : DOM_NX;
        i32 ndz = (glbZ==0) ? 0 : -DOM_NX*DOM_NY;
        i32 pdz = (glbZ==DOM_NZ-1) ? 0 : DOM_NX*DOM_NY;
        const i32 glbIdx = glbX + DOM_NX * (glbY + DOM_NY * glbZ);
        constexpr i32 nCell = DOM_NX * DOM_NY * DOM_NZ;
        constexpr real_t invTau = INV_TAU;
#else
        i32 ndx = (glbX==0) ? 0 : -1;
        i32 pdx = (glbX==param.domX-1) ? 0 : 1;
        i32 ndy = (glbY==0) ? 0 : -param.domX;
        i32 pdy = (glbY==param.domY-1) ? 0 : param.domX;
        i32 ndz = (glbZ==0) ? 0 : -param.domX * param.domY;
        i32 pdz = (glbZ==param.domZ-1) ? 0 : param.domX * param.domY;
        const i32 glbIdx = glbX + param.domX * (glbY + param.domY * glbZ);
        const i32 nCell = param.domX * param.domY * param.domZ;
        const real_t invTau = param.invTau;
#endif // USE_STATIC_CONFIG
        real_t rhon, vxn, vyn, vzn;
        real_t fn[NDIR];
        const flag_t flagn = param.flagBuf[locIdx];

        if((flagn & LOAD_DDF_BIT)==LOAD_DDF_BIT)
        {
            pullLoad<NDIR>(
                false, glbIdx, ndx, pdx, ndy, pdy, ndz, pdz, nCell, std::begin(fn), param.srcBuf
            );
        }

        if((flagn & REV_LOAD_DDF_BIT)==REV_LOAD_DDF_BIT)
        {
            pullLoad<NDIR>(
                true, glbIdx, ndx, pdx, ndy, pdy, ndz, pdz, nCell, std::begin(fn), param.srcBuf
            );
        }

        if((flagn & EQU_DDF_BIT)==EQU_DDF_BIT)
        {
            bgk::collision<NDIR>(
                invTau, rhon, vxn, vyn, vzn, std::begin(fn)
            );
        }

        if((flagn & STORE_DDF_BIT)==STORE_DDF_BIT)
        {
            store<NDIR>(
                false, locIdx, BLOCKING_DIM.x * BLOCKING_DIM.y * BLOCKING_DIM.z, std::begin(fn), param.blkBuf0
            );
        }
    }

    __global__ __launch_bounds__(BLOCK_DIM.x * BLOCK_DIM.y * BLOCK_DIM.z)
    void dynamicBlockBasedKernelLastIter(const BlockBasedKernelParam __grid_constant__ param)
    {
        const i32 locX = BLOCK_DIM.x * blockIdx.x + threadIdx.x;
        const i32 locY = BLOCK_DIM.y * blockIdx.y + threadIdx.y;
        const i32 locZ = BLOCK_DIM.z * blockIdx.z + threadIdx.z;
        const i32 locIdx = locX + BLOCKING_DIM.x * (locY + BLOCKING_DIM.y * locZ);
        const i32 glbX = locX + param.xOff;
        const i32 glbY = locY + param.yOff;
        const i32 glbZ = locZ + param.zOff;
        const i32 ndx = (locX==0) ? 0 : -1;
        const i32 pdx = (locX==BLOCKING_DIM.x-1) ? 0 : 1;
        const i32 ndy = (locY==0) ? 0 : -BLOCKING_DIM.x;
        const i32 pdy = (locY==BLOCKING_DIM.y-1) ? 0 : BLOCKING_DIM.x;
        const i32 ndz = (locZ==0) ? 0 : -BLOCKING_DIM.x * BLOCKING_DIM.y;
        const i32 pdz = (locZ==BLOCKING_DIM.z-1) ? 0 : BLOCKING_DIM.x * BLOCKING_DIM.y;

#if USE_STATIC_CONFIG
        constexpr real_t invTau = INV_TAU;
        constexpr i32 nCell = DOM_NX * DOM_NY * DOM_NZ;
        const i32 glbIdx = glbX + DOM_NX * (glbY + DOM_NY * glbZ);
#else
        const real_t invTau = param.invTau;
        const i32 nCell = param.domX * param.domY * param.domZ;
        const i32 glbIdx = glbX + param.domX * (glbY + param.domY * glbZ);
#endif // USE_STATIC_CONFIG

        real_t rhon, vxn, vyn, vzn;
        real_t fn[NDIR];
        const flag_t flagn = param.flagBuf[locIdx];

        if((flagn & LOAD_DDF_BIT)==LOAD_DDF_BIT)
        {
            pullLoad<NDIR>(
                false, locIdx, ndx, pdx, ndy, pdy, ndz, pdz, BLOCKING_DIM.x * BLOCKING_DIM.y * BLOCKING_DIM.z, std::begin(fn), param.blkBuf0
            );
        }

        if((flagn & REV_LOAD_DDF_BIT)==REV_LOAD_DDF_BIT)
        {
            pullLoad<NDIR>(
                true, locIdx, ndx, pdx, ndy, pdy, ndz, pdz, BLOCKING_DIM.x * BLOCKING_DIM.y * BLOCKING_DIM.z, std::begin(fn), param.blkBuf0
            );
        }

        if((flagn & COLLIDE_BIT)==COLLIDE_BIT)
        {
            bgk::collision<NDIR>(
                invTau, rhon, vxn, vyn, vzn, std::begin(fn)
            );
        }

        if((flagn & (STORE_DDF_BIT | CORRECT_BIT))==(STORE_DDF_BIT | CORRECT_BIT))
        {
            store<NDIR>(
                false, glbIdx, nCell, std::begin(fn), param.dstBuf
            );
        }
    }

    __global__ __launch_bounds__(BLOCK_DIM.x * BLOCK_DIM.y * BLOCK_DIM.z)
    void dynamicBlockBasedKernelInnerIter(const BlockBasedKernelParam __grid_constant__ param)
    {

        const i32 locX = BLOCK_DIM.x * blockIdx.x + threadIdx.x;
        const i32 locY = BLOCK_DIM.y * blockIdx.y + threadIdx.y;
        const i32 locZ = BLOCK_DIM.z * blockIdx.z + threadIdx.z;
        const i32 locIdx = locX + BLOCKING_DIM.x * (locY + BLOCKING_DIM.y * locZ);
        
        const i32 ndx = (locX==0) ? 0 : -1;
        const i32 pdx = (locX==BLOCKING_DIM.x-1) ? 0 : 1;
        const i32 ndy = (locY==0) ? 0 : -BLOCKING_DIM.x;
        const i32 pdy = (locY==BLOCKING_DIM.y-1) ? 0 : BLOCKING_DIM.x;
        const i32 ndz = (locZ==0) ? 0 : -BLOCKING_DIM.x * BLOCKING_DIM.y;
        const i32 pdz = (locZ==BLOCKING_DIM.z-1) ? 0 : BLOCKING_DIM.x * BLOCKING_DIM.y;

#if USE_STATIC_CONFIG
        constexpr real_t invTau = INV_TAU;
#else
        const real_t invTau = param.invTau;
#endif  // USE_STATIC_CONFIG

        real_t rhon, vxn, vyn, vzn;
        real_t fn[NDIR];
        const flag_t flagn = param.flagBuf[locIdx];

        if((flagn & LOAD_DDF_BIT)==LOAD_DDF_BIT)
        {
            pullLoad<NDIR>(
                false, locIdx, ndx, pdx, ndy, pdy, ndz, pdz, BLOCKING_DIM.x * BLOCKING_DIM.y * BLOCKING_DIM.z, std::begin(fn), param.blkBuf0
            );
        }

        if((flagn & REV_LOAD_DDF_BIT)==REV_LOAD_DDF_BIT)
        {
            pullLoad<NDIR>(
                true, locIdx, ndx, pdx, ndy, pdy, ndz, pdz, BLOCKING_DIM.x * BLOCKING_DIM.y * BLOCKING_DIM.z, std::begin(fn), param.blkBuf0
            );
        }

        if((flagn & COLLIDE_BIT)==COLLIDE_BIT)
        {
            bgk::collision<NDIR>(
                invTau, rhon, vxn, vyn, vzn, std::begin(fn)
            );
        }

        if((flagn & (CORRECT_BIT | STORE_DDF_BIT))==(CORRECT_BIT | STORE_DDF_BIT))
        {
            store<NDIR>(
                false, locIdx, BLOCKING_DIM.x * BLOCKING_DIM.y * BLOCKING_DIM.z, std::begin(fn), param.dstBuf
            );
        }
    }

    __global__ __launch_bounds__(BLOCK_DIM.x * BLOCK_DIM.y * BLOCK_DIM.z)
    void dynamicBlockBasedKernelSingleIter(const BlockBasedKernelParam __grid_constant__ param)
    {
        const i32 locX = BLOCK_DIM.x * blockIdx.x + threadIdx.x;
        const i32 locY = BLOCK_DIM.y * blockIdx.y + threadIdx.y;
        const i32 locZ = BLOCK_DIM.z * blockIdx.z + threadIdx.z;
        const i32 locIdx = locX + BLOCKING_DIM.x * (locY + BLOCKING_DIM.y * locZ);
        const i32 glbX = param.xOff;
        const i32 glbY = param.yOff;
        const i32 glbZ = param.zOff;

#if USE_STATIC_CONFIG
        const i32 ndx = (glbX==0) ? 0 : -1;
        const i32 pdx = (glbX==DOM_NX-1) ? 0 : 1;
        const i32 ndy = (glbY==0) ? 0 : -DOM_NX;
        const i32 pdy = (glbY==DOM_NY-1) ? 0 : DOM_NX;
        const i32 ndz = (glbZ==0) ? 0 : -DOM_NX*DOM_NY;
        const i32 pdz = (glbZ==DOM_NZ-1) ? 0 : DOM_NX*DOM_NY;
        const i32 glbIdx = glbX + DOM_NX * (glbY + DOM_NY * glbZ);
        constexpr real_t invTau = INV_TAU;
        constexpr real_t nCell = DOM_NX * DOM_NY * DOM_NZ;
#else
        const i32 ndx = (glbX==0) ? 0 : -1;
        const i32 pdx = (glbX==param.domX-1) ? 0 : 1;
        const i32 ndy = (glbY==0) ? 0 : -param.domX;
        const i32 pdy = (glbY==param.domY-1) ? 0 : param.domX;
        const i32 ndz = (glbZ==0) ? 0 : -param.domX * param.domY;
        const i32 pdz = (glbZ==param.domZ-1) ? 0 : param.domX * param.domY;
        const i32 glbIdx = glbX + param.domX * (glbY + param.domY * glbZ);
        const real_t invTau = param.invTau;
        const real_t nCell = param.domX * param.domY * param.domZ;
#endif // USE_STATIC_CONFIG
        real_t rhon, vxn, vyn, vzn;
        real_t fn[NDIR];
        const flag_t flagn = param.flagBuf[locIdx];

        if((flagn & LOAD_DDF_BIT)==LOAD_DDF_BIT)
        {
            pullLoad<NDIR>(
                false, glbIdx, ndx, pdx, ndy, pdy, ndz, pdz, nCell, std::begin(fn), param.srcBuf
            );
        }

        if((flagn & REV_LOAD_DDF_BIT)==REV_LOAD_DDF_BIT)
        {
            pullLoad<NDIR>(
                true, glbIdx, ndx, pdx, ndy, pdy, ndz, pdz, nCell, std::begin(fn), param.srcBuf
            );
        }

        if((flagn & COLLIDE_BIT)==COLLIDE_BIT)
        {
            bgk::collision<NDIR>(
                invTau, rhon, vxn, vyn, vzn, std::begin(fn)
            );
        }

        if((flagn & (CORRECT_BIT | STORE_DDF_BIT))==(CORRECT_BIT | STORE_DDF_BIT))
        {
            store<NDIR>(
                false, glbIdx, nCell, std::begin(fn), param.dstBuf
            );
        }
    }
}