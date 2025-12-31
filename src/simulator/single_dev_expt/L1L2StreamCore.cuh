#pragma once

#include "local_config.hpp"
#include "velocity_set.hpp"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace culbm::core::L1L2StreamCore
{
    constexpr std::uint32_t BLK_X_FRONT_BIT = 1 << 0;
    constexpr std::uint32_t BLK_X_BACK_BIT  = 1 << 1;
    constexpr std::uint32_t BLK_Y_FRONT_BIT = 1 << 2;
    constexpr std::uint32_t BLK_Y_BACK_BIT  = 1 << 3;
    constexpr std::uint32_t BLK_Z_FRONT_BIT = 1 << 4;
    constexpr std::uint32_t BLK_Z_BACK_BIT  = 1 << 5;

    __forceinline__ __device__ std::uint32_t getBlockFlag3D()
    {
        const std::uint32_t xbits = (threadIdx.x==0) ? BLK_X_FRONT_BIT : (threadIdx.x==(blockDim.x-1)) ? BLK_X_BACK_BIT : 0;
        const std::uint32_t ybits = (threadIdx.y==0) ? BLK_Y_FRONT_BIT : (threadIdx.y==(blockDim.y-1)) ? BLK_Y_BACK_BIT : 0;
        const std::uint32_t zbits = (threadIdx.z==0) ? BLK_Z_FRONT_BIT : (threadIdx.z==(blockDim.z-1)) ? BLK_Z_BACK_BIT : 0;
        return xbits | ybits | zbits;
    }

    template<int DX, int DY, int DZ>
    consteval std::uint32_t getBlockMask3D()
    {
        static_assert(-1<=DX and DX<=1);
        static_assert(-1<=DY and DY<=1);
        static_assert(-1<=DZ and DZ<=1);

        const std::uint32_t xmask = (DX==-1) ? BLK_X_FRONT_BIT : (DX==1) ? BLK_X_BACK_BIT : 0;
        const std::uint32_t ymask = (DY==-1) ? BLK_Y_FRONT_BIT : (DY==1) ? BLK_Y_BACK_BIT : 0;
        const std::uint32_t zmask = (DZ==-1) ? BLK_Z_FRONT_BIT : (DZ==1) ? BLK_Z_BACK_BIT : 0;

        return xmask | ymask | zmask;
    }    

    template<std::int32_t NDIR>
    struct StreamCore3D
    {
        using VelSet = culbm::basic::detail::VelSet3D<NDIR>;

        static __device__ __forceinline__ void stream(real_t* fn, real_t* blkDDFBuf, real_t* swapDDFBuf)
        {
            const std::uint32_t blkFlag = getBlockFlag3D();

            auto thisGrid = cg::this_grid();
            auto thisWarp = cg::tiled_partition<32>(cg::this_thread_block());

            //data exchange by shared memory
            if constexpr (NDIR==27)
            {
                const std::int32_t blkIdx   = (blockDim.y*threadIdx.z+threadIdx.y)*blockDim.x+threadIdx.x;
                const std::int32_t blkN     = blockDim.z * blockDim.y * blockDim.x;
                //store f0 (x:-,y:-,z:-)
                blkDDFBuf[ 0*blkN+blkIdx] = fn[ 0];
                //store f1 (x:0,y:-,z:-)
                blkDDFBuf[ 1*blkN+blkIdx] = fn[ 1];
                //store f2 (x:+,y:-,z:-)
                blkDDFBuf[ 2*blkN+blkIdx] = fn[ 2];
                //store f3 (x:-,y:0,z:-)
                blkDDFBuf[ 3*blkN+blkIdx] = fn[ 3];
                //store f4 (x:0,y:0,z:-)
                blkDDFBuf[ 4*blkN+blkIdx] = fn[ 4];
                //store f5 (x:+,y:0,z:-)
                blkDDFBuf[ 5*blkN+blkIdx] = fn[ 5];
                //store f6 (x:-,y:+,z:-)
                blkDDFBuf[ 6*blkN+blkIdx] = fn[ 6];
                //store f7 (x:0,y:+,z:-)
                blkDDFBuf[ 7*blkN+blkIdx] = fn[ 7];
                //store f8 (x:+,y:+,z:-)
                blkDDFBuf[ 8*blkN+blkIdx] = fn[ 8];
                //store f9 (x:-,y:-,z:0)
                blkDDFBuf[ 9*blkN+blkIdx] = fn[ 9];
                //store f10(x:0,y:-,z:0)
                blkDDFBuf[10*blkN+blkIdx] = fn[10];
                //store f11(x:+,y:-,z:0)
                blkDDFBuf[11*blkN+blkIdx] = fn[11];
                //store f12(x:-,y:0,z:0) use warp shfl
                //f13(x:0,y:0,z:0) don't need stream
                //store f14(x:+,y:0,z:0) use warp shfl
                //store f15(x:-,y:+,z:0)
                blkDDFBuf[12*blkN+blkIdx] = fn[15];
                //store f16(x:0,y:+,z:0)
                blkDDFBuf[13*blkN+blkIdx] = fn[16];
                //store f17(x:+,y:+,z:0)
                blkDDFBuf[14*blkN+blkIdx] = fn[17];
                //store f18(x:-,y:-,z:+)
                blkDDFBuf[15*blkN+blkIdx] = fn[18];
                //store f19(x:0,y:-,z:+)
                blkDDFBuf[16*blkN+blkIdx] = fn[19];
                //store f20(x:+,y:-,z:+)
                blkDDFBuf[17*blkN+blkIdx] = fn[20];
                //store f21(x:-,y:0,z:+)
                blkDDFBuf[18*blkN+blkIdx] = fn[21];
                //store f22(x:0,y:0,z:+)
                blkDDFBuf[19*blkN+blkIdx] = fn[22];
                //store f23(x:+,y:0,z:+)
                blkDDFBuf[20*blkN+blkIdx] = fn[23];
                //store f24(x:-,y:+,z:+)
                blkDDFBuf[21*blkN+blkIdx] = fn[24];
                //store f25(x:0,y:+,z:+)
                blkDDFBuf[22*blkN+blkIdx] = fn[25];
                //store f26(x:+,y:+,z:+)
                blkDDFBuf[23*blkN+blkIdx] = fn[26];
            }

            //data exchange by l2 cache
            if constexpr (NDIR==27)
            {
                const std::int32_t gridIdxX = blockDim.x * blockIdx.x + threadIdx.x;
                const std::int32_t gridIdxY = blockDim.y * blockIdx.y + threadIdx.y;
                const std::int32_t gridIdxZ = blockDim.z * blockIdx.z + threadIdx.z;
                const std::int32_t gridNX   = gridDim.x * blockDim.x;
                const std::int32_t gridNY   = gridDim.y * blockDim.y;
                const std::int32_t gridNZ   = gridDim.z * blockDim.z;
                const std::int32_t plnZYIdx = gridNY*gridIdxZ+gridIdxY;
                const std::int32_t plnZXIdx = gridNX*gridIdxZ+gridIdxX;
                const std::int32_t plnYXIdx = gridNX*gridIdxY+gridIdxX;
                const std::int32_t plnZYN   = gridNZ*gridNY;
                const std::int32_t plnZXN   = gridNZ*gridNX;
                const std::int32_t plnYXN   = gridNY*gridNX;
                const std::int32_t xBufN    = (gridDim.x+1)*plnZYN;
                const std::int32_t yBufN    = (gridDim.y+1)*plnZXN;
                const std::int32_t zBufN    = (gridDim.z+1)*plnYXN;
                //f0 (x:-,y:-,z:-)
                if((blkFlag & getBlockMask3D<-1,-1,-1>())!=0)
                {
                    const std::int32_t xoff = 0*xBufN+ 0*yBufN+ 0*zBufN+blockIdx.x*plnZYN+plnZYIdx;
                    const std::int32_t yoff = 1*xBufN+ 0*yBufN+ 0*zBufN+blockIdx.y*plnZXN+plnZXIdx;
                    const std::int32_t zoff = 1*xBufN+ 1*yBufN+ 0*zBufN+blockIdx.z*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : (threadIdx.y==0) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_FRONT_BIT)!=0) ? yoff : zoff;
                    swapDDFBuf[off] = fn[ 0];
                }
                //f1 (x:0,y:-,z:-)
                if((blkFlag & getBlockMask3D< 0,-1,-1>())!=0)
                {
                    const std::int32_t yoff = 1*xBufN+ 1*yBufN+ 1*zBufN+blockIdx.y*plnZXN+plnZXIdx;
                    const std::int32_t zoff = 1*xBufN+ 2*yBufN+ 1*zBufN+blockIdx.z*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.y==0) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_Y_FRONT_BIT)!=0) ? yoff : zoff;
                    swapDDFBuf[off] = fn[ 1];
                }
                //f2 (x:+,y:-,z:-)
                if((blkFlag & getBlockMask3D< 1,-1,-1>())!=0)
                {
                    const std::int32_t xoff = 1*xBufN+ 2*yBufN+ 2*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx;
                    const std::int32_t yoff = 2*xBufN+ 2*yBufN+ 2*zBufN+blockIdx.y*plnZXN+plnZXIdx;
                    const std::int32_t zoff = 2*xBufN+ 3*yBufN+ 2*zBufN+blockIdx.z*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : (threadIdx.y==0) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_FRONT_BIT)!=0) ? yoff : zoff;
                    swapDDFBuf[off] = fn[ 2];
                }
                //f3 (x:-,y:0,z:-)
                if((blkFlag & getBlockMask3D<-1, 0,-1>())!=0)
                {
                    const std::int32_t xoff = 2*xBufN+ 3*yBufN+ 3*zBufN+blockIdx.x*plnZYN+plnZYIdx;
                    const std::int32_t zoff = 3*xBufN+ 3*yBufN+ 3*zBufN+blockIdx.z*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : zoff;
                    swapDDFBuf[off] = fn[ 3];
                }
                //f4 (x:0,y:0,z:-)
                if((blkFlag & getBlockMask3D< 0, 0,-1>())!=0)
                {
                    const std::int32_t zoff = 3*xBufN+ 3*yBufN+ 4*zBufN+blockIdx.z*plnYXN+plnYXIdx;
                    swapDDFBuf[zoff] = fn[ 4];
                }
                //f5 (x:+,y:0,z:-)
                if((blkFlag & getBlockMask3D< 1, 0,-1>())!=0)
                {
                    const std::int32_t xoff = 3*xBufN+ 3*yBufN+ 5*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx;
                    const std::int32_t zoff = 4*xBufN+ 3*yBufN+ 5*zBufN+blockIdx.z*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : zoff;
                    swapDDFBuf[off] = fn[ 5];
                }
                //f6 (x:-,y:+,z:-)
                if((blkFlag & getBlockMask3D<-1, 1,-1>())!=0)
                {
                    const std::int32_t xoff = 4*xBufN+ 3*yBufN+ 6*zBufN+blockIdx.x*plnZYN+plnZYIdx;
                    const std::int32_t yoff = 5*xBufN+ 3*yBufN+ 6*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx;
                    const std::int32_t zoff = 5*xBufN+ 4*yBufN+ 6*zBufN+blockIdx.z*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : (threadIdx.y==(blockDim.y-1)) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_BACK_BIT)!=0) ? yoff : zoff;
                    swapDDFBuf[off] = fn[ 6];
                }
                //f7 (x:0,y:+,z:-)
                if((blkFlag & getBlockMask3D< 0, 1,-1>())!=0)
                {
                    const std::int32_t yoff = 5*xBufN+ 4*yBufN+ 7*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx;
                    const std::int32_t zoff = 5*xBufN+ 5*yBufN+ 7*zBufN+blockIdx.z*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.y==(blockDim.y-1)) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_Y_BACK_BIT)!=0) ? yoff : zoff;
                    swapDDFBuf[off] = fn[ 7];
                }
                //f8 (x:+,y:+,z:-)
                if((blkFlag & getBlockMask3D< 1, 1,-1>())!=0)
                {
                    const std::int32_t xoff = 5*xBufN+ 5*yBufN+ 8*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx;
                    const std::int32_t yoff = 6*xBufN+ 5*yBufN+ 8*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx;
                    const std::int32_t zoff = 6*xBufN+ 6*yBufN+ 8*zBufN+blockIdx.z*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : (threadIdx.y==(blockDim.y-1)) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_BACK_BIT)!=0) ? yoff : zoff;
                    swapDDFBuf[off] = fn[ 8];
                }
                //f9 (x:-,y:-,z:0)
                if((blkFlag & getBlockMask3D<-1,-1, 0>())!=0)
                {
                    const std::int32_t xoff = 6*xBufN+ 6*yBufN+ 9*zBufN+blockIdx.x*plnZYN+plnZYIdx;
                    const std::int32_t yoff = 7*xBufN+ 6*yBufN+ 9*zBufN+blockIdx.y*plnZXN+plnZXIdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : yoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : yoff;
                    swapDDFBuf[off] = fn[ 9];
                }
                //f10(x:0,y:-,z:0)
                if((blkFlag & getBlockMask3D< 0,-1, 0>())!=0)
                {
                    const std::int32_t yoff = 7*xBufN+ 7*yBufN+ 9*zBufN+blockIdx.y*plnZXN+plnZXIdx;
                    swapDDFBuf[yoff] = fn[10];
                }
                //f11(x:+,y:-,z:0)
                if((blkFlag & getBlockMask3D< 1,-1, 0>())!=0)
                {
                    const std::int32_t xoff = 7*xBufN+ 8*yBufN+ 9*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx;
                    const std::int32_t yoff = 8*xBufN+ 8*yBufN+ 9*zBufN+blockIdx.y*plnZXN+plnZXIdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : yoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : yoff;
                    swapDDFBuf[off] = fn[11];
                }
                //f12(x:-,y:0,z:0)
                if((blkFlag & getBlockMask3D<-1, 0, 0>())!=0)
                {
                    const std::int32_t xoff = 8*xBufN+ 9*yBufN+ 9*zBufN+blockIdx.x*plnZYN+plnZYIdx;
                    swapDDFBuf[xoff] = fn[12];
                }
                //f14(x:+,y:0,z:0)
                if((blkFlag & getBlockMask3D< 1, 0, 0>())!=0)
                {
                    const std::int32_t xoff = 9*xBufN+ 9*yBufN+ 9*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx;
                    swapDDFBuf[xoff] = fn[14];
                }
                //f15(x:-,y:+,z:0)
                if((blkFlag & getBlockMask3D<-1, 1, 0>())!=0)
                {
                    const std::int32_t xoff =10*xBufN+ 9*yBufN+ 9*zBufN+blockIdx.x*plnZYN+plnZYIdx;
                    const std::int32_t yoff =11*xBufN+ 9*yBufN+ 9*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : yoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : yoff;
                    swapDDFBuf[off] = fn[15];
                }
                //f16(x:0,y:+,z:0)
                if((blkFlag & getBlockMask3D< 0, 1, 0>())!=0)
                {
                    const std::int32_t yoff =11*xBufN+10*yBufN+ 9*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx;
                    swapDDFBuf[yoff] = fn[16];
                }
                //f17(x:+,y:+,z:0)
                if((blkFlag & getBlockMask3D< 1, 1, 0>())!=0)
                {
                    const std::int32_t xoff =11*xBufN+11*yBufN+ 9*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx;
                    const std::int32_t yoff =12*xBufN+11*yBufN+ 9*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : yoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : yoff;
                    swapDDFBuf[off] = fn[17];
                }
                //f18(x:-,y:-,z:+)
                if((blkFlag & getBlockMask3D<-1,-1, 1>())!=0)
                {
                    const std::int32_t xoff =12*xBufN+12*yBufN+ 9*zBufN+blockIdx.x*plnZYN+plnZYIdx;
                    const std::int32_t yoff =13*xBufN+12*yBufN+ 9*zBufN+blockIdx.y*plnZXN+plnZXIdx;
                    const std::int32_t zoff =13*xBufN+13*yBufN+ 9*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : (threadIdx.y==0) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_FRONT_BIT)!=0) ? yoff : zoff;
                    swapDDFBuf[off] = fn[18];
                }
                //f19(x:0,y:-,z:+)
                if((blkFlag & getBlockMask3D< 0,-1, 1>())!=0)
                {
                    const std::int32_t yoff =13*xBufN+13*yBufN+10*zBufN+blockIdx.y*plnZXN+plnZXIdx;
                    const std::int32_t zoff =13*xBufN+14*yBufN+10*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.y==0) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_Y_FRONT_BIT)!=0) ? yoff : zoff;
                    swapDDFBuf[off] = fn[19];
                }
                //f20(x:+,y:-,z:+)
                if((blkFlag & getBlockMask3D< 1,-1, 1>())!=0)
                {
                    const std::int32_t xoff =13*xBufN+14*yBufN+11*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx;
                    const std::int32_t yoff =14*xBufN+14*yBufN+11*zBufN+blockIdx.y*plnZXN+plnZXIdx;
                    const std::int32_t zoff =14*xBufN+15*yBufN+11*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : (threadIdx.y==0) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_FRONT_BIT)!=0) ? yoff : zoff;
                    swapDDFBuf[off] = fn[20];
                }
                //f21(x:-,y:0,z:+)
                if((blkFlag & getBlockMask3D<-1, 0, 1>())!=0)
                {
                    const std::int32_t xoff =14*xBufN+15*yBufN+12*zBufN+blockIdx.x*plnZYN+plnZYIdx;
                    const std::int32_t zoff =15*xBufN+15*yBufN+12*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : zoff;
                    swapDDFBuf[off] = fn[21];
                }
                //f22(x:0,y:0,z:+)
                if((blkFlag & getBlockMask3D< 0, 0, 1>())!=0)
                {
                    const std::int32_t zoff =15*xBufN+15*yBufN+13*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx;
                    swapDDFBuf[zoff] = fn[22];
                }
                //f23(x:+,y:0,z:+)
                if((blkFlag & getBlockMask3D< 1, 0, 1>())!=0)
                {
                    const std::int32_t xoff =15*xBufN+15*yBufN+14*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx;
                    const std::int32_t zoff =16*xBufN+15*yBufN+14*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : zoff;
                    swapDDFBuf[off] = fn[23];
                }
                //f24(x:-,y:+,z:+)
                if((blkFlag & getBlockMask3D<-1, 1, 1>())!=0)
                {
                    const std::int32_t xoff =16*xBufN+15*yBufN+15*zBufN+blockIdx.x*plnZYN+plnZYIdx;
                    const std::int32_t yoff =17*xBufN+15*yBufN+15*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx;
                    const std::int32_t zoff =17*xBufN+16*yBufN+15*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : (threadIdx.y==(blockDim.y-1)) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_BACK_BIT)!=0) ? yoff : zoff;
                    swapDDFBuf[off] = fn[24];
                }
                //f25(x:0,y:+,z:+)
                if((blkFlag & getBlockMask3D< 0, 1, 1>())!=0)
                {
                    const std::int32_t yoff =17*xBufN+16*yBufN+16*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx;
                    const std::int32_t zoff =17*xBufN+17*yBufN+16*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.y==(blockDim.y-1)) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_Y_BACK_BIT)!=0) ? yoff : zoff;
                    swapDDFBuf[off] = fn[25];
                }
                //f26(x:+,y:+,z:+)
                if((blkFlag & getBlockMask3D< 1, 1, 1>())!=0)
                {
                    const std::int32_t xoff =17*xBufN+17*yBufN+17*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx;
                    const std::int32_t yoff =18*xBufN+17*yBufN+17*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx;
                    const std::int32_t zoff =18*xBufN+18*yBufN+17*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : (threadIdx.y==(blockDim.y-1)) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_BACK_BIT)!=0) ? yoff : zoff;
                    swapDDFBuf[off] = fn[26];
                }
            }

            thisGrid.sync();

            //data exchange by shared memory
            if constexpr (NDIR==27)
            {
                const std::int32_t blkIdx   = (blockDim.y*threadIdx.z+threadIdx.y)*blockDim.x+threadIdx.x;
                const std::int32_t blkN     = blockDim.z * blockDim.y * blockDim.x;
                //const std::int32_t blkNdx = (threadIdx.x==0) ? static_cast<std::int32_t>(blockDim.x-1) : -1;
                //const std::int32_t blkPdx = (threadIdx.x==(blockDim.x-1)) ? static_cast<std::int32_t>(1-blockDim.x) : 1;
                // const std::int32_t blkNdy = (threadIdx.y==0) ? 0 : -static_cast<std::int32_t>(blockDim.x);
                // const std::int32_t blkPdy = (threadIdx.y==(blockDim.y-1)) ? 0 : static_cast<std::int32_t>(blockDim.x);
                // const std::int32_t blkNdz = (threadIdx.z==0) ? 0 : -static_cast<std::int32_t>(blockDim.y*blockDim.x);
                // const std::int32_t blkPdz = (threadIdx.z==(blockDim.z-1)) ? 0 : static_cast<std::int32_t>(blockDim.y*blockDim.x);

                const std::int32_t blkNdx = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? static_cast<std::int32_t>(blockDim.x-1) : -1;
                const std::int32_t blkPdx = ((blkFlag & BLK_X_BACK_BIT)!=0) ? static_cast<std::int32_t>(1-blockDim.x) : 1;
                const std::int32_t blkNdy = ((blkFlag & BLK_Y_FRONT_BIT)!=0) ?  0 : -static_cast<std::int32_t>(blockDim.x);
                const std::int32_t blkPdy = ((blkFlag & BLK_Y_BACK_BIT)!=0) ? 0 : static_cast<std::int32_t>(blockDim.x);
                const std::int32_t blkNdz = ((blkFlag & BLK_Z_FRONT_BIT)!=0) ? 0 : -static_cast<std::int32_t>(blockDim.y*blockDim.x);
                const std::int32_t blkPdz = ((blkFlag & BLK_Z_BACK_BIT)!=0) ? 0 : static_cast<std::int32_t>(blockDim.y*blockDim.x);

                //load f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
                fn[ 0] = blkDDFBuf[ 0*blkN+blkIdx+blkPdx+blkPdy+blkPdz];
                //load f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
                fn[ 1] = blkDDFBuf[ 1*blkN+blkIdx       +blkPdy+blkPdz];
                //load f2 (x:+,y:-,z:-) from neighbor (x:-,y:+,z:+)
                fn[ 2] = blkDDFBuf[ 2*blkN+blkIdx+blkNdx+blkPdy+blkPdz];
                //load f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
                fn[ 3] = blkDDFBuf[ 3*blkN+blkIdx+blkPdx       +blkPdz];
                //load f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
                fn[ 4] = blkDDFBuf[ 4*blkN+blkIdx              +blkPdz];
                //load f5 (x:+,y:0,z:-) from neighbor (x:-,y:0,z:+)
                fn[ 5] = blkDDFBuf[ 5*blkN+blkIdx+blkNdx       +blkPdz];
                //load f6 (x:-,y:+,z:-) from neighbor (x:+,y:-,z:+)
                fn[ 6] = blkDDFBuf[ 6*blkN+blkIdx+blkPdx+blkNdy+blkPdz];
                //load f7 (x:0,y:+,z:-) from neighbor (x:0,y:-,z:+)
                fn[ 7] = blkDDFBuf[ 7*blkN+blkIdx       +blkNdy+blkPdz];
                //load f8 (x:+,y:+,z:-) from neighbor (x:-,y:-,z:+)
                fn[ 8] = blkDDFBuf[ 8*blkN+blkIdx+blkNdx+blkNdy+blkPdz];

                //load f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
                fn[ 9] = blkDDFBuf[ 9*blkN+blkIdx+blkPdx+blkPdy       ];
                //load f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
                fn[10] = blkDDFBuf[10*blkN+blkIdx       +blkPdy       ];
                //load f11(x:+,y:-,z:0) from neighbor (x:-,y:+,z:0)
                fn[11] = blkDDFBuf[11*blkN+blkIdx+blkNdx+blkPdy       ];
                //load f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
                fn[12] = thisWarp.shfl_down(fn[12], 1);
                //load f14(x:+,y:0,z:0) from neighbor (x:-,y:0,z:0)
                fn[14] = thisWarp.shfl_up(fn[14], 1);
                //load f15(x:-,y:+,z:0) from neighbor (x:+,y:-,z:0)
                fn[15] = blkDDFBuf[12*blkN+blkIdx+blkPdx+blkNdy       ];
                //load f16(x:0,y:+,z:0) from neighbor (x:0,y:-,z:0)
                fn[16] = blkDDFBuf[13*blkN+blkIdx       +blkNdy       ];
                //load f17(x:+,y:+,z:0) from neighbor (x:-,y:-,z:0)
                fn[17] = blkDDFBuf[14*blkN+blkIdx+blkNdx+blkNdy       ];

                //load f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:-)
                fn[18] = blkDDFBuf[15*blkN+blkIdx+blkPdx+blkPdy+blkNdz];
                //load f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:-)
                fn[19] = blkDDFBuf[16*blkN+blkIdx       +blkPdy+blkNdz];
                //load f20(x:+,y:-,z:+) from neighbor (x:-,y:+,z:-)
                fn[20] = blkDDFBuf[17*blkN+blkIdx+blkNdx+blkPdy+blkNdz];
                //load f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:-)
                fn[21] = blkDDFBuf[18*blkN+blkIdx+blkPdx       +blkNdz];
                //load f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:-)
                fn[22] = blkDDFBuf[19*blkN+blkIdx              +blkNdz];
                //load f23(x:+,y:0,z:+) from neighbor (x:-,y:0,z:-)
                fn[23] = blkDDFBuf[20*blkN+blkIdx+blkNdx       +blkNdz];
                //load f24(x:-,y:+,z:+) from neighbor (x:+,y:-,z:-)
                fn[24] = blkDDFBuf[21*blkN+blkIdx+blkPdx+blkNdy+blkNdz];
                //load f25(x:0,y:+,z:+) from neighbor (x:0,y:-,z:-)
                fn[25] = blkDDFBuf[22*blkN+blkIdx       +blkNdy+blkNdz];
                //load f26(x:+,y:+,z:+) from neighbor (x:-,y:-,z:-)
                fn[26] = blkDDFBuf[23*blkN+blkIdx+blkNdx+blkNdy+blkNdz];
            }
            
            //data exchange by l2 cache
            if constexpr (NDIR==27)
            {
                const std::int32_t gridIdxX = blockDim.x * blockIdx.x + threadIdx.x;
                const std::int32_t gridIdxY = blockDim.y * blockIdx.y + threadIdx.y;
                const std::int32_t gridIdxZ = blockDim.z * blockIdx.z + threadIdx.z;
                const std::int32_t gridNX   = gridDim.x * blockDim.x;
                const std::int32_t gridNY   = gridDim.y * blockDim.y;
                const std::int32_t gridNZ   = gridDim.z * blockDim.z;

                const std::int32_t plnZYN   = gridNZ*gridNY;
                const std::int32_t plnZXN   = gridNZ*gridNX;
                const std::int32_t plnYXN   = gridNY*gridNX;
                const std::int32_t xBufN    = (gridDim.x+1)*plnZYN;
                const std::int32_t yBufN    = (gridDim.y+1)*plnZXN;
                const std::int32_t zBufN    = (gridDim.z+1)*plnYXN;

                const std::int32_t plnZYIdx = gridNY*gridIdxZ+gridIdxY;
                const std::int32_t plnZXIdx = gridNX*gridIdxZ+gridIdxX;
                const std::int32_t plnYXIdx = gridNX*gridIdxY+gridIdxX;

                const std::int32_t plnZYNdy = (gridIdxZ==0) ? 0 : -gridNY;
                const std::int32_t plnZYPdy = (gridIdxZ==(gridNZ-1)) ? 0 : gridNY;
                const std::int32_t plnZYNdx = (gridIdxY==0) ? 0 : -1;
                const std::int32_t plnZYPdx = (gridIdxY==(gridNY-1)) ? 0 : 1;

                const std::int32_t plnZXNdy = (gridIdxZ==0) ? 0 : -gridNX;
                const std::int32_t plnZXPdy = (gridIdxZ==(gridNZ-1)) ? 0 : gridNX;
                const std::int32_t plnZXNdx = (gridIdxX==0) ? 0 : -1;
                const std::int32_t plnZXPdx = (gridIdxX==(gridNX-1)) ? 0 : 1;

                const std::int32_t plnYXNdy = (gridIdxY==0) ? 0 : -gridNX;
                const std::int32_t plnYXPdy = (gridIdxY==(gridNY-1)) ? 0 : gridNX;
                const std::int32_t plnYXNdx = (gridIdxX==0) ? 0 : -1;
                const std::int32_t plnYXPdx = (gridIdxX==(gridNX-1)) ? 0 : 1;
                //load f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
                if((blkFlag & getBlockMask3D< 1, 1, 1>())!=0)
                {
                    const std::int32_t xoff = 0*xBufN+ 0*yBufN+ 0*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx+plnZYPdy+plnZYPdx;
                    const std::int32_t yoff = 1*xBufN+ 0*yBufN+ 0*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx+plnZXPdy+plnZXPdx;
                    const std::int32_t zoff = 1*xBufN+ 1*yBufN+ 0*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx+plnYXPdy+plnYXPdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : (threadIdx.y==(blockDim.y-1)) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_BACK_BIT)!=0) ? yoff : zoff;
                    fn[ 0] = swapDDFBuf[off];
                }
                //load f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
                if((blkFlag & getBlockMask3D< 0, 1, 1>())!=0)
                {
                    const std::int32_t yoff = 1*xBufN+ 1*yBufN+ 1*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx+plnZXPdy;
                    const std::int32_t zoff = 1*xBufN+ 2*yBufN+ 1*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx+plnYXPdy;
                    //const std::int32_t off = (threadIdx.y==(blockDim.y-1)) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_Y_BACK_BIT)!=0) ? yoff : zoff;
                    fn[ 1] = swapDDFBuf[off];
                }
                //load f2 (x:+,y:-,z:-) from neighbor (x:-,y:+,z:+)
                if((blkFlag & getBlockMask3D<-1,1,1>())!=0)
                {
                    const std::int32_t xoff = 1*xBufN+ 2*yBufN+ 2*zBufN+blockIdx.x*plnZYN+plnZYIdx+plnZYPdy+plnZYPdx;
                    const std::int32_t yoff = 2*xBufN+ 2*yBufN+ 2*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx+plnZXPdy+plnZXNdx;
                    const std::int32_t zoff = 2*xBufN+ 3*yBufN+ 2*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx+plnYXPdy+plnYXNdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : (threadIdx.y==(blockDim.y-1)) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_BACK_BIT)!=0) ? yoff : zoff;
                    fn[ 2] = swapDDFBuf[off];
                }
                //load f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
                if((blkFlag & getBlockMask3D< 1, 0, 1>())!=0)
                {
                    const std::int32_t xoff = 2*xBufN+ 3*yBufN+ 3*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx+plnZYPdy;
                    const std::int32_t zoff = 3*xBufN+ 3*yBufN+ 3*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx+plnYXPdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : zoff;
                    fn[ 3] = swapDDFBuf[off];
                }
                //load f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
                if((blkFlag & getBlockMask3D< 0, 0, 1>())!=0)
                {
                    const std::int32_t zoff = 3*xBufN+ 3*yBufN+ 4*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx;
                    fn[ 4] = swapDDFBuf[zoff];
                }
                //load f5 (x:+,y:0,z:-) from neighbor (x:-,y:0,z:+)
                if((blkFlag & getBlockMask3D<-1, 0, 1>())!=0)
                {
                    const std::int32_t xoff = 3*xBufN+ 3*yBufN+ 5*zBufN+blockIdx.x*plnZYN+plnZYIdx+plnZYPdy;
                    const std::int32_t zoff = 4*xBufN+ 3*yBufN+ 5*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx+plnYXNdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : zoff;
                    fn[ 5] = swapDDFBuf[off];
                }
                //load f6 (x:-,y:+,z:-) from neighbor (x:+,y:-,z:+)
                if((blkFlag & getBlockMask3D< 1,-1, 1>())!=0)
                {
                    const std::int32_t xoff = 4*xBufN+ 3*yBufN+ 6*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx+plnZYPdy+plnZYNdx;
                    const std::int32_t yoff = 5*xBufN+ 3*yBufN+ 6*zBufN+blockIdx.y*plnZXN+plnZXIdx+plnZXPdy+plnZXPdx;
                    const std::int32_t zoff = 5*xBufN+ 4*yBufN+ 6*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx+plnYXNdy+plnYXPdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : (threadIdx.y==0) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_FRONT_BIT)!=0) ? yoff : zoff;
                    fn[ 6] = swapDDFBuf[off];
                }
                //load f7 (x:0,y:+,z:-) from neighbor (x:0,y:-,z:+)
                if((blkFlag & getBlockMask3D< 0,-1, 1>())!=0)
                {
                    const std::int32_t yoff = 5*xBufN+ 4*yBufN+ 7*zBufN+blockIdx.y*plnZXN+plnZXIdx+plnZXPdy;
                    const std::int32_t zoff = 5*xBufN+ 5*yBufN+ 7*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx+plnYXNdy;
                    //const std::int32_t off = (threadIdx.y==0) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_Y_FRONT_BIT)!=0) ? yoff : zoff;
                    fn[ 7] = swapDDFBuf[off];
                }
                //load f8 (x:+,y:+,z:-) from neighbor (x:-,y:-,z:+)
                if((blkFlag & getBlockMask3D<-1,-1, 1>())!=0)
                {
                    const std::int32_t xoff = 5*xBufN+ 5*yBufN+ 8*zBufN+blockIdx.x*plnZYN+plnZYIdx+plnZYPdy+plnZYNdx;
                    const std::int32_t yoff = 6*xBufN+ 5*yBufN+ 8*zBufN+blockIdx.y*plnZXN+plnZXIdx+plnZXPdy+plnZXNdx;
                    const std::int32_t zoff = 6*xBufN+ 6*yBufN+ 8*zBufN+(blockIdx.z+1)*plnYXN+plnYXIdx+plnYXNdy+plnYXNdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : (threadIdx.y==0) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_FRONT_BIT)!=0) ? yoff : zoff;
                    fn[ 8] = swapDDFBuf[off];
                }
                //load f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
                if((blkFlag & getBlockMask3D< 1, 1, 0>())!=0)
                {
                    const std::int32_t xoff = 6*xBufN+ 6*yBufN+ 9*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx+plnZYPdx;
                    const std::int32_t yoff = 7*xBufN+ 6*yBufN+ 9*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx+plnZXPdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : yoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : yoff;
                    fn[ 9] = swapDDFBuf[off];
                }
                //load f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
                if((blkFlag & getBlockMask3D< 0, 1, 0>())!=0)
                {
                    const std::int32_t yoff = 7*xBufN+ 7*yBufN+ 9*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx;
                    fn[10] = swapDDFBuf[yoff];
                }
                //load f11(x:+,y:-,z:0) from neighbor (x:-,y:+,z:0)
                if((blkFlag & getBlockMask3D<-1, 1, 0>())!=0)
                {
                    const std::int32_t xoff = 7*xBufN+ 8*yBufN+ 9*zBufN+blockIdx.x*plnZYN+plnZYIdx+plnZYPdx;
                    const std::int32_t yoff = 8*xBufN+ 8*yBufN+ 9*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx+plnZXNdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : yoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : yoff;
                    fn[11] = swapDDFBuf[off];
                }
                //load f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
                if((blkFlag & getBlockMask3D< 1, 0, 0>())!=0)
                {
                    const std::int32_t xoff = 8*xBufN+ 9*yBufN+ 9*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx;
                    fn[12] = swapDDFBuf[xoff];
                }
                //load f14(x:+,y:0,z:0) from neighbor (x:-,y:0,z:0)
                if((blkFlag & getBlockMask3D<-1, 0, 0>())!=0)
                {
                    const std::int32_t xoff = 9*xBufN+ 9*yBufN+ 9*zBufN+blockIdx.x*plnZYN+plnZYIdx;
                    fn[14] = swapDDFBuf[xoff];
                }
                //load f15(x:-,y:+,z:0) from neighbor (x:+,y:-,z:0)
                if((blkFlag & getBlockMask3D< 1,-1,0>())!=0)
                {
                    const std::int32_t xoff =10*xBufN+ 9*yBufN+ 9*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx+plnZYNdx;
                    const std::int32_t yoff =11*xBufN+ 9*yBufN+ 9*zBufN+blockIdx.y*plnZXN+plnZXIdx+plnZXPdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : yoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : yoff;
                    fn[15] = swapDDFBuf[off];
                }
                //load f16(x:0,y:+,z:0) from neighbor (x:0,y:-,z:0)
                if((blkFlag & getBlockMask3D< 0,-1, 0>())!=0)
                {
                    const std::int32_t yoff =11*xBufN+10*yBufN+ 9*zBufN+blockIdx.y*plnZXN+plnZXIdx;
                    fn[16] = swapDDFBuf[yoff];
                }
                //load f17(x:+,y:+,z:0) from neighbor (x:-,y:-,z:0)
                if((blkFlag & getBlockMask3D<-1,-1, 0>())!=0)
                {
                    const std::int32_t xoff =11*xBufN+11*yBufN+ 9*zBufN+blockIdx.x*plnZYN+plnZYIdx+plnZYNdx;
                    const std::int32_t yoff =12*xBufN+11*yBufN+ 9*zBufN+blockIdx.y*plnZXN+plnZXIdx+plnZXNdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : yoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : yoff;
                    fn[17] = swapDDFBuf[off];
                }
                //load f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:-)
                if((blkFlag & getBlockMask3D< 1, 1,-1>())!=0)
                {
                    const std::int32_t xoff =12*xBufN+12*yBufN+ 9*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx+plnZYNdy+plnZYPdx;
                    const std::int32_t yoff =13*xBufN+12*yBufN+ 9*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx+plnZXNdy+plnZXPdx;
                    const std::int32_t zoff =13*xBufN+13*yBufN+ 9*zBufN+blockIdx.z*plnYXN+plnYXIdx+plnYXPdy+plnYXPdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : (threadIdx.y==(blockDim.y-1)) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_BACK_BIT)!=0) ? yoff : zoff;
                    fn[18] = swapDDFBuf[off];
                }
                //load f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:-)
                if((blkFlag & getBlockMask3D< 0, 1,-1>())!=0)
                {
                    const std::int32_t yoff =13*xBufN+13*yBufN+10*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx+plnZXNdy;
                    const std::int32_t zoff =13*xBufN+14*yBufN+10*zBufN+blockIdx.z*plnYXN+plnYXIdx+plnYXPdy;
                    //const std::int32_t off = (threadIdx.y==(blockDim.y-1)) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_Y_BACK_BIT)!=0) ? yoff : zoff;
                    fn[19] = swapDDFBuf[off]; 
                }
                //load f20(x:+,y:-,z:+) from neighbor (x:-,y:+,z:-)
                if((blkFlag & getBlockMask3D<-1, 1,-1>())!=0)
                {
                    const std::int32_t xoff =13*xBufN+14*yBufN+11*zBufN+blockIdx.x*plnZYN+plnZYIdx+plnZYNdy+plnZYPdx;
                    const std::int32_t yoff =14*xBufN+14*yBufN+11*zBufN+(blockIdx.y+1)*plnZXN+plnZXIdx+plnZXNdy+plnZXNdx;
                    const std::int32_t zoff =14*xBufN+15*yBufN+11*zBufN+blockIdx.z*plnYXN+plnYXIdx+plnYXPdy+plnYXNdx;
                    //const std::int32_t off =(threadIdx.x==0) ? xoff : (threadIdx.y==(blockDim.y-1)) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_BACK_BIT)!=0) ? yoff : zoff;
                    fn[20] = swapDDFBuf[off];
                }
                //load f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:-)
                if((blkFlag & getBlockMask3D< 1, 0,-1>())!=0)
                {
                    const std::int32_t xoff =14*xBufN+15*yBufN+12*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx+plnZYNdy;
                    const std::int32_t zoff =15*xBufN+15*yBufN+12*zBufN+blockIdx.z*plnYXN+plnYXIdx+plnYXPdx;
                    //const std::int32_t off =(threadIdx.x==(blockDim.x-1)) ? xoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : zoff;
                    fn[21] = swapDDFBuf[off];
                }
                //load f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:-)
                if((blkFlag & getBlockMask3D< 0, 0,-1>())!=0)
                {
                    const std::int32_t zoff =15*xBufN+15*yBufN+13*zBufN+blockIdx.z*plnYXN+plnYXIdx;
                    fn[22] = swapDDFBuf[zoff];
                }
                //load f23(x:+,y:0,z:+) from neighbor (x:-,y:0,z:-)
                if((blkFlag & getBlockMask3D<-1, 0,-1>())!=0)
                {
                    const std::int32_t xoff =15*xBufN+15*yBufN+14*zBufN+blockIdx.x*plnZYN+plnZYIdx+plnZYNdy;
                    const std::int32_t zoff =16*xBufN+15*yBufN+14*zBufN+blockIdx.z*plnYXN+plnYXIdx+plnYXNdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : zoff;
                    fn[23] = swapDDFBuf[off];
                }
                //load f24(x:-,y:+,z:+) from neighbor (x:+,y:-,z:-)
                if((blkFlag & getBlockMask3D< 1,-1,-1>())!=0)
                {
                    const std::int32_t xoff =16*xBufN+15*yBufN+15*zBufN+(blockIdx.x+1)*plnZYN+plnZYIdx+plnZYNdy+plnZYNdx;
                    const std::int32_t yoff =17*xBufN+15*yBufN+15*zBufN+blockIdx.y*plnZXN+plnZXIdx+plnZXNdy+plnZXPdx;
                    const std::int32_t zoff =17*xBufN+16*yBufN+15*zBufN+blockIdx.z*plnYXN+plnYXIdx+plnYXNdy+plnYXPdx;
                    //const std::int32_t off = (threadIdx.x==(blockDim.x-1)) ? xoff : (threadIdx.y==0) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_BACK_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_FRONT_BIT)!=0) ? yoff : zoff;
                    fn[24] = swapDDFBuf[off];
                }
                //load f25(x:0,y:+,z:+) from neighbor (x:0,y:-,z:-)
                if((blkFlag & getBlockMask3D< 0,-1,-1>())!=0)
                {
                    const std::int32_t yoff =17*xBufN+16*yBufN+16*zBufN+blockIdx.y*plnZXN+plnZXIdx+plnZXNdy;
                    const std::int32_t zoff =17*xBufN+17*yBufN+16*zBufN+blockIdx.z*plnYXN+plnYXIdx+plnYXNdy;
                    //const std::int32_t off = (threadIdx.y==0) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_Y_FRONT_BIT)!=0) ? yoff : zoff;
                    fn[25] = swapDDFBuf[off];
                }
                //load f26(x:+,y:+,z:+) from neighbor (x:-,y:-,z:-)
                if((blkFlag & getBlockMask3D<-1,-1,-1>())!=0)
                {
                    const std::int32_t xoff =17*xBufN+17*yBufN+17*zBufN+blockIdx.x*plnZYN+plnZYIdx+plnZYNdy+plnZYNdx;
                    const std::int32_t yoff =18*xBufN+17*yBufN+17*zBufN+blockIdx.y*plnZXN+plnZXIdx+plnZXNdy+plnZXNdx;
                    const std::int32_t zoff =18*xBufN+18*yBufN+17*zBufN+blockIdx.z*plnYXN+plnYXIdx+plnYXNdy+plnYXNdx;
                    //const std::int32_t off = (threadIdx.x==0) ? xoff : (threadIdx.y==0) ? yoff : zoff;
                    const std::int32_t off = ((blkFlag & BLK_X_FRONT_BIT)!=0) ? xoff : ((blkFlag & BLK_Y_FRONT_BIT)!=0) ? yoff : zoff;
                    fn[26] = swapDDFBuf[off];
                }
            }
        }
    };

}