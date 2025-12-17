#include "kernel.cuh"
#include "L1L2StreamCore.cuh"
#include "device_function.hpp"



namespace gf::simulator::single_dev_expt
{
    __global__ __launch_bounds__(1024) void
    HaloBlockingL1L2D3Q27PullKernel(const HaloBlockingL1L2Param __grid_constant__ param)
    {
        extern __shared__ real_t blkDDFBuf [];

        const idx_t blkx = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t blky = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t blkz = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t blknx = gridDim.x * blockDim.x;
        const idx_t blkny = gridDim.y * blockDim.y;
        const idx_t blki = (blkny * blkz + blky) * blknx + blkx;

        const idx_t glbx = param.offx + blkx;
        const idx_t glby = param.offy + blky;
        const idx_t glbz = param.offz + blkz;
        const idx_t glbi = (param.glbny * glbz + glby) * param.glbnx + glbx;
        const idx_t glbn = param.glbnx * param.glbny * param.glbnz;

        const flag_t flagi = param.blkFlagBuf[blki];
        //real_t rhoi = 1, vxi = 0, vyi = 0, vzi = 0;

        real_t fni[27];

        //first global load
        if((flagi & LOAD_DDF_BIT)!=0)
        {
            const idx_t glbndx = (glbx==0) ? 0 : -1;
            const idx_t glbpdx = (glbx==param.glbnx-1) ? 0 : 1;
            const idx_t glbndy = (glby==0) ? 0 : -param.glbnx;
            const idx_t glbpdy = (glby==param.glbny-1) ? 0 : param.glbnx;
            const idx_t glbndz = (glbz==0) ? 0 : -param.glbnx * param.glbny;
            const idx_t glbpdz = (glbz==param.glbnz-1) ? 0 : param.glbnx * param.glbny;

            //f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
            fni[ 0] = param.glbSrcDDFBuf[ 0*glbn+glbi+glbpdx+glbpdy+glbpdz];
            //f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 1] = param.glbSrcDDFBuf[ 1*glbn+glbi       +glbpdy+glbpdz];
            //f2 (x:+,y:-,z:-) from neighbor (x:-,y:+,z:+)
            fni[ 2] = param.glbSrcDDFBuf[ 2*glbn+glbi+glbndx+glbpdy+glbpdz];
            //f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 3] = param.glbSrcDDFBuf[ 3*glbn+glbi+glbpdx       +glbpdz];
            //f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 4] = param.glbSrcDDFBuf[ 4*glbn+glbi              +glbpdz];
            //f5 (x:+,y:0,z:-) from neighbor (x:-,y:0,z:+)
            fni[ 5] = param.glbSrcDDFBuf[ 5*glbn+glbi+glbndx       +glbpdz];
            //f6 (x:-,y:+,z:-) from neighbor (x:+,y:-,z:+)
            fni[ 6] = param.glbSrcDDFBuf[ 6*glbn+glbi+glbpdx+glbndy+glbpdz];
            //f7 (x:0,y:+,z:-) from neighbor (x:0,y:-,z:+)
            fni[ 7] = param.glbSrcDDFBuf[ 7*glbn+glbi       +glbndy+glbpdz];
            //f8 (x:+,y:+,z:-) from neighbor (x:-,y:-,z:+)
            fni[ 8] = param.glbSrcDDFBuf[ 8*glbn+glbi+glbndx+glbndy+glbpdz];

            //f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
            fni[ 9] = param.glbSrcDDFBuf[ 9*glbn+glbi+glbpdx+glbpdy       ];
            //f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[10] = param.glbSrcDDFBuf[10*glbn+glbi       +glbpdy       ];
            //f11(x:+,y:-,z:0) from neighbor (x:-,y:+,z:0)
            fni[11] = param.glbSrcDDFBuf[11*glbn+glbi+glbndx+glbpdy       ];
            //f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
            fni[12] = param.glbSrcDDFBuf[12*glbn+glbi+glbpdx              ];
            //f13(x:0,y:0,z:0) from neighbor (x:0,y:0,z:0)
            fni[13] = param.glbSrcDDFBuf[13*glbn+glbi                     ];
            //f14(x:+,y:0,z:0) from neighbor (x:-,y:0,z:0)
            fni[14] = param.glbSrcDDFBuf[14*glbn+glbi+glbndx              ];
            //f15(x:-,y:+,z:0) from neighbor (x:+,y:-,z:0)
            fni[15] = param.glbSrcDDFBuf[15*glbn+glbi+glbpdx+glbndy       ];
            //f16(x:0,y:+,z:0) from neighbor (x:0,y:-,z:0)
            fni[16] = param.glbSrcDDFBuf[16*glbn+glbi       +glbndy       ];
            //f17(x:+,y:+,z:0) from neighbor (x:-,y:-,z:0)
            fni[17] = param.glbSrcDDFBuf[17*glbn+glbi+glbndx+glbndy       ];

            //f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:-)
            fni[18] = param.glbSrcDDFBuf[18*glbn+glbi+glbpdx+glbpdy+glbndz];
            //f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:-)
            fni[19] = param.glbSrcDDFBuf[19*glbn+glbi       +glbpdy+glbndz];
            //f20(x:+,y:-,z:+) from neighbor (x:-,y:+,z:-)
            fni[20] = param.glbSrcDDFBuf[20*glbn+glbi+glbndx+glbpdy+glbndz];
            //f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:-)
            fni[21] = param.glbSrcDDFBuf[21*glbn+glbi+glbpdx       +glbndz];
            //f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:-)
            fni[22] = param.glbSrcDDFBuf[22*glbn+glbi              +glbndz];
            //f23(x:+,y:0,z:+) from neighbor (x:-,y:0,z:-)
            fni[23] = param.glbSrcDDFBuf[23*glbn+glbi+glbndx       +glbndz];
            //f24(x:-,y:+,z:+) from neighbor (x:+,y:-,z:-)
            fni[24] = param.glbSrcDDFBuf[24*glbn+glbi+glbpdx+glbndy+glbndz];
            //f25(x:0,y:+,z:+) from neighbor (x:0,y:-,z:-)
            fni[25] = param.glbSrcDDFBuf[25*glbn+glbi       +glbndy+glbndz];
            //f26(x:+,y:+,z:+) from neighbor (x:-,y:-,z:-)
            fni[26] = param.glbSrcDDFBuf[26*glbn+glbi+glbndx+glbndy+glbndz];
        }

        if((flagi & EQU_DDF_BIT)!=0)
        {
            const real_t rhoi    = param.glbRhoBuf[glbi];
            const real_t vxi     = param.glbVxBuf[glbi];
            const real_t vyi     = param.glbVyBuf[glbi];
            const real_t vzi     = param.glbVzBuf[glbi];
            gf::lbm_core::bgk::calcEqu<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
        }

        if((flagi & BOUNCE_BACK_BIT)!=0)
        {
            std::swap(fni[ 0], fni[26]);
            std::swap(fni[ 1], fni[25]);
            std::swap(fni[ 2], fni[24]);
            std::swap(fni[ 3], fni[23]);
            std::swap(fni[ 4], fni[22]);
            std::swap(fni[ 5], fni[21]);
            std::swap(fni[ 6], fni[20]);
            std::swap(fni[ 7], fni[19]);
            std::swap(fni[ 8], fni[18]);
            std::swap(fni[ 9], fni[17]);
            std::swap(fni[10], fni[16]);
            std::swap(fni[11], fni[15]);
            std::swap(fni[12], fni[14]);
        }

        if((flagi & COLLIDE_BIT)!=0)
        {
            gf::lbm_core::bgk::collision2<27>(param.invTau, std::begin(fni));
        }

        for(idx_t iter=1 ; iter<param.nloop ; ++iter)
        {
            gf::core::L1L2StreamCore::StreamCore3D<27>::stream(std::begin(fni), blkDDFBuf, param.glbSwapDDFBuf);

            if((flagi & EQU_DDF_BIT)!=0)
            {
                const real_t rhoi = param.glbRhoBuf[glbi];
                const real_t vxi = param.glbVxBuf[glbi];
                const real_t vyi = param.glbVyBuf[glbi];
                const real_t vzi = param.glbVzBuf[glbi];
                gf::lbm_core::bgk::calcEqu<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
            }

            if((flagi & BOUNCE_BACK_BIT)!=0)
            {
                std::swap(fni[ 0], fni[26]);
                std::swap(fni[ 1], fni[25]);
                std::swap(fni[ 2], fni[24]);
                std::swap(fni[ 3], fni[23]);
                std::swap(fni[ 4], fni[22]);
                std::swap(fni[ 5], fni[21]);
                std::swap(fni[ 6], fni[20]);
                std::swap(fni[ 7], fni[19]);
                std::swap(fni[ 8], fni[18]);
                std::swap(fni[ 9], fni[17]);
                std::swap(fni[10], fni[16]);
                std::swap(fni[11], fni[15]);
                std::swap(fni[12], fni[14]);
            }

            if((flagi & COLLIDE_BIT)!=0)
            {
                gf::lbm_core::bgk::collision2<27>(param.invTau, std::begin(fni));
            }
        }

        //last global store
        if((flagi & (STORE_DDF_BIT | CORRECT_BIT))==(STORE_DDF_BIT | CORRECT_BIT))
        {
            param.glbDstDDFBuf[ 0*glbn+glbi] = fni[ 0];
            param.glbDstDDFBuf[ 1*glbn+glbi] = fni[ 1];
            param.glbDstDDFBuf[ 2*glbn+glbi] = fni[ 2];
            param.glbDstDDFBuf[ 3*glbn+glbi] = fni[ 3];
            param.glbDstDDFBuf[ 4*glbn+glbi] = fni[ 4];
            param.glbDstDDFBuf[ 5*glbn+glbi] = fni[ 5];
            param.glbDstDDFBuf[ 6*glbn+glbi] = fni[ 6];
            param.glbDstDDFBuf[ 7*glbn+glbi] = fni[ 7];
            param.glbDstDDFBuf[ 8*glbn+glbi] = fni[ 8];

            param.glbDstDDFBuf[ 9*glbn+glbi] = fni[ 9];
            param.glbDstDDFBuf[10*glbn+glbi] = fni[10];
            param.glbDstDDFBuf[11*glbn+glbi] = fni[11];
            param.glbDstDDFBuf[12*glbn+glbi] = fni[12];
            param.glbDstDDFBuf[13*glbn+glbi] = fni[13];
            param.glbDstDDFBuf[14*glbn+glbi] = fni[14];
            param.glbDstDDFBuf[15*glbn+glbi] = fni[15];
            param.glbDstDDFBuf[16*glbn+glbi] = fni[16];
            param.glbDstDDFBuf[17*glbn+glbi] = fni[17];

            param.glbDstDDFBuf[18*glbn+glbi] = fni[18];
            param.glbDstDDFBuf[19*glbn+glbi] = fni[19];
            param.glbDstDDFBuf[20*glbn+glbi] = fni[20];
            param.glbDstDDFBuf[21*glbn+glbi] = fni[21];
            param.glbDstDDFBuf[22*glbn+glbi] = fni[22];
            param.glbDstDDFBuf[23*glbn+glbi] = fni[23];
            param.glbDstDDFBuf[24*glbn+glbi] = fni[24];
            param.glbDstDDFBuf[25*glbn+glbi] = fni[25];
            param.glbDstDDFBuf[26*glbn+glbi] = fni[26];
        }
    }

    __global__ __launch_bounds__(1024) void
    HaloBlockingL1L2D3Q27DumpKernel(const HaloBlockingL1L2Param __grid_constant__ param)
    {
        const idx_t blkx = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t blky = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t blkz = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t blknx = gridDim.x * blockDim.x;
        const idx_t blkny = gridDim.y * blockDim.y;
        const idx_t blki = (blkny*blkz+blky)*blknx+blkx;
        const idx_t glbx = param.offx + blkx;
        const idx_t glby = param.offy + blky;
        const idx_t glbz = param.offz + blkz;
        const idx_t glbi = (param.glbny * glbz + glby) * param.glbnx + glbx;
        const idx_t glbn = param.glbnx * param.glbny * param.glbnz;

        const flag_t flagi = param.blkFlagBuf[blki];
        real_t rhoi, vxi, vyi, vzi;
        real_t fni[27];

        if((flagi & CORRECT_BIT)!=0)
        {
            const idx_t glbndx = (glbx==0) ? 0 : -1;
            const idx_t glbpdx = (glbx==param.glbnx-1) ? 0 : 1;
            const idx_t glbndy = (glby==0) ? 0 : -param.glbnx;
            const idx_t glbpdy = (glby==param.glbny-1) ? 0 : param.glbnx;
            const idx_t glbndz = (glbz==0) ? 0 : -param.glbnx * param.glbny;
            const idx_t glbpdz = (glbz==param.glbnz-1) ? 0 : param.glbnx * param.glbny;

            //f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
            fni[ 0] = param.glbSrcDDFBuf[ 0*glbn+glbi+glbpdx+glbpdy+glbpdz];
            //f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 1] = param.glbSrcDDFBuf[ 1*glbn+glbi       +glbpdy+glbpdz];
            //f2 (x:+,y:-,z:-) from neighbor (x:-,y:+,z:+)
            fni[ 2] = param.glbSrcDDFBuf[ 2*glbn+glbi+glbndx+glbpdy+glbpdz];
            //f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 3] = param.glbSrcDDFBuf[ 3*glbn+glbi+glbpdx       +glbpdz];
            //f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 4] = param.glbSrcDDFBuf[ 4*glbn+glbi              +glbpdz];
            //f5 (x:+,y:0,z:-) from neighbor (x:-,y:0,z:+)
            fni[ 5] = param.glbSrcDDFBuf[ 5*glbn+glbi+glbndx       +glbpdz];
            //f6 (x:-,y:+,z:-) from neighbor (x:+,y:-,z:+)
            fni[ 6] = param.glbSrcDDFBuf[ 6*glbn+glbi+glbpdx+glbndy+glbpdz];
            //f7 (x:0,y:+,z:-) from neighbor (x:0,y:-,z:+)
            fni[ 7] = param.glbSrcDDFBuf[ 7*glbn+glbi       +glbndy+glbpdz];
            //f8 (x:+,y:+,z:-) from neighbor (x:-,y:-,z:+)
            fni[ 8] = param.glbSrcDDFBuf[ 8*glbn+glbi+glbndx+glbndy+glbpdz];

            //f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
            fni[ 9] = param.glbSrcDDFBuf[ 9*glbn+glbi+glbpdx+glbpdy       ];
            //f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[10] = param.glbSrcDDFBuf[10*glbn+glbi       +glbpdy       ];
            //f11(x:+,y:-,z:0) from neighbor (x:-,y:+,z:0)
            fni[11] = param.glbSrcDDFBuf[11*glbn+glbi+glbndx+glbpdy       ];
            //f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
            fni[12] = param.glbSrcDDFBuf[12*glbn+glbi+glbpdx              ];
            //f13(x:0,y:0,z:0) from neighbor (x:0,y:0,z:0)
            fni[13] = param.glbSrcDDFBuf[13*glbn+glbi                     ];
            //f14(x:+,y:0,z:0) from neighbor (x:-,y:0,z:0)
            fni[14] = param.glbSrcDDFBuf[14*glbn+glbi+glbndx              ];
            //f15(x:-,y:+,z:0) from neighbor (x:+,y:-,z:0)
            fni[15] = param.glbSrcDDFBuf[15*glbn+glbi+glbpdx+glbndy       ];
            //f16(x:0,y:+,z:0) from neighbor (x:0,y:-,z:0)
            fni[16] = param.glbSrcDDFBuf[16*glbn+glbi       +glbndy       ];
            //f17(x:+,y:+,z:0) from neighbor (x:-,y:-,z:0)
            fni[17] = param.glbSrcDDFBuf[17*glbn+glbi+glbndx+glbndy       ];

            //f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:-)
            fni[18] = param.glbSrcDDFBuf[18*glbn+glbi+glbpdx+glbpdy+glbndz];
            //f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:-)
            fni[19] = param.glbSrcDDFBuf[19*glbn+glbi       +glbpdy+glbndz];
            //f20(x:+,y:-,z:+) from neighbor (x:-,y:+,z:-)
            fni[20] = param.glbSrcDDFBuf[20*glbn+glbi+glbndx+glbpdy+glbndz];
            //f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:-)
            fni[21] = param.glbSrcDDFBuf[21*glbn+glbi+glbpdx       +glbndz];
            //f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:-)
            fni[22] = param.glbSrcDDFBuf[22*glbn+glbi              +glbndz];
            //f23(x:+,y:0,z:+) from neighbor (x:-,y:0,z:-)
            fni[23] = param.glbSrcDDFBuf[23*glbn+glbi+glbndx       +glbndz];
            //f24(x:-,y:+,z:+) from neighbor (x:+,y:-,z:-)
            fni[24] = param.glbSrcDDFBuf[24*glbn+glbi+glbpdx+glbndy+glbndz];
            //f25(x:0,y:+,z:+) from neighbor (x:0,y:-,z:-)
            fni[25] = param.glbSrcDDFBuf[25*glbn+glbi       +glbndy+glbndz];
            //f26(x:+,y:+,z:+) from neighbor (x:-,y:-,z:-)
            fni[26] = param.glbSrcDDFBuf[26*glbn+glbi+glbndx+glbndy+glbndz];

            gf::lbm_core::bgk::calcRhoU<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
        }

        if((flagi & (CORRECT_BIT | DUMP_RHO_BIT))==(CORRECT_BIT | DUMP_RHO_BIT))
        {
            param.glbRhoBuf[glbi] = rhoi;
        }

        if((flagi & (CORRECT_BIT | DUMP_VX_BIT))==(CORRECT_BIT | DUMP_VX_BIT))
        {
            param.glbVxBuf[glbi] = vxi;
        }

        if((flagi & (CORRECT_BIT | DUMP_VY_BIT))==(CORRECT_BIT | DUMP_VY_BIT))
        {
            param.glbVyBuf[glbi] = vyi;
        }

        if((flagi & (CORRECT_BIT | DUMP_VZ_BIT))==(CORRECT_BIT | DUMP_VZ_BIT))
        {
            param.glbVzBuf[glbi] = vzi;
        }
    }

    __global__ __launch_bounds__(1024) void
    HaloBlockingStaticL2D3Q27PullKernel(const HaloBlockingStaticL2Param __grid_constant__ param)
    {
        const idx_t blkx = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t blky = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t blkz = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t blknx = gridDim.x * blockDim.x;
        const idx_t blkny = gridDim.y * blockDim.y;
        const idx_t blknz = gridDim.z * blockDim.z;
        const idx_t blki = (blkny * blkz + blky) * blknx + blkx;
        const idx_t blkn = blknx * blkny * blknz;
        const idx_t glbx = param.offx + blkx;
        const idx_t glby = param.offy + blky;
        const idx_t glbz = param.offz + blkz;
        const idx_t glbi = (param.glbny * glbz + glby) * param.glbnx + glbx;
        const idx_t glbn = param.glbnx * param.glbny * param.glbnz;

        const flag_t flagi = param.blkFlagBuf[blki];
        real_t fni[27];

        if((flagi & LOAD_DDF_BIT)!=0)
        {
            const idx_t glbndx = (glbx==0) ? 0 : -1;
            const idx_t glbpdx = (glbx==param.glbnx-1) ? 0 : 1;
            const idx_t glbndy = (glby==0) ? 0 : -param.glbnx;
            const idx_t glbpdy = (glby==param.glbny-1) ? 0 : param.glbnx;
            const idx_t glbndz = (glbz==0) ? 0 : -param.glbnx * param.glbny;
            const idx_t glbpdz = (glbz==param.glbnz-1) ? 0 : param.glbnx * param.glbny;

            //f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
            fni[ 0] = param.glbSrcDDFBuf[ 0*glbn+glbi+glbpdx+glbpdy+glbpdz];
            //f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 1] = param.glbSrcDDFBuf[ 1*glbn+glbi       +glbpdy+glbpdz];
            //f2 (x:+,y:-,z:-) from neighbor (x:-,y:+,z:+)
            fni[ 2] = param.glbSrcDDFBuf[ 2*glbn+glbi+glbndx+glbpdy+glbpdz];
            //f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 3] = param.glbSrcDDFBuf[ 3*glbn+glbi+glbpdx       +glbpdz];
            //f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 4] = param.glbSrcDDFBuf[ 4*glbn+glbi              +glbpdz];
            //f5 (x:+,y:0,z:-) from neighbor (x:-,y:0,z:+)
            fni[ 5] = param.glbSrcDDFBuf[ 5*glbn+glbi+glbndx       +glbpdz];
            //f6 (x:-,y:+,z:-) from neighbor (x:+,y:-,z:+)
            fni[ 6] = param.glbSrcDDFBuf[ 6*glbn+glbi+glbpdx+glbndy+glbpdz];
            //f7 (x:0,y:+,z:-) from neighbor (x:0,y:-,z:+)
            fni[ 7] = param.glbSrcDDFBuf[ 7*glbn+glbi       +glbndy+glbpdz];
            //f8 (x:+,y:+,z:-) from neighbor (x:-,y:-,z:+)
            fni[ 8] = param.glbSrcDDFBuf[ 8*glbn+glbi+glbndx+glbndy+glbpdz];

            //f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
            fni[ 9] = param.glbSrcDDFBuf[ 9*glbn+glbi+glbpdx+glbpdy       ];
            //f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[10] = param.glbSrcDDFBuf[10*glbn+glbi       +glbpdy       ];
            //f11(x:+,y:-,z:0) from neighbor (x:-,y:+,z:0)
            fni[11] = param.glbSrcDDFBuf[11*glbn+glbi+glbndx+glbpdy       ];
            //f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
            fni[12] = param.glbSrcDDFBuf[12*glbn+glbi+glbpdx              ];
            //f13(x:0,y:0,z:0) from neighbor (x:0,y:0,z:0)
            fni[13] = param.glbSrcDDFBuf[13*glbn+glbi                     ];
            //f14(x:+,y:0,z:0) from neighbor (x:-,y:0,z:0)
            fni[14] = param.glbSrcDDFBuf[14*glbn+glbi+glbndx              ];
            //f15(x:-,y:+,z:0) from neighbor (x:+,y:-,z:0)
            fni[15] = param.glbSrcDDFBuf[15*glbn+glbi+glbpdx+glbndy       ];
            //f16(x:0,y:+,z:0) from neighbor (x:0,y:-,z:0)
            fni[16] = param.glbSrcDDFBuf[16*glbn+glbi       +glbndy       ];
            //f17(x:+,y:+,z:0) from neighbor (x:-,y:-,z:0)
            fni[17] = param.glbSrcDDFBuf[17*glbn+glbi+glbndx+glbndy       ];

            //f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:-)
            fni[18] = param.glbSrcDDFBuf[18*glbn+glbi+glbpdx+glbpdy+glbndz];
            //f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:-)
            fni[19] = param.glbSrcDDFBuf[19*glbn+glbi       +glbpdy+glbndz];
            //f20(x:+,y:-,z:+) from neighbor (x:-,y:+,z:-)
            fni[20] = param.glbSrcDDFBuf[20*glbn+glbi+glbndx+glbpdy+glbndz];
            //f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:-)
            fni[21] = param.glbSrcDDFBuf[21*glbn+glbi+glbpdx       +glbndz];
            //f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:-)
            fni[22] = param.glbSrcDDFBuf[22*glbn+glbi              +glbndz];
            //f23(x:+,y:0,z:+) from neighbor (x:-,y:0,z:-)
            fni[23] = param.glbSrcDDFBuf[23*glbn+glbi+glbndx       +glbndz];
            //f24(x:-,y:+,z:+) from neighbor (x:+,y:-,z:-)
            fni[24] = param.glbSrcDDFBuf[24*glbn+glbi+glbpdx+glbndy+glbndz];
            //f25(x:0,y:+,z:+) from neighbor (x:0,y:-,z:-)
            fni[25] = param.glbSrcDDFBuf[25*glbn+glbi       +glbndy+glbndz];
            //f26(x:+,y:+,z:+) from neighbor (x:-,y:-,z:-)
            fni[26] = param.glbSrcDDFBuf[26*glbn+glbi+glbndx+glbndy+glbndz];
        }

        if((flagi & EQU_DDF_BIT)!=0)
        {
            const real_t rhoi = param.glbRhoBuf[glbi];
            const real_t vxi  = param.glbVxBuf[glbi];
            const real_t vyi  = param.glbVyBuf[glbi];
            const real_t vzi  = param.glbVzBuf[glbi];
            gf::lbm_core::bgk::calcEqu<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
        }

        if((flagi & BOUNCE_BACK_BIT)!=0)
        {
            std::swap(fni[ 0], fni[26]);
            std::swap(fni[ 1], fni[25]);
            std::swap(fni[ 2], fni[24]);
            std::swap(fni[ 3], fni[23]);
            std::swap(fni[ 4], fni[22]);
            std::swap(fni[ 5], fni[21]);
            std::swap(fni[ 6], fni[20]);
            std::swap(fni[ 7], fni[19]);
            std::swap(fni[ 8], fni[18]);
            std::swap(fni[ 9], fni[17]);
            std::swap(fni[10], fni[16]);
            std::swap(fni[11], fni[15]);
            std::swap(fni[12], fni[14]);
        }

        if((flagi & COLLIDE_BIT)!=0)
        {
            gf::lbm_core::bgk::collision2<27>(param.invTau, std::begin(fni));
        }

        for(idx_t iter=1 ; iter<param.nloop ; ++iter)
        {
            if((flagi & STORE_DDF_BIT)!=0)
            {
                real_t* blkDDFBuf = ((iter & 1) != 0) ? param.blkDDFBuf0 : param.blkDDFBuf1;
                blkDDFBuf[ 0*blkn+blki] = fni[ 0];
                blkDDFBuf[ 1*blkn+blki] = fni[ 1];
                blkDDFBuf[ 2*blkn+blki] = fni[ 2];
                blkDDFBuf[ 3*blkn+blki] = fni[ 3];
                blkDDFBuf[ 4*blkn+blki] = fni[ 4];
                blkDDFBuf[ 5*blkn+blki] = fni[ 5];
                blkDDFBuf[ 6*blkn+blki] = fni[ 6];
                blkDDFBuf[ 7*blkn+blki] = fni[ 7];
                blkDDFBuf[ 8*blkn+blki] = fni[ 8];
                blkDDFBuf[ 9*blkn+blki] = fni[ 9];
                blkDDFBuf[10*blkn+blki] = fni[10];
                blkDDFBuf[11*blkn+blki] = fni[11];
                blkDDFBuf[12*blkn+blki] = fni[12];
                //blkDDFBuf[13*blkn+blki] = fni[13];
                blkDDFBuf[14*blkn+blki] = fni[14];
                blkDDFBuf[15*blkn+blki] = fni[15];
                blkDDFBuf[16*blkn+blki] = fni[16];
                blkDDFBuf[17*blkn+blki] = fni[17];
                blkDDFBuf[18*blkn+blki] = fni[18];
                blkDDFBuf[19*blkn+blki] = fni[19];
                blkDDFBuf[20*blkn+blki] = fni[20];
                blkDDFBuf[21*blkn+blki] = fni[21];
                blkDDFBuf[22*blkn+blki] = fni[22];
                blkDDFBuf[23*blkn+blki] = fni[23];
                blkDDFBuf[24*blkn+blki] = fni[24];
                blkDDFBuf[25*blkn+blki] = fni[25];
                blkDDFBuf[26*blkn+blki] = fni[26];
            }

            cg::this_grid().sync();

            if((flagi & LOAD_DDF_BIT)!=0)
            {
                const idx_t blkndx = (blkx==0) ? 0 : -1;
                const idx_t blkpdx = (blkx==blknx-1) ? 0 : 1;
                const idx_t blkndy = (blky==0) ? 0 : -blknx;
                const idx_t blkpdy = (blky==blkny-1) ? 0 : blknx;
                const idx_t blkndz = (blkz==0) ? 0 : -blknx * blkny;
                const idx_t blkpdz = (blkz==blknz-1) ? 0 : blknx * blkny;
                const real_t* blkDDFBuf = ((iter & 1) != 0) ? param.blkDDFBuf0 : param.blkDDFBuf1;

                //f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
                fni[ 0] = blkDDFBuf[ 0*blkn+blki+blkpdx+blkpdy+blkpdz];
                //f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
                fni[ 1] = blkDDFBuf[ 1*blkn+blki       +blkpdy+blkpdz];
                //f2 (x:+,y:-,z:-) from neighbor (x:-,y:+,z:+)
                fni[ 2] = blkDDFBuf[ 2*blkn+blki+blkndx+blkpdx+blkpdz];
                //f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
                fni[ 3] = blkDDFBuf[ 3*blkn+blki+blkpdx       +blkpdz];
                //f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
                fni[ 4] = blkDDFBuf[ 4*blkn+blki              +blkpdz];
                //f5 (x:+,y:0,z:-) from neighbor (x:-,y:0,z:+)
                fni[ 5] = blkDDFBuf[ 5*blkn+blki+blkndx       +blkpdz];
                //f6 (x:-,y:+,z:-) from neighbor (x:+,y:-,z:+)
                fni[ 6] = blkDDFBuf[ 6*blkn+blki+blkpdx+blkndy+blkpdz];
                //f7 (x:0,y:+,z:-) from neighbor (x:0,y:-,z:+)
                fni[ 7] = blkDDFBuf[ 7*blkn+blki       +blkndy+blkpdz];
                //f8 (x:+,y:+,z:-) from neighbor (x:-,y:-,z:+)
                fni[ 8] = blkDDFBuf[ 8*blkn+blki+blkndx+blkndy+blkpdz];

                //f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
                fni[ 9] = blkDDFBuf[ 9*blkn+blki+blkpdx+blkpdy       ];
                //f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
                fni[10] = blkDDFBuf[10*blkn+blki       +blkpdy       ];
                //f11(x:+,y:-,z:0) from neighbor (x:-,y:+,z:0)
                fni[11] = blkDDFBuf[11*blkn+blki+blkndx+blkpdy       ];
                //f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
                fni[12] = blkDDFBuf[12*blkn+blki+blkpdx              ];
                //f13(x:0,y:0,z:0) from neighbor (x:0,y:0,z:0)
                //fni[13] = blkDDFBuf[13*blkn+blki                     ];
                //f14(x:+,y:0,z:0) from neighbor (x:-,y:0,z:0)
                fni[14] = blkDDFBuf[14*blkn+blki+blkndx              ];
                //f15(x:-,y:+,z:0) from neighbor (x:+,y:-,z:0)
                fni[15] = blkDDFBuf[15*blkn+blki+blkpdx+blkndy       ];
                //f16(x:0,y:+,z:0) from neighbor (x:0,y:-,z:0)
                fni[16] = blkDDFBuf[16*blkn+blki       +blkndy       ];
                //f17(x:+,y:+,z:0) from neighbor (x:-,y:-,z:0)
                fni[17] = blkDDFBuf[17*blkn+blki+blkndx+blkndy       ];

                //f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:-)
                fni[18] = blkDDFBuf[18*blkn+blki+blkpdx+blkpdy+blkndz];
                //f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:-)
                fni[19] = blkDDFBuf[19*blkn+blki       +blkpdy+blkndz];
                //f20(x:+,y:-,z:+) from neighbor (x:-,y:+,z:-)
                fni[20] = blkDDFBuf[20*blkn+blki+blkndx+blkpdy+blkndz];
                //f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:-)
                fni[21] = blkDDFBuf[21*blkn+blki+blkpdx       +blkndz];
                //f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:-)
                fni[22] = blkDDFBuf[22*blkn+blki              +blkndz];
                //f23(x:+,y:0,z:+) from neighbor (x:-,y:0,z:-)
                fni[23] = blkDDFBuf[23*blkn+blki+blkndx       +blkndz];
                //f24(x:-,y:+,z:+) from neighbor (x:+,y:-,z:-)
                fni[24] = blkDDFBuf[24*blkn+blki+blkpdx+blkndy+blkndz];
                //f25(x:0,y:+,z:+) from neighbor (x:0,y:-,z:-)
                fni[25] = blkDDFBuf[25*blkn+blki       +blkndy+blkndz];
                //f26(x:+,y:+,z:+) from neighbor (x:-,y:-,z:-)
                fni[26] = blkDDFBuf[26*blkn+blki+blkndx+blkndy+blkndz];


            }

            if((flagi & EQU_DDF_BIT)!=0)
            {
                const real_t rhoi = param.glbRhoBuf[glbi];
                const real_t vxi  = param.glbVxBuf[glbi];
                const real_t vyi  = param.glbVyBuf[glbi];
                const real_t vzi  = param.glbVzBuf[glbi];
                gf::lbm_core::bgk::calcEqu<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
            }

            if((flagi & BOUNCE_BACK_BIT)!=0)
            {
                std::swap(fni[ 0], fni[26]);
                std::swap(fni[ 1], fni[25]);
                std::swap(fni[ 2], fni[24]);
                std::swap(fni[ 3], fni[23]);
                std::swap(fni[ 4], fni[22]);
                std::swap(fni[ 5], fni[21]);
                std::swap(fni[ 6], fni[20]);
                std::swap(fni[ 7], fni[19]);
                std::swap(fni[ 8], fni[18]);
                std::swap(fni[ 9], fni[17]);
                std::swap(fni[10], fni[16]);
                std::swap(fni[11], fni[15]);
                std::swap(fni[12], fni[14]); 
            }

            if((flagi & COLLIDE_BIT)!=0)
            {
                gf::lbm_core::bgk::collision2<27>(param.invTau, std::begin(fni));
            }
        }

        if((flagi & (STORE_DDF_BIT | CORRECT_BIT))==(STORE_DDF_BIT | CORRECT_BIT))
        {
            param.glbDstDDFBuf[ 0*glbn+glbi] = fni[ 0];
            param.glbDstDDFBuf[ 1*glbn+glbi] = fni[ 1];
            param.glbDstDDFBuf[ 2*glbn+glbi] = fni[ 2];
            param.glbDstDDFBuf[ 3*glbn+glbi] = fni[ 3];
            param.glbDstDDFBuf[ 4*glbn+glbi] = fni[ 4];
            param.glbDstDDFBuf[ 5*glbn+glbi] = fni[ 5];
            param.glbDstDDFBuf[ 6*glbn+glbi] = fni[ 6];
            param.glbDstDDFBuf[ 7*glbn+glbi] = fni[ 7];
            param.glbDstDDFBuf[ 8*glbn+glbi] = fni[ 8];
            param.glbDstDDFBuf[ 9*glbn+glbi] = fni[ 9];
            param.glbDstDDFBuf[10*glbn+glbi] = fni[10];
            param.glbDstDDFBuf[11*glbn+glbi] = fni[11];
            param.glbDstDDFBuf[12*glbn+glbi] = fni[12];
            param.glbDstDDFBuf[13*glbn+glbi] = fni[13];
            param.glbDstDDFBuf[14*glbn+glbi] = fni[14];
            param.glbDstDDFBuf[15*glbn+glbi] = fni[15];
            param.glbDstDDFBuf[16*glbn+glbi] = fni[16];
            param.glbDstDDFBuf[17*glbn+glbi] = fni[17];
            param.glbDstDDFBuf[18*glbn+glbi] = fni[18];
            param.glbDstDDFBuf[19*glbn+glbi] = fni[19];
            param.glbDstDDFBuf[20*glbn+glbi] = fni[20];
            param.glbDstDDFBuf[21*glbn+glbi] = fni[21];
            param.glbDstDDFBuf[22*glbn+glbi] = fni[22];
            param.glbDstDDFBuf[23*glbn+glbi] = fni[23];
            param.glbDstDDFBuf[24*glbn+glbi] = fni[24];
            param.glbDstDDFBuf[25*glbn+glbi] = fni[25];
            param.glbDstDDFBuf[26*glbn+glbi] = fni[26];
        }
    }

    __global__ __launch_bounds__(1024) void
    HaloBlockingStaticL2D3Q27DumpKernel(const HaloBlockingStaticL2Param __grid_constant__ param)
    {
        const idx_t blkx = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t blky = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t blkz = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t blknx = gridDim.x * blockDim.x;
        const idx_t blkny = gridDim.y * blockDim.y;
        const idx_t blki = (blkny*blkz+blky)*blknx+blkx;
        const idx_t glbx = param.offx + blkx;
        const idx_t glby = param.offy + blky;
        const idx_t glbz = param.offz + blkz;
        const idx_t glbi = (param.glbny * glbz + glby) * param.glbnx + glbx;
        const idx_t glbn = param.glbnx * param.glbny * param.glbnz;

        const flag_t flagi = param.blkFlagBuf[blki];
        real_t rhoi, vxi, vyi, vzi;
        real_t fni[27];

        if((flagi & CORRECT_BIT)!=0)
        {
            const idx_t glbndx = (glbx==0) ? 0 : -1;
            const idx_t glbpdx = (glbx==param.glbnx-1) ? 0 : 1;
            const idx_t glbndy = (glby==0) ? 0 : -param.glbnx;
            const idx_t glbpdy = (glby==param.glbny-1) ? 0 : param.glbnx;
            const idx_t glbndz = (glbz==0) ? 0 : -param.glbnx * param.glbny;
            const idx_t glbpdz = (glbz==param.glbnz-1) ? 0 : param.glbnx * param.glbny;

            //f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
            fni[ 0] = param.glbSrcDDFBuf[ 0*glbn+glbi+glbpdx+glbpdy+glbpdz];
            //f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 1] = param.glbSrcDDFBuf[ 1*glbn+glbi       +glbpdy+glbpdz];
            //f2 (x:+,y:-,z:-) from neighbor (x:-,y:+,z:+)
            fni[ 2] = param.glbSrcDDFBuf[ 2*glbn+glbi+glbndx+glbpdy+glbpdz];
            //f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 3] = param.glbSrcDDFBuf[ 3*glbn+glbi+glbpdx       +glbpdz];
            //f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 4] = param.glbSrcDDFBuf[ 4*glbn+glbi              +glbpdz];
            //f5 (x:+,y:0,z:-) from neighbor (x:-,y:0,z:+)
            fni[ 5] = param.glbSrcDDFBuf[ 5*glbn+glbi+glbndx       +glbpdz];
            //f6 (x:-,y:+,z:-) from neighbor (x:+,y:-,z:+)
            fni[ 6] = param.glbSrcDDFBuf[ 6*glbn+glbi+glbpdx+glbndy+glbpdz];
            //f7 (x:0,y:+,z:-) from neighbor (x:0,y:-,z:+)
            fni[ 7] = param.glbSrcDDFBuf[ 7*glbn+glbi       +glbndy+glbpdz];
            //f8 (x:+,y:+,z:-) from neighbor (x:-,y:-,z:+)
            fni[ 8] = param.glbSrcDDFBuf[ 8*glbn+glbi+glbndx+glbndy+glbpdz];

            //f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
            fni[ 9] = param.glbSrcDDFBuf[ 9*glbn+glbi+glbpdx+glbpdy       ];
            //f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[10] = param.glbSrcDDFBuf[10*glbn+glbi       +glbpdy       ];
            //f11(x:+,y:-,z:0) from neighbor (x:-,y:+,z:0)
            fni[11] = param.glbSrcDDFBuf[11*glbn+glbi+glbndx+glbpdy       ];
            //f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
            fni[12] = param.glbSrcDDFBuf[12*glbn+glbi+glbpdx              ];
            //f13(x:0,y:0,z:0) from neighbor (x:0,y:0,z:0)
            fni[13] = param.glbSrcDDFBuf[13*glbn+glbi                     ];
            //f14(x:+,y:0,z:0) from neighbor (x:-,y:0,z:0)
            fni[14] = param.glbSrcDDFBuf[14*glbn+glbi+glbndx              ];
            //f15(x:-,y:+,z:0) from neighbor (x:+,y:-,z:0)
            fni[15] = param.glbSrcDDFBuf[15*glbn+glbi+glbpdx+glbndy       ];
            //f16(x:0,y:+,z:0) from neighbor (x:0,y:-,z:0)
            fni[16] = param.glbSrcDDFBuf[16*glbn+glbi       +glbndy       ];
            //f17(x:+,y:+,z:0) from neighbor (x:-,y:-,z:0)
            fni[17] = param.glbSrcDDFBuf[17*glbn+glbi+glbndx+glbndy       ];

            //f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:-)
            fni[18] = param.glbSrcDDFBuf[18*glbn+glbi+glbpdx+glbpdy+glbndz];
            //f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:-)
            fni[19] = param.glbSrcDDFBuf[19*glbn+glbi       +glbpdy+glbndz];
            //f20(x:+,y:-,z:+) from neighbor (x:-,y:+,z:-)
            fni[20] = param.glbSrcDDFBuf[20*glbn+glbi+glbndx+glbpdy+glbndz];
            //f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:-)
            fni[21] = param.glbSrcDDFBuf[21*glbn+glbi+glbpdx       +glbndz];
            //f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:-)
            fni[22] = param.glbSrcDDFBuf[22*glbn+glbi              +glbndz];
            //f23(x:+,y:0,z:+) from neighbor (x:-,y:0,z:-)
            fni[23] = param.glbSrcDDFBuf[23*glbn+glbi+glbndx       +glbndz];
            //f24(x:-,y:+,z:+) from neighbor (x:+,y:-,z:-)
            fni[24] = param.glbSrcDDFBuf[24*glbn+glbi+glbpdx+glbndy+glbndz];
            //f25(x:0,y:+,z:+) from neighbor (x:0,y:-,z:-)
            fni[25] = param.glbSrcDDFBuf[25*glbn+glbi       +glbndy+glbndz];
            //f26(x:+,y:+,z:+) from neighbor (x:-,y:-,z:-)
            fni[26] = param.glbSrcDDFBuf[26*glbn+glbi+glbndx+glbndy+glbndz];

            gf::lbm_core::bgk::calcRhoU<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
        }

        if((flagi & (CORRECT_BIT | DUMP_RHO_BIT))==(CORRECT_BIT | DUMP_RHO_BIT))
        {
            param.glbRhoBuf[glbi] = rhoi;
        }

        if((flagi & (CORRECT_BIT | DUMP_VX_BIT))==(CORRECT_BIT | DUMP_VX_BIT))
        {
            param.glbVxBuf[glbi] = vxi;
        }

        if((flagi & (CORRECT_BIT | DUMP_VY_BIT))==(CORRECT_BIT | DUMP_VY_BIT))
        {
            param.glbVyBuf[glbi] = vyi;
        }

        if((flagi & (CORRECT_BIT | DUMP_VZ_BIT))==(CORRECT_BIT | DUMP_VZ_BIT))
        {
            param.glbVzBuf[glbi] = vzi;
        }
    }

    template<>
    __global__ __launch_bounds__(1024) void
    HaloBlockingDynamicL2D3Q27InplaceFirstKernel<false>(const HaloBlockingDynamicL2InplaceParam __grid_constant__ param)
    {
        const idx_t blkx = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t blky = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t blkz = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t blknx = gridDim.x * blockDim.x;
        const idx_t blkny = gridDim.y * blockDim.y;
        const idx_t blki = (blkny * blkz + blky) * blknx + blkx;
        const idx_t glbx = param.offx + blkx;
        const idx_t glby = param.offy + blky;
        const idx_t glbz = param.offz + blkz;
        const idx_t glbi = (param.glbny * glbz + glby) * param.glbnx + glbx;
        const idx_t glbn = param.glbnx * param.glbny * param.glbnz;
        
        const flag_t flagi = param.blkFlagBuf[blki];
        real_t fni[27];

        if((flagi & LOAD_DDF_BIT)!=0)
        {
            const idx_t glbndx = (glbx==0) ? 0 : -1;
            const idx_t glbpdx = (glbx==param.glbnx-1) ? 0 : 1;
            const idx_t glbndy = (glby==0) ? 0 : -param.glbnx;
            const idx_t glbpdy = (glby==param.glbny-1) ? 0 : param.glbnx;
            const idx_t glbndz = (glbz==0) ? 0 : -param.glbnx * param.glbny;
            const idx_t glbpdz = (glbz==param.glbnz-1) ? 0 : param.glbnx * param.glbny;

            //f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
            fni[ 0] = param.glbSrcDDFBuf[ 0*glbn+glbi+glbpdx+glbpdy+glbpdz];
            //f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 1] = param.glbSrcDDFBuf[ 1*glbn+glbi       +glbpdy+glbpdz];
            //f2 (x:+,y:-,z:-) from neighbor (x:-,y:+,z:+)
            fni[ 2] = param.glbSrcDDFBuf[ 2*glbn+glbi+glbndx+glbpdy+glbpdz];
            //f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 3] = param.glbSrcDDFBuf[ 3*glbn+glbi+glbpdx       +glbpdz];
            //f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 4] = param.glbSrcDDFBuf[ 4*glbn+glbi              +glbpdz];
            //f5 (x:+,y:0,z:-) from neighbor (x:-,y:0,z:+)
            fni[ 5] = param.glbSrcDDFBuf[ 5*glbn+glbi+glbndx       +glbpdz];
            //f6 (x:-,y:+,z:-) from neighbor (x:+,y:-,z:+)
            fni[ 6] = param.glbSrcDDFBuf[ 6*glbn+glbi+glbpdx+glbndy+glbpdz];
            //f7 (x:0,y:+,z:-) from neighbor (x:0,y:-,z:+)
            fni[ 7] = param.glbSrcDDFBuf[ 7*glbn+glbi       +glbndy+glbpdz];
            //f8 (x:+,y:+,z:-) from neighbor (x:-,y:-,z:+)
            fni[ 8] = param.glbSrcDDFBuf[ 8*glbn+glbi+glbndx+glbndy+glbpdz];

            //f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
            fni[ 9] = param.glbSrcDDFBuf[ 9*glbn+glbi+glbpdx+glbpdy       ];
            //f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[10] = param.glbSrcDDFBuf[10*glbn+glbi       +glbpdy       ];
            //f11(x:+,y:-,z:0) from neighbor (x:-,y:+,z:0)
            fni[11] = param.glbSrcDDFBuf[11*glbn+glbi+glbndx+glbpdy       ];
            //f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
            fni[12] = param.glbSrcDDFBuf[12*glbn+glbi+glbpdx              ];
            //f13(x:0,y:0,z:0) from neighbor (x:0,y:0,z:0)
            fni[13] = param.glbSrcDDFBuf[13*glbn+glbi                     ];
            //f14(x:+,y:0,z:0) from neighbor (x:-,y:0,z:0)
            fni[14] = param.glbSrcDDFBuf[14*glbn+glbi+glbndx              ];
            //f15(x:-,y:+,z:0) from neighbor (x:+,y:-,z:0)
            fni[15] = param.glbSrcDDFBuf[15*glbn+glbi+glbpdx+glbndy       ];
            //f16(x:0,y:+,z:0) from neighbor (x:0,y:-,z:0)
            fni[16] = param.glbSrcDDFBuf[16*glbn+glbi       +glbndy       ];
            //f17(x:+,y:+,z:0) from neighbor (x:-,y:-,z:0)
            fni[17] = param.glbSrcDDFBuf[17*glbn+glbi+glbndx+glbndy       ];

            //f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:-)
            fni[18] = param.glbSrcDDFBuf[18*glbn+glbi+glbpdx+glbpdy+glbndz];
            //f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:-)
            fni[19] = param.glbSrcDDFBuf[19*glbn+glbi       +glbpdy+glbndz];
            //f20(x:+,y:-,z:+) from neighbor (x:-,y:+,z:-)
            fni[20] = param.glbSrcDDFBuf[20*glbn+glbi+glbndx+glbpdy+glbndz];
            //f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:-)
            fni[21] = param.glbSrcDDFBuf[21*glbn+glbi+glbpdx       +glbndz];
            //f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:-)
            fni[22] = param.glbSrcDDFBuf[22*glbn+glbi              +glbndz];
            //f23(x:+,y:0,z:+) from neighbor (x:-,y:0,z:-)
            fni[23] = param.glbSrcDDFBuf[23*glbn+glbi+glbndx       +glbndz];
            //f24(x:-,y:+,z:+) from neighbor (x:+,y:-,z:-)
            fni[24] = param.glbSrcDDFBuf[24*glbn+glbi+glbpdx+glbndy+glbndz];
            //f25(x:0,y:+,z:+) from neighbor (x:0,y:-,z:-)
            fni[25] = param.glbSrcDDFBuf[25*glbn+glbi       +glbndy+glbndz];
            //f26(x:+,y:+,z:+) from neighbor (x:-,y:-,z:-)
            fni[26] = param.glbSrcDDFBuf[26*glbn+glbi+glbndx+glbndy+glbndz];
        }

        if((flagi & EQU_DDF_BIT)!=0)
        {
            const real_t rhoi   = param.glbRhoBuf[glbi];
            const real_t vxi    = param.glbVxBuf[glbi];
            const real_t vyi    = param.glbVyBuf[glbi];
            const real_t vzi    = param.glbVzBuf[glbi];
            gf::lbm_core::bgk::calcEqu<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
        }

        if((flagi & BOUNCE_BACK_BIT)!=0)
        {
            std::swap(fni[ 0], fni[26]);
            std::swap(fni[ 1], fni[25]);
            std::swap(fni[ 2], fni[24]);
            std::swap(fni[ 3], fni[23]);
            std::swap(fni[ 4], fni[22]);
            std::swap(fni[ 5], fni[21]);
            std::swap(fni[ 6], fni[20]);
            std::swap(fni[ 7], fni[19]);
            std::swap(fni[ 8], fni[18]);
            std::swap(fni[ 9], fni[17]);
            std::swap(fni[10], fni[16]);
            std::swap(fni[11], fni[15]);
            std::swap(fni[12], fni[14]);            
        }

        if((flagi & COLLIDE_BIT)!=0)
        {
            gf::lbm_core::bgk::collision2<27>(param.invTau, std::begin(fni));
        }

        if((flagi & STORE_DDF_BIT)!=0)
        {
            const idx_t blknz = gridDim.z * blockDim.z;
            const idx_t blkn = blknx * blkny * blknz;
            const idx_t blkpdx = (blkx==blknx-1) ? 1-blknx : 1;
            const idx_t blkpdy = (blky==blkny-1) ? (1-blkny)*blknx : blknx;
            const idx_t blkpdz = (blkz==blknz-1) ? (1-blknz)*blknx*blkny : blknx*blkny;
            //EsoTwist Even Store Rules:
            //Don't change direction
            //store f0 (x:-,y:-,z:-) to current lattice
            param.blkDDFBuf[ 0*blkn+blki                     ] = fni[ 0];
            //store f1 (x:0,y:-,z:-) to current lattice
            param.blkDDFBuf[ 1*blkn+blki                     ] = fni[ 1];
            //store f2 (x:+,y:-,z:-) to neighbor (x:+,y:0,z:0)
            param.blkDDFBuf[ 2*blkn+blki+blkpdx              ] = fni[ 2];
            //store f3 (x:-,y:0,z:-) to current lattice
            param.blkDDFBuf[ 3*blkn+blki                     ] = fni[ 3];
            //store f4 (x:0,y:0,z:-) to current lattice
            param.blkDDFBuf[ 4*blkn+blki                     ] = fni[ 4];
            //store f5 (x:+,y:0,z:-) to neighbor (x:+,y:0,z:0)
            param.blkDDFBuf[ 5*blkn+blki+blkpdx              ] = fni[ 5];
            //store f6 (x:-,y:+,z:-) to neighbor (x:0,y:+,z:0)
            param.blkDDFBuf[ 6*blkn+blki       +blkpdy       ] = fni[ 6];
            //store f7 (x:0,y:+,z:-) to neighbor (x:0,y:+,z:0)
            param.blkDDFBuf[ 7*blkn+blki       +blkpdy       ] = fni[ 7];
            //store f8 (x:+,y:+,z:-) to neighbor (x:+,y:+,z:0)
            param.blkDDFBuf[ 8*blkn+blki+blkpdx+blkpdy       ] = fni[ 8];

            //store f9 (x:-,y:-,z:0) to current lattice
            param.blkDDFBuf[ 9*blkn+blki                     ] = fni[ 9];
            //store f10(x:0,y:-,z:0) to current lattice
            param.blkDDFBuf[10*blkn+blki                     ] = fni[10];
            //store f11(x:+,y:-,z:0) to neighbor (x:+,y:0,z:0)
            param.blkDDFBuf[11*blkn+blki+blkpdx              ] = fni[11];
            //store f12(x:-,y:0,z:0) to current lattice
            param.blkDDFBuf[12*blkn+blki                     ] = fni[12];
            //store f13(x:0,y:0,z:0) to current lattice
            param.blkDDFBuf[13*blkn+blki                     ] = fni[13];
            //store f14(x:+,y:0,z:0) to neighbor (x:+,y:0,z:0)
            param.blkDDFBuf[14*blkn+blki+blkpdx              ] = fni[14];
            //store f15(x:-,y:+,z:0) to neighbor (x:0,y:+,z:0)
            param.blkDDFBuf[15*blkn+blki       +blkpdy       ] = fni[15];
            //store f16(x:0,y:+,z:0) to neighbor (x:0,y:+,z:0)
            param.blkDDFBuf[16*blkn+blki       +blkpdy       ] = fni[16];
            //store f17(x:+,y:+,z:0) to neighbor (x:+,y:+,z:0)
            param.blkDDFBuf[17*blkn+blki+blkpdx+blkpdy       ] = fni[17];

            //store f18(x:-,y:-,z:+) to neighbor (x:0,y:0,z:+)
            param.blkDDFBuf[18*blkn+blki              +blkpdz] = fni[18];
            //store f19(x:0,y:-,z:+) to neighbor (x:0,y:0,z:+)
            param.blkDDFBuf[19*blkn+blki              +blkpdz] = fni[19];
            //store f20(x:+,y:-,z:+) to neighbor (x:+,y:0,z:+)
            param.blkDDFBuf[20*blkn+blki+blkpdx       +blkpdz] = fni[20];
            //store f21(x:-,y:0,z:+) to neighbor (x:0,y:0,z:+)
            param.blkDDFBuf[21*blkn+blki              +blkpdz] = fni[21];
            //store f22(x:0,y:0,z:+) to neighbor (x:0,y:0,z:+)
            param.blkDDFBuf[22*blkn+blki              +blkpdz] = fni[22];
            //store f23(x:+,y:0,z:+) to neighbor (x:+,y:0,z:+)
            param.blkDDFBuf[23*blkn+blki+blkpdx       +blkpdz] = fni[23];
            //store f24(x:-,y:+,z:+) to neighbor (x:0,y:+,z:+)
            param.blkDDFBuf[24*blkn+blki       +blkpdy+blkpdz] = fni[24];
            //store f25(x:0,y:+,z:+) to neighbor (x:0,y:+,z:+)
            param.blkDDFBuf[25*blkn+blki       +blkpdy+blkpdz] = fni[25];
            //store f26(x:+,y:+,z:+) to neighbor (x:+,y:+,z:+)
            param.blkDDFBuf[26*blkn+blki+blkpdx+blkpdy+blkpdz] = fni[26];
        }
    }

    template<>
    __global__ __launch_bounds__(1024) void
    HaloBlockingDynamicL2D3Q27InplaceFirstKernel<true>(const HaloBlockingDynamicL2InplaceParam __grid_constant__ param)
    {
        const idx_t blkx = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t blky = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t blkz = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t blknx = gridDim.x * blockDim.x;
        const idx_t blkny = gridDim.y * blockDim.y;
        const idx_t blki = (blkny * blkz + blky) * blknx + blkx;
        const idx_t glbx = param.offx + blkx;
        const idx_t glby = param.offy + blky;
        const idx_t glbz = param.offz + blkz;
        const idx_t glbi = (param.glbny * glbz + glby) * param.glbnx + glbx;
        const idx_t glbn = param.glbnx * param.glbny * param.glbnz;
        
        const flag_t flagi = param.blkFlagBuf[blki];
        real_t fni[27];

        if((flagi & LOAD_DDF_BIT)!=0)
        {
            const idx_t glbndx = (glbx==0) ? 0 : -1;
            const idx_t glbpdx = (glbx==param.glbnx-1) ? 0 : 1;
            const idx_t glbndy = (glby==0) ? 0 : -param.glbnx;
            const idx_t glbpdy = (glby==param.glbny-1) ? 0 : param.glbnx;
            const idx_t glbndz = (glbz==0) ? 0 : -param.glbnx * param.glbny;
            const idx_t glbpdz = (glbz==param.glbnz-1) ? 0 : param.glbnx * param.glbny;

            //f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
            fni[ 0] = param.glbSrcDDFBuf[ 0*glbn+glbi+glbpdx+glbpdy+glbpdz];
            //f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 1] = param.glbSrcDDFBuf[ 1*glbn+glbi       +glbpdy+glbpdz];
            //f2 (x:+,y:-,z:-) from neighbor (x:-,y:+,z:+)
            fni[ 2] = param.glbSrcDDFBuf[ 2*glbn+glbi+glbndx+glbpdy+glbpdz];
            //f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 3] = param.glbSrcDDFBuf[ 3*glbn+glbi+glbpdx       +glbpdz];
            //f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 4] = param.glbSrcDDFBuf[ 4*glbn+glbi              +glbpdz];
            //f5 (x:+,y:0,z:-) from neighbor (x:-,y:0,z:+)
            fni[ 5] = param.glbSrcDDFBuf[ 5*glbn+glbi+glbndx       +glbpdz];
            //f6 (x:-,y:+,z:-) from neighbor (x:+,y:-,z:+)
            fni[ 6] = param.glbSrcDDFBuf[ 6*glbn+glbi+glbpdx+glbndy+glbpdz];
            //f7 (x:0,y:+,z:-) from neighbor (x:0,y:-,z:+)
            fni[ 7] = param.glbSrcDDFBuf[ 7*glbn+glbi       +glbndy+glbpdz];
            //f8 (x:+,y:+,z:-) from neighbor (x:-,y:-,z:+)
            fni[ 8] = param.glbSrcDDFBuf[ 8*glbn+glbi+glbndx+glbndy+glbpdz];

            //f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
            fni[ 9] = param.glbSrcDDFBuf[ 9*glbn+glbi+glbpdx+glbpdy       ];
            //f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[10] = param.glbSrcDDFBuf[10*glbn+glbi       +glbpdy       ];
            //f11(x:+,y:-,z:0) from neighbor (x:-,y:+,z:0)
            fni[11] = param.glbSrcDDFBuf[11*glbn+glbi+glbndx+glbpdy       ];
            //f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
            fni[12] = param.glbSrcDDFBuf[12*glbn+glbi+glbpdx              ];
            //f13(x:0,y:0,z:0) from neighbor (x:0,y:0,z:0)
            fni[13] = param.glbSrcDDFBuf[13*glbn+glbi                     ];
            //f14(x:+,y:0,z:0) from neighbor (x:-,y:0,z:0)
            fni[14] = param.glbSrcDDFBuf[14*glbn+glbi+glbndx              ];
            //f15(x:-,y:+,z:0) from neighbor (x:+,y:-,z:0)
            fni[15] = param.glbSrcDDFBuf[15*glbn+glbi+glbpdx+glbndy       ];
            //f16(x:0,y:+,z:0) from neighbor (x:0,y:-,z:0)
            fni[16] = param.glbSrcDDFBuf[16*glbn+glbi       +glbndy       ];
            //f17(x:+,y:+,z:0) from neighbor (x:-,y:-,z:0)
            fni[17] = param.glbSrcDDFBuf[17*glbn+glbi+glbndx+glbndy       ];

            //f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:-)
            fni[18] = param.glbSrcDDFBuf[18*glbn+glbi+glbpdx+glbpdy+glbndz];
            //f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:-)
            fni[19] = param.glbSrcDDFBuf[19*glbn+glbi       +glbpdy+glbndz];
            //f20(x:+,y:-,z:+) from neighbor (x:-,y:+,z:-)
            fni[20] = param.glbSrcDDFBuf[20*glbn+glbi+glbndx+glbpdy+glbndz];
            //f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:-)
            fni[21] = param.glbSrcDDFBuf[21*glbn+glbi+glbpdx       +glbndz];
            //f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:-)
            fni[22] = param.glbSrcDDFBuf[22*glbn+glbi              +glbndz];
            //f23(x:+,y:0,z:+) from neighbor (x:-,y:0,z:-)
            fni[23] = param.glbSrcDDFBuf[23*glbn+glbi+glbndx       +glbndz];
            //f24(x:-,y:+,z:+) from neighbor (x:+,y:-,z:-)
            fni[24] = param.glbSrcDDFBuf[24*glbn+glbi+glbpdx+glbndy+glbndz];
            //f25(x:0,y:+,z:+) from neighbor (x:0,y:-,z:-)
            fni[25] = param.glbSrcDDFBuf[25*glbn+glbi       +glbndy+glbndz];
            //f26(x:+,y:+,z:+) from neighbor (x:-,y:-,z:-)
            fni[26] = param.glbSrcDDFBuf[26*glbn+glbi+glbndx+glbndy+glbndz];
        }

        if((flagi & EQU_DDF_BIT)!=0)
        {
            const real_t rhoi   = param.glbRhoBuf[glbi];
            const real_t vxi    = param.glbVxBuf[glbi];
            const real_t vyi    = param.glbVyBuf[glbi];
            const real_t vzi    = param.glbVzBuf[glbi];
            gf::lbm_core::bgk::calcEqu<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
        }

        if((flagi & BOUNCE_BACK_BIT)!=0)
        {
            std::swap(fni[ 0], fni[26]);
            std::swap(fni[ 1], fni[25]);
            std::swap(fni[ 2], fni[24]);
            std::swap(fni[ 3], fni[23]);
            std::swap(fni[ 4], fni[22]);
            std::swap(fni[ 5], fni[21]);
            std::swap(fni[ 6], fni[20]);
            std::swap(fni[ 7], fni[19]);
            std::swap(fni[ 8], fni[18]);
            std::swap(fni[ 9], fni[17]);
            std::swap(fni[10], fni[16]);
            std::swap(fni[11], fni[15]);
            std::swap(fni[12], fni[14]);            
        }

        if((flagi & COLLIDE_BIT)!=0)
        {
            gf::lbm_core::bgk::collision2<27>(param.invTau, std::begin(fni));
        }

        if((flagi & STORE_DDF_BIT)!=0)
        {
            param.glbDstDDFBuf[ 0*glbn+glbi] = fni[ 0];
            param.glbDstDDFBuf[ 1*glbn+glbi] = fni[ 1];
            param.glbDstDDFBuf[ 2*glbn+glbi] = fni[ 2];
            param.glbDstDDFBuf[ 3*glbn+glbi] = fni[ 3];
            param.glbDstDDFBuf[ 4*glbn+glbi] = fni[ 4];
            param.glbDstDDFBuf[ 5*glbn+glbi] = fni[ 5];
            param.glbDstDDFBuf[ 6*glbn+glbi] = fni[ 6];
            param.glbDstDDFBuf[ 7*glbn+glbi] = fni[ 7];
            param.glbDstDDFBuf[ 8*glbn+glbi] = fni[ 8];
            param.glbDstDDFBuf[ 9*glbn+glbi] = fni[ 9];
            param.glbDstDDFBuf[10*glbn+glbi] = fni[10];
            param.glbDstDDFBuf[11*glbn+glbi] = fni[11];
            param.glbDstDDFBuf[12*glbn+glbi] = fni[12];
            param.glbDstDDFBuf[13*glbn+glbi] = fni[13];
            param.glbDstDDFBuf[14*glbn+glbi] = fni[14];
            param.glbDstDDFBuf[15*glbn+glbi] = fni[15];
            param.glbDstDDFBuf[16*glbn+glbi] = fni[16];
            param.glbDstDDFBuf[17*glbn+glbi] = fni[17];
            param.glbDstDDFBuf[18*glbn+glbi] = fni[18];
            param.glbDstDDFBuf[19*glbn+glbi] = fni[19];
            param.glbDstDDFBuf[20*glbn+glbi] = fni[20];
            param.glbDstDDFBuf[21*glbn+glbi] = fni[21];
            param.glbDstDDFBuf[22*glbn+glbi] = fni[22];
            param.glbDstDDFBuf[23*glbn+glbi] = fni[23];
            param.glbDstDDFBuf[24*glbn+glbi] = fni[24];
            param.glbDstDDFBuf[25*glbn+glbi] = fni[25];
            param.glbDstDDFBuf[26*glbn+glbi] = fni[26];
        }
    }

    template<>
    __global__ __launch_bounds__(1024) void 
    HaloBlockingDynamicL2D3Q27InplaceMiddleKernel<false>(const HaloBlockingDynamicL2InplaceParam __grid_constant__ param)
    {
        const idx_t blkx = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t blky = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t blkz = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t blknx = gridDim.x * blockDim.x;
        const idx_t blkny = gridDim.y * blockDim.y;
        const idx_t blknz = gridDim.z * blockDim.z;
        const idx_t blki = (blkny * blkz + blky) * blknx + blkx;
        const idx_t blkn = blknx * blkny * blknz;
        const idx_t glbx = param.offx + blkx;
        const idx_t glby = param.offy + blky;
        const idx_t glbz = param.offz + blkz;
        const idx_t glbi = (param.glbny * glbz + glby) * param.glbnx + glbx;
        const idx_t glbn = param.glbnx * param.glbny * param.glbnz;
        const idx_t blkpdx = (blkx==blknx-1) ? 1-blknx : 1;
        const idx_t blkpdy = (blky==blkny-1) ? (1-blkny)*blknx : blknx;
        const idx_t blkpdz = (blkz==blknz-1) ? (1-blknz)*blknx*blkny : blknx*blkny;

        const flag_t flagi = param.blkFlagBuf[blki];
        real_t fni[27];

        if((flagi & LOAD_DDF_BIT)!=0)
        {   
            //EsoTwist Odd Load Rules:
            //Don' t change direction
            //load f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
            fni[ 0] = param.blkDDFBuf[ 0*blkn+blki+blkpdx+blkpdy+blkpdz];
            //load f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 1] = param.blkDDFBuf[ 1*blkn+blki       +blkpdy+blkpdz];
            //load f2 (x:+,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 2] = param.blkDDFBuf[ 2*blkn+blki       +blkpdy+blkpdz];
            //load f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 3] = param.blkDDFBuf[ 3*blkn+blki+blkpdx       +blkpdz];
            //load f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 4] = param.blkDDFBuf[ 4*blkn+blki              +blkpdz];
            //load f5 (x:+,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 5] = param.blkDDFBuf[ 5*blkn+blki              +blkpdz];
            //load f6 (x:-,y:+,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 6] = param.blkDDFBuf[ 6*blkn+blki+blkpdx       +blkpdz];
            //load f7 (x:0,y:+,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 7] = param.blkDDFBuf[ 7*blkn+blki              +blkpdz];
            //load f8 (x:+,y:+,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 8] = param.blkDDFBuf[ 8*blkn+blki              +blkpdz];

            //load f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
            fni[ 9] = param.blkDDFBuf[ 9*blkn+blki+blkpdx+blkpdy       ];
            //load f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[10] = param.blkDDFBuf[10*blkn+blki       +blkpdy       ];
            //load f11(x:+,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[11] = param.blkDDFBuf[11*blkn+blki       +blkpdy       ];
            //load f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
            fni[12] = param.blkDDFBuf[12*blkn+blki+blkpdx              ];
            //load f13(x:0,y:0,z:0) from current lattice
            fni[13] = param.blkDDFBuf[13*blkn+blki                     ];
            //load f14(x:+,y:0,z:0) from current lattice
            fni[14] = param.blkDDFBuf[14*blkn+blki                     ];
            //load f15(x:-,y:+,z:0) from neighbor (x:+,y:0,z:0)
            fni[15] = param.blkDDFBuf[15*blkn+blki+blkpdx              ];
            //load f16(x:0,y:+,z:0) from current lattice
            fni[16] = param.blkDDFBuf[16*blkn+blki                     ];
            //load f17(x:+,y:+,z:0) from current lattice
            fni[17] = param.blkDDFBuf[17*blkn+blki                     ];

            //load f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:0)
            fni[18] = param.blkDDFBuf[18*blkn+blki+blkpdx+blkpdy       ];
            //load f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:0)
            fni[19] = param.blkDDFBuf[19*blkn+blki       +blkpdy       ];
            //load f20(x:+,y:-,z:+) from neighbor (x:0,y:+,z:0)
            fni[20] = param.blkDDFBuf[20*blkn+blki       +blkpdy       ];
            //load f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:0)
            fni[21] = param.blkDDFBuf[21*blkn+blki+blkpdx              ];
            //load f22(x:0,y:0,z:+) from current lattice
            fni[22] = param.blkDDFBuf[22*blkn+blki                     ];
            //load f23(x:+,y:0,z:+) from current lattice
            fni[23] = param.blkDDFBuf[23*blkn+blki                     ];
            //load f24(x:-,y:+,z:+) from neighbor (x:+,y:0,z:0)
            fni[24] = param.blkDDFBuf[24*blkn+blki+blkpdx              ];
            //load f25(x:0,y:+,z:+) from current lattice
            fni[25] = param.blkDDFBuf[25*blkn+blki                     ];
            //load f26(x:+,y:+,z:+) from current lattice
            fni[26] = param.blkDDFBuf[26*blkn+blki                     ];
        }

        if((flagi & EQU_DDF_BIT)!=0)
        {
            const real_t rhoi   = param.glbRhoBuf[glbi];
            const real_t vxi    = param.glbVxBuf[glbi];
            const real_t vyi    = param.glbVyBuf[glbi];
            const real_t vzi    = param.glbVzBuf[glbi];
            gf::lbm_core::bgk::calcEqu<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
        }

        if((flagi & BOUNCE_BACK_BIT)!=0)
        {
            std::swap(fni[ 0], fni[26]);
            std::swap(fni[ 1], fni[25]);
            std::swap(fni[ 2], fni[24]);
            std::swap(fni[ 3], fni[23]);
            std::swap(fni[ 4], fni[22]);
            std::swap(fni[ 5], fni[21]);
            std::swap(fni[ 6], fni[20]);
            std::swap(fni[ 7], fni[19]);
            std::swap(fni[ 8], fni[18]);
            std::swap(fni[ 9], fni[17]);
            std::swap(fni[10], fni[16]);
            std::swap(fni[11], fni[15]);
            std::swap(fni[12], fni[14]);            
        }

        if((flagi & COLLIDE_BIT)!=0)
        {
            gf::lbm_core::bgk::collision2<27>(param.invTau, std::begin(fni));
        }

        if((flagi & STORE_DDF_BIT)!=0)
        {
            //EsoTwist Odd Store Rules:
            //Reverse all direction
            //store f0 (x:-,y:-,z:-) to current lattice
            param.blkDDFBuf[26*blkn+blki                     ] = fni[ 0];
            //store f1 (x:0,y:-,z:-) to current lattice
            param.blkDDFBuf[25*blkn+blki                     ] = fni[ 1];
            //store f2 (x:+,y:-,z:-) to neighbor (x:+,y:0,z:0)
            param.blkDDFBuf[24*blkn+blki+blkpdx              ] = fni[ 2];
            //store f3 (x:-,y:0,z:-) to current lattice
            param.blkDDFBuf[23*blkn+blki                     ] = fni[ 3];
            //store f4 (x:0,y:0,z:-) to current lattice
            param.blkDDFBuf[22*blkn+blki                     ] = fni[ 4];
            //store f5 (x:+,y:0,z:-) to neighbor (x:+,y:0,z:0)
            param.blkDDFBuf[21*blkn+blki+blkpdx              ] = fni[ 5];
            //store f6 (x:-,y:+,z:-) to neighbor (x:0,y:+,z:0)
            param.blkDDFBuf[20*blkn+blki       +blkpdy       ] = fni[ 6];
            //store f7 (x:0,y:+,z:-) to neighbor (x:0,y:+,z:0)
            param.blkDDFBuf[19*blkn+blki       +blkpdy       ] = fni[ 7];
            //store f8 (x:+,y:+,z:-) to neighbor (x:+,y:+,z:0)
            param.blkDDFBuf[18*blkn+blki+blkpdx+blkpdy       ] = fni[ 8];

            //store f9 (x:-,y:-,z:0) to current lattice
            param.blkDDFBuf[17*blkn+blki                     ] = fni[ 9];
            //store f10(x:0,y:-,z:0) to current lattice
            param.blkDDFBuf[16*blkn+blki                     ] = fni[10];
            //store f11(x:+,y:-,z:0) to neighbor (x:+,y:0,z:0)
            param.blkDDFBuf[15*blkn+blki+blkpdx              ] = fni[11];
            //store f12(x:-,y:0,z:0) to current lattice
            param.blkDDFBuf[14*blkn+blki                     ] = fni[12];
            //store f13(x:0,y:0,z:0) to current lattice
            param.blkDDFBuf[13*blkn+blki                     ] = fni[13];
            //store f14(x:+,y:0,z:0) to neighbor (x:+,y:0,z:0)
            param.blkDDFBuf[12*blkn+blki+blkpdx              ] = fni[14];
            //store f15(x:-,y:+,z:0) to neighbor (x:0,y:+,z:0)
            param.blkDDFBuf[11*blkn+blki       +blkpdy       ] = fni[15];
            //store f16(x:0,y:+,z:0) to neighbor (x:0,y:+,z:0)
            param.blkDDFBuf[10*blkn+blki       +blkpdy       ] = fni[16];
            //store f17(x:+,y:+,z:0) to neighbor (x:+,y:+,z:0)
            param.blkDDFBuf[ 9*blkn+blki+blkpdx+blkpdy       ] = fni[17];

            //store f18(x:-,y:-,z:+) to neighbor (x:0,y:0,z:+)
            param.blkDDFBuf[ 8*blkn+blki              +blkpdz] = fni[18];
            //store f19(x:0,y:-,z:+) to neighbor (x:0,y:0,z:+)
            param.blkDDFBuf[ 7*blkn+blki              +blkpdz] = fni[19];
            //store f20(x:+,y:-,z:+) to neighbor (x:+,y:0,z:+)
            param.blkDDFBuf[ 6*blkn+blki+blkpdx       +blkpdz] = fni[20];
            //store f21(x:-,y:0,z:+) to neighbor (x:0,y:0,z:+)
            param.blkDDFBuf[ 5*blkn+blki              +blkpdz] = fni[21];
            //store f22(x:0,y:0,z:+) to neighbor (x:0,y:0,z:+)
            param.blkDDFBuf[ 4*blkn+blki              +blkpdz] = fni[22];
            //store f23(x:+,y:0,z:+) to neighbor (x:+,y:0,z:+)
            param.blkDDFBuf[ 3*blkn+blki+blkpdx       +blkpdz] = fni[23];
            //store f24(x:-,y:+,z:+) to neighbor (x:0,y:+,z:+)
            param.blkDDFBuf[ 2*blkn+blki       +blkpdy+blkpdz] = fni[24];
            //store f25(x:0,y:+,z:+) to neighbor (x:0,y:+,z:+)
            param.blkDDFBuf[ 1*blkn+blki       +blkpdy+blkpdz] = fni[25];
            //store f26(x:+,y:+,z:+) to neighbor (x:+,y:+,z:+)
            param.blkDDFBuf[ 0*blkn+blki+blkpdx+blkpdy+blkpdz] = fni[26];
        }
    }

    template<>
    __global__ __launch_bounds__(1024) void
    HaloBlockingDynamicL2D3Q27InplaceMiddleKernel<true>(const HaloBlockingDynamicL2InplaceParam __grid_constant__ param)
    {
        const idx_t blkx = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t blky = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t blkz = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t blknx = gridDim.x * blockDim.x;
        const idx_t blkny = gridDim.y * blockDim.y;
        const idx_t blknz = gridDim.z * blockDim.z;
        const idx_t blki = (blkny * blkz + blky) * blknx + blkx;
        const idx_t blkn = blknx * blkny * blknz;
        const idx_t glbx = param.offx + blkx;
        const idx_t glby = param.offy + blky;
        const idx_t glbz = param.offz + blkz;
        const idx_t glbi = (param.glbny * glbz + glby) * param.glbnx + glbx;
        const idx_t glbn = param.glbnx * param.glbny * param.glbnz;
        const idx_t blkpdx = (blkx==blknx-1) ? 1-blknx : 1;
        const idx_t blkpdy = (blky==blkny-1) ? (1-blkny)*blknx : blknx;
        const idx_t blkpdz = (blkz==blknz-1) ? (1-blknz)*blknx*blkny : blknx*blkny;

        const flag_t flagi = param.blkFlagBuf[blki];
        real_t fni[27];

        if((flagi & LOAD_DDF_BIT)!=0)
        {
            //EsoTwist Even Load Rules:
            //Reverse all direction
            //load f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
            fni[ 0] = param.blkDDFBuf[26*blkn+blki+blkpdx+blkpdy+blkpdz];
            //load f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 1] = param.blkDDFBuf[25*blkn+blki       +blkpdy+blkpdz];
            //loaf f2 (x:+,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 2] = param.blkDDFBuf[24*blkn+blki       +blkpdy+blkpdz];
            //load f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 3] = param.blkDDFBuf[23*blkn+blki+blkpdx       +blkpdz];
            //load f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 4] = param.blkDDFBuf[22*blkn+blki              +blkpdz];
            //load f5 (x:+,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 5] = param.blkDDFBuf[21*blkn+blki              +blkpdz];
            //load f6 (x:-,y:+,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 6] = param.blkDDFBuf[20*blkn+blki+blkpdx       +blkpdz];
            //load f7 (x:0,y:+,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 7] = param.blkDDFBuf[19*blkn+blki              +blkpdz];
            //load f8 (x:+,y:+,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 8] = param.blkDDFBuf[18*blkn+blki              +blkpdz];

            //load f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
            fni[ 9] = param.blkDDFBuf[17*blkn+blki+blkpdx+blkpdy       ];
            //load f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[10] = param.blkDDFBuf[16*blkn+blki       +blkpdy       ];
            //load f11(x:+,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[11] = param.blkDDFBuf[15*blkn+blki       +blkpdy       ];
            //load f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
            fni[12] = param.blkDDFBuf[14*blkn+blki+blkpdx              ];
            //load f13(x:0,y:0,z:0) from current lattice
            fni[13] = param.blkDDFBuf[13*blkn+blki                     ];
            //load f14(x:+,y:0,z:0) from current lattice
            fni[14] = param.blkDDFBuf[12*blkn+blki                     ];
            //load f15(x:-,y:+,z:0) from neighbor (x:+,y:0,z:0)
            fni[15] = param.blkDDFBuf[11*blkn+blki+blkpdx              ];
            //load f16(x:0,y:+,z:0) from current lattice
            fni[16] = param.blkDDFBuf[10*blkn+blki                     ];
            //load f17(x:+,y:+,z:0) from current lattice
            fni[17] = param.blkDDFBuf[ 9*blkn+blki                     ];

            //load f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:0)
            fni[18] = param.blkDDFBuf[ 8*blkn+blki+blkpdx+blkpdy       ];
            //load f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:0)
            fni[19] = param.blkDDFBuf[ 7*blkn+blki       +blkpdy       ];
            //load f20(x:+,y:-,z:+) from neighbor (x:0,y:+,z:0)
            fni[20] = param.blkDDFBuf[ 6*blkn+blki       +blkpdy       ];
            //load f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:0)
            fni[21] = param.blkDDFBuf[ 5*blkn+blki+blkpdx              ];
            //load f22(x:0,y:0,z:+) from current lattice
            fni[22] = param.blkDDFBuf[ 4*blkn+blki                     ];
            //load f23(x:+,y:0,z:+) from current lattice
            fni[23] = param.blkDDFBuf[ 3*blkn+blki                     ];
            //load f24(x:-,y:+,z:+) from neighbor (x:+,y:0,z:0)
            fni[24] = param.blkDDFBuf[ 2*blkn+blki+blkpdx              ];
            //load f25(x:0,y:+,z:+) from current lattice
            fni[25] = param.blkDDFBuf[ 1*blkn+blki                     ];
            //load f26(x:+,y:+,z:+) from current lattice
            fni[26] = param.blkDDFBuf[ 0*blkn+blki                     ];
        }

        if((flagi & EQU_DDF_BIT)!=0)
        {
            const real_t rhoi   = param.glbRhoBuf[glbi];
            const real_t vxi    = param.glbVxBuf[glbi];
            const real_t vyi    = param.glbVyBuf[glbi];
            const real_t vzi    = param.glbVzBuf[glbi];
            gf::lbm_core::bgk::calcEqu<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
        }

        if((flagi & BOUNCE_BACK_BIT)!=0)
        {
            std::swap(fni[ 0], fni[26]);
            std::swap(fni[ 1], fni[25]);
            std::swap(fni[ 2], fni[24]);
            std::swap(fni[ 3], fni[23]);
            std::swap(fni[ 4], fni[22]);
            std::swap(fni[ 5], fni[21]);
            std::swap(fni[ 6], fni[20]);
            std::swap(fni[ 7], fni[19]);
            std::swap(fni[ 8], fni[18]);
            std::swap(fni[ 9], fni[17]);
            std::swap(fni[10], fni[16]);
            std::swap(fni[11], fni[15]);
            std::swap(fni[12], fni[14]);            
        }

        if((flagi & COLLIDE_BIT)!=0)
        {
            gf::lbm_core::bgk::collision2<27>(param.invTau, std::begin(fni));
        }

        if((flagi & STORE_DDF_BIT)!=0)
        {
            //EsoTwist Even Store Rules:
            //Don't change direction
            //store f0 (x:-,y:-,z:-) to current lattice
            param.blkDDFBuf[ 0*blkn+blki                     ] = fni[ 0];
            //store f1 (x:0,y:-,z:-) to current lattice
            param.blkDDFBuf[ 1*blkn+blki                     ] = fni[ 1];
            //store f2 (x:+,y:-,z:-) to neighbor (x:+,y:0,z:0)
            param.blkDDFBuf[ 2*blkn+blki+blkpdx              ] = fni[ 2];
            //store f3 (x:-,y:0,z:-) to current lattice
            param.blkDDFBuf[ 3*blkn+blki                     ] = fni[ 3];
            //store f4 (x:0,y:0,z:-) to current lattice
            param.blkDDFBuf[ 4*blkn+blki                     ] = fni[ 4];
            //store f5 (x:+,y:0,z:-) to neighbor (x:+,y:0,z:0)
            param.blkDDFBuf[ 5*blkn+blki+blkpdx              ] = fni[ 5];
            //store f6 (x:-,y:+,z:-) to neighbor (x:0,y:+,z:0)
            param.blkDDFBuf[ 6*blkn+blki       +blkpdy       ] = fni[ 6];
            //store f7 (x:0,y:+,z:-) to neighbor (x:0,y:+,z:0)
            param.blkDDFBuf[ 7*blkn+blki       +blkpdy       ] = fni[ 7];
            //store f8 (x:+,y:+,z:-) to neighbor (x:+,y:+,z:0)
            param.blkDDFBuf[ 8*blkn+blki+blkpdx+blkpdy       ] = fni[ 8];

            //store f9 (x:-,y:-,z:0) to current lattice
            param.blkDDFBuf[ 9*blkn+blki                     ] = fni[ 9];
            //store f10(x:0,y:-,z:0) to current lattice
            param.blkDDFBuf[10*blkn+blki                     ] = fni[10];
            //store f11(x:+,y:-,z:0) to neighbor (x:+,y:0,z:0)
            param.blkDDFBuf[11*blkn+blki+blkpdx              ] = fni[11];
            //store f12(x:-,y:0,z:0) to current lattice
            param.blkDDFBuf[12*blkn+blki                     ] = fni[12];
            //store f13(x:0,y:0,z:0) to current lattice
            param.blkDDFBuf[13*blkn+blki                     ] = fni[13];
            //store f14(x:+,y:0,z:0) to neighbor (x:+,y:0,z:0)
            param.blkDDFBuf[14*blkn+blki+blkpdx              ] = fni[14];
            //store f15(x:-,y:+,z:0) to neighbor (x:0,y:+,z:0)
            param.blkDDFBuf[15*blkn+blki       +blkpdy       ] = fni[15];
            //store f16(x:0,y:+,z:0) to neighbor (x:0,y:+,z:0)
            param.blkDDFBuf[16*blkn+blki       +blkpdy       ] = fni[16];
            //store f17(x:+,z:+,z:0) to neighbor (x:+,y:+,z:0)
            param.blkDDFBuf[17*blkn+blki+blkpdx+blkpdy       ] = fni[17];

            //store f18(x:-,y:-,z:+) to neighbor (x:0,y:+,z:+)
            param.blkDDFBuf[18*blkn+blki       +blkpdy+blkpdz] = fni[18];
            //store f19(x:0,y:-,z:+) to neighbor (x:0,y:0,z:+)
            param.blkDDFBuf[19*blkn+blki              +blkpdz] = fni[19];
            //store f20(x:+,y:-,z:+) to neighbor (x:+,y:0,z:+)
            param.blkDDFBuf[20*blkn+blki+blkpdx       +blkpdz] = fni[20];
            //store f21(x:-,y:0,z:+) to neighbor (x:0,y:0,z:+)
            param.blkDDFBuf[21*blkn+blki              +blkpdz] = fni[21];
            //store f22(x:0,y:0,z:+) to neighbor (x:0,y:0,z:+)
            param.blkDDFBuf[22*blkn+blki              +blkpdz] = fni[22];
            //store f23(x:+,y:0,z:+) to neighbor (x:+,y:0,z:+)
            param.blkDDFBuf[23*blkn+blki+blkpdx       +blkpdz] = fni[23];
            //store f24(x:-,y:+,z:+) to neighbor (x:0,y:+,z:+)
            param.blkDDFBuf[24*blkn+blki       +blkpdy+blkpdz] = fni[24];
            //store f25(x:0,y:+,z:+) to neighbor (x:0,y:+,z:+)
            param.blkDDFBuf[25*blkn+blki       +blkpdy+blkpdz] = fni[25];
            //store f26(x:+,y:+,z:+) to neighbor (x:+,y:+,z:+)
            param.blkDDFBuf[26*blkn+blki+blkpdx+blkpdy+blkpdz] = fni[26];
        }
    }

    template<>
    __global__ __launch_bounds__(1024) void 
    HaloBlockingDynamicL2D3Q27InplaceLastKernel<false>(const HaloBlockingDynamicL2InplaceParam __grid_constant__ param)
    {
        const idx_t blkx = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t blky = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t blkz = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t blknx = gridDim.x * blockDim.x;
        const idx_t blkny = gridDim.y * blockDim.y;
        const idx_t blknz = gridDim.z * blockDim.z;
        const idx_t blkn = blknx * blkny * blknz;
        const idx_t blki = (blkny * blkz + blky) * blknx + blkx;
        const idx_t glbx = param.offx + blkx;
        const idx_t glby = param.offy + blky;
        const idx_t glbz = param.offz + blkz;
        const idx_t glbi = (param.glbny * glbz + glby) * param.glbnx + glbx;
        const idx_t glbn = param.glbnx * param.glbny * param.glbnz;
        
        const flag_t flagi = param.blkFlagBuf[blki];
        real_t fni[27];

        if((flagi & LOAD_DDF_BIT)!=0)
        {
            const idx_t blkpdx = (blkx==blknx-1) ? 1-blknx : 1;
            const idx_t blkpdy = (blky==blkny-1) ? (1-blkny)*blknx : blknx;
            const idx_t blkpdz = (blkz==blknz-1) ? (1-blknz)*blknx*blkny : blknx*blkny;
            //EsoTwist Odd Load Rules:
            //Don' t change direction
            //load f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
            fni[ 0] = param.blkDDFBuf[ 0*blkn+blki+blkpdx+blkpdy+blkpdz];
            //load f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 1] = param.blkDDFBuf[ 1*blkn+blki       +blkpdy+blkpdz];
            //load f2 (x:+,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 2] = param.blkDDFBuf[ 2*blkn+blki       +blkpdy+blkpdz];
            //load f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 3] = param.blkDDFBuf[ 3*blkn+blki+blkpdx       +blkpdz];
            //load f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 4] = param.blkDDFBuf[ 4*blkn+blki              +blkpdz];
            //load f5 (x:+,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 5] = param.blkDDFBuf[ 5*blkn+blki              +blkpdz];
            //load f6 (x:-,y:+,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 6] = param.blkDDFBuf[ 6*blkn+blki+blkpdx       +blkpdz];
            //load f7 (x:0,y:+,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 7] = param.blkDDFBuf[ 7*blkn+blki              +blkpdz];
            //load f8 (x:+,y:+,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 8] = param.blkDDFBuf[ 8*blkn+blki              +blkpdz];

            //load f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
            fni[ 9] = param.blkDDFBuf[ 9*blkn+blki+blkpdx+blkpdy       ];
            //load f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[10] = param.blkDDFBuf[10*blkn+blki       +blkpdy       ];
            //load f11(x:+,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[11] = param.blkDDFBuf[11*blkn+blki       +blkpdy       ];
            //load f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
            fni[12] = param.blkDDFBuf[12*blkn+blki+blkpdx              ];
            //load f13(x:0,y:0,z:0) from current lattice
            fni[13] = param.blkDDFBuf[13*blkn+blki                     ];
            //load f14(x:+,y:0,z:0) from current lattice
            fni[14] = param.blkDDFBuf[14*blkn+blki                     ];
            //load f15(x:-,y:+,z:0) from neighbor (x:+,y:0,z:0)
            fni[15] = param.blkDDFBuf[15*blkn+blki+blkpdx              ];
            //load f16(x:0,y:+,z:0) from current lattice
            fni[16] = param.blkDDFBuf[16*blkn+blki                     ];
            //load f17(x:+,y:+,z:0) from current lattice
            fni[17] = param.blkDDFBuf[17*blkn+blki                     ];

            //load f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:0)
            fni[18] = param.blkDDFBuf[18*blkn+blki+blkpdx+blkpdy       ];
            //load f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:0)
            fni[19] = param.blkDDFBuf[19*blkn+blki       +blkpdy       ];
            //load f20(x:+,y:-,z:+) from neighbor (x:0,y:+,z:0)
            fni[20] = param.blkDDFBuf[20*blkn+blki       +blkpdy       ];
            //load f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:0)
            fni[21] = param.blkDDFBuf[21*blkn+blki+blkpdx              ];
            //load f22(x:0,y:0,z:+) from current lattice
            fni[22] = param.blkDDFBuf[22*blkn+blki                     ];
            //load f23(x:+,y:0,z:+) from current lattice
            fni[23] = param.blkDDFBuf[23*blkn+blki                     ];
            //load f24(x:-,y:+,z:+) from neighbor (x:+,y:0,z:0)
            fni[24] = param.blkDDFBuf[24*blkn+blki+blkpdx              ];
            //load f25(x:0,y:+,z:+) from current lattice
            fni[25] = param.blkDDFBuf[25*blkn+blki                     ];
            //load f26(x:+,y:+,z:+) from current lattice
            fni[26] = param.blkDDFBuf[26*blkn+blki                     ];
        }

        if((flagi & EQU_DDF_BIT)!=0)
        {
            const real_t rhoi = param.glbRhoBuf[glbi];
            const real_t vxi = param.glbVxBuf[glbi];
            const real_t vyi = param.glbVyBuf[glbi];
            const real_t vzi = param.glbVzBuf[glbi];
            gf::lbm_core::bgk::calcEqu<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
        }

        if((flagi & BOUNCE_BACK_BIT)!=0)
        {
            std::swap(fni[ 0], fni[26]);
            std::swap(fni[ 1], fni[25]);
            std::swap(fni[ 2], fni[24]);
            std::swap(fni[ 3], fni[23]);
            std::swap(fni[ 4], fni[22]);
            std::swap(fni[ 5], fni[21]);
            std::swap(fni[ 6], fni[20]);
            std::swap(fni[ 7], fni[19]);
            std::swap(fni[ 8], fni[18]);
            std::swap(fni[ 9], fni[17]);
            std::swap(fni[10], fni[16]);
            std::swap(fni[11], fni[15]);
            std::swap(fni[12], fni[14]); 
        }

        if((flagi & COLLIDE_BIT)!=0)
        {
            gf::lbm_core::bgk::collision2<27>(param.invTau, std::begin(fni));
        }

        if((flagi & (CORRECT_BIT | STORE_DDF_BIT))==(CORRECT_BIT | STORE_DDF_BIT))
        {
            param.glbDstDDFBuf[ 0*glbn+glbi] = fni[ 0];
            param.glbDstDDFBuf[ 1*glbn+glbi] = fni[ 1];
            param.glbDstDDFBuf[ 2*glbn+glbi] = fni[ 2];
            param.glbDstDDFBuf[ 3*glbn+glbi] = fni[ 3];
            param.glbDstDDFBuf[ 4*glbn+glbi] = fni[ 4];
            param.glbDstDDFBuf[ 5*glbn+glbi] = fni[ 5];
            param.glbDstDDFBuf[ 6*glbn+glbi] = fni[ 6];
            param.glbDstDDFBuf[ 7*glbn+glbi] = fni[ 7];
            param.glbDstDDFBuf[ 8*glbn+glbi] = fni[ 8];
            param.glbDstDDFBuf[ 9*glbn+glbi] = fni[ 9];
            param.glbDstDDFBuf[10*glbn+glbi] = fni[10];
            param.glbDstDDFBuf[11*glbn+glbi] = fni[11];
            param.glbDstDDFBuf[12*glbn+glbi] = fni[12];
            param.glbDstDDFBuf[13*glbn+glbi] = fni[13];
            param.glbDstDDFBuf[14*glbn+glbi] = fni[14];
            param.glbDstDDFBuf[15*glbn+glbi] = fni[15];
            param.glbDstDDFBuf[16*glbn+glbi] = fni[16];
            param.glbDstDDFBuf[17*glbn+glbi] = fni[17];
            param.glbDstDDFBuf[18*glbn+glbi] = fni[18];
            param.glbDstDDFBuf[19*glbn+glbi] = fni[19];
            param.glbDstDDFBuf[20*glbn+glbi] = fni[20];
            param.glbDstDDFBuf[21*glbn+glbi] = fni[21];
            param.glbDstDDFBuf[22*glbn+glbi] = fni[22];
            param.glbDstDDFBuf[23*glbn+glbi] = fni[23];
            param.glbDstDDFBuf[24*glbn+glbi] = fni[24];
            param.glbDstDDFBuf[25*glbn+glbi] = fni[25];
            param.glbDstDDFBuf[26*glbn+glbi] = fni[26];
        }
    }

    template<>
    __global__ __launch_bounds__(1024) void 
    HaloBlockingDynamicL2D3Q27InplaceLastKernel<true>(const HaloBlockingDynamicL2InplaceParam __grid_constant__ param)
    {
        const idx_t blkx = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t blky = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t blkz = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t blknx = gridDim.x * blockDim.x;
        const idx_t blkny = gridDim.y * blockDim.y;
        const idx_t blknz = gridDim.z * blockDim.z;
        const idx_t blkn = blknx * blkny * blknz;
        const idx_t blki = (blkny * blkz + blky) * blknx + blkx;
        const idx_t glbx = param.offx + blkx;
        const idx_t glby = param.offy + blky;
        const idx_t glbz = param.offz + blkz;
        const idx_t glbi = (param.glbny * glbz + glby) * param.glbnx + glbx;
        const idx_t glbn = param.glbnx * param.glbny * param.glbnz;
        
        const flag_t flagi = param.blkFlagBuf[blki];
        real_t fni[27];

        if((flagi & LOAD_DDF_BIT)!=0)
        {
            const idx_t blkpdx = (blkx==blknx-1) ? 1-blknx : 1;
            const idx_t blkpdy = (blky==blkny-1) ? (1-blkny)*blknx : blknx;
            const idx_t blkpdz = (blkz==blknz-1) ? (1-blknz)*blknx*blkny : blknx*blkny;
            //EsoTwist Even Load Rules:
            //Reverse all direction
            //load f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
            fni[ 0] = param.blkDDFBuf[26*blkn+blki+blkpdx+blkpdy+blkpdz];
            //load f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 1] = param.blkDDFBuf[25*blkn+blki       +blkpdy+blkpdz];
            //loaf f2 (x:+,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 2] = param.blkDDFBuf[24*blkn+blki       +blkpdy+blkpdz];
            //load f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 3] = param.blkDDFBuf[23*blkn+blki+blkpdx       +blkpdz];
            //load f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 4] = param.blkDDFBuf[22*blkn+blki              +blkpdz];
            //load f5 (x:+,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 5] = param.blkDDFBuf[21*blkn+blki              +blkpdz];
            //load f6 (x:-,y:+,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 6] = param.blkDDFBuf[20*blkn+blki+blkpdx       +blkpdz];
            //load f7 (x:0,y:+,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 7] = param.blkDDFBuf[19*blkn+blki              +blkpdz];
            //load f8 (x:+,y:+,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 8] = param.blkDDFBuf[18*blkn+blki              +blkpdz];

            //load f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
            fni[ 9] = param.blkDDFBuf[17*blkn+blki+blkpdx+blkpdy       ];
            //load f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[10] = param.blkDDFBuf[16*blkn+blki       +blkpdy       ];
            //load f11(x:+,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[11] = param.blkDDFBuf[15*blkn+blki       +blkpdy       ];
            //load f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
            fni[12] = param.blkDDFBuf[14*blkn+blki+blkpdx              ];
            //load f13(x:0,y:0,z:0) from current lattice
            fni[13] = param.blkDDFBuf[13*blkn+blki                     ];
            //load f14(x:+,y:0,z:0) from current lattice
            fni[14] = param.blkDDFBuf[12*blkn+blki                     ];
            //load f15(x:-,y:+,z:0) from neighbor (x:+,y:0,z:0)
            fni[15] = param.blkDDFBuf[11*blkn+blki+blkpdx              ];
            //load f16(x:0,y:+,z:0) from current lattice
            fni[16] = param.blkDDFBuf[10*blkn+blki                     ];
            //load f17(x:+,y:+,z:0) from current lattice
            fni[17] = param.blkDDFBuf[ 9*blkn+blki                     ];

            //load f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:0)
            fni[18] = param.blkDDFBuf[ 8*blkn+blki+blkpdx+blkpdy       ];
            //load f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:0)
            fni[19] = param.blkDDFBuf[ 7*blkn+blki       +blkpdy       ];
            //load f20(x:+,y:-,z:+) from neighbor (x:0,y:+,z:0)
            fni[20] = param.blkDDFBuf[ 6*blkn+blki       +blkpdy       ];
            //load f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:0)
            fni[21] = param.blkDDFBuf[ 5*blkn+blki+blkpdx              ];
            //load f22(x:0,y:0,z:+) from current lattice
            fni[22] = param.blkDDFBuf[ 4*blkn+blki                     ];
            //load f23(x:+,y:0,z:+) from current lattice
            fni[23] = param.blkDDFBuf[ 3*blkn+blki                     ];
            //load f24(x:-,y:+,z:+) from neighbor (x:+,y:0,z:0)
            fni[24] = param.blkDDFBuf[ 2*blkn+blki+blkpdx              ];
            //load f25(x:0,y:+,z:+) from current lattice
            fni[25] = param.blkDDFBuf[ 1*blkn+blki                     ];
            //load f26(x:+,y:+,z:+) from current lattice
            fni[26] = param.blkDDFBuf[ 0*blkn+blki                     ];
        }

        if((flagi & EQU_DDF_BIT)!=0)
        {
            const real_t rhoi = param.glbRhoBuf[glbi];
            const real_t vxi = param.glbVxBuf[glbi];
            const real_t vyi = param.glbVyBuf[glbi];
            const real_t vzi = param.glbVzBuf[glbi];
            gf::lbm_core::bgk::calcEqu<27>(rhoi, vxi, vyi, vzi, std::begin(fni));
        }

        if((flagi & BOUNCE_BACK_BIT)!=0)
        {
            std::swap(fni[ 0], fni[26]);
            std::swap(fni[ 1], fni[25]);
            std::swap(fni[ 2], fni[24]);
            std::swap(fni[ 3], fni[23]);
            std::swap(fni[ 4], fni[22]);
            std::swap(fni[ 5], fni[21]);
            std::swap(fni[ 6], fni[20]);
            std::swap(fni[ 7], fni[19]);
            std::swap(fni[ 8], fni[18]);
            std::swap(fni[ 9], fni[17]);
            std::swap(fni[10], fni[16]);
            std::swap(fni[11], fni[15]);
            std::swap(fni[12], fni[14]); 
        }

        if((flagi &  COLLIDE_BIT)!=0)
        {
            gf::lbm_core::bgk::collision2<27>(param.invTau, std::begin(fni));
        }

        if((flagi & (CORRECT_BIT | STORE_DDF_BIT))==(CORRECT_BIT | STORE_DDF_BIT))
        {
            param.glbDstDDFBuf[ 0*glbn+glbi] = fni[ 0];
            param.glbDstDDFBuf[ 1*glbn+glbi] = fni[ 1];
            param.glbDstDDFBuf[ 2*glbn+glbi] = fni[ 2];
            param.glbDstDDFBuf[ 3*glbn+glbi] = fni[ 3];
            param.glbDstDDFBuf[ 4*glbn+glbi] = fni[ 4];
            param.glbDstDDFBuf[ 5*glbn+glbi] = fni[ 5];
            param.glbDstDDFBuf[ 6*glbn+glbi] = fni[ 6];
            param.glbDstDDFBuf[ 7*glbn+glbi] = fni[ 7];
            param.glbDstDDFBuf[ 8*glbn+glbi] = fni[ 8];
            param.glbDstDDFBuf[ 9*glbn+glbi] = fni[ 9];
            param.glbDstDDFBuf[10*glbn+glbi] = fni[10];
            param.glbDstDDFBuf[11*glbn+glbi] = fni[11];
            param.glbDstDDFBuf[12*glbn+glbi] = fni[12];
            param.glbDstDDFBuf[13*glbn+glbi] = fni[13];
            param.glbDstDDFBuf[14*glbn+glbi] = fni[14];
            param.glbDstDDFBuf[15*glbn+glbi] = fni[15];
            param.glbDstDDFBuf[16*glbn+glbi] = fni[16];
            param.glbDstDDFBuf[17*glbn+glbi] = fni[17];
            param.glbDstDDFBuf[18*glbn+glbi] = fni[18];
            param.glbDstDDFBuf[19*glbn+glbi] = fni[19];
            param.glbDstDDFBuf[20*glbn+glbi] = fni[20];
            param.glbDstDDFBuf[21*glbn+glbi] = fni[21];
            param.glbDstDDFBuf[22*glbn+glbi] = fni[22];
            param.glbDstDDFBuf[23*glbn+glbi] = fni[23];
            param.glbDstDDFBuf[24*glbn+glbi] = fni[24];
            param.glbDstDDFBuf[25*glbn+glbi] = fni[25];
            param.glbDstDDFBuf[26*glbn+glbi] = fni[26];
        }        
    }

    __global__ __launch_bounds__(1024) void
    HaloBlockingDynamicL2D3Q27InplaceDumpKernel(const HaloBlockingDynamicL2InplaceParam __grid_constant__ param)
    {
        const idx_t blkx = blockDim.x * blockIdx.x + threadIdx.x;
        const idx_t blky = blockDim.y * blockIdx.y + threadIdx.y;
        const idx_t blkz = blockDim.z * blockIdx.z + threadIdx.z;
        const idx_t blknx = gridDim.x * blockDim.x;
        const idx_t blkny = gridDim.y * blockDim.y;
        const idx_t blki = (blkny*blkz+blky)*blknx+blkx;
        const idx_t glbx = param.offx + blkx;
        const idx_t glby = param.offy + blky;
        const idx_t glbz = param.offz + blkz;
        const idx_t glbi = (param.glbny * glbz + glby) * param.glbnx + glbx;
        const idx_t glbn = param.glbnx * param.glbny * param.glbnz;

        const flag_t flagi = param.blkFlagBuf[blki];
        real_t rhoi, vxi, vyi, vzi;
        real_t fni[27];

        if((flagi & CORRECT_BIT)!=0)
        {
            const idx_t glbndx = (glbx==0) ? 0 : -1;
            const idx_t glbpdx = (glbx==param.glbnx-1) ? 0 : 1;
            const idx_t glbndy = (glby==0) ? 0 : -param.glbnx;
            const idx_t glbpdy = (glby==param.glbny-1) ? 0 : param.glbnx;
            const idx_t glbndz = (glbz==0) ? 0 : -param.glbnx * param.glbny;
            const idx_t glbpdz = (glbz==param.glbnz-1) ? 0 : param.glbnx * param.glbny;

            //f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
            fni[ 0] = param.glbSrcDDFBuf[ 0*glbn+glbi+glbpdx+glbpdy+glbpdz];
            //f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
            fni[ 1] = param.glbSrcDDFBuf[ 1*glbn+glbi       +glbpdy+glbpdz];
            //f2 (x:+,y:-,z:-) from neighbor (x:-,y:+,z:+)
            fni[ 2] = param.glbSrcDDFBuf[ 2*glbn+glbi+glbndx+glbpdy+glbpdz];
            //f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
            fni[ 3] = param.glbSrcDDFBuf[ 3*glbn+glbi+glbpdx       +glbpdz];
            //f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
            fni[ 4] = param.glbSrcDDFBuf[ 4*glbn+glbi              +glbpdz];
            //f5 (x:+,y:0,z:-) from neighbor (x:-,y:0,z:+)
            fni[ 5] = param.glbSrcDDFBuf[ 5*glbn+glbi+glbndx       +glbpdz];
            //f6 (x:-,y:+,z:-) from neighbor (x:+,y:-,z:+)
            fni[ 6] = param.glbSrcDDFBuf[ 6*glbn+glbi+glbpdx+glbndy+glbpdz];
            //f7 (x:0,y:+,z:-) from neighbor (x:0,y:-,z:+)
            fni[ 7] = param.glbSrcDDFBuf[ 7*glbn+glbi       +glbndy+glbpdz];
            //f8 (x:+,y:+,z:-) from neighbor (x:-,y:-,z:+)
            fni[ 8] = param.glbSrcDDFBuf[ 8*glbn+glbi+glbndx+glbndy+glbpdz];

            //f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
            fni[ 9] = param.glbSrcDDFBuf[ 9*glbn+glbi+glbpdx+glbpdy       ];
            //f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
            fni[10] = param.glbSrcDDFBuf[10*glbn+glbi       +glbpdy       ];
            //f11(x:+,y:-,z:0) from neighbor (x:-,y:+,z:0)
            fni[11] = param.glbSrcDDFBuf[11*glbn+glbi+glbndx+glbpdy       ];
            //f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
            fni[12] = param.glbSrcDDFBuf[12*glbn+glbi+glbpdx              ];
            //f13(x:0,y:0,z:0) from neighbor (x:0,y:0,z:0)
            fni[13] = param.glbSrcDDFBuf[13*glbn+glbi                     ];
            //f14(x:+,y:0,z:0) from neighbor (x:-,y:0,z:0)
            fni[14] = param.glbSrcDDFBuf[14*glbn+glbi+glbndx              ];
            //f15(x:-,y:+,z:0) from neighbor (x:+,y:-,z:0)
            fni[15] = param.glbSrcDDFBuf[15*glbn+glbi+glbpdx+glbndy       ];
            //f16(x:0,y:+,z:0) from neighbor (x:0,y:-,z:0)
            fni[16] = param.glbSrcDDFBuf[16*glbn+glbi       +glbndy       ];
            //f17(x:+,y:+,z:0) from neighbor (x:-,y:-,z:0)
            fni[17] = param.glbSrcDDFBuf[17*glbn+glbi+glbndx+glbndy       ];

            //f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:-)
            fni[18] = param.glbSrcDDFBuf[18*glbn+glbi+glbpdx+glbpdy+glbndz];
            //f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:-)
            fni[19] = param.glbSrcDDFBuf[19*glbn+glbi       +glbpdy+glbndz];
            //f20(x:+,y:-,z:+) from neighbor (x:-,y:+,z:-)
            fni[20] = param.glbSrcDDFBuf[20*glbn+glbi+glbndx+glbpdy+glbndz];
            //f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:-)
            fni[21] = param.glbSrcDDFBuf[21*glbn+glbi+glbpdx       +glbndz];
            //f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:-)
            fni[22] = param.glbSrcDDFBuf[22*glbn+glbi              +glbndz];
            //f23(x:+,y:0,z:+) from neighbor (x:-,y:0,z:-)
            fni[23] = param.glbSrcDDFBuf[23*glbn+glbi+glbndx       +glbndz];
            //f24(x:-,y:+,z:+) from neighbor (x:+,y:-,z:-)
            fni[24] = param.glbSrcDDFBuf[24*glbn+glbi+glbpdx+glbndy+glbndz];
            //f25(x:0,y:+,z:+) from neighbor (x:0,y:-,z:-)
            fni[25] = param.glbSrcDDFBuf[25*glbn+glbi       +glbndy+glbndz];
            //f26(x:+,y:+,z:+) from neighbor (x:-,y:-,z:-)
            fni[26] = param.glbSrcDDFBuf[26*glbn+glbi+glbndx+glbndy+glbndz];

            gf::lbm_core::bgk::calcRhoU<27>(rhoi, vxi, vyi, vzi, std::begin(fni));            
        }

        if((flagi & (CORRECT_BIT | DUMP_RHO_BIT))==(CORRECT_BIT | DUMP_RHO_BIT))
        {
            param.glbRhoBuf[glbi] = rhoi;
        }

        if((flagi & (CORRECT_BIT | DUMP_VX_BIT))==(CORRECT_BIT | DUMP_VX_BIT))
        {
            param.glbVxBuf[glbi] = vxi;
        }

        if((flagi & (CORRECT_BIT | DUMP_VY_BIT))==(CORRECT_BIT | DUMP_VY_BIT))
        {
            param.glbVyBuf[glbi] = vyi;
        }

        if((flagi & (CORRECT_BIT | DUMP_VZ_BIT))==(CORRECT_BIT | DUMP_VZ_BIT))
        {
            param.glbVzBuf[glbi] = vzi;
        }
    }
}