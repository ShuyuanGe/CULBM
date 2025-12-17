#include <format>
#include <assert.h>
#include <cstdint>
#include <iostream>
#include <cuda_runtime.h>
#include <cu_exception.cuh>

template<int Q, bool Even2Odd>
__global__ void setKernel(float* blkDDFBuf);

template<int Q, bool Even2Odd>
__global__ void testKernel(const float* blkDDFBuf);

template<int Q, bool IsEven>
__device__ __forceinline__ void inplaceStore(int i, int n, int nbrpdx, int nbrpdy, int nbrpdz, const float* fni, float* blkDDFBuf);

template<int Q, bool IsEven>
__device__ __forceinline__ void inplaceLoad(int i, int n, int nbrpdx, int nbrpdy, int nbrpdz, float* fni, const float *blkDDFBuf);

template<int Q, int I=0>
__device__ __forceinline__ bool check(int blkx, int blknx, int blky, int blkny, int blkz, int blknz, const float* fni)
{
    const int srcv = std::bit_cast<int>(fni[I]);
    const int srcx = srcv % blknx;
    const int srcy = (srcv / blknx) % blkny;
    const int srcz = srcv / (blknx * blkny);
    bool res = true;

    if constexpr (Q==27)
    {
        if constexpr (I==0)
        {
            //(x:-,y:-,z:-)
            if(blkx<blknx-1 and blky<blkny-1 and blkz<blknz-1)
            {
                res = ((srcx==blkx+1) and (srcy==blky+1) and (srcz==blkz+1));
            }
        }

        if constexpr (I==1)
        {
            //(x:0,y:-,z:-)
            if(blky<blkny-1 and blkz<blknz-1)
            {
                res = ((srcx==blkx) and (srcy==blky+1) and (srcz==blkz+1));
            }
        }

        if constexpr (I==2)
        {
            //(x:+,y:-,z:-)
            if(0<blkx and blky<blkny-1 and blkz<blknz-1)
            {
                res = ((srcx==blkx-1) and (srcy==blky+1) and (srcz==blkz+1));
            }
        }

        if constexpr (I==3)
        {
            //(x:-,y:0,z:-)
            if(blkx<blknx-1 and blkz<blknz-1)
            {
                res = ((srcx==blkx+1) and (srcy==blky) and (srcz==blkz+1));
            }
        }

        if constexpr (I==4)
        {
            //(x:0,y:0,z:-)
            if(blkz<blknz-1)
            {
                res = ((srcx==blkx) and (srcy==blky) and (srcz==blkz+1));
            }
        }

        if constexpr (I==5)
        {
            //(x:+,y:0,z:-)
            if(0<blkx and blkz<blknz-1)
            {
                res = ((srcx==blkx-1) and (srcy==blky) and (srcz==blkz+1));
            }
        }

        if constexpr (I==6)
        {
            //(x:-,y:+,z:-)
            if(blkx<blknx-1 and 0<blky and blkz<blknz-1)
            {
                res = ((srcx==blkx+1) and (srcy==blky-1) and (srcz==blkz+1));
            }
        }

        if constexpr (I==7)
        {
            //(x:0,y:+,z:-)
            if(0<blky and blkz<blknz-1)
            {
                res = ((srcx==blkx) and (srcy==blky-1) and (srcz==blkz+1));
            }
        }

        if constexpr (I==8)
        {
            //(x:+,y:+,z:-)
            if(0<blkx and 0<blky and blkz<blknz-1)
            {
                res = ((srcx==blkx-1) and (srcy==blky-1) and (srcz==blkz+1));
            }
        }

        if constexpr (I==9)
        {
            //(x:-,y:-,z:0)
            if(blkx<blknx-1 and blky<blkny-1)
            {
                res = ((srcx==blkx+1) and (srcy==blky+1) and (srcz==blkz));
            }
        }

        if constexpr (I==10)
        {
            //(x:0,y:-,z:0)
            if(blky<blkny-1)
            {
                res = ((srcx==blkx) and (srcy==blky+1) and (srcz==blkz));
            }
        }

        if constexpr (I==11)
        {
            //(x:+,y:-,z:0)
            if(0<blkx and blky<blkny-1)
            {
                res = ((srcx==blkx-1) and (srcy==blky+1) and (srcz==blkz));
            }
        }

        if constexpr (I==12)
        {
            //(x:-,y:0,z:0)
            if(blkx<blknx-1)
            {
                res = ((srcx==blkx+1) and (srcy==blky) and (srcz==blkz));
            }
        }

        if constexpr (I==13)
        {
            res = ((srcx==blkx) and (srcy==blky) and (srcz==blkz));
        }

        if constexpr (I==14)
        {
            //(x:+,y:0,z:0)
            if(0<blkx)
            {
                res = ((srcx==blkx-1) and (srcy==blky) and (srcz==blkz));
            }
        }

        if constexpr (I==15)
        {
            //(x:-,y:+,z:0)
            if(blkx<blknx-1 and 0<blky)
            {
                res = ((srcx==blkx+1) and (srcy==blky-1) and (srcz==blkz));
            }
        }

        if constexpr (I==16)
        {
            //(x:0,y:+,z:0)
            if(0<blky)
            {
                res = ((srcx==blkx) and (srcy==blky-1) and (srcz==blkz));
            }
        }

        if constexpr (I==17)
        {
            //(x:+,y:+,z:0)
            if(0<blkx and 0<blky)
            {
                res = ((srcx==blkx-1) and (srcy==blky-1) and (srcz==blkz));
            }
        }

        if constexpr (I==18)
        {
            //(x:-,y:-,z:+)
            if(blkx<blknx-1 and blky<blkny-1 and 0<blkz)
            {
                res = ((srcx==blkx+1) and (srcy==blky+1) and (srcz==blkz-1));
            }
        }

        if constexpr (I==19)
        {
            //(x:0,y:-,z:+)
            if(blky<blkny-1 and 0<blkz)
            {
                res = ((srcx==blkx) and (srcy==blky+1) and (srcz==blkz-1));
            }
        }

        if constexpr (I==20)
        {
            //(x:+,y:-,z:+)
            if(0<blkx and blky<blkny-1 and 0<blkz)
            {
                res = ((srcx==blkx-1) and (srcy==blky+1) and (srcz==blkz-1));
            }
        }

        if constexpr (I==21)
        {
            //(x:-,y:0,z:+)
            if(blkx<blknx-1 and 0<blkz)
            {
                res = ((srcx==blkx+1) and (srcy==blky) and (srcz==blkz-1));
            }
        }

        if constexpr (I==22)
        {
            //(x:0,y:0,z:+)
            if(0<blkz)
            {
                res = ((srcx==blkx) and (srcy==blky) and (srcz==blkz-1));
            }
        }

        if constexpr (I==23)
        {
            //(x:+,y:0,z:+)
            if(0<blkx and 0<blkz)
            {
                res = ((srcx==blkx-1) and (srcy==blky) and (srcz==blkz-1));
            }
        }

        if constexpr (I==24)
        {
            //(x:-,y:+,z:+)
            if(blkx<blknx-1 and 0<blky and 0<blkz)
            {
                res = ((srcx==blkx+1) and (srcy==blky-1) and (srcz==blkz-1));
            }
        }

        if constexpr (I==25)
        {
            //(x:0,y:+,z:+)
            if(0<blky and 0<blkz)
            {
                res = ((srcx==blkx) and (srcy==blky-1) and (srcz==blkz-1));
            }
        }

        if constexpr (I==26)
        {
            //(x:+,y:+,z:+)
            if(0<blkx and 0<blky and 0<blkz)
            {
                res = ((srcx==blkx-1) and (srcy==blky-1) and (srcz==blkz-1));
            }
        }
    }

    if constexpr (I<Q-1)
    {
        res = (res and check<Q, I+1>(blkx, blknx, blky, blkny, blkz, blknz, fni));
    }

    return res;
}


int main()
{
    try
    {
        constexpr dim3 gridDim  { 2,  6, 38};
        constexpr dim3 blockDim {32, 16,  2};
        constexpr dim3 tileDim {gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z};
        constexpr int tileSize = tileDim.x * tileDim.y * tileDim.z;
        constexpr int Q = 27;
        constexpr bool Even2Odd = false;

        float* blkDDFBuf = nullptr;
        CU_CHECK(cudaMalloc(&blkDDFBuf, sizeof(float)*Q*tileSize));
        void* kernelArgs[1] = {(void*)&blkDDFBuf};

        CU_CHECK(cudaLaunchKernel((const void*)&setKernel<Q, Even2Odd>, gridDim, blockDim, std::begin(kernelArgs), 0, 0));
        CU_CHECK(cudaLaunchKernel((const void*)&testKernel<Q, Even2Odd>, gridDim, blockDim, std::begin(kernelArgs), 0, 0));
        CU_CHECK(cudaStreamSynchronize(0));

        std::cout << std::format("Inplace Stream ({}) Correctness Check Pass!", Even2Odd ? "Even Store->Odd Load" : "Odd Store->Even Load") << std::endl;

        CU_CHECK(cudaFree(blkDDFBuf));
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        exit(1);
    }
    return 0;
}

template<int Q, bool Even2Odd>
__global__ void setKernel(float* blkDDFBuf)
{
    const int blkx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blky = blockIdx.y * blockDim.y + threadIdx.y;
    const int blkz = blockIdx.z * blockDim.z + threadIdx.z;
    const int blknx = gridDim.x * blockDim.x;
    const int blkny = gridDim.y * blockDim.y;
    const int blknz = gridDim.z * blockDim.z;
    
    const int blki = (blkny * blkz + blky) * blknx + blkx;
    const int blkn = blknx * blkny * blknz;

    const int nbrpdx = (blkx==(blknx-1)) ? (1-blknx) : 1;
    const int nbrpdy = (blky==(blkny-1)) ? (1-blkny)*blknx : blknx;
    const int nbrpdz = (blkz==(blknz-1)) ? (1-blknz)*blknx*blkny : blknx*blkny;

    float fni[Q];
    std::fill_n(std::begin(fni), Q, std::bit_cast<float>(blki));

    inplaceStore<Q, Even2Odd>(blki, blkn, nbrpdx, nbrpdy, nbrpdz, std::begin(fni), blkDDFBuf);
}

template<int Q, bool Even2Odd>
__global__ void testKernel(const float* blkDDFBuf)
{
    const int blkx = blockIdx.x * blockDim.x + threadIdx.x;
    const int blky = blockIdx.y * blockDim.y + threadIdx.y;
    const int blkz = blockIdx.z * blockDim.z + threadIdx.z;
    const int blknx = gridDim.x * blockDim.x;
    const int blkny = gridDim.y * blockDim.y;
    const int blknz = gridDim.z * blockDim.z;

    const int blki = (blkny * blkz + blky) * blknx + blkx;
    const int blkn = blknx * blkny * blknz;

    const int nbrpdx = (blkx==(blknx-1)) ? (1-blknx) : 1;
    const int nbrpdy = (blky==(blkny-1)) ? (1-blkny)*blknx : blknx;
    const int nbrpdz = (blkz==(blknz-1)) ? (1-blknz)*blknx*blkny : blknx*blkny;

    float fni[Q];
    inplaceLoad<Q, not Even2Odd>(blki, blkn, nbrpdx, nbrpdy, nbrpdz, std::begin(fni), blkDDFBuf);

    const bool res = check<Q, 0>(blkx, blknx, blky, blkny, blkz, blknz, std::begin(fni));
    if(not res)
    {
        printf("Test Failed\n");
    }
}

template<int Q, bool IsEven>
__device__ __forceinline__ void inplaceStore(int i, int n, int nbrpdx, int nbrpdy, int nbrpdz, const float* fni, float* blkDDFBuf)
{
    if constexpr ((Q==27) and (IsEven))
    {
        //EsoTwist Even Store Rules:
        //Don't change direction
        //store f0 (x:-,y:-,z:-) to current lattice
        blkDDFBuf[ 0*n+i                     ] = fni[ 0];
        //store f1 (x:0,y:-,z:-) to current lattice
        blkDDFBuf[ 1*n+i                     ] = fni[ 1];
        //store f2 (x:+,y:-,z:-) to neighbor (x:+,y:0,z:0)
        blkDDFBuf[ 2*n+i+nbrpdx              ] = fni[ 2];
        //store f3 (x:-,y:0,z:-) to current lattice
        blkDDFBuf[ 3*n+i                     ] = fni[ 3];
        //store f4 (x:0,y:0,z:-) to current lattice
        blkDDFBuf[ 4*n+i                     ] = fni[ 4];
        //store f5 (x:+,y:0,z:-) to neighbor (x:+,y:0,z:0)
        blkDDFBuf[ 5*n+i+nbrpdx              ] = fni[ 5];
        //store f6 (x:-,y:+,z:-) to neighbor (x:0,y:+,z:0)
        blkDDFBuf[ 6*n+i       +nbrpdy       ] = fni[ 6];
        //store f7 (x:0,y:+,z:-) to neighbor (x:0,y:+,z:0)
        blkDDFBuf[ 7*n+i       +nbrpdy       ] = fni[ 7];
        //store f8 (x:+,y:+,z:-) to neighbor (x:+,y:+,z:0)
        blkDDFBuf[ 8*n+i+nbrpdx+nbrpdy       ] = fni[ 8];

        //store f9 (x:-,y:-,z:0) to current lattice
        blkDDFBuf[ 9*n+i                     ] = fni[ 9];
        //store f10(x:0,y:-,z:0) to current lattice
        blkDDFBuf[10*n+i                     ] = fni[10];
        //store f11(x:+,y:-,z:0) to neighbor (x:+,y:0,z:0)
        blkDDFBuf[11*n+i+nbrpdx              ] = fni[11];
        //store f12(x:-,y:0,z:0) to current lattice
        blkDDFBuf[12*n+i                     ] = fni[12];
        //store f13(x:0,y:0,z:0) to current lattice
        blkDDFBuf[13*n+i                     ] = fni[13];
        //store f14(x:+,y:0,z:0) to neighbor (x:+,y:0,z:0)
        blkDDFBuf[14*n+i+nbrpdx              ] = fni[14];
        //store f15(x:-,y:+,z:0) to neighbor (x:0,y:+,z:0)
        blkDDFBuf[15*n+i       +nbrpdy       ] = fni[15];
        //store f16(x:0,y:+,z:0) to neighbor (x:0,y:+,z:0)
        blkDDFBuf[16*n+i       +nbrpdy       ] = fni[16];
        //store f17(x:+,y:+,z:0) to neighbor (x:+,y:+,z:0)
        blkDDFBuf[17*n+i+nbrpdx+nbrpdy       ] = fni[17];

        //store f18(x:-,y:-,z:+) to neighbor (x:0,y:0,z:+)
        blkDDFBuf[18*n+i              +nbrpdz] = fni[18];
        //store f19(x:0,y:-,z:+) to neighbor (x:0,y:0,z:+)
        blkDDFBuf[19*n+i              +nbrpdz] = fni[19];
        //store f20(x:+,y:-,z:+) to neighbor (x:+,y:0,z:+)
        blkDDFBuf[20*n+i+nbrpdx       +nbrpdz] = fni[20];
        //store f21(x:-,y:0,z:+) to neighbor (x:0,y:0,z:+)
        blkDDFBuf[21*n+i              +nbrpdz] = fni[21];
        //store f22(x:0,y:0,z:+) to neighbor (x:0,y:0,z:+)
        blkDDFBuf[22*n+i              +nbrpdz] = fni[22];
        //store f23(x:+,y:0,z:+) to neighbor (x:+,y:0,z:+)
        blkDDFBuf[23*n+i+nbrpdx       +nbrpdz] = fni[23];
        //store f24(x:-,y:+,z:+) to neighbor (x:0,y:+,z:+)
        blkDDFBuf[24*n+i       +nbrpdy+nbrpdz] = fni[24];
        //store f25(x:0,y:+,z:+) to neighbor (x:0,y:+,z:+)
        blkDDFBuf[25*n+i       +nbrpdy+nbrpdz] = fni[25];
        //store f26(x:+,y:+,z:+) to neighbor (x:+,y:+,z:+)
        blkDDFBuf[26*n+i+nbrpdx+nbrpdy+nbrpdz] = fni[26];
    }

    if constexpr ((Q==27) and (not IsEven))
    {
        //EsoTwist Odd Store Rules:
        //Reverse all directions
        //store f0 (x:-,y:-,z:-) to current lattice
        blkDDFBuf[26*n+i                     ] = fni[ 0];
        //store f1 (x:0,y:-,z:-) to current lattice
        blkDDFBuf[25*n+i                     ] = fni[ 1];
        //store f2 (x:+,y:-,z:-) to neighbor (x:+,y:0,z:0)
        blkDDFBuf[24*n+i+nbrpdx              ] = fni[ 2];
        //store f3 (x:-,y:0,z:-) to current lattice
        blkDDFBuf[23*n+i                     ] = fni[ 3];
        //store f4 (x:0,y:0,z:-) to current lattice
        blkDDFBuf[22*n+i                     ] = fni[ 4];
        //store f5 (x:+,y:0,z:-) to neighbor (x:+,y:0,z:0)
        blkDDFBuf[21*n+i+nbrpdx              ] = fni[ 5];
        //store f6 (x:-,y:+,z:-) to neighbor (x:0,y:+,z:0)
        blkDDFBuf[20*n+i       +nbrpdy       ] = fni[ 6];
        //store f7 (x:0,y:+,z:-) to neighbor (x:0,y:+,z:0)
        blkDDFBuf[19*n+i       +nbrpdy       ] = fni[ 7];
        //store f8 (x:+,y:+,z:-) to neighbor (x:+,y:+,z:0)
        blkDDFBuf[18*n+i+nbrpdx+nbrpdy       ] = fni[ 8];

        //store f9 (x:-,y:-,z:0) to current lattice
        blkDDFBuf[17*n+i                     ] = fni[ 9];
        //store f10(x:0,y:-,z:0) to current lattice
        blkDDFBuf[16*n+i                     ] = fni[10];
        //store f11(x:+,y:-,z:0) to neighbor (x:+,y:0,z:0)
        blkDDFBuf[15*n+i+nbrpdx              ] = fni[11];
        //store f12(x:-,y:0,z:0) to current lattice
        blkDDFBuf[14*n+i                     ] = fni[12];
        //store f13(x:0,y:0,z:0) to current lattice
        blkDDFBuf[13*n+i                     ] = fni[13];
        //store f14(x:+,y:0,z:0) to neighbor (x:+,y:0,z:0)
        blkDDFBuf[12*n+i+nbrpdx              ] = fni[14];
        //store f15(x:-,y:+,z:0) to neighbor (x:0,y:+,z:0)
        blkDDFBuf[11*n+i       +nbrpdy       ] = fni[15];
        //store f16(x:0,y:+,z:0) to neighbor (x:0,y:+,z:0)
        blkDDFBuf[10*n+i       +nbrpdy       ] = fni[16];
        //store f17(x:+,y:+,z:0) to neighbor (x:+,y:+,z:0)
        blkDDFBuf[ 9*n+i+nbrpdx+nbrpdy       ] = fni[17];

        //store f18(x:-,y:-,z:+) to neighbor (x:0,y:0,z:+)
        blkDDFBuf[ 8*n+i              +nbrpdz] = fni[18];
        //store f19(x:0,y:-,z:+) to neighbor (x:0,y:0,z:+)
        blkDDFBuf[ 7*n+i              +nbrpdz] = fni[19];
        //store f20(x:+,y:-,z:+) to neighbor (x:+,y:0,z:+)
        blkDDFBuf[ 6*n+i+nbrpdx       +nbrpdz] = fni[20];
        //store f21(x:-,y:0,z:+) to neighbor (x:0,y:0,z:+)
        blkDDFBuf[ 5*n+i              +nbrpdz] = fni[21];
        //store f22(x:0,y:0,z:+) to neighbor (x:0,y:0,z:+)
        blkDDFBuf[ 4*n+i              +nbrpdz] = fni[22];
        //store f23(x:+,y:0,z:+) to neighbor (x:+,y:0,z:+)
        blkDDFBuf[ 3*n+i+nbrpdx       +nbrpdz] = fni[23];
        //store f24(x:-,y:+,z:+) to neighbor (x:0,y:+,z:+)
        blkDDFBuf[ 2*n+i       +nbrpdy+nbrpdz] = fni[24];
        //store f25(x:0,y:+,z:+) to neighbor (x:0,y:+,z:+)
        blkDDFBuf[ 1*n+i       +nbrpdy+nbrpdz] = fni[25];
        //store f26(x:+,y:+,z:+) to neighbor (x:+,y:+,z:+)
        blkDDFBuf[ 0*n+i+nbrpdx+nbrpdy+nbrpdz] = fni[26];
    }
}

template<int Q, bool IsEven>
__device__ __forceinline__ void inplaceLoad(int i, int n, int nbrpdx, int nbrpdy, int nbrpdz, float* fni, const float *blkDDFBuf)
{
    if constexpr ((Q==27) and (IsEven))
    {
        //EsoTwist Even Load Rules
        //Reverse all directions
        //load f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
        fni[ 0] = blkDDFBuf[26*n+i+nbrpdx+nbrpdy+nbrpdz];
        //load f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
        fni[ 1] = blkDDFBuf[25*n+i       +nbrpdy+nbrpdz];
        //loaf f2 (x:+,y:-,z:-) from neighbor (x:0,y:+,z:+)
        fni[ 2] = blkDDFBuf[24*n+i       +nbrpdy+nbrpdz];
        //load f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
        fni[ 3] = blkDDFBuf[23*n+i+nbrpdx       +nbrpdz];
        //load f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
        fni[ 4] = blkDDFBuf[22*n+i              +nbrpdz];
        //load f5 (x:+,y:0,z:-) from neighbor (x:0,y:0,z:+)
        fni[ 5] = blkDDFBuf[21*n+i              +nbrpdz];
        //load f6 (x:-,y:+,z:-) from neighbor (x:+,y:0,z:+)
        fni[ 6] = blkDDFBuf[20*n+i+nbrpdx       +nbrpdz];
        //load f7 (x:0,y:+,z:-) from neighbor (x:0,y:0,z:+)
        fni[ 7] = blkDDFBuf[19*n+i              +nbrpdz];
        //load f8 (x:+,y:+,z:-) from neighbor (x:0,y:0,z:+)
        fni[ 8] = blkDDFBuf[18*n+i              +nbrpdz];

        //load f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
        fni[ 9] = blkDDFBuf[17*n+i+nbrpdx+nbrpdy       ];
        //load f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
        fni[10] = blkDDFBuf[16*n+i       +nbrpdy       ];
        //load f11(x:+,y:-,z:0) from neighbor (x:0,y:+,z:0)
        fni[11] = blkDDFBuf[15*n+i       +nbrpdy       ];
        //load f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
        fni[12] = blkDDFBuf[14*n+i+nbrpdx              ];
        //load f13(x:0,y:0,z:0) from current lattice
        fni[13] = blkDDFBuf[13*n+i                     ];
        //load f14(x:+,y:0,z:0) from current lattice
        fni[14] = blkDDFBuf[12*n+i                     ];
        //load f15(x:-,y:+,z:0) from neighbor (x:+,y:0,z:0)
        fni[15] = blkDDFBuf[11*n+i+nbrpdx              ];
        //load f16(x:0,y:+,z:0) from current lattice
        fni[16] = blkDDFBuf[10*n+i                     ];
        //load f17(x:+,y:+,z:0) from current lattice
        fni[17] = blkDDFBuf[ 9*n+i                     ];

        //load f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:0)
        fni[18] = blkDDFBuf[ 8*n+i+nbrpdx+nbrpdy       ];
        //load f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:0)
        fni[19] = blkDDFBuf[ 7*n+i       +nbrpdy       ];
        //load f20(x:+,y:-,z:+) from neighbor (x:0,y:+,z:0)
        fni[20] = blkDDFBuf[ 6*n+i       +nbrpdy       ];
        //load f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:0)
        fni[21] = blkDDFBuf[ 5*n+i+nbrpdx              ];
        //load f22(x:0,y:0,z:+) from current lattice
        fni[22] = blkDDFBuf[ 4*n+i                     ];
        //load f23(x:+,y:0,z:+) from current lattice
        fni[23] = blkDDFBuf[ 3*n+i                     ];
        //load f24(x:-,y:+,z:+) from neighbor (x:+,y:0,z:0)
        fni[24] = blkDDFBuf[ 2*n+i+nbrpdx              ];
        //load f25(x:0,y:+,z:+) from current lattice
        fni[25] = blkDDFBuf[ 1*n+i                     ];
        //load f26(x:+,y:+,z:+) from current lattice
        fni[26] = blkDDFBuf[ 0*n+i                     ];
    }

    if constexpr ((Q==27) and (not IsEven))
    {
        //EsoTwist Odd Load Rules:
        //Don' t change direction
        //load f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
        fni[ 0] = blkDDFBuf[ 0*n+i+nbrpdx+nbrpdy+nbrpdz];
        //load f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
        fni[ 1] = blkDDFBuf[ 1*n+i       +nbrpdy+nbrpdz];
        //load f2 (x:+,y:-,z:-) from neighbor (x:0,y:+,z:+)
        fni[ 2] = blkDDFBuf[ 2*n+i       +nbrpdy+nbrpdz];
        //load f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
        fni[ 3] = blkDDFBuf[ 3*n+i+nbrpdx       +nbrpdz];
        //load f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
        fni[ 4] = blkDDFBuf[ 4*n+i              +nbrpdz];
        //load f5 (x:+,y:0,z:-) from neighbor (x:0,y:0,z:+)
        fni[ 5] = blkDDFBuf[ 5*n+i              +nbrpdz];
        //load f6 (x:-,y:+,z:-) from neighbor (x:+,y:0,z:+)
        fni[ 6] = blkDDFBuf[ 6*n+i+nbrpdx       +nbrpdz];
        //load f7 (x:0,y:+,z:-) from neighbor (x:0,y:0,z:+)
        fni[ 7] = blkDDFBuf[ 7*n+i              +nbrpdz];
        //load f8 (x:+,y:+,z:-) from neighbor (x:0,y:0,z:+)
        fni[ 8] = blkDDFBuf[ 8*n+i              +nbrpdz];

        //load f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
        fni[ 9] = blkDDFBuf[ 9*n+i+nbrpdx+nbrpdy       ];
        //load f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
        fni[10] = blkDDFBuf[10*n+i       +nbrpdy       ];
        //load f11(x:+,y:-,z:0) from neighbor (x:0,y:+,z:0)
        fni[11] = blkDDFBuf[11*n+i       +nbrpdy       ];
        //load f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
        fni[12] = blkDDFBuf[12*n+i+nbrpdx              ];
        //load f13(x:0,y:0,z:0) from current lattice
        fni[13] = blkDDFBuf[13*n+i                     ];
        //load f14(x:+,y:0,z:0) from current lattice
        fni[14] = blkDDFBuf[14*n+i                     ];
        //load f15(x:-,y:+,z:0) from neighbor (x:+,y:0,z:0)
        fni[15] = blkDDFBuf[15*n+i+nbrpdx              ];
        //load f16(x:0,y:+,z:0) from current lattice
        fni[16] = blkDDFBuf[16*n+i                     ];
        //load f17(x:+,y:+,z:0) from current lattice
        fni[17] = blkDDFBuf[17*n+i                     ];

        //load f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:0)
        fni[18] = blkDDFBuf[18*n+i+nbrpdx+nbrpdy       ];
        //load f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:0)
        fni[19] = blkDDFBuf[19*n+i       +nbrpdy       ];
        //load f20(x:+,y:-,z:+) from neighbor (x:0,y:+,z:0)
        fni[20] = blkDDFBuf[20*n+i       +nbrpdy       ];
        //load f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:0)
        fni[21] = blkDDFBuf[21*n+i+nbrpdx              ];
        //load f22(x:0,y:0,z:+) from current lattice
        fni[22] = blkDDFBuf[22*n+i                     ];
        //load f23(x:+,y:0,z:+) from current lattice
        fni[23] = blkDDFBuf[23*n+i                     ];
        //load f24(x:-,y:+,z:+) from neighbor (x:+,y:0,z:0)
        fni[24] = blkDDFBuf[24*n+i+nbrpdx              ];
        //load f25(x:0,y:+,z:+) from current lattice
        fni[25] = blkDDFBuf[25*n+i                     ];
        //load f26(x:+,y:+,z:+) from current lattice
        fni[26] = blkDDFBuf[26*n+i                     ];
    }
}

