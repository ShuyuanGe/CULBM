#include <bit>
#include <cstdint>
#include <assert.h>
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include "cu_exception.cuh"
#include "velocity_set.hpp"
#include <cooperative_groups.h>
#include "mix_block_stream_core.cuh"

using ddf_t = float;
using flag_t = std::uint32_t;

namespace cg = cooperative_groups;

struct StreamNNNKernelParam
{
    ddf_t* swapBuf = nullptr;
};

__global__ void StreamNNNKernel(const StreamNNNKernelParam __grid_constant__ param);

template<
    u32 DIR, 
    std::int32_t DX = culbm::basic::VelSet3D::template getDx<DIR>(),
    std::int32_t DY = culbm::basic::VelSet3D::template getDy<DIR>(), 
    std::int32_t DZ = culbm::basic::VelSet3D::template getDz<DIR>()
>
__device__ __forceinline__ void check(
    ddf_t fn, 
    std::int32_t glbx,  std::int32_t glby,  std::int32_t glbz, 
    std::int32_t glbnx, std::int32_t glbny, std::int32_t glbnz
)
{
    const std::int32_t srcidx = std::bit_cast<std::int32_t>(fn);
    const std::int32_t srcx = srcidx % glbnx;
    const std::int32_t srcy = (srcidx / glbnx) % glbny;
    const std::int32_t srcz = srcidx / (glbnx * glbny);
    const std::int32_t x_ = (DX==-1) ? glbnx-1 : (DX==1) ? 0 : glbnx;
    const std::int32_t y_ = (DY==-1) ? glbny-1 : (DY==1) ? 0 : glbny;
    const std::int32_t z_ = (DZ==-1) ? glbnz-1 : (DZ==1) ? 0 : glbnz;
    if(glbx!=x_ and glby!=y_ and glbz!=z_)
    {
        const bool valid = (srcx==glbx-DX) and (srcy==glby-DY) and (srcz==glbz-DZ);
        assert(valid);
    }
}

constexpr std::int32_t nTest = 5;

int main()
{
    try
    {
        CU_CHECK(cudaSetDevice(0));
        constexpr dim3 gridDim  {2,3,19};
        constexpr dim3 blockDim {32,16,2};
        cudaStream_t stream;
        cudaEvent_t start, end;
        StreamNNNKernelParam param;
        constexpr idx_t swapBufSize = sizeof(float) * 18 *
            (
                (gridDim.x+1)*gridDim.z*blockDim.z*gridDim.y*blockDim.y + 
                (gridDim.y+1)*gridDim.z*blockDim.z*gridDim.x*blockDim.x +
                (gridDim.z+1)*gridDim.y*blockDim.y*gridDim.x*blockDim.x
            );
        constexpr idx_t blkDDFBufSize = sizeof(float)*blockDim.z*blockDim.y*blockDim.x*24;
        CU_CHECK(cudaFuncSetAttribute((const void*)&StreamNNNKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, blkDDFBufSize));
        CU_CHECK(cudaMalloc(&param.swapBuf, swapBufSize));
        CU_CHECK(cudaStreamCreate(&stream));
        CU_CHECK(cudaEventCreate(&start));
        CU_CHECK(cudaEventCreate(&end));
        void* kernelArgs[1] = { (void*)&param };
        CU_CHECK(cudaLaunchCooperativeKernel((const void*)&StreamNNNKernel, gridDim, blockDim, std::begin(kernelArgs), blkDDFBufSize, stream));

        CU_CHECK(cudaEventRecord(start, stream));
        CU_CHECK(cudaLaunchCooperativeKernel((const void*)&StreamNNNKernel, gridDim, blockDim, std::begin(kernelArgs), blkDDFBufSize, stream));
        CU_CHECK(cudaEventRecord(end, stream));

        CU_CHECK(cudaEventSynchronize(end));
        float ms;
        CU_CHECK(cudaEventElapsedTime(&ms, start, end));
        const float mlups = ((gridDim.x*gridDim.y*gridDim.z*blockDim.x*blockDim.y*blockDim.z) * 1e-6f) / (ms / 1000);
        printf("speed = %.4f (MLUPS)\n", mlups);

        CU_CHECK(cudaEventDestroy(end));
        CU_CHECK(cudaEventDestroy(start));
        CU_CHECK(cudaStreamDestroy(stream));
        CU_CHECK(cudaFree(param.swapBuf));
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}

__global__ __launch_bounds__(1024) void 
StreamNNNKernel(const StreamNNNKernelParam __grid_constant__ param)
{
    extern __shared__ ddf_t blkDDFBuf[];
    const std::int32_t glbx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::int32_t glby = blockIdx.y * blockDim.y + threadIdx.y;
    const std::int32_t glbz = blockIdx.z * blockDim.z + threadIdx.z;
    const std::int32_t glbnx = gridDim.x * blockDim.x;
    const std::int32_t glbny = gridDim.y * blockDim.y;
    const std::int32_t glbnz = gridDim.z * blockDim.z;
    const std::int32_t glbidx = glbx + glbnx * (glby + glbny * glbz);

    ddf_t fn[27];
    std::fill_n(std::begin(fn), 27, std::bit_cast<ddf_t>(glbidx));


    culbm::simulator::single_dev::mix_block::StreamCore3D<27>::stream(std::begin(fn), blkDDFBuf, param.swapBuf);


    check< 0>(fn[ 0], glbx, glby, glbz, glbnx, glbny, glbnz);
    check< 1>(fn[ 1], glbx, glby, glbz, glbnx, glbny, glbnz);
    check< 2>(fn[ 2], glbx, glby, glbz, glbnx, glbny, glbnz);
    check< 3>(fn[ 3], glbx, glby, glbz, glbnx, glbny, glbnz);
    check< 4>(fn[ 4], glbx, glby, glbz, glbnx, glbny, glbnz);
    check< 5>(fn[ 5], glbx, glby, glbz, glbnx, glbny, glbnz);
    check< 6>(fn[ 6], glbx, glby, glbz, glbnx, glbny, glbnz);
    check< 7>(fn[ 7], glbx, glby, glbz, glbnx, glbny, glbnz);
    check< 8>(fn[ 8], glbx, glby, glbz, glbnx, glbny, glbnz);

    check< 9>(fn[ 9], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<10>(fn[10], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<11>(fn[11], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<12>(fn[12], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<14>(fn[14], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<15>(fn[15], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<16>(fn[16], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<17>(fn[17], glbx, glby, glbz, glbnx, glbny, glbnz);

    check<18>(fn[18], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<19>(fn[19], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<20>(fn[20], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<21>(fn[21], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<22>(fn[22], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<23>(fn[23], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<24>(fn[24], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<25>(fn[25], glbx, glby, glbz, glbnx, glbny, glbnz);
    check<26>(fn[26], glbx, glby, glbz, glbnx, glbny, glbnz);
}