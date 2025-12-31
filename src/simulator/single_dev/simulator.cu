#include <format>
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>
#include <thrust/fill.h>
#include "simulator.hpp"
#include <cuda_runtime.h>
#include "cu_exception.cuh"
#include <thrust/device_ptr.h>
#include "blocking_alogrithm.hpp"
#include "block_based_kernel.cuh"

namespace culbm::simulator::single_dev
{
    class Simulator::Data
    {
        friend class Simulator;
        private:
            std::shared_ptr<Options> _opts;
            Vec3<u32> _blkDim;
            Vec3<u32> _numBlk;
            cudaMemPool_t _devMemPool;
            cudaStream_t _stream;
            cudaEvent_t _start, _end;
            ddf_t* _glbDDFSrcBuf = nullptr;
            ddf_t* _glbDDFDstBuf = nullptr;
            flag_t* _locFlagBufArr = nullptr;
            real_t* _glbRhoBuf = nullptr;
            real_t* _glbVxBuf = nullptr;
            real_t* _glbVyBuf = nullptr;
            real_t* _glbVzBuf = nullptr;
            ddf_t* _l2DDFBuf0 = nullptr;
            ddf_t* _l2DDFBuf1 = nullptr;
        public:

            explicit Data(std::shared_ptr<Options> opts) : _opts(opts)
            {
                using namespace culbm::basic;
                using namespace culbm::blocking_core;
                using namespace block_based_config;

                if(_opts->kernelType == Options::KernelType::staticKernel)
                {
                    _blkDim.x = BLOCK_DIM.x * GRID_DIM.x;
                    _blkDim.y = BLOCK_DIM.y * GRID_DIM.y;
                    _blkDim.z = BLOCK_DIM.z * GRID_DIM.z;
                    assert(validBlkAxisConfig<u32>(opts->domDim.x, _blkDim.x, BLOCKING_ITER));
                    assert(validBlkAxisConfig<u32>(opts->domDim.y, _blkDim.y, BLOCKING_ITER));
                    assert(validBlkAxisConfig<u32>(opts->domDim.z, _blkDim.z, BLOCKING_ITER));
                    _numBlk.x = calcBlkNum<u32>(opts->domDim.x, _blkDim.x, BLOCKING_ITER);
                    _numBlk.y = calcBlkNum<u32>(opts->domDim.y, _blkDim.y, BLOCKING_ITER);
                    _numBlk.z = calcBlkNum<u32>(opts->domDim.z, _blkDim.z, BLOCKING_ITER);
                }

                if(_opts->kernelType == Options::KernelType::dynamicKernel)
                {
                    _blkDim.x = opts->dyKernelParam->blkDim.x * opts->dyKernelParam->gridDim.x;
                    _blkDim.y = opts->dyKernelParam->blkDim.y * opts->dyKernelParam->gridDim.y;
                    _blkDim.z = opts->dyKernelParam->blkDim.z * opts->dyKernelParam->gridDim.z;
                    assert(validBlkAxisConfig<u32>(opts->domDim.x, _blkDim.x, opts->dyKernelParam->iter));
                    assert(validBlkAxisConfig<u32>(opts->domDim.y, _blkDim.y, opts->dyKernelParam->iter));
                    assert(validBlkAxisConfig<u32>(opts->domDim.z, _blkDim.z, opts->dyKernelParam->iter));
                    _numBlk.x = calcBlkNum<u32>(opts->domDim.x, _blkDim.x, opts->dyKernelParam->iter);
                    _numBlk.y = calcBlkNum<u32>(opts->domDim.y, _blkDim.y, opts->dyKernelParam->iter);
                    _numBlk.z = calcBlkNum<u32>(opts->domDim.z, _blkDim.z, opts->dyKernelParam->iter);
                }

                cu::check(cudaSetDevice(_opts->deviceId));
                cu::check(cudaStreamCreate(&_stream));
                cu::check(cudaEventCreate(&_start)); 
                cu::check(cudaEventCreate(&_end));

                auto allocDevMem = [=,this]() -> void
                {
                    cudaMemPoolProps props {};
                    props.allocType = cudaMemAllocationTypePinned;
                    props.handleTypes = cudaMemHandleTypeNone;
                    props.location.type = cudaMemLocationTypeDevice;
                    props.location.id = _opts->deviceId;
                    cu::check(cudaMemPoolCreate(&this->_devMemPool, &props));
                    const idx_t domCell = opts->domDim.x * opts->domDim.y * opts->domDim.z;
                    const idx_t blkNum  = _numBlk.x * _numBlk.y * _numBlk.z;
                    const idx_t blkSize = _blkDim.x * _blkDim.y * _blkDim.z;
                    cu::check(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&_glbDDFSrcBuf), sizeof(ddf_t)*NDIR*domCell, _devMemPool, _stream));
                    cu::check(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&_glbDDFDstBuf), sizeof(ddf_t)*NDIR*domCell, _devMemPool, _stream));
                    cu::check(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&_locFlagBufArr), sizeof(flag_t)*blkNum*blkSize, _devMemPool, _stream));
                    cu::check(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&_glbRhoBuf), sizeof(real_t)*domCell, _devMemPool, _stream));
                    cu::check(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&_glbVxBuf), sizeof(real_t)*domCell, _devMemPool, _stream));
                    cu::check(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&_glbVyBuf), sizeof(real_t)*domCell, _devMemPool, _stream));
                    cu::check(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&_glbVzBuf), sizeof(real_t)*domCell, _devMemPool, _stream));
                    cu::check(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&_l2DDFBuf0), sizeof(ddf_t)*NDIR*blkSize, _devMemPool, _stream));
                    cu::check(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&_l2DDFBuf1), sizeof(ddf_t)*NDIR*blkSize, _devMemPool, _stream));
                    cu::check(cudaStreamSynchronize(_stream));
                };

                allocDevMem();
            }

            Data(const Data&) = delete;

            ~Data() noexcept
            {
                using namespace culbm::basic;

                auto deAllocDevMem = [=, this]() -> void
                {
                    cu::check(cudaFreeAsync(reinterpret_cast<void*>(_l2DDFBuf1), _stream));
                    cu::check(cudaFreeAsync(reinterpret_cast<void*>(_l2DDFBuf0), _stream));
                    cu::check(cudaFreeAsync(reinterpret_cast<void*>(_glbVzBuf), _stream));
                    cu::check(cudaFreeAsync(reinterpret_cast<void*>(_glbVyBuf), _stream));
                    cu::check(cudaFreeAsync(reinterpret_cast<void*>(_glbVxBuf), _stream));
                    cu::check(cudaFreeAsync(reinterpret_cast<void*>(_glbRhoBuf), _stream));
                    cu::check(cudaFreeAsync(reinterpret_cast<void*>(_locFlagBufArr), _stream));
                    cu::check(cudaFreeAsync(reinterpret_cast<void*>(_glbDDFDstBuf), _stream));
                    cu::check(cudaFreeAsync(reinterpret_cast<void*>(_glbDDFSrcBuf), _stream));
                    cu::check(cudaStreamSynchronize(_stream));
                    cu::check(cudaMemPoolDestroy(_devMemPool));
                };

                deAllocDevMem();

                cu::check(cudaEventDestroy(_end));
                cu::check(cudaEventDestroy(_start));
                cu::check(cudaStreamDestroy(_stream));
            }
    };

    Simulator::Simulator(std::shared_ptr<Options> opts)
        : _data(std::make_unique<Data>(opts))
    {
        using namespace culbm::basic;

        auto initRhoUEqu = [=, this]() -> void
        {
            using namespace culbm::lbm_core;
            const idx_t domSize = _data->_opts->domDim.x * _data->_opts->domDim.y * _data->_opts->domDim.z;
            const real_t dftRho = 1, dftVx = 0, dftVy = 0, dftVz = 0;
            ddf_t fn[NDIR];
            bgk::calcEqu<NDIR>(dftRho, dftVx, dftVy, dftVz, std::begin(fn));
            thrust::device_ptr<real_t> rhoPtr = thrust::device_pointer_cast(_data->_glbRhoBuf);
            thrust::device_ptr<real_t> vxPtr = thrust::device_pointer_cast(_data->_glbVxBuf);
            thrust::device_ptr<real_t> vyPtr = thrust::device_pointer_cast(_data->_glbVyBuf);
            thrust::device_ptr<real_t> vzPtr = thrust::device_pointer_cast(_data->_glbVzBuf);
            thrust::device_ptr<real_t> srcDDFPtr = thrust::device_pointer_cast(_data->_glbDDFSrcBuf);
            thrust::device_ptr<real_t> dstDDFPtr = thrust::device_pointer_cast(_data->_glbDDFDstBuf);
            thrust::fill_n(rhoPtr, domSize, dftRho);
            thrust::fill_n(vxPtr, domSize, dftVx);
            thrust::fill_n(vyPtr, domSize, dftVy);
            thrust::fill_n(vzPtr, domSize, dftVz);
            for(idx_t dir = 0 ; dir < NDIR ; ++dir)
            {
                thrust::fill_n(srcDDFPtr+dir*domSize, domSize, fn[dir]);
                thrust::fill_n(dstDDFPtr+dir*domSize, domSize, fn[dir]);
            }
        };

        initRhoUEqu();

        auto getDomFlag = [=, this]() -> std::vector<flag_t>
        {
            const u32 domCell = opts->domDim.x * opts->domDim.y * opts->domDim.z;
            std::vector<flag_t> domFlag(domCell, FLUID_FLAG);

            for(u32 cellIdx=0 ; cellIdx<domCell ; ++cellIdx)
            {
                const u32 x = cellIdx % opts->domDim.x;
                const u32 y = (cellIdx / opts->domDim.x) % opts->domDim.y;
                const u32 z = cellIdx / (opts->domDim.x * opts->domDim.y);

                if((x == 0) or (x == opts->domDim.x-1) or (y == 0) or (y == opts->domDim.y-1) or (z == 0) or (z == opts->domDim.z-1))
                {
                    domFlag[cellIdx] = BOUNCE_BACK_FLAG;
                }
                
            }
            return domFlag;
        };

        auto domFlag = getDomFlag();

        auto blockDomFlag = [=, this, &domFlag]() -> std::vector<flag_t>
        {
            using namespace block_based_config;
            using namespace culbm::blocking_core;
            const u32 blkSize = _data->_blkDim.x * _data->_blkDim.y * _data->_blkDim.z;
            const u32 numBlk  = _data->_numBlk.x * _data->_numBlk.y * _data->_numBlk.z;
            std::vector<flag_t> blkFlagArr(numBlk * blkSize, 0);

            i64 istIdx = 0;

            for(u32 blkIdxZ=0 ; blkIdxZ<_data->_numBlk.z ; ++blkIdxZ)
            {
                const i64 blkZStBegin = calcValidPrev<i64>(blkIdxZ,  _data->_blkDim.z, _data->_numBlk.z, BLOCKING_ITER, _data->_opts->domDim.z);
                const i64 blkZStEnd   = calcValidPrev<i64>(blkIdxZ+1,_data->_blkDim.z, _data->_numBlk.z, BLOCKING_ITER, _data->_opts->domDim.z);
                const i64 blkZLdBegin = std::max<i64>(blkZStBegin-(BLOCKING_ITER-1), 0);
                const i64 blkZLdEnd   = std::min<i64>(blkZStEnd+(BLOCKING_ITER-1), _data->_opts->domDim.z);
                for(u32 blkIdxY=0 ; blkIdxY<_data->_numBlk.y ; ++blkIdxY)
                {
                    const i64 blkYStBegin = calcValidPrev<i64>(blkIdxY,  _data->_blkDim.y, _data->_numBlk.y, BLOCKING_ITER, _data->_opts->domDim.y);
                    const i64 blkYStEnd   = calcValidPrev<i64>(blkIdxY+1,_data->_blkDim.y, _data->_numBlk.y, BLOCKING_ITER, _data->_opts->domDim.y);
                    const i64 blkYLdBegin = std::max<i64>(blkYStBegin-(BLOCKING_ITER-1), 0);
                    const i64 blkYLdEnd   = std::min<i64>(blkYStEnd+(BLOCKING_ITER-1), _data->_opts->domDim.y);
                    for(u32 blkIdxX=0 ; blkIdxX<_data->_numBlk.x ; ++blkIdxX)
                    {
                        const i64 blkXStBegin = calcValidPrev<i64>(blkIdxX,  _data->_blkDim.x, _data->_numBlk.x, BLOCKING_ITER, _data->_opts->domDim.x);
                        const i64 blkXStEnd   = calcValidPrev<i64>(blkIdxX+1,_data->_blkDim.x, _data->_numBlk.x, BLOCKING_ITER, _data->_opts->domDim.x);
                        const i64 blkXLdBegin = std::max<i64>(blkXStBegin-(BLOCKING_ITER-1), 0);
                        const i64 blkXLdEnd   = std::min<i64>(blkXStEnd+(BLOCKING_ITER-1), _data->_opts->domDim.x);

                        for(i64 blkOffZ=0, glbOffZ=blkZLdBegin ; glbOffZ<blkZLdEnd ; ++blkOffZ, ++glbOffZ)
                        {
                            for(i64 blkOffY=0, glbOffY=blkYLdBegin ; glbOffY<blkYLdEnd ; ++blkOffY, ++glbOffY)
                            {
                                for(i64 blkOffX=0, glbOffX=blkXLdBegin ; glbOffX<blkXLdEnd ; ++blkOffX, ++glbOffX)
                                {
                                    const i64 glbOff = glbOffX + _data->_opts->domDim.x * (glbOffY + _data->_opts->domDim.y * glbOffZ);
                                    const i64 blkOff = blkOffX + _data->_blkDim.x * (blkOffY + _data->_blkDim.y * blkOffZ);

                                    if(
                                        blkZStBegin<=glbOffZ and glbOffZ<blkZStEnd and
                                        blkYStBegin<=glbOffY and glbOffY<blkYStEnd and
                                        blkXStBegin<=glbOffX and glbOffX<blkXStEnd
                                    )
                                    {
                                        blkFlagArr[istIdx+blkOff] = domFlag[glbOff] | CORRECT_BIT;
                                    }
                                    else
                                    {
                                        blkFlagArr[istIdx+blkOff] = domFlag[glbOff];
                                    }
                                }
                            }
                        }
                        istIdx += blkSize;
                    }
                }
            }

            return blkFlagArr;
        };
        
        auto blkFlagArr = blockDomFlag();

        cu::check(cudaMemcpy(reinterpret_cast<void*>(_data->_locFlagBufArr), reinterpret_cast<const void*>(blkFlagArr.data()), blkFlagArr.size() * sizeof(flag_t), cudaMemcpyDefault));
    }

    void Simulator::run(idx_t batch_step)
    {
        using namespace culbm::basic;
        using namespace culbm::lbm_core;
        using namespace culbm::blocking_core;
        using namespace block_based_config;

        BlockBasedKernelParam param
        {
            .xOff = 0, .yOff = 0, .zOff = 0, 
            .srcBuf = _data->_glbDDFSrcBuf, .dstBuf = _data->_glbDDFDstBuf, 
            .flagBuf = nullptr, 
            .rhoBuf = _data->_glbRhoBuf, 
            .vxBuf  = _data->_glbVxBuf, 
            .vyBuf  = _data->_glbVyBuf, 
            .vzBuf  = _data->_glbVzBuf, 
            .blkBuf0 = _data->_l2DDFBuf0, 
            .blkBuf1 = _data->_l2DDFBuf1
        };

#if !USE_STATIC_CONFIG
        param.domX = _data->_opts->domDim.x;
        param.domY = _data->_opts->domDim.y;
        param.domZ = _data->_opts->domDim.z;
#endif // USE_STATIC_CONFIG
        constexpr dim3 gridDim {GRID_DIM.x, GRID_DIM.y, GRID_DIM.z};
        constexpr dim3 blockDim {BLOCK_DIM.x, BLOCK_DIM.y, BLOCK_DIM.z};

        void* kernelArgs[1] = {reinterpret_cast<void*>(&param)};

        cu::check(cudaEventRecord(_data->_start, _data->_stream));

        const u32 blkSize = _data->_blkDim.x * _data->_blkDim.y * _data->_blkDim.z;
        const u32 blkNum = _data->_numBlk.x * _data->_numBlk.y * _data->_numBlk.z;
        for(u32 locStep=0 ; locStep<batch_step ; ++locStep)
        {
            param.flagBuf = _data->_locFlagBufArr;

            for(u32 blkIdx=0 ; blkIdx<blkNum ; ++blkIdx)
            {
                const u32 blkIdxX = blkIdx % _data->_numBlk.x;
                const u32 blkIdxY = (blkIdx / _data->_numBlk.x) % _data->_numBlk.y;
                const u32 blkIdxZ = blkIdx / (_data->_numBlk.x * _data->_numBlk.y);
                param.xOff = std::max<i32>(calcValidPrev<i32>(blkIdxX, _data->_blkDim.x, _data->_numBlk.x, BLOCKING_ITER, _data->_opts->domDim.x)-(BLOCKING_ITER-1), 0);
                param.yOff = std::max<i32>(calcValidPrev<i32>(blkIdxY, _data->_blkDim.y, _data->_numBlk.y, BLOCKING_ITER, _data->_opts->domDim.y)-(BLOCKING_ITER-1), 0);
                param.zOff = std::max<i32>(calcValidPrev<i32>(blkIdxZ, _data->_blkDim.z, _data->_numBlk.z, BLOCKING_ITER, _data->_opts->domDim.z)-(BLOCKING_ITER-1), 0);
                switch(_data->_opts->kernelType)
                {
                    case Options::KernelType::staticKernel:
                    {
                        cu::check(cudaLaunchCooperativeKernel((const void*)&staticBlockBasedKernel, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                        break;                    
                    }
                    case Options::KernelType::dynamicKernel:
                    {
                        if(BLOCKING_ITER == 1)
                        {
                            cu::check(cudaLaunchCooperativeKernel((const void*)&dynamicBlockBasedKernelSingleIter, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                        }
                        else
                        {
                            cu::check(cudaLaunchKernel((const void*)&dynamicBlockBasedKernelFirstIter, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                            #pragma unroll
                            for(i32 iter=1 ; iter<BLOCKING_ITER-1 ; ++iter)
                            {
                                cu::check(cudaLaunchKernel((const void*)&dynamicBlockBasedKernelInnerIter, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                                std::swap(_data->_l2DDFBuf0, _data->_l2DDFBuf1);
                            }
                            cu::check(cudaLaunchKernel((const void*)&dynamicBlockBasedKernelLastIter, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                        }
                        break;
                    }
                }

                param.flagBuf += blkSize;
            }

            std::swap(_data->_glbDDFSrcBuf, _data->_glbDDFDstBuf);
        }

        cu::check(cudaEventRecord(_data->_end, _data->_stream));
        cu::check(cudaEventSynchronize(_data->_end));
        float ms;
        cu::check(cudaEventElapsedTime(&ms, _data->_start, _data->_end));
        const u32 domSize = _data->_opts->domDim.x * _data->_opts->domDim.y * _data->_opts->domDim.z;
        const float mlups = (static_cast<float>(domSize) / 1024 / 1024) / (ms / 1000) * batch_step * BLOCKING_ITER;
        printf("[Info] speed = %.4f (MLUPS)\n", mlups);
    }

    Simulator::~Simulator() {}
}