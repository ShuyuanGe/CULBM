#include <format>
#include <vector>
#include "vec3.hpp"
#include <ostream>
#include "CLI11.hpp"
#include "config.hpp"
#include "kernel.cuh"
#include "logger.hpp"
#include "simulator.hpp"
#include <thrust/fill.h>
#include <cuda_runtime.h>
#include "velocity_set.hpp"
#include "cu_exception.cuh"
#include <thrust/execution_policy.h>

namespace gf::simulator::multi_dev
{
    class Simulator::Data
    {
        friend class Simulator;

        struct SingleDevData
        {
            flag_t *flagBuf = nullptr;
            real_t *rhoBuf  = nullptr;
            real_t *vxBuf   = nullptr;
            real_t *vyBuf   = nullptr;
            real_t *vzBuf   = nullptr;
            ddf_t  *ddfBuf0 = nullptr;
            ddf_t  *ddfBuf1 = nullptr;
            cudaStream_t stream;
            cudaEvent_t start, end;
        };

        private:
            std::uint32_t _step = 0;
            std::uint32_t _dStep = 0;
            float _invTau = 0;
            std::uint32_t _nStep;
            gf::basic::Vec3<std::uint32_t> _devDim;
            gf::basic::Vec3<std::uint32_t> _blockDim;
            gf::basic::Vec3<std::uint32_t> _gridDim;
            std::vector<SingleDevData>     _singleDevData;
        public:
            Data(int argc, char** argv)
            {
                CLI::App app {"Multiple Device Fluid Solver"};
                app.add_option("--invTau", _invTau, "Reciprocal of Tau.")
                    ->default_val(0.5);
                app.add_option("--nstep", _nStep, "Total time steps of the simulation.")
                    ->default_val(1000);
                app.add_option("--dstep", _dStep, "Delta time steps of the simulation.")
                    ->default_val(100);
                app.add_option("--devDim", _devDim.data, "Dimension of GPUs involved in the simulation.")
                    ->default_val(std::array<std::uint32_t,3>{1,1,1});
                app.add_option("--blkDim", _blockDim.data, "Dimension of blocks per grid.")
                    ->default_val(std::array<std::uint32_t,3>{32, 8, 4});
                app.add_option("--gridDim", _gridDim.data, "Dimension of grids per kernel.")
                    ->default_val(std::array<std::uint32_t,3>{16, 32, 64});
                try
                {
                    app.parse(argc, argv);
                }
                catch(const CLI::ParseError& e)
                {
                    std::exit(app.exit(e));
                }
                _singleDevData.resize(getDevNum());
            }

            std::uint32_t getDevNum() const
            {
                return _devDim.x * _devDim.y * _devDim.z;
            }

            gf::basic::Vec3<std::uint32_t> getDomainDim() const
            {
                return
                {
                    _blockDim.x * _gridDim.x, 
                    _blockDim.y * _gridDim.y,
                    _blockDim.z * _gridDim.z
                };
            }

            std::uint32_t getDomainSize() const
            {
                const auto domainDim = getDomainDim();
                return domainDim.x * domainDim.y * domainDim.z;
            }

            dim3 getBlockDim() const
            {
                return { _blockDim.x, _blockDim.y, _blockDim.z };
            }

            dim3 getGridDim() const
            {
                return { _gridDim.x, _gridDim.y, _gridDim.z };
            }

            std::ostream& log(std::ostream& os) const
            {
                auto domainDim = getDomainDim();
                os << std::format(
                    "InvTau: {}\nTotal Time Step: {}\nDevice Dimension: [{},{},{}]\nSingle Device Domain Dimension: [{},{},{}]\nAll Device Domain Dimension: [{},{},{}]\n",
                    _invTau, _nStep,
                    _devDim.x, _devDim.y, _devDim.z, 
                    domainDim.x, domainDim.y, domainDim.z, 
                    _devDim.x * domainDim.x, _devDim.y * domainDim.y, _devDim.z * domainDim.z
                ) << std::endl;
                return os;
            }
    };

    Simulator::Simulator(int argc, char** argv) : 
        _data(std::make_unique<Data>(argc, argv)), 
        _pool(_data->getDevNum())
    {
        _data->log(std::cout);

        auto initDevData = [this](std::uint32_t devId, gf::basic::Logger& logger, std::barrier<>& barrier)
        {
            CU_CHECK(cudaSetDevice(devId));
            const gf::basic::Vec3<std::uint32_t> devDim = _data->_devDim;
            const std::int32_t devIdxX = devId%devDim.x;
            const std::int32_t devIdxY = (devId/devDim.x)%devDim.y;
            const std::int32_t devIdxZ = devId/(devDim.x * devDim.y);
            for(std::uint32_t dir=0 ; dir<NDIR ; ++dir)
            {
                if(dir == gf::basic::VelSet3D::getCentIdx()) continue;
                const std::int32_t nbrIdxX = devIdxX + gf::basic::VelSet3D::getDxArr()[dir];
                const std::int32_t nbrIdxY = devIdxY + gf::basic::VelSet3D::getDyArr()[dir];
                const std::int32_t nbrIdxZ = devIdxZ + gf::basic::VelSet3D::getDzArr()[dir];
                if(
                    0<=nbrIdxX and nbrIdxX<devDim.x and 
                    0<=nbrIdxY and nbrIdxY<devDim.y and 
                    0<=nbrIdxZ and nbrIdxZ<devDim.z
                )
                {
                    const std::int32_t nbrIdx = nbrIdxX+devDim.x*(nbrIdxY+devDim.y*nbrIdxZ);
                    CU_CHECK(cudaDeviceEnablePeerAccess(nbrIdx, 0));
                }
            }

            LOG_INFO(logger, "Initialize cuda environment successfully!");

            barrier.arrive_and_wait();

            const std::uint32_t domainSize = _data->getDomainSize();
            Data::SingleDevData& singleDevData = _data->_singleDevData[devId];
            CU_CHECK(cudaMalloc(&singleDevData.flagBuf, sizeof(flag_t)*domainSize));
            CU_CHECK(cudaMalloc(&singleDevData.rhoBuf,  sizeof(real_t)*domainSize));
            CU_CHECK(cudaMalloc(&singleDevData.vxBuf,   sizeof(real_t)*domainSize));
            CU_CHECK(cudaMalloc(&singleDevData.vyBuf,   sizeof(real_t)*domainSize));
            CU_CHECK(cudaMalloc(&singleDevData.vzBuf,   sizeof(real_t)*domainSize));
            CU_CHECK(cudaMalloc(&singleDevData.ddfBuf0, sizeof(ddf_t)*NDIR*domainSize));
            CU_CHECK(cudaMalloc(&singleDevData.ddfBuf1, sizeof(ddf_t)*NDIR*domainSize));
            CU_CHECK(cudaStreamCreate(&singleDevData.stream));
            CU_CHECK(cudaEventCreate(&singleDevData.start));
            CU_CHECK(cudaEventCreate(&singleDevData.end));

            LOG_INFO(logger, "Allocate device memory successfully!");

            barrier.arrive_and_wait();

            InitKernelParam<27> param;
            void *kernelArgs[1] = {nullptr};
            param.devDim = _data->_devDim;
            param.devIdx.x = devIdxX;
            param.devIdx.y = devIdxY;
            param.devIdx.z = devIdxZ;
            param.flagBuf = singleDevData.flagBuf;
            param.rhoBuf  = singleDevData.rhoBuf;
            param.vxBuf   = singleDevData.vxBuf;
            param.vyBuf   = singleDevData.vyBuf;
            param.vzBuf   = singleDevData.vzBuf;
            param.srcDDFBuf = singleDevData.ddfBuf0;
            param.dstDDFBuf = singleDevData.ddfBuf1;
            
            kernelArgs[0] = reinterpret_cast<void*>(&param);
            CU_CHECK(
                cudaLaunchKernel(
                    (const void*)&D3Q27BGKInitKernel, 
                    _data->getGridDim(), 
                    _data->getBlockDim(), 
                    std::begin(kernelArgs), 
                    0, 
                    singleDevData.stream
                )
            );

            CU_CHECK(cudaStreamSynchronize(singleDevData.stream));

            LOG_INFO(logger, "Set device data successfully!");
        };


        _pool.addTask(initDevData);
        _pool.waitAll();
    }

    void Simulator::run()
    {
        auto simulateLoop = [this](std::uint32_t devIdx, gf::basic::Logger& logger, std::barrier<>& barrier)
        {
            const auto devDim = _data->_devDim;
            KernelParam<27> evenParam;
            KernelParam<27> oddParam;
            void* kernelArgs[1] = {nullptr};

            evenParam.invTau  = oddParam.invTau  = _data->_invTau;
            evenParam.flagBuf = oddParam.flagBuf = _data->_singleDevData[devIdx].flagBuf;
            evenParam.rhoBuf  = oddParam.rhoBuf  = _data->_singleDevData[devIdx].rhoBuf;
            evenParam.vxBuf   = oddParam.vxBuf   = _data->_singleDevData[devIdx].vxBuf;
            evenParam.vyBuf   = oddParam.vyBuf   = _data->_singleDevData[devIdx].vyBuf;
            evenParam.vzBuf   = oddParam.vzBuf   = _data->_singleDevData[devIdx].vzBuf;
            evenParam.dstDDFBuf = _data->_singleDevData[devIdx].ddfBuf1;
            oddParam.dstDDFBuf  = _data->_singleDevData[devIdx].ddfBuf0;

            const std::int32_t devIdxX = devIdx%devDim.x;
            const std::int32_t devIdxY = (devIdx/devDim.x)%devDim.y;
            const std::int32_t devIdxZ = devIdx/(devDim.x*devDim.y);

            for(std::uint32_t dir=0 ; dir<27 ; ++dir)
            {
                using VelSet = gf::basic::detail::VelSet3D<27>;
                const std::int32_t nbrIdxX = std::min<std::int32_t>(devDim.x-1, std::max<std::int32_t>(0, devIdxX+VelSet::getDxArr()[dir]));
                const std::int32_t nbrIdxY = std::min<std::int32_t>(devDim.y-1, std::max<std::int32_t>(0, devIdxY+VelSet::getDyArr()[dir]));
                const std::int32_t nbrIdxZ = std::min<std::int32_t>(devDim.z-1, std::max<std::int32_t>(0, devIdxZ+VelSet::getDzArr()[dir]));
                const std::int32_t nbrIdx  = nbrIdxX+devDim.x*(nbrIdxY+devDim.y*nbrIdxZ);
                evenParam.srcDDFBuf[dir] = _data->_singleDevData[nbrIdx].ddfBuf0;
                oddParam.srcDDFBuf[dir]  = _data->_singleDevData[nbrIdx].ddfBuf1;
            }

            barrier.arrive_and_wait();

            auto& singleDevData = _data->_singleDevData[devIdx];
            const std::uint32_t stepBnd = std::min<std::uint32_t>(_data->_nStep, _data->_step+_data->_dStep);

            CU_CHECK(cudaEventRecord(singleDevData.start, singleDevData.stream));

            for(std::uint32_t step = _data->_step ; step<stepBnd ; ++step)
            {
                kernelArgs[0] = ((step%2)==0 ? (void*)&evenParam : (void*)&oddParam);

                CU_CHECK(
                    cudaLaunchKernel(
                        (const void*)&D3Q27BGKKernel, 
                        _data->getGridDim(), 
                        _data->getBlockDim(), 
                        std::begin(kernelArgs),
                        0,
                        singleDevData.stream
                    )
                );

                CU_CHECK(cudaStreamSynchronize(singleDevData.stream));

                barrier.arrive_and_wait();
            }

            CU_CHECK(cudaEventRecord(singleDevData.end, singleDevData.stream));

            float ms;
            CU_CHECK(cudaEventSynchronize(singleDevData.end));
            CU_CHECK(cudaEventElapsedTime(&ms, singleDevData.start, singleDevData.end));
            const float mlups = ((_data->getDomainSize())*1e-6f) / (ms/1000) * (stepBnd - _data->_step);
            LOG_INFO(logger, std::format("speed: {:.4f} (MLUPS)", mlups));
        };

        while(_data->_step < _data->_nStep)
        {
            const auto step = std::min<std::int32_t>(_data->_nStep-_data->_step, _data->_dStep);
            _pool.addTask(simulateLoop);
            _pool.waitAll();
            _data->_step += step;
        }
    }

    Simulator::~Simulator()
    {
        auto deinitDevData = [this](std::uint32_t devId, gf::basic::Logger& logger, std::barrier<>& barrier)
        {
            Data::SingleDevData& singleDevData = _data->_singleDevData[devId];
            CU_CHECK(cudaEventDestroy(singleDevData.end));
            CU_CHECK(cudaEventDestroy(singleDevData.start));
            CU_CHECK(cudaStreamDestroy(singleDevData.stream));
            CU_CHECK(cudaFree(singleDevData.ddfBuf1));
            singleDevData.ddfBuf1 = nullptr;
            CU_CHECK(cudaFree(singleDevData.ddfBuf0));
            singleDevData.ddfBuf0 = nullptr;
            CU_CHECK(cudaFree(singleDevData.vzBuf));
            singleDevData.vzBuf = nullptr;
            CU_CHECK(cudaFree(singleDevData.vyBuf));
            singleDevData.vyBuf = nullptr;
            CU_CHECK(cudaFree(singleDevData.vxBuf));
            singleDevData.vzBuf = nullptr;
            CU_CHECK(cudaFree(singleDevData.rhoBuf));
            singleDevData.rhoBuf = nullptr;
            CU_CHECK(cudaFree(singleDevData.flagBuf));
            singleDevData.flagBuf = nullptr;
            LOG_INFO(logger, "Deallocate device memory successfully!");
        };

        _pool.addTask(deinitDevData);
    }
}