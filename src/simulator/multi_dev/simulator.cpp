#include <array>
#include <format>
#include <vector>
#include <filesystem>
#include "types.hpp"
#include <print>
#include "CLI11.hpp"
#include "flags.hpp"
#include "kernel.cuh"
#include "method/dispatch.hpp"
#include "logger.hpp"
#include "simulator.hpp"
#include <cuda_runtime.h>
#include "stencil.hpp"
#include "exception.cuh"

namespace culbm::simulator::multi_dev
{
    namespace error = culbm::basic::error;

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
            real_t _invTau = 0;
            int _q = 27;
            std::string _collision = "bgk";
            Kernels _kernels;
            std::uint32_t _nStep;
            uint3 _devDim{};
            uint3 _blockDim{};
            uint3 _gridDim{};
            std::string _dumpFolder = "data/multi_dev_output";
            bool _dumpRho = false;
            bool _dumpVx = false;
            bool _dumpVy = false;
            bool _dumpVz = false;
            std::vector<SingleDevData>     _singleDevData;
        public:
            Data(int argc, char** argv)
            {
                CLI::App app {"Multiple Device Fluid Solver"};
                using DimValues = std::array<decltype(uint3::x), 3>;
                const auto addDimOption = [&](const char* name, uint3& dim, DimValues defaultValue, const char* description) {
                    return app.add_option_function<DimValues>(name, [&dim](const auto& values) {
                        dim = {values[0], values[1], values[2]};
                    }, description)->run_callback_for_default()->default_val(defaultValue);
                };
                app.add_option("--invTau", _invTau, "Reciprocal of Tau.")
                    ->default_val(0.5);
                app.add_option("--velSet", _q, "Velocity set: 15, 19, or 27.")
                    ->default_val(27)->check(CLI::IsMember({15, 19, 27}));
                app.add_option("--collision", _collision, "Collision operator: bgk, mrt, or cumulant.")
                    ->default_val("bgk")->check(CLI::IsMember({"bgk", "mrt", "cumulant"}));
                app.add_option("--nstep", _nStep, "Total time steps of the simulation.")
                    ->default_val(1000);
                app.add_option("--dstep", _dStep, "Delta time steps of the simulation.")
                    ->default_val(100);
                addDimOption("--devDim", _devDim, {1, 1, 1}, "Dimension of GPUs involved in the simulation.");
                addDimOption("--blkDim", _blockDim, {32, 8, 4}, "Dimension of blocks per grid.");
                addDimOption("--gridDim", _gridDim, {16, 32, 64}, "Dimension of grids per kernel.");
                app.add_option("--dumpFolder", _dumpFolder, "Folder to store dumped fields.")
                    ->default_val("data/multi_dev_output");
                app.add_flag("--dumpRho", _dumpRho, "Dump density every `dstep` steps.");
                app.add_flag("--dumpVx", _dumpVx, "Dump vx every `dstep` steps.");
                app.add_flag("--dumpVy", _dumpVy, "Dump vy every `dstep` steps.");
                app.add_flag("--dumpVz", _dumpVz, "Dump vz every `dstep` steps.");
                try
                {
                    app.parse(argc, argv);
                }
                catch (const CLI::ParseError& e)
                {
                    std::exit(app.exit(e));
                }
                _kernels = culbm::core::method::dispatch(_q, _collision, []<int NDir, culbm::core::method::Kind Kind>() {
                    return kernels<NDir, Kind>();
                });
                _singleDevData.resize(getDevNum());
            }

            std::uint32_t getDevNum() const
            {
                return _devDim.x * _devDim.y * _devDim.z;
            }

            uint3 getDomainDim() const
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
                return _blockDim;
            }

            dim3 getGridDim() const
            {
                return _gridDim;
            }

            void log() const
            {
                auto domainDim = getDomainDim();
                std::println(
                    "InvTau: {}\nTotal Time Step: {}\nDevice Dimension: [{},{},{}]\nSingle Device Domain Dimension: [{},{},{}]\nAll Device Domain Dimension: [{},{},{}]\n",
                    _invTau, _nStep,
                    _devDim.x, _devDim.y, _devDim.z,
                    domainDim.x, domainDim.y, domainDim.z,
                    _devDim.x * domainDim.x, _devDim.y * domainDim.y, _devDim.z * domainDim.z
                );
            }
    };

    Simulator::Simulator(int argc, char** argv) :
        _data(std::make_unique<Data>(argc, argv)),
        _pool(_data->getDevNum())
    {
        _data->log();

        auto initDevData = [this](std::uint32_t devId, culbm::basic::Logger& logger, std::barrier<>& barrier)
        {
            error::check(cudaSetDevice(devId));
            const uint3 devDim = _data->_devDim;
            const std::int32_t devIdxX = devId % devDim.x;
            const std::int32_t devIdxY = (devId / devDim.x) % devDim.y;
            const std::int32_t devIdxZ = devId / (devDim.x * devDim.y);
            using PeerDirections = culbm::core::Stencil<27>;
            for (std::uint32_t dir = 0 ; dir < 27 ; ++dir)
            {
                const auto d = PeerDirections::directions[dir];
                if (d.x == 0 and d.y == 0 and d.z == 0) continue;
                const std::int32_t nbrIdxX = devIdxX + d.x;
                const std::int32_t nbrIdxY = devIdxY + d.y;
                const std::int32_t nbrIdxZ = devIdxZ + d.z;
                if (
                    0 <= nbrIdxX and nbrIdxX < devDim.x and
                    0 <= nbrIdxY and nbrIdxY < devDim.y and
                    0 <= nbrIdxZ and nbrIdxZ < devDim.z
                )
                {
                    const std::int32_t nbrIdx = nbrIdxX + devDim.x * (nbrIdxY + devDim.y * nbrIdxZ);
                    error::check(cudaDeviceEnablePeerAccess(nbrIdx, 0));
                }
            }

            logger.info("Initialize cuda environment successfully!");

            barrier.arrive_and_wait();

            const auto domainSize = _data->getDomainSize();
            Data::SingleDevData& singleDevData = _data->_singleDevData[devId];
            error::check(cudaMalloc(&singleDevData.flagBuf, sizeof(flag_t) * domainSize));
            error::check(cudaMalloc(&singleDevData.rhoBuf,  sizeof(real_t) * domainSize));
            error::check(cudaMalloc(&singleDevData.vxBuf,   sizeof(real_t) * domainSize));
            error::check(cudaMalloc(&singleDevData.vyBuf,   sizeof(real_t) * domainSize));
            error::check(cudaMalloc(&singleDevData.vzBuf,   sizeof(real_t) * domainSize));
            error::check(cudaMalloc(&singleDevData.ddfBuf0, sizeof(ddf_t) * _data->_q * domainSize));
            error::check(cudaMalloc(&singleDevData.ddfBuf1, sizeof(ddf_t) * _data->_q * domainSize));
            error::check(cudaStreamCreate(&singleDevData.stream));
            error::check(cudaEventCreate(&singleDevData.start));
            error::check(cudaEventCreate(&singleDevData.end));

            logger.info("Allocate device memory successfully!");

            barrier.arrive_and_wait();

            InitKernelParam param;
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

            void* kernelArgs[] = {&param};
            error::check(
                cudaLaunchKernel(
                    _data->_kernels.init,
                    _data->getGridDim(),
                    _data->getBlockDim(),
                    kernelArgs,
                    0,
                    singleDevData.stream
                )
            );

            error::check(cudaStreamSynchronize(singleDevData.stream));

            logger.info("Set device data successfully!");
        };


        _pool.addTask(initDevData);
        _pool.waitAll();
    }

    void Simulator::run()
    {
        auto dumpRes = [this](std::uint32_t devIdx,
                              std::uint32_t stepBnd,
                              std::int32_t devIdxX,
                              std::int32_t devIdxY,
                              std::int32_t devIdxZ)
        {
            if (not (_data->_dumpRho || _data->_dumpVx || _data->_dumpVy || _data->_dumpVz)) return;

            const auto dumpFolder = std::filesystem::path{_data->_dumpFolder};
            std::filesystem::create_directories(dumpFolder);

            const auto nbytes = _data->getDomainSize() * sizeof(real_t);
            auto hostBuf = std::make_unique<char[]>(nbytes);
            const auto& singleDevData = _data->_singleDevData[devIdx];

            auto dumpField = [&](bool enabled, std::string_view prefix, const void *devicePtr)
            {
                if (not enabled) return;

                const auto filePath = dumpFolder / std::format("{}_{}_dev_{}_{}_{}.dat", prefix, stepBnd, devIdxX, devIdxY, devIdxZ);

                if (std::ofstream f(filePath, std::ios::binary); f)
                {
                    error::check(cudaMemcpy(hostBuf.get(), devicePtr, nbytes, cudaMemcpyKind::cudaMemcpyDefault));
                    f.write(hostBuf.get(), nbytes);
                }
                else
                {
                    throw std::runtime_error(std::format("Could not open the file: {}.\n", filePath.string()));
                }
            };

            dumpField(_data->_dumpRho, "rho", singleDevData.rhoBuf);
            dumpField(_data->_dumpVx,  "vx",  singleDevData.vxBuf);
            dumpField(_data->_dumpVy,  "vy",  singleDevData.vyBuf);
            dumpField(_data->_dumpVz,  "vz",  singleDevData.vzBuf);
        };

        auto simulateLoop = [this, dumpRes](std::uint32_t devIdx, culbm::basic::Logger& logger, std::barrier<>& barrier)
        {
            const auto devDim = _data->_devDim;
            auto& singleDevData = _data->_singleDevData[devIdx];
            KernelParam evenParam;
            KernelParam oddParam;
            void* kernelArgs[1] = {nullptr};

            evenParam.invTau  = oddParam.invTau  = _data->_invTau;
            evenParam.flagBuf = oddParam.flagBuf = singleDevData.flagBuf;
            evenParam.rhoBuf  = oddParam.rhoBuf  = singleDevData.rhoBuf;
            evenParam.vxBuf   = oddParam.vxBuf   = singleDevData.vxBuf;
            evenParam.vyBuf   = oddParam.vyBuf   = singleDevData.vyBuf;
            evenParam.vzBuf   = oddParam.vzBuf   = singleDevData.vzBuf;
            evenParam.dstDDFBuf = singleDevData.ddfBuf1;
            oddParam.dstDDFBuf  = singleDevData.ddfBuf0;

            const std::int32_t devIdxX = devIdx % devDim.x;
            const std::int32_t devIdxY = (devIdx / devDim.x) % devDim.y;
            const std::int32_t devIdxZ = devIdx / (devDim.x * devDim.y);

            for (std::uint32_t dir = 0 ; dir < 27 ; ++dir)
            {
                const auto d = culbm::core::Stencil<27>::directions[dir];
                const auto nbrIdxX = std::min<std::int32_t>(devDim.x - 1, std::max<std::int32_t>(0, devIdxX + d.x));
                const auto nbrIdxY = std::min<std::int32_t>(devDim.y - 1, std::max<std::int32_t>(0, devIdxY + d.y));
                const auto nbrIdxZ = std::min<std::int32_t>(devDim.z - 1, std::max<std::int32_t>(0, devIdxZ + d.z));
                const std::int32_t nbrIdx  = nbrIdxX + devDim.x * (nbrIdxY + devDim.y * nbrIdxZ);
                evenParam.srcDDFBuf[dir] = _data->_singleDevData[nbrIdx].ddfBuf0;
                oddParam.srcDDFBuf[dir]  = _data->_singleDevData[nbrIdx].ddfBuf1;
            }

            barrier.arrive_and_wait();

            const auto stepBnd = std::min<std::uint32_t>(_data->_nStep, _data->_step + _data->_dStep);

            error::check(cudaEventRecord(singleDevData.start, singleDevData.stream));

            for (std::uint32_t step = _data->_step ; step < stepBnd ; ++step)
            {
                kernelArgs[0] = ((step % 2) == 0 ? (void*)&evenParam : (void*)&oddParam);

                error::check(
                    cudaLaunchKernel(
                        _data->_kernels.step,
                        _data->getGridDim(),
                        _data->getBlockDim(),
                        kernelArgs,
                        0,
                        singleDevData.stream
                    )
                );

                error::check(cudaStreamSynchronize(singleDevData.stream));

                barrier.arrive_and_wait();
            }

            error::check(cudaEventRecord(singleDevData.end, singleDevData.stream));

            float ms;
            error::check(cudaEventSynchronize(singleDevData.end));
            error::check(cudaEventElapsedTime(&ms, singleDevData.start, singleDevData.end));
            const float mlups = ((_data->getDomainSize()) * 1e-6f) / (ms / 1000) * (stepBnd - _data->_step);
            logger.info(std::format("speed: {:.4f} (MLUPS)", mlups));

            dumpRes(devIdx, stepBnd, devIdxX, devIdxY, devIdxZ);
        };

        while (_data->_step < _data->_nStep)
        {
            const auto step = std::min<std::int32_t>(_data->_nStep - _data->_step, _data->_dStep);
            _pool.addTask(simulateLoop);
            _pool.waitAll();
            _data->_step += step;
        }
    }

    Simulator::~Simulator()
    {
        auto deinitDevData = [this](std::uint32_t devId, culbm::basic::Logger& logger, std::barrier<>& barrier)
        {
            Data::SingleDevData& singleDevData = _data->_singleDevData[devId];
            error::check(cudaEventDestroy(singleDevData.end));
            error::check(cudaEventDestroy(singleDevData.start));
            error::check(cudaStreamDestroy(singleDevData.stream));
            error::check(cudaFree(singleDevData.ddfBuf1));
            singleDevData.ddfBuf1 = nullptr;
            error::check(cudaFree(singleDevData.ddfBuf0));
            singleDevData.ddfBuf0 = nullptr;
            error::check(cudaFree(singleDevData.vzBuf));
            singleDevData.vzBuf = nullptr;
            error::check(cudaFree(singleDevData.vyBuf));
            singleDevData.vyBuf = nullptr;
            error::check(cudaFree(singleDevData.vxBuf));
            singleDevData.vzBuf = nullptr;
            error::check(cudaFree(singleDevData.rhoBuf));
            singleDevData.rhoBuf = nullptr;
            error::check(cudaFree(singleDevData.flagBuf));
            singleDevData.flagBuf = nullptr;
            logger.info("Deallocate device memory successfully!");
        };

        _pool.addTask(deinitDevData);
    }
}
