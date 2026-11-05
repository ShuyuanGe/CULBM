#include <array>
#include <cstddef>
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
            std::uint32_t step = 0;
            std::uint32_t stepsPerBatch = 0;
            real_t invTau = 0;
            int q = 27;
            std::string collision = "bgk";
            Kernels kernels;
            std::uint32_t targetStep;
            uint3 devDim{};
            uint3 blockDim{};
            uint3 gridDim{};
            std::string dumpFolder = "data/multi_dev_output";
            bool dumpRho = false;
            bool dumpVx = false;
            bool dumpVy = false;
            bool dumpVz = false;
            std::vector<SingleDevData>     devices;
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
                app.add_option("--invTau", invTau, "Reciprocal of Tau.")
                    ->default_val(0.5);
                app.add_option("--velSet", q, "Velocity set: 15, 19, or 27.")
                    ->default_val(27)->check(CLI::IsMember({15, 19, 27}));
                app.add_option("--collision", collision, "Collision operator: bgk, mrt, or cumulant.")
                    ->default_val("bgk")->check(CLI::IsMember({"bgk", "mrt", "cumulant"}));
                app.add_option("--nstep", targetStep, "Total time steps of the simulation.")
                    ->default_val(1000);
                app.add_option("--dstep", stepsPerBatch, "Delta time steps of the simulation.")
                    ->default_val(100);
                addDimOption("--devDim", devDim, {1, 1, 1}, "Dimension of GPUs involved in the simulation.");
                addDimOption("--blkDim", blockDim, {32, 8, 4}, "Dimension of blocks per grid.");
                addDimOption("--gridDim", gridDim, {16, 32, 64}, "Dimension of grids per kernel.");
                app.add_option("--dumpFolder", dumpFolder, "Folder to store dumped fields.")
                    ->default_val("data/multi_dev_output");
                app.add_flag("--dumpRho", dumpRho, "Dump density every `dstep` steps.");
                app.add_flag("--dumpVx", dumpVx, "Dump vx every `dstep` steps.");
                app.add_flag("--dumpVy", dumpVy, "Dump vy every `dstep` steps.");
                app.add_flag("--dumpVz", dumpVz, "Dump vz every `dstep` steps.");
                try
                {
                    app.parse(argc, argv);
                }
                catch (const CLI::ParseError& e)
                {
                    std::exit(app.exit(e));
                }
                kernels = culbm::core::method::dispatch(q, collision, []<int NDir, culbm::core::method::Kind Kind>() {
                    return culbm::simulator::multi_dev::kernels<NDir, Kind>();
                });
                devices.resize(getDevNum());
            }

            std::uint32_t getDevNum() const
            {
                return devDim.x * devDim.y * devDim.z;
            }

            uint3 getDomainDim() const
            {
                return
                {
                    blockDim.x * gridDim.x,
                    blockDim.y * gridDim.y,
                    blockDim.z * gridDim.z
                };
            }

            std::size_t getDomainSize() const
            {
                const auto domainDim = getDomainDim();
                return static_cast<std::size_t>(domainDim.x) * domainDim.y * domainDim.z;
            }

            dim3 getBlockDim() const
            {
                return blockDim;
            }

            dim3 getGridDim() const
            {
                return gridDim;
            }

            void log() const
            {
                auto domainDim = getDomainDim();
                std::println(
                    "InvTau: {}\nTotal Time Step: {}\nDevice Dimension: [{},{},{}]\nSingle Device Domain Dimension: [{},{},{}]\nAll Device Domain Dimension: [{},{},{}]\n",
                    invTau, targetStep,
                    devDim.x, devDim.y, devDim.z,
                    domainDim.x, domainDim.y, domainDim.z,
                    devDim.x * domainDim.x, devDim.y * domainDim.y, devDim.z * domainDim.z
                );
            }
    };

    Simulator::Simulator(int argc, char** argv) :
        data(std::make_unique<Data>(argc, argv)),
        pool(data->getDevNum())
    {
        data->log();

        auto initDevData = [this](std::uint32_t devId, culbm::basic::Logger& logger, std::barrier<>& barrier)
        {
            error::check(cudaSetDevice(devId));
            const uint3 devDim = data->devDim;
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

            const auto domainSize = data->getDomainSize();
            Data::SingleDevData& device = data->devices[devId];
            error::check(cudaMalloc(&device.flagBuf, sizeof(flag_t) * domainSize));
            error::check(cudaMalloc(&device.rhoBuf,  sizeof(real_t) * domainSize));
            error::check(cudaMalloc(&device.vxBuf,   sizeof(real_t) * domainSize));
            error::check(cudaMalloc(&device.vyBuf,   sizeof(real_t) * domainSize));
            error::check(cudaMalloc(&device.vzBuf,   sizeof(real_t) * domainSize));
            error::check(cudaMalloc(&device.ddfBuf0, sizeof(ddf_t) * data->q * domainSize));
            error::check(cudaMalloc(&device.ddfBuf1, sizeof(ddf_t) * data->q * domainSize));
            error::check(cudaStreamCreate(&device.stream));
            error::check(cudaEventCreate(&device.start));
            error::check(cudaEventCreate(&device.end));

            logger.info("Allocate device memory successfully!");

            barrier.arrive_and_wait();

            InitKernelParam param;
            param.devDim = data->devDim;
            param.devIdx.x = devIdxX;
            param.devIdx.y = devIdxY;
            param.devIdx.z = devIdxZ;
            param.flagBuf = device.flagBuf;
            param.rhoBuf  = device.rhoBuf;
            param.vxBuf   = device.vxBuf;
            param.vyBuf   = device.vyBuf;
            param.vzBuf   = device.vzBuf;
            param.srcDDFBuf = device.ddfBuf0;
            param.dstDDFBuf = device.ddfBuf1;

            void* kernelArgs[] = {&param};
            error::check(
                cudaLaunchKernel(
                    data->kernels.init,
                    data->getGridDim(),
                    data->getBlockDim(),
                    kernelArgs,
                    0,
                    device.stream
                )
            );

            error::check(cudaStreamSynchronize(device.stream));

            logger.info("Set device data successfully!");
        };


        pool.addTask(initDevData);
        pool.waitAll();
    }

    void Simulator::run()
    {
        auto dumpRes = [this](std::uint32_t devIdx,
                              std::uint32_t stepBnd,
                              std::int32_t devIdxX,
                              std::int32_t devIdxY,
                              std::int32_t devIdxZ)
        {
            if (not (data->dumpRho || data->dumpVx || data->dumpVy || data->dumpVz)) return;

            const auto dumpFolder = std::filesystem::path{data->dumpFolder};
            std::filesystem::create_directories(dumpFolder);

            const auto nbytes = data->getDomainSize() * sizeof(real_t);
            auto hostBuf = std::make_unique<char[]>(nbytes);
            const auto& device = data->devices[devIdx];

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

            dumpField(data->dumpRho, "rho", device.rhoBuf);
            dumpField(data->dumpVx,  "vx",  device.vxBuf);
            dumpField(data->dumpVy,  "vy",  device.vyBuf);
            dumpField(data->dumpVz,  "vz",  device.vzBuf);
        };

        auto simulateLoop = [this, dumpRes](std::uint32_t devIdx, culbm::basic::Logger& logger, std::barrier<>& barrier)
        {
            const auto devDim = data->devDim;
            auto& device = data->devices[devIdx];
            KernelParam evenParam;
            KernelParam oddParam;
            void* kernelArgs[1] = {nullptr};

            evenParam.invTau  = oddParam.invTau  = data->invTau;
            evenParam.flagBuf = oddParam.flagBuf = device.flagBuf;
            evenParam.rhoBuf  = oddParam.rhoBuf  = device.rhoBuf;
            evenParam.vxBuf   = oddParam.vxBuf   = device.vxBuf;
            evenParam.vyBuf   = oddParam.vyBuf   = device.vyBuf;
            evenParam.vzBuf   = oddParam.vzBuf   = device.vzBuf;
            evenParam.dstDDFBuf = device.ddfBuf1;
            oddParam.dstDDFBuf  = device.ddfBuf0;

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
                evenParam.srcDDFBuf[dir] = data->devices[nbrIdx].ddfBuf0;
                oddParam.srcDDFBuf[dir]  = data->devices[nbrIdx].ddfBuf1;
            }

            barrier.arrive_and_wait();

            const auto stepBnd = std::min<std::uint32_t>(data->targetStep, data->step + data->stepsPerBatch);

            error::check(cudaEventRecord(device.start, device.stream));

            for (std::uint32_t step = data->step ; step < stepBnd ; ++step)
            {
                kernelArgs[0] = ((step % 2) == 0 ? &evenParam : &oddParam);

                error::check(
                    cudaLaunchKernel(
                        data->kernels.step,
                        data->getGridDim(),
                        data->getBlockDim(),
                        kernelArgs,
                        0,
                        device.stream
                    )
                );

                error::check(cudaStreamSynchronize(device.stream));

                barrier.arrive_and_wait();
            }

            error::check(cudaEventRecord(device.end, device.stream));

            float ms;
            error::check(cudaEventSynchronize(device.end));
            error::check(cudaEventElapsedTime(&ms, device.start, device.end));
            const float mlups = ((data->getDomainSize()) * 1e-6f) / (ms / 1000) * (stepBnd - data->step);
            logger.info(std::format("speed: {:.4f} (MLUPS)", mlups));

            dumpRes(devIdx, stepBnd, devIdxX, devIdxY, devIdxZ);
        };

        while (data->step < data->targetStep)
        {
            const auto step = std::min<std::int32_t>(data->targetStep - data->step, data->stepsPerBatch);
            pool.addTask(simulateLoop);
            pool.waitAll();
            data->step += step;
        }
    }

    Simulator::~Simulator()
    {
        auto deinitDevData = [this](std::uint32_t devId, culbm::basic::Logger& logger, std::barrier<>& barrier)
        {
            Data::SingleDevData& device = data->devices[devId];
            error::check(cudaEventDestroy(device.end));
            error::check(cudaEventDestroy(device.start));
            error::check(cudaStreamDestroy(device.stream));
            error::check(cudaFree(device.ddfBuf1));
            device.ddfBuf1 = nullptr;
            error::check(cudaFree(device.ddfBuf0));
            device.ddfBuf0 = nullptr;
            error::check(cudaFree(device.vzBuf));
            device.vzBuf = nullptr;
            error::check(cudaFree(device.vyBuf));
            device.vyBuf = nullptr;
            error::check(cudaFree(device.vxBuf));
            device.vzBuf = nullptr;
            error::check(cudaFree(device.rhoBuf));
            device.rhoBuf = nullptr;
            error::check(cudaFree(device.flagBuf));
            device.flagBuf = nullptr;
            logger.info("Deallocate device memory successfully!");
        };

        pool.addTask(deinitDevData);
    }
}
