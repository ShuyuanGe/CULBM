#include <array>
#include <cstddef>
#include <deque>
#include <format>
#include <optional>
#include <filesystem>
#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/memory_pool>
#include <cuda/std/span>
#include "types.hpp"
#include <print>
#include "CLI11.hpp"
#include "flags.hpp"
#include "kernel.cuh"
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
            ::cuda::stream stream;
            ::cuda::device_memory_pool pool;
            ::cuda::timed_event start, end;
            ::cuda::device_buffer<flag_t> flagBuf;
            ::cuda::device_buffer<real_t> rhoBuf, vxBuf, vyBuf, vzBuf;
            ::cuda::device_buffer<ddf_t> ddfBuf0, ddfBuf1;

            SingleDevData(int id, std::size_t size, int q) :
                stream(::cuda::device_ref{id}), pool(::cuda::device_ref{id}),
                start(::cuda::device_ref{id}), end(::cuda::device_ref{id}),
                flagBuf(stream, pool.as_ref(), size, ::cuda::no_init),
                rhoBuf(stream, pool.as_ref(), size, ::cuda::no_init),
                vxBuf(stream, pool.as_ref(), size, ::cuda::no_init),
                vyBuf(stream, pool.as_ref(), size, ::cuda::no_init),
                vzBuf(stream, pool.as_ref(), size, ::cuda::no_init),
                ddfBuf0(stream, pool.as_ref(), q * size, ::cuda::no_init),
                ddfBuf1(stream, pool.as_ref(), q * size, ::cuda::no_init) {}
        };

        private:
            std::uint32_t step = 0;
            std::uint32_t stepsPerBatch = 0;
            real_t invTau = 0;
            int q = 27;
            std::string collision = "bgk";
            Kernels kernels;
            std::uint32_t targetStep;
            dim3 devDim{};
            dim3 blockDim{};
            dim3 gridDim{};
            std::string dumpFolder = "data/multi_dev_output";
            bool dumpRho = false;
            bool dumpVx = false;
            bool dumpVy = false;
            bool dumpVz = false;
            std::deque<std::optional<SingleDevData>> devices;
        public:
            Data(int argc, char** argv)
            {
                CLI::App app {"Multiple Device Fluid Solver"};
                using DimValues = std::array<decltype(dim3::x), 3>;
                const auto addDimOption = [&](const char* name, dim3& dim, DimValues defaultValue, const char* description) {
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
                if (getDevNum() > ::cuda::devices.size())
                    throw std::invalid_argument("Requested more devices than are available.");
                kernels = makeKernels(q, collision);
                devices.resize(getDevNum());
            }

            std::uint32_t getDevNum() const
            {
                return devDim.x * devDim.y * devDim.z;
            }

            dim3 getDomainDim() const
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
            const auto devDim = data->devDim;
            const std::int32_t devIdxX = devId % devDim.x;
            const std::int32_t devIdxY = (devId / devDim.x) % devDim.y;
            const std::int32_t devIdxZ = devId / (devDim.x * devDim.y);
            auto& device = data->devices[devId].emplace(devId, data->getDomainSize(), data->q);
            using PeerDirections = culbm::core::Stencil<27>;
            for (dir_t dir = 0 ; dir < 27 ; ++dir)
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
                    device.pool.enable_access_from(::cuda::device_ref{nbrIdx});
                }
            }

            logger.info("Initialize cuda environment successfully!");

            barrier.arrive_and_wait();

            logger.info("Allocate device memory successfully!");

            barrier.arrive_and_wait();

            InitKernelParam param;
            param.devDim = data->devDim;
            param.devIdx.x = devIdxX;
            param.devIdx.y = devIdxY;
            param.devIdx.z = devIdxZ;
            param.flagBuf = device.flagBuf.data();
            param.rhoBuf  = device.rhoBuf.data();
            param.vxBuf   = device.vxBuf.data();
            param.vyBuf   = device.vyBuf.data();
            param.vzBuf   = device.vzBuf.data();
            param.srcDDFBuf = device.ddfBuf0.data();
            param.dstDDFBuf = device.ddfBuf1.data();

            const auto launch = ::cuda::make_config(::cuda::grid_dims(data->gridDim), ::cuda::block_dims(data->blockDim));
            ::cuda::launch(device.stream, launch, data->kernels.init, param);
            device.stream.sync();

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

            const auto size = data->getDomainSize();
            auto hostBuf = std::make_unique_for_overwrite<real_t[]>(size);
            const auto& device = *data->devices[devIdx];

            auto dumpField = [&](bool enabled, std::string_view prefix, const real_t* devicePtr)
            {
                if (not enabled) return;

                const auto filePath = dumpFolder / std::format("{}_{}_dev_{}_{}_{}.dat", prefix, stepBnd, devIdxX, devIdxY, devIdxZ);

                if (std::ofstream f(filePath, std::ios::binary); f)
                {
                    ::cuda::copy_bytes(device.stream, ::cuda::std::span{devicePtr, size}, ::cuda::std::span{hostBuf.get(), size});
                    device.stream.sync();
                    f.write(reinterpret_cast<const char*>(hostBuf.get()), size * sizeof(real_t));
                }
                else
                {
                    throw std::runtime_error(std::format("Could not open the file: {}.\n", filePath.string()));
                }
            };

            dumpField(data->dumpRho, "rho", device.rhoBuf.data());
            dumpField(data->dumpVx,  "vx",  device.vxBuf.data());
            dumpField(data->dumpVy,  "vy",  device.vyBuf.data());
            dumpField(data->dumpVz,  "vz",  device.vzBuf.data());
        };

        auto simulateLoop = [this, dumpRes](std::uint32_t devIdx, culbm::basic::Logger& logger, std::barrier<>& barrier)
        {
            const auto devDim = data->devDim;
            auto& device = *data->devices[devIdx];
            KernelParam evenParam{
                .invTau = data->invTau,
                .flagBuf = device.flagBuf.data(),
                .rhoBuf = device.rhoBuf.data(),
                .vxBuf = device.vxBuf.data(),
                .vyBuf = device.vyBuf.data(),
                .vzBuf = device.vzBuf.data(),
                .dstDDFBuf = device.ddfBuf1.data()
            };
            auto oddParam = evenParam;
            oddParam.dstDDFBuf = device.ddfBuf0.data();
            const auto launch = ::cuda::make_config(::cuda::grid_dims(data->gridDim), ::cuda::block_dims(data->blockDim));

            const std::int32_t devIdxX = devIdx % devDim.x;
            const std::int32_t devIdxY = (devIdx / devDim.x) % devDim.y;
            const std::int32_t devIdxZ = devIdx / (devDim.x * devDim.y);

            for (dir_t dir = 0 ; dir < 27 ; ++dir)
            {
                const auto d = culbm::core::Stencil<27>::directions[dir];
                const auto nbrIdxX = std::min<std::int32_t>(devDim.x - 1, std::max(0, devIdxX + d.x));
                const auto nbrIdxY = std::min<std::int32_t>(devDim.y - 1, std::max(0, devIdxY + d.y));
                const auto nbrIdxZ = std::min<std::int32_t>(devDim.z - 1, std::max(0, devIdxZ + d.z));
                const std::int32_t nbrIdx  = nbrIdxX + devDim.x * (nbrIdxY + devDim.y * nbrIdxZ);
                evenParam.srcDDFBuf[dir] = data->devices[nbrIdx]->ddfBuf0.data();
                oddParam.srcDDFBuf[dir]  = data->devices[nbrIdx]->ddfBuf1.data();
            }

            barrier.arrive_and_wait();

            const auto stepBnd = std::min(data->targetStep, data->step + data->stepsPerBatch);

            device.start.record(device.stream);

            for (std::uint32_t step = data->step ; step < stepBnd ; ++step)
            {
                ::cuda::launch(device.stream, launch, data->kernels.step, (step % 2) == 0 ? evenParam : oddParam);
                device.stream.sync();

                barrier.arrive_and_wait();
            }

            device.end.record(device.stream);
            device.end.sync();
            const float ms = (device.end - device.start).count() * 1e-6f;
            const float mlups = ((data->getDomainSize()) * 1e-6f) / (ms / 1000) * (stepBnd - data->step);
            logger.info(std::format("speed: {:.4f} (MLUPS)", mlups));

            dumpRes(devIdx, stepBnd, devIdxX, devIdxY, devIdxZ);
        };

        while (data->step < data->targetStep)
        {
            const auto step = std::min(data->targetStep - data->step, data->stepsPerBatch);
            pool.addTask(simulateLoop);
            pool.waitAll();
            data->step += step;
        }
    }

    Simulator::~Simulator()
    {
        auto deinitDevData = [this](std::uint32_t devId, culbm::basic::Logger& logger, std::barrier<>& barrier)
        {
            error::check(cudaSetDevice(devId));
            auto& device = *data->devices[devId];
            device.ddfBuf1.destroy();
            device.ddfBuf0.destroy();
            device.vzBuf.destroy();
            device.vyBuf.destroy();
            device.vxBuf.destroy();
            device.rhoBuf.destroy();
            device.flagBuf.destroy();
            device.stream.sync();
            data->devices[devId].reset();
            logger.info("Deallocate device memory successfully!");
        };

        pool.addTask(deinitDevData);
    }
}
