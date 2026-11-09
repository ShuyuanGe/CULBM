#include <filesystem>
#include <format>
#include <memory>
#include <utility>
#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include "core/stencil.hpp"
#include "gpu/error.hpp"
#include "state.hpp"
#include "sim/output.hpp"
#include <tuple>

namespace culbm::sim::multi
{
    State::Device::Device(int id, std::size_t size, int q) :
        stream(::cuda::device_ref{id}), pool(::cuda::device_ref{id}),
        startEvent(::cuda::device_ref{id}), endEvent(::cuda::device_ref{id}),
        flagBuf(stream, pool.as_ref(), size, ::cuda::no_init),
        rhoBuf(stream, pool.as_ref(), size, ::cuda::no_init),
        vxBuf(stream, pool.as_ref(), size, ::cuda::no_init),
        vyBuf(stream, pool.as_ref(), size, ::cuda::no_init),
        vzBuf(stream, pool.as_ref(), size, ::cuda::no_init),
        ddf0(stream, pool.as_ref(), q * size, ::cuda::no_init),
        ddf1(stream, pool.as_ref(), q * size, ::cuda::no_init) {}

    State::State(Config config) :
        config(std::move(config)), devices(this->config.deviceCount()) {}

    void State::init(std::uint32_t devId, void (*initKernel)(InitKernelParam), util::Logger& logger, std::barrier<>& barrier)
    {
        gpu::check(cudaSetDevice(devId));
        const auto devDim = config.devDim;
        const auto position = config.coordinates(devId);
        const idx_t devIdxX = position.x, devIdxY = position.y, devIdxZ = position.z;
        auto& device = devices[devId].emplace(devId, config.localSize(), config.q);
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
        param.devDim = config.devDim;
        param.devPos.x = devIdxX;
        param.devPos.y = devIdxY;
        param.devPos.z = devIdxZ;
        param.flagBuf = device.flagBuf.data();
        param.rhoBuf  = device.rhoBuf.data();
        param.vxBuf   = device.vxBuf.data();
        param.vyBuf   = device.vyBuf.data();
        param.vzBuf   = device.vzBuf.data();
        param.srcDdf = device.ddf0.data();
        param.dstDdf = device.ddf1.data();

        const auto launch = ::cuda::make_config(::cuda::grid_dims(config.gridDim), ::cuda::block_dims(config.blockDim));
        ::cuda::launch(device.stream, launch, initKernel, param);
        device.stream.sync();

        logger.info("Set device data successfully!");
    }

    void State::write(std::uint32_t devId, std::uint32_t endStep, uint3 position) const
    {
        if (not (config.dumpRho || config.dumpVx || config.dumpVy || config.dumpVz)) return;

        const auto dumpFolder = std::filesystem::path{config.dumpFolder};
        std::filesystem::create_directories(dumpFolder);

        const auto size = config.localSize();
        auto hostBuf = std::make_unique_for_overwrite<real_t[]>(size);
        const auto& device = *devices[devId];

        for (const auto& [enabled, prefix, devicePtr] : {
            std::tuple{config.dumpRho, "rho", device.rhoBuf.data()},
            std::tuple{config.dumpVx,  "vx",  device.vxBuf.data()},
            std::tuple{config.dumpVy,  "vy",  device.vyBuf.data()},
            std::tuple{config.dumpVz,  "vz",  device.vzBuf.data()}
        })
        {
            if (not enabled) continue;
            const auto path = dumpFolder / std::format("{}_{}_dev_{}_{}_{}.dat", prefix, endStep, position.x, position.y, position.z);
            writeField(device.stream, devicePtr, {hostBuf.get(), size}, path);
        }
    }

    void State::clear(std::uint32_t devId, util::Logger& logger)
    {
        gpu::check(cudaSetDevice(devId));
        auto& device = *devices[devId];
        device.ddf1.destroy();
        device.ddf0.destroy();
        device.vzBuf.destroy();
        device.vyBuf.destroy();
        device.vxBuf.destroy();
        device.rhoBuf.destroy();
        device.flagBuf.destroy();
        device.stream.sync();
        devices[devId].reset();
        logger.info("Deallocate device memory successfully!");
    }
}
