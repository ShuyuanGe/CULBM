#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include "core/stencil.hpp"
#include "gpu/error.hpp"
#include "state.hpp"

namespace culbm::sim::multi
{
    Simulator::State::State(Config config) :
        config(std::move(config)), kernels(makeKernels(this->config.q, this->config.collision)),
        devices(this->config.deviceCount()) {}

    void Simulator::State::init(std::uint32_t devId, util::Logger& logger, std::barrier<>& barrier)
    {
        gpu::check(cudaSetDevice(devId));
        const auto devDim = config.devDim;
        const auto position = config.coordinates(devId);
        const idx_t devIdxX = position.x, devIdxY = position.y, devIdxZ = position.z;
        auto& device = devices[devId].emplace(devId, config.domainSize(), config.q);
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
        param.devIdx.x = devIdxX;
        param.devIdx.y = devIdxY;
        param.devIdx.z = devIdxZ;
        param.flagBuf = device.flagBuf.data();
        param.rhoBuf  = device.rhoBuf.data();
        param.vxBuf   = device.vxBuf.data();
        param.vyBuf   = device.vyBuf.data();
        param.vzBuf   = device.vzBuf.data();
        param.srcDdf = device.ddf0.data();
        param.dstDdf = device.ddf1.data();

        const auto launch = ::cuda::make_config(::cuda::grid_dims(config.gridDim), ::cuda::block_dims(config.blockDim));
        ::cuda::launch(device.stream, launch, kernels.init, param);
        device.stream.sync();

        logger.info("Set device data successfully!");
    }

    void Simulator::State::write(std::uint32_t devIdx, std::uint32_t stepBnd, uint3 position) const
    {
        if (not (config.dumpRho || config.dumpVx || config.dumpVy || config.dumpVz)) return;

        const auto dumpFolder = std::filesystem::path{config.dumpFolder};
        std::filesystem::create_directories(dumpFolder);

        const auto size = config.domainSize();
        auto hostBuf = std::make_unique_for_overwrite<real_t[]>(size);
        const auto& device = *devices[devIdx];

        auto dumpField = [&](bool enabled, std::string_view prefix, const real_t* devicePtr)
        {
            if (not enabled) return;

            const auto filePath = dumpFolder / std::format("{}_{}_dev_{}_{}_{}.dat", prefix, stepBnd, position.x, position.y, position.z);

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

        dumpField(config.dumpRho, "rho", device.rhoBuf.data());
        dumpField(config.dumpVx,  "vx",  device.vxBuf.data());
        dumpField(config.dumpVy,  "vy",  device.vyBuf.data());
        dumpField(config.dumpVz,  "vz",  device.vzBuf.data());
    }

    void Simulator::State::advance(std::uint32_t devIdx, std::uint32_t stepBnd, util::Logger& logger, std::barrier<>& barrier)
    {
        const auto devDim = config.devDim;
        auto& device = *devices[devIdx];
        KernelParam evenParam{
            .invTau = config.invTau,
            .flagBuf = device.flagBuf.data(),
            .rhoBuf = device.rhoBuf.data(),
            .vxBuf = device.vxBuf.data(),
            .vyBuf = device.vyBuf.data(),
            .vzBuf = device.vzBuf.data(),
            .dstDdf = device.ddf1.data()
        };
        auto oddParam = evenParam;
        oddParam.dstDdf = device.ddf0.data();
        const auto launch = ::cuda::make_config(::cuda::grid_dims(config.gridDim), ::cuda::block_dims(config.blockDim));

        const auto position = config.coordinates(devIdx);
        const idx_t devIdxX = position.x, devIdxY = position.y, devIdxZ = position.z;

        for (dir_t dir = 0 ; dir < 27 ; ++dir)
        {
            const auto d = culbm::core::Stencil<27>::directions[dir];
            const auto nbrIdxX = std::clamp<std::int32_t>(devIdxX + d.x, 0, devDim.x - 1);
            const auto nbrIdxY = std::clamp<std::int32_t>(devIdxY + d.y, 0, devDim.y - 1);
            const auto nbrIdxZ = std::clamp<std::int32_t>(devIdxZ + d.z, 0, devDim.z - 1);
            const std::int32_t nbrIdx  = nbrIdxX + devDim.x * (nbrIdxY + devDim.y * nbrIdxZ);
            evenParam.srcDdf[dir] = devices[nbrIdx]->ddf0.data();
            oddParam.srcDdf[dir]  = devices[nbrIdx]->ddf1.data();
        }

        barrier.arrive_and_wait();

        device.start.record(device.stream);

        for (std::uint32_t current = step ; current < stepBnd ; ++current)
        {
            ::cuda::launch(device.stream, launch, kernels.step, (current % 2) == 0 ? evenParam : oddParam);
            device.stream.sync();

            barrier.arrive_and_wait();
        }

        device.end.record(device.stream);
        device.end.sync();
        const float ms = (device.end - device.start).count() * 1e-6f;
        const float mlups = ((config.domainSize()) * 1e-6f) / (ms / 1000) * (stepBnd - step);
        logger.info(std::format("speed: {:.4f} (MLUPS)", mlups));

        write(devIdx, stepBnd, position);
    }

    void Simulator::State::clear(std::uint32_t devId, util::Logger& logger)
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
