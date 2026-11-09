#include <algorithm>
#include <format>
#include <memory>
#include <print>
#include <utility>
#include <cuda/launch>
#include <cuda/std/chrono>
#include "core/stencil.hpp"
#include "kernel.cuh"
#include "simulator.hpp"
#include "state.hpp"
#include "util/workers.hpp"

namespace culbm::sim::multi
{
    struct Simulator::Impl
    {
        State state;
        Kernels kernels;
        std::uint32_t step = 0;
        util::Workers workers;

        explicit Impl(Config config) :
            state(std::move(config)), kernels(Kernels::make(state.config.q, state.config.collision)),
            workers(state.config.deviceCount())
        {
            const auto& c = state.config;
            const auto localDim = c.localDim();
            std::println(
                "InvTau: {}\nTotal Time Step: {}\nDevice Dimension: [{},{},{}]\nSingle Device Domain Dimension: [{},{},{}]\nAll Device Domain Dimension: [{},{},{}]\n",
                c.invTau, c.targetStep,
                c.devDim.x, c.devDim.y, c.devDim.z,
                localDim.x, localDim.y, localDim.z,
                c.devDim.x * localDim.x, c.devDim.y * localDim.y, c.devDim.z * localDim.z
            );
            workers.submit([this](std::uint32_t id, util::Logger& logger, std::barrier<>& barrier) {
                state.init(id, kernels.init, logger, barrier);
            });
            workers.wait();
        }

        void advance(std::uint32_t devId, std::uint32_t endStep, util::Logger& logger, std::barrier<>& barrier)
        {
            const auto& config = state.config;
            const auto devDim = config.devDim;
            auto& device = *state.devices[devId];
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

            const auto position = config.coordinates(devId);
            const idx_t devIdxX = position.x, devIdxY = position.y, devIdxZ = position.z;

            for (dir_t dir = 0 ; dir < 27 ; ++dir)
            {
                const auto d = culbm::core::Stencil<27>::directions[dir];
                const auto nbrIdxX = std::clamp<std::int32_t>(devIdxX + d.x, 0, devDim.x - 1);
                const auto nbrIdxY = std::clamp<std::int32_t>(devIdxY + d.y, 0, devDim.y - 1);
                const auto nbrIdxZ = std::clamp<std::int32_t>(devIdxZ + d.z, 0, devDim.z - 1);
                const std::int32_t nbrIdx  = nbrIdxX + devDim.x * (nbrIdxY + devDim.y * nbrIdxZ);
                evenParam.srcDdf[dir] = state.devices[nbrIdx]->ddf0.data();
                oddParam.srcDdf[dir]  = state.devices[nbrIdx]->ddf1.data();
            }

            barrier.arrive_and_wait();

            device.startEvent.record(device.stream);

            for (std::uint32_t current = step ; current < endStep ; ++current)
            {
                ::cuda::launch(device.stream, launch, kernels.step, (current % 2) == 0 ? evenParam : oddParam);
                device.stream.sync();

                barrier.arrive_and_wait();
            }

            device.endEvent.record(device.stream);
            device.endEvent.sync();
            const float seconds = ::cuda::std::chrono::duration<float>{device.endEvent - device.startEvent}.count();
            const float mlups = config.localSize() * 1e-6f / seconds * (endStep - step);
            logger.info(std::format("speed: {:.4f} (MLUPS)", mlups));

            state.write(devId, endStep, position);
        }

        void run()
        {
            const auto& config = state.config;
            while (step < config.targetStep)
            {
                const auto endStep = step + std::min(config.targetStep - step, config.stepsPerBatch);
                workers.submit([this, endStep](std::uint32_t id, util::Logger& logger, std::barrier<>& barrier) {
                    advance(id, endStep, logger, barrier);
                });
                workers.wait();
                step = endStep;
            }
        }

        ~Impl()
        {
            // Workers drain this task before State is destroyed.
            workers.submit([this](std::uint32_t id, util::Logger& logger, std::barrier<>&) {
                state.clear(id, logger);
            });
        }
    };

    Simulator::Simulator(Config config) : impl(std::make_unique<Impl>(std::move(config))) {}

    void Simulator::run() { impl->run(); }

    Simulator::~Simulator() = default;
}
