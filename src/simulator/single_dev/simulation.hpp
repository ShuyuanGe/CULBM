#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <cuda_runtime.h>
#include "types.hpp"
#include "execution.hpp"
#include "cuda.hpp"

namespace culbm::simulator::single_dev
{
    struct SimulationConfig
    {
        int deviceId = 0;
        std::uint32_t stepsPerBatch = 0, targetStep = 0;
        real_t invTau = 0;
        uint3 blockDim{}, gridDim{}, domainDim{};
        StrategySelection strategy = PlainScheme::TwoFieldPull;
        int q = 27;
        std::string collision = "bgk";
        std::string initStateFolder, dumpFolder;
        bool dumpRho = false, dumpVx = false, dumpVy = false, dumpVz = false;
        bool interiorTilesOnly = false;
        int cacheHint = 1;
        bool disablePersistentL2 = false;
        std::int32_t innerLoop = 0;

        std::size_t domainSize() const noexcept;
        uint3 launchExtent() const noexcept;
        std::uint32_t tileSize() const noexcept;
        uint3 tileCounts() const;
    };

    struct SimulationState
    {
        const SimulationConfig& config;
        std::uint32_t step = 0;
        basic::cuda::Stream stream;
        basic::cuda::Event startEvent, endEvent;
        basic::cuda::DeviceBuffer<real_t> rhoBuf, vxBuf, vyBuf, vzBuf;

        explicit SimulationState(const SimulationConfig& config);
        void init();
    };

    SimulationConfig makeConfig(int argc, char** argv);
    void simulate(const SimulationConfig& config);
}
