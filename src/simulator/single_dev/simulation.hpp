#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <cuda/buffer>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>
#include <cuda_runtime.h>
#include "types.hpp"
#include "execution.hpp"

namespace culbm::simulator::single_dev
{
    struct SimulationConfig
    {
        int deviceId = 0;
        std::uint32_t stepsPerBatch = 0, targetStep = 0;
        real_t invTau = 0;
        dim3 blockDim{}, gridDim{}, domainDim{};
        StrategySelection strategy = PlainScheme::TwoFieldPull;
        int q = 27;
        std::string collision = "bgk";
        std::string initFolder, dumpFolder;
        bool dumpRho = false, dumpVx = false, dumpVy = false, dumpVz = false;
        bool interiorTilesOnly = false;
        int cacheHint = 1;
        bool disablePersistentL2 = false;
        std::int32_t innerLoop = 0;

        static SimulationConfig parse(int argc, char** argv);

        std::size_t domainSize() const noexcept;
        dim3 launchExtent() const noexcept;
        std::uint32_t tileSize() const noexcept;
        dim3 tileCounts() const;
    };

    struct SimulationState
    {
        const SimulationConfig& config;
        std::uint32_t step = 0;
        ::cuda::device_ref device;
        ::cuda::stream stream;
        ::cuda::device_memory_pool pool;
        ::cuda::timed_event startEvent, endEvent;
        ::cuda::device_buffer<real_t> rhoBuf, vxBuf, vyBuf, vzBuf;

        explicit SimulationState(const SimulationConfig& config);
    };
}
