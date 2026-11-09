#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector_types.h>
#include "core/types.hpp"
#include "mode.hpp"

namespace culbm::sim::single
{
    struct Config
    {
        int deviceId = 0;
        std::uint32_t stepsPerBatch = 0, targetStep = 0;
        real_t invTau = 0;
        dim3 blockDim{}, gridDim{}, domainDim{};
        Mode mode = Mode::Pull;
        int q = 27;
        std::string collision = "bgk";
        std::string initFolder, dumpFolder;
        bool dumpRho = false, dumpVx = false, dumpVy = false, dumpVz = false;
        bool interiorTilesOnly = false;
        int cacheHint = 1;
        bool disablePersistentL2 = false;
        std::int32_t innerLoop = 0;

        static Config parse(int argc, char** argv);

        std::size_t domainSize() const noexcept;
        dim3 launchExtent() const noexcept;
        std::uint32_t tileSize() const noexcept;
        dim3 tileCounts() const;
    };
}
