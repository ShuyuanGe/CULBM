#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector_types.h>
#include "core/types.hpp"

namespace culbm::sim::multi
{
    struct Config
    {
        std::uint32_t stepsPerBatch = 0, targetStep;
        real_t invTau = 0;
        int q = 27;
        std::string collision = "bgk";
        dim3 devDim{}, blockDim{}, gridDim{};
        std::string dumpFolder = "data/multi_dev_output";
        bool dumpRho = false, dumpVx = false, dumpVy = false, dumpVz = false;

        static Config parse(int argc, char** argv);
        std::uint32_t deviceCount() const noexcept;
        dim3 domainDim() const noexcept;
        std::size_t domainSize() const noexcept;
        uint3 coordinates(std::uint32_t index) const noexcept;
        void log() const;
    };
}
