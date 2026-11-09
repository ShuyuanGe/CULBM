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
        std::uint32_t stepsPerBatch = 0, targetStep = 0;
        real_t invTau = 0;
        int q = 27;
        std::string collision = "bgk";
        dim3 devDim{}, blockDim{}, gridDim{};
        std::string dumpFolder;
        bool dumpRho = false, dumpVx = false, dumpVy = false, dumpVz = false;

        static Config parse(int argc, char** argv);
        std::uint32_t deviceCount() const noexcept;
        dim3 localDim() const noexcept;
        std::size_t localSize() const noexcept;
        uint3 coordinates(std::uint32_t index) const noexcept;
    };
}
