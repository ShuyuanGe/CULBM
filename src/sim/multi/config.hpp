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
        std::uint32_t steps_per_batch = 0, target_step = 0;
        real_t inv_tau = 0;
        int q = 27;
        std::string collision = "bgk";
        dim3 dev_dim{}, block_dim{}, grid_dim{};
        std::string dump_folder;
        bool dump_rho = false, dump_vx = false, dump_vy = false, dump_vz = false;

        static Config parse(int argc, char** argv);
        std::uint32_t device_count() const noexcept;
        dim3 local_dim() const noexcept;
        std::size_t local_size() const noexcept;
        uint3 coordinates(std::uint32_t index) const noexcept;
    };
}
