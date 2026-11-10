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
        int device_id = 0;
        std::uint32_t steps_per_batch = 0, target_step = 0;
        real_t inv_tau = 0;
        dim3 block_dim{}, grid_dim{}, global_dim{};
        Mode mode = Mode::pull;
        int q = 27;
        std::string collision = "bgk";
        std::string init_folder, dump_folder;
        bool dump_rho = false, dump_vx = false, dump_vy = false, dump_vz = false;
        bool interior_tiles_only = false;
        int cache_hint = 1;
        bool disable_persistent_l2 = false;
        std::int32_t inner_steps = 0;

        static Config parse(int argc, char** argv);

        std::size_t global_size() const noexcept;
        dim3 launch_dim() const noexcept;
        std::uint32_t tile_size() const noexcept;
        dim3 tile_counts() const;
    };
}
