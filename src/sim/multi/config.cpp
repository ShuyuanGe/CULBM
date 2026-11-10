#include <stdexcept>
#include <cuda/devices>
#include "sim/cli.hpp"
#include "config.hpp"

namespace culbm::sim::multi
{
    Config Config::parse(int argc, char** argv)
    {
        Config config;
        CLI::App app {"Multiple Device Fluid Solver"};
        app.add_option("--inv-tau", config.inv_tau, "Reciprocal of Tau.")
            ->default_val(0.5);
        app.add_option("--q", config.q, "Velocity set: 15, 19, or 27.")
            ->default_val(27)->check(CLI::IsMember({15, 19, 27}));
        app.add_option("--collision", config.collision, "Collision operator: bgk, mrt, or cumulant.")
            ->default_val("bgk")->check(CLI::IsMember({"bgk", "mrt", "cumulant"}));
        app.add_option("--steps", config.target_step, "Total time steps of the simulation.")
            ->required();
        app.add_option("--batch-steps", config.steps_per_batch, "Steps per MLUPS report and field dump.")
            ->required()->check(CLI::PositiveNumber);
        cli::add_dim_option(app, "--device-grid", config.dev_dim, "Dimension of GPUs involved in the simulation.")->required();
        cli::add_dim_option(app, "--block", config.block_dim, "Dimension of blocks per grid.")->required();
        cli::add_dim_option(app, "--grid", config.grid_dim, "Dimension of grids per kernel.")->required();
        app.add_option("--output", config.dump_folder, "Output folder; required when dumping fields.");
        app.add_flag("--dump-rho", config.dump_rho, "Dump density every `--batch-steps` steps.");
        app.add_flag("--dump-vx", config.dump_vx, "Dump vx every `--batch-steps` steps.");
        app.add_flag("--dump-vy", config.dump_vy, "Dump vy every `--batch-steps` steps.");
        app.add_flag("--dump-vz", config.dump_vz, "Dump vz every `--batch-steps` steps.");
        cli::parse(app, argc, argv);
        if ((config.dump_rho or config.dump_vx or config.dump_vy or config.dump_vz) and config.dump_folder.empty())
            throw std::invalid_argument("--output is required when dumping fields.");
        if (config.device_count() > ::cuda::devices.size())
            throw std::invalid_argument("Requested more devices than are available.");
        return config;
    }

    std::uint32_t Config::device_count() const noexcept
    {
        return dev_dim.x * dev_dim.y * dev_dim.z;
    }

    dim3 Config::local_dim() const noexcept
    {
        return
        {
            block_dim.x * grid_dim.x,
            block_dim.y * grid_dim.y,
            block_dim.z * grid_dim.z
        };
    }

    std::size_t Config::local_size() const noexcept
    {
        const auto local_dim = this->local_dim();
        return static_cast<std::size_t>(local_dim.x) * local_dim.y * local_dim.z;
    }

    uint3 Config::coordinates(std::uint32_t index) const noexcept
    {
        return {index % dev_dim.x, (index / dev_dim.x) % dev_dim.y, index / (dev_dim.x * dev_dim.y)};
    }
}
