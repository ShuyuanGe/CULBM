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
        app.add_option("--inv-tau", config.invTau, "Reciprocal of Tau.")
            ->default_val(0.5);
        app.add_option("--q", config.q, "Velocity set: 15, 19, or 27.")
            ->default_val(27)->check(CLI::IsMember({15, 19, 27}));
        app.add_option("--collision", config.collision, "Collision operator: bgk, mrt, or cumulant.")
            ->default_val("bgk")->check(CLI::IsMember({"bgk", "mrt", "cumulant"}));
        app.add_option("--steps", config.targetStep, "Total time steps of the simulation.")
            ->required();
        app.add_option("--batch-steps", config.stepsPerBatch, "Steps per MLUPS report and field dump.")
            ->required()->check(CLI::PositiveNumber);
        cli::addDimOption(app, "--device-grid", config.devDim, "Dimension of GPUs involved in the simulation.")->required();
        cli::addDimOption(app, "--block", config.blockDim, "Dimension of blocks per grid.")->required();
        cli::addDimOption(app, "--grid", config.gridDim, "Dimension of grids per kernel.")->required();
        app.add_option("--output", config.dumpFolder, "Output folder; required when dumping fields.");
        app.add_flag("--dump-rho", config.dumpRho, "Dump density every `--batch-steps` steps.");
        app.add_flag("--dump-vx", config.dumpVx, "Dump vx every `--batch-steps` steps.");
        app.add_flag("--dump-vy", config.dumpVy, "Dump vy every `--batch-steps` steps.");
        app.add_flag("--dump-vz", config.dumpVz, "Dump vz every `--batch-steps` steps.");
        cli::parse(app, argc, argv);
        if ((config.dumpRho or config.dumpVx or config.dumpVy or config.dumpVz) and config.dumpFolder.empty())
            throw std::invalid_argument("--output is required when dumping fields.");
        if (config.deviceCount() > ::cuda::devices.size())
            throw std::invalid_argument("Requested more devices than are available.");
        return config;
    }

    std::uint32_t Config::deviceCount() const noexcept
    {
        return devDim.x * devDim.y * devDim.z;
    }

    dim3 Config::localDim() const noexcept
    {
        return
        {
            blockDim.x * gridDim.x,
            blockDim.y * gridDim.y,
            blockDim.z * gridDim.z
        };
    }

    std::size_t Config::localSize() const noexcept
    {
        const auto localDim = this->localDim();
        return static_cast<std::size_t>(localDim.x) * localDim.y * localDim.z;
    }

    uint3 Config::coordinates(std::uint32_t index) const noexcept
    {
        return {index % devDim.x, (index / devDim.x) % devDim.y, index / (devDim.x * devDim.y)};
    }
}
