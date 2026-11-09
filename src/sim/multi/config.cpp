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
            ->default_val(1000);
        app.add_option("--batch-steps", config.stepsPerBatch, "Delta time steps of the simulation.")
            ->default_val(100)->check(CLI::PositiveNumber);
        cli::addDimOption(app, "--device-grid", config.devDim, {1, 1, 1}, "Dimension of GPUs involved in the simulation.");
        cli::addDimOption(app, "--block", config.blockDim, {32, 8, 4}, "Dimension of blocks per grid.");
        cli::addDimOption(app, "--grid", config.gridDim, {16, 32, 64}, "Dimension of grids per kernel.");
        app.add_option("--output", config.dumpFolder, "Folder to store dumped fields.")
            ->default_val("data/multi_dev_output");
        app.add_flag("--dump-rho", config.dumpRho, "Dump density every `--batch-steps` steps.");
        app.add_flag("--dump-vx", config.dumpVx, "Dump vx every `--batch-steps` steps.");
        app.add_flag("--dump-vy", config.dumpVy, "Dump vy every `--batch-steps` steps.");
        app.add_flag("--dump-vz", config.dumpVz, "Dump vz every `--batch-steps` steps.");
        cli::parse(app, argc, argv);
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
