#include <array>
#include <cstdlib>
#include <print>
#include <stdexcept>
#include <cuda/devices>
#include "CLI11.hpp"
#include "config.hpp"

namespace culbm::sim::multi
{
    Config Config::parse(int argc, char** argv)
    {
        Config config;
        CLI::App app {"Multiple Device Fluid Solver"};
        using DimValues = std::array<decltype(dim3::x), 3>;
        const auto addDimOption = [&](const char* name, dim3& dim, DimValues defaultValue, const char* description) {
            return app.add_option_function<DimValues>(name, [&dim](const auto& values) {
                dim = {values[0], values[1], values[2]};
            }, description)->run_callback_for_default()->default_val(defaultValue);
        };
        app.add_option("--invTau", config.invTau, "Reciprocal of Tau.")
            ->default_val(0.5);
        app.add_option("--velSet", config.q, "Velocity set: 15, 19, or 27.")
            ->default_val(27)->check(CLI::IsMember({15, 19, 27}));
        app.add_option("--collision", config.collision, "Collision operator: bgk, mrt, or cumulant.")
            ->default_val("bgk")->check(CLI::IsMember({"bgk", "mrt", "cumulant"}));
        app.add_option("--nstep", config.targetStep, "Total time steps of the simulation.")
            ->default_val(1000);
        app.add_option("--dstep", config.stepsPerBatch, "Delta time steps of the simulation.")
            ->default_val(100);
        addDimOption("--devDim", config.devDim, {1, 1, 1}, "Dimension of GPUs involved in the simulation.");
        addDimOption("--blkDim", config.blockDim, {32, 8, 4}, "Dimension of blocks per grid.");
        addDimOption("--gridDim", config.gridDim, {16, 32, 64}, "Dimension of grids per kernel.");
        app.add_option("--dumpFolder", config.dumpFolder, "Folder to store dumped fields.")
            ->default_val("data/multi_dev_output");
        app.add_flag("--dumpRho", config.dumpRho, "Dump density every `dstep` steps.");
        app.add_flag("--dumpVx", config.dumpVx, "Dump vx every `dstep` steps.");
        app.add_flag("--dumpVy", config.dumpVy, "Dump vy every `dstep` steps.");
        app.add_flag("--dumpVz", config.dumpVz, "Dump vz every `dstep` steps.");
        try
        {
            app.parse(argc, argv);
        }
        catch (const CLI::ParseError& e)
        {
            std::exit(app.exit(e));
        }
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
        const auto domainDim = this->localDim();
        return static_cast<std::size_t>(domainDim.x) * domainDim.y * domainDim.z;
    }

    uint3 Config::coordinates(std::uint32_t index) const noexcept
    {
        return {index % devDim.x, (index / devDim.x) % devDim.y, index / (devDim.x * devDim.y)};
    }
}
