#include <print>
#include <stdexcept>
#include <utility>
#include "sim/cli.hpp"
#include "core/blocking.hpp"
#include "config.hpp"

namespace culbm::sim::single
{
    namespace
    {
        struct Options
        {
            Config config;
            std::string stream, opt, buffer;

            Options(int argc, char** argv);
            void validate() const;
            Mode mode() const;
        };

        Options::Options(int argc, char** argv)
        {
            CLI::App app {"Single Device Fluid Solver"};
            app.add_option("--device", config.deviceId, "Device index of the GPU")
                ->default_val(0);

            app.add_option("--batch-steps", config.stepsPerBatch, "Delta time steps of the simulation.")
                ->default_val(100);

            app.add_option("--steps", config.targetStep, "Total time steps of the simulation.")
                ->default_val(1000);

            app.add_option("--inv-tau", config.invTau, "Reciprocal of Tau.")
                ->default_val(0.5);

            cli::addDimOption(app, "--block", config.blockDim, {32, 16, 2}, "Dimension of blocks per grid.");

            auto* gridOption = cli::addDimOption(app, "--grid", config.gridDim, {2, 3, 19}, "Dimension of grids per kernel.");

            cli::addDimOption(app, "--domain", config.globalDim, {256, 256, 256}, "Dimension of the global simulation domain.");

            app.add_option("--stream-policy", stream, "Streaming strategy for the computation buffer: pull or esotwist.")
                ->default_val("pull")->check(CLI::IsMember({"pull", "esotwist"}));

            app.add_option("--opt-policy", opt, "Computation organization: plain or blocking.")
                ->default_val("plain")->check(CLI::IsMember({"plain", "blocking"}));

            app.add_option("--cache-hint", config.cacheHint, "Cache hints for in-place L2 kernels: 0 = normal, 1 = hinted.")
                ->default_val(1)->check(CLI::Range(0, 1));
            app.add_flag("--no-persistent-l2", config.disablePersistentL2, "Do not configure the persisting L2 limit or access-policy window.");

            app.add_option("--q", config.q, "Velocity set: 15, 19, or 27.")
                ->default_val(27)->check(CLI::IsMember({15, 19, 27}));
            app.add_option("--collision", config.collision, "Collision operator: bgk, mrt, or cumulant.")
                ->default_val("bgk")->check(CLI::IsMember({"bgk", "mrt", "cumulant"}));

            app.add_option("--init", config.initFolder, "The folder contains initial state of the simulation.");

            app.add_option("--output", config.dumpFolder, "The folder contains final output.")
                ->default_val("data/left_inlet_right_outlet_256_256_256_output");

            app.add_flag("--dump-rho", config.dumpRho, "Enable dump the density every `--batch-steps` time steps in the simulation.");
            app.add_flag("--dump-vx", config.dumpVx, "Enable dumping the x-axis velocity every `--batch-steps` time steps in the simulation.");
            app.add_flag("--dump-vy", config.dumpVy, "Enable dumping the y-axis velocity every `--batch-steps` time steps in the simulation.");
            app.add_flag("--dump-vz", config.dumpVz, "Enable dumping the z-axis velocity every `--batch-steps` time steps in the simulation.");
            app.add_option("--ddf-buffer", buffer, "Global DDF storage: single, double, or ring.")
                ->default_val("double")->check(CLI::IsMember({"single", "double", "ring"}));
            app.add_flag("--interior-tiles-only", config.interiorTilesOnly, "Run only tiles that are non-boundary along all three axes.");

            app.add_option("--inner-steps", config.innerLoop, "# of iterations in halo-blocking.")
                ->default_val(5);

            cli::parse(app, argc, argv);

            if (opt == "plain" and gridOption->count() == 0 and
               config.globalDim.x % config.blockDim.x == 0 and config.globalDim.y % config.blockDim.y == 0 and config.globalDim.z % config.blockDim.z == 0)
            {
                config.gridDim = {config.globalDim.x / config.blockDim.x, config.globalDim.y / config.blockDim.y, config.globalDim.z / config.blockDim.z};
            }
        }

        void Options::validate() const
        {
            if (config.interiorTilesOnly and opt != "blocking")
            {
                throw std::runtime_error("--interior-tiles-only requires --opt-policy blocking.");
            }

            if (config.interiorTilesOnly)
            {
                const auto tileCounts = config.tileCounts();
                if (tileCounts.x < 3 or tileCounts.y < 3 or tileCounts.z < 3)
                {
                    throw std::runtime_error("--interior-tiles-only requires at least three tiles along each axis.");
                }
            }

            if (config.interiorTilesOnly and (config.dumpRho or config.dumpVx or config.dumpVy or config.dumpVz))
            {
                std::println("Warning: --interior-tiles-only is intended for profiling; dumped results are incorrect.");
            }
        }

        Mode Options::mode() const
        {
            if (stream == "pull" and opt == "plain" and buffer == "double")
                return Mode::Pull;
            if (stream == "esotwist" and opt == "plain" and buffer == "single")
                return Mode::Twist;
            if (stream == "esotwist" and opt == "blocking")
            {
                if (buffer == "double") return Mode::BlockDouble;
                if (buffer == "ring") return Mode::BlockRing;
            }
            throw std::invalid_argument("Supported <stream-policy, opt-policy, ddf-buffer> combinations: "
                                        "<pull, plain, double>, <esotwist, plain, single>, "
                                        "<esotwist, blocking, double>, <esotwist, blocking, ring>.");
        }
    }

    Config Config::parse(int argc, char** argv)
    {
        Options options(argc, argv);
        options.config.mode = options.mode();
        options.validate();
        return std::move(options.config);
    }

    std::size_t Config::globalSize() const noexcept
    {
        return static_cast<std::size_t>(globalDim.x) * globalDim.y * globalDim.z;
    }

    dim3 Config::launchDim() const noexcept
    {
        return {gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z};
    }

    std::uint32_t Config::tileSize() const noexcept
    {
        const auto dim = launchDim();
        return dim.x * dim.y * dim.z;
    }

    dim3 Config::tileCounts() const
    {
        using namespace culbm::core::blocking;
        const auto dim = launchDim();
        const bool valid =
            validAxis<std::uint32_t>(globalDim.x, dim.x, innerLoop) and
            validAxis<std::uint32_t>(globalDim.y, dim.y, innerLoop) and
            validAxis<std::uint32_t>(globalDim.z, dim.z, innerLoop);
        if (not valid)
        {
            throw std::invalid_argument("<Domain Dimension, Blocking Dimension and InnerLoop> don't match!");
        }
        return {
            tileCount<std::uint32_t>(globalDim.x, dim.x, innerLoop),
            tileCount<std::uint32_t>(globalDim.y, dim.y, innerLoop),
            tileCount<std::uint32_t>(globalDim.z, dim.z, innerLoop)
        };
    }
}
