#include <array>
#include <cstdlib>
#include <print>
#include <stdexcept>
#include <utility>
#include "CLI11.hpp"
#include "blocking.hpp"
#include "simulation.hpp"

namespace culbm::simulator::single_dev
{
    struct ParsedOptions
    {
        enum class StreamPolicy { PullStream, InplaceStream };
        enum class OptPolicy { Plain, OverlappedBlocking = 3 };
        enum class BufferOption { Double, Ring, EsoTwist };

        SimulationConfig config;
        StreamPolicy stream = StreamPolicy::PullStream;
        OptPolicy opt = OptPolicy::Plain;
        BufferOption buffer = BufferOption::Double;

        ParsedOptions(int argc, char** argv);
        void validate() const;
        StrategySelection strategy() const;
    };

    ParsedOptions::ParsedOptions(int argc, char** argv)
    {
        CLI::App app {"Single Device Fluid Solver"};
        using DimValues = std::array<decltype(uint3::x), 3>;
        const auto addDimOption = [&](const char* name, uint3& dim, DimValues defaultValue, const char* description) {
            return app.add_option_function<DimValues>(name, [&dim](const auto& values) {
                dim = {values[0], values[1], values[2]};
            }, description)->run_callback_for_default()->default_val(defaultValue);
        };
        app.add_option("--devId", config.deviceId, "Device index of the GPU")
            ->default_val(0);

        app.add_option("--dstep", config.stepsPerBatch, "Delta time steps of the simulation.")
            ->default_val(100);

        app.add_option("--nstep", config.targetStep, "Total time steps of the simulation.")
            ->default_val(1000);

        app.add_option("--invTau", config.invTau, "Reciprocal of Tau.")
            ->default_val(0.5);

        addDimOption("--blockDim", config.blockDim, {32, 16, 2}, "Dimension of blocks per grid.");

        auto* gridOption = addDimOption("--gridDim", config.gridDim, {2, 3, 19}, "Dimension of grids per kernel.");

        addDimOption("--domainDim", config.domainDim, {256, 256, 256}, "Dimension of the domain.");

        app.add_option("--streamPolicy", stream, "The stream policy of the solver. {[0]:PullStream, [1]:InplaceStream}.")
            ->default_val(StreamPolicy::PullStream)->check(CLI::IsMember({0, 1}));

        app.add_option("--optPolicy", opt, "Optimization: 0 = plain, 3 = overlapped blocking.")
            ->default_val(OptPolicy::Plain)->check(CLI::IsMember({0, 3}));

        app.add_option("--cacheHint", config.cacheHint, "Cache hints for in-place L2 kernels: 0 = normal, 1 = hinted.")
            ->default_val(1)->check(CLI::Range(0, 1));
        app.add_flag("--no-persistent-l2", config.disablePersistentL2, "Do not configure the persisting L2 limit or access-policy window.");

        app.add_option("--velSet", config.q, "Velocity set: 15, 19, or 27.")
            ->default_val(27)->check(CLI::IsMember({15, 19, 27}));
        app.add_option("--collision", config.collision, "Collision operator: bgk, mrt, or cumulant.")
            ->default_val("bgk")->check(CLI::IsMember({"bgk", "mrt", "cumulant"}));

        app.add_option("--initStateFolder", config.initFolder, "The folder contains initial state of the simulation.");

        app.add_option("--dumpFolder", config.dumpFolder, "The folder contains final output.")
            ->default_val("data/left_inlet_right_outlet_256_256_256_output");

        app.add_flag("--dumpRho", config.dumpRho, "Enable dump the density every `dstep` time steps in the simulation.");
        app.add_flag("--dumpVx", config.dumpVx, "Enable dumping the x-axis velocity every `dstep` time steps in the simulation.");
        app.add_flag("--dumpVy", config.dumpVy, "Enable dumping the y-axis velocity every `dstep` time steps in the simulation.");
        app.add_flag("--dumpVz", config.dumpVz, "Enable dumping the z-axis velocity every `dstep` time steps in the simulation.");
        app.add_option("--ddfBuffer", buffer, "DDF buffer mode: double, ring, or esotwist.")
            ->transform(CLI::CheckedTransformer({{"double", "0"}, {"ring", "1"}, {"esotwist", "2"}}))
            ->default_val(BufferOption::Double);
        app.add_flag("--interior-tiles-only", config.interiorTilesOnly, "Run only tiles that are non-boundary along all three axes.");

        app.add_option("--innerLoop", config.innerLoop, "# of iterations in halo-blocking.")
            ->default_val(5);

        try
        {
            app.parse(argc, argv);
        }
        catch (const CLI::ParseError& e)
        {
            std::exit(app.exit(e));
        }

        if (opt == OptPolicy::Plain and gridOption->count() == 0 and
           config.domainDim.x % config.blockDim.x == 0 and config.domainDim.y % config.blockDim.y == 0 and config.domainDim.z % config.blockDim.z == 0)
        {
            config.gridDim = {config.domainDim.x / config.blockDim.x, config.domainDim.y / config.blockDim.y, config.domainDim.z / config.blockDim.z};
        }
    }

    void ParsedOptions::validate() const
    {
        if ((opt == OptPolicy::Plain and stream != StreamPolicy::PullStream) or
           (opt == OptPolicy::OverlappedBlocking and stream != StreamPolicy::InplaceStream))
        {
            throw std::invalid_argument("Supported combinations are --streamPolicy 0 --optPolicy 0 and --streamPolicy 1 --optPolicy 3.");
        }

        if (buffer == BufferOption::Ring and (stream != StreamPolicy::InplaceStream or opt != OptPolicy::OverlappedBlocking))
        {
            throw std::runtime_error("--ddfBuffer ring requires --streamPolicy 1 --optPolicy 3.");
        }

        if (buffer == BufferOption::EsoTwist and opt != OptPolicy::Plain)
            throw std::runtime_error("--ddfBuffer esotwist requires plain CUDA.");

        if (config.interiorTilesOnly and (stream != StreamPolicy::InplaceStream or opt != OptPolicy::OverlappedBlocking))
        {
            throw std::runtime_error("--interior-tiles-only requires --streamPolicy 1 --optPolicy 3.");
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

    StrategySelection ParsedOptions::strategy() const
    {
        if (opt == OptPolicy::Plain)
        {
            if (buffer == BufferOption::Double)        return PlainScheme::TwoFieldPull;
            else if (buffer == BufferOption::EsoTwist) return PlainScheme::EsoTwist;
        }
        else if (opt == OptPolicy::OverlappedBlocking)
        {
            if (buffer == BufferOption::Double)    return BlockingGlobalBuffer::Double;
            else if (buffer == BufferOption::Ring) return BlockingGlobalBuffer::Ring;
        }
        std::unreachable();
    }

    SimulationConfig makeConfig(int argc, char** argv)
    {
        ParsedOptions options(argc, argv);
        options.validate();
        options.config.strategy = options.strategy();
        return std::move(options.config);
    }

    std::size_t SimulationConfig::domainSize() const noexcept
    {
        return static_cast<std::size_t>(domainDim.x) * domainDim.y * domainDim.z;
    }

    uint3 SimulationConfig::launchExtent() const noexcept
    {
        return {gridDim.x * blockDim.x, gridDim.y * blockDim.y, gridDim.z * blockDim.z};
    }

    std::uint32_t SimulationConfig::tileSize() const noexcept
    {
        const auto dim = launchExtent();
        return dim.x * dim.y * dim.z;
    }

    uint3 SimulationConfig::tileCounts() const
    {
        using namespace culbm::core::blocking;
        const auto dim = launchExtent();
        const bool valid =
            validAxis<std::uint32_t>(domainDim.x, dim.x, innerLoop) and
            validAxis<std::uint32_t>(domainDim.y, dim.y, innerLoop) and
            validAxis<std::uint32_t>(domainDim.z, dim.z, innerLoop);
        if (not valid)
        {
            throw std::invalid_argument("<Domain Dimension, Blocking Dimension and InnerLoop> don't match!");
        }
        return {
            tileCount<std::uint32_t>(domainDim.x, dim.x, innerLoop),
            tileCount<std::uint32_t>(domainDim.y, dim.y, innerLoop),
            tileCount<std::uint32_t>(domainDim.z, dim.z, innerLoop)
        };
    }
}
