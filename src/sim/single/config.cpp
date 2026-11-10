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
            app.add_option("--device", config.device_id, "Device index of the GPU")
                ->default_val(0);

            app.add_option("--batch-steps", config.steps_per_batch, "Steps per MLUPS report and field dump.")
                ->required()->check(CLI::PositiveNumber);

            app.add_option("--steps", config.target_step, "Total time steps of the simulation.")
                ->required();

            app.add_option("--inv-tau", config.inv_tau, "Reciprocal of Tau.")
                ->default_val(0.5);

            cli::add_dim_option(app, "--block", config.block_dim, "Dimension of blocks per grid.")->required();

            auto* grid_option = cli::add_dim_option(app, "--grid", config.grid_dim, "Grid dimensions; required for blocking and when the plain grid cannot be derived.");

            cli::add_dim_option(app, "--domain", config.global_dim, "Dimension of the global simulation domain.")->required();

            app.add_option("--stream-policy", stream, "Streaming strategy for the computation buffer: pull or esotwist.")
                ->required()->check(CLI::IsMember({"pull", "esotwist"}));

            app.add_option("--opt-policy", opt, "Computation organization: plain or blocking.")
                ->required()->check(CLI::IsMember({"plain", "blocking"}));

            app.add_option("--cache-hint", config.cache_hint, "Cache hints for in-place L2 kernels: 0 = normal, 1 = hinted.")
                ->default_val(1)->check(CLI::Range(0, 1));
            app.add_flag("--no-persistent-l2", config.disable_persistent_l2, "Do not configure the persisting L2 limit or access-policy window.");

            app.add_option("--q", config.q, "Velocity set: 15, 19, or 27.")
                ->default_val(27)->check(CLI::IsMember({15, 19, 27}));
            app.add_option("--collision", config.collision, "Collision operator: bgk, mrt, or cumulant.")
                ->default_val("bgk")->check(CLI::IsMember({"bgk", "mrt", "cumulant"}));

            app.add_option("--init", config.init_folder, "The folder contains initial state of the simulation.");

            app.add_option("--output", config.dump_folder, "Output folder; required when dumping fields.");

            app.add_flag("--dump-rho", config.dump_rho, "Enable dump the density every `--batch-steps` time steps in the simulation.");
            app.add_flag("--dump-vx", config.dump_vx, "Enable dumping the x-axis velocity every `--batch-steps` time steps in the simulation.");
            app.add_flag("--dump-vy", config.dump_vy, "Enable dumping the y-axis velocity every `--batch-steps` time steps in the simulation.");
            app.add_flag("--dump-vz", config.dump_vz, "Enable dumping the z-axis velocity every `--batch-steps` time steps in the simulation.");
            app.add_flag("--dump-ddf", config.dump_ddf, "Dump raw full-domain DDF every `--batch-steps` time steps.");
            app.add_option("--ddf-buffer", buffer, "Global DDF storage: single, double, or ring.")
                ->required()->check(CLI::IsMember({"single", "double", "ring"}));
            app.add_flag("--interior-tiles-only", config.interior_tiles_only, "Run only tiles that are non-boundary along all three axes.");

            auto* inner_option = app.add_option("--inner-steps", config.inner_steps, "Iterations in halo-blocking; required for blocking.")
                ->check(CLI::PositiveNumber);

            cli::parse(app, argc, argv);

            if ((config.dump_rho or config.dump_vx or config.dump_vy or config.dump_vz or config.dump_ddf) and config.dump_folder.empty())
                throw std::invalid_argument("--output is required when dumping fields.");
            if (opt == "blocking" and (grid_option->count() == 0 or inner_option->count() == 0))
                throw std::invalid_argument("--grid and --inner-steps are required for blocking.");
            if (opt == "plain" and grid_option->count() == 0 and
                (config.global_dim.x % config.block_dim.x != 0 or config.global_dim.y % config.block_dim.y != 0 or config.global_dim.z % config.block_dim.z != 0))
                throw std::invalid_argument("--grid is required when --domain is not divisible by --block.");

            if (opt == "plain" and grid_option->count() == 0 and
                config.global_dim.x % config.block_dim.x == 0 and config.global_dim.y % config.block_dim.y == 0 and config.global_dim.z % config.block_dim.z == 0)
            {
                config.grid_dim = {config.global_dim.x / config.block_dim.x, config.global_dim.y / config.block_dim.y, config.global_dim.z / config.block_dim.z};
            }
        }

        void Options::validate() const
        {
            if (config.interior_tiles_only and opt != "blocking")
            {
                throw std::runtime_error("--interior-tiles-only requires --opt-policy blocking.");
            }

            if (opt == "blocking")
            {
                const auto tile_counts = config.tile_counts();
                if (config.interior_tiles_only and (tile_counts.x < 3 or tile_counts.y < 3 or tile_counts.z < 3))
                {
                    throw std::runtime_error("--interior-tiles-only requires at least three tiles along each axis.");
                }
            }

            if (config.interior_tiles_only and (config.dump_rho or config.dump_vx or config.dump_vy or config.dump_vz or config.dump_ddf))
            {
                std::println("Warning: --interior-tiles-only is intended for profiling; dumped results are incorrect.");
            }
        }

        Mode Options::mode() const
        {
            if (stream == "pull" and opt == "plain" and buffer == "double")
                return Mode::pull;
            if (stream == "esotwist" and opt == "plain" and buffer == "single")
                return Mode::twist;
            if (stream == "esotwist" and opt == "blocking")
            {
                if (buffer == "double") return Mode::block_double;
                if (buffer == "ring") return Mode::block_ring;
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

    std::size_t Config::global_size() const noexcept
    {
        return static_cast<std::size_t>(global_dim.x) * global_dim.y * global_dim.z;
    }

    dim3 Config::launch_dim() const noexcept
    {
        return {grid_dim.x * block_dim.x, grid_dim.y * block_dim.y, grid_dim.z * block_dim.z};
    }

    std::uint32_t Config::tile_size() const noexcept
    {
        const auto dim = launch_dim();
        return dim.x * dim.y * dim.z;
    }

    dim3 Config::tile_counts() const
    {
        using namespace culbm::core::blocking;
        const auto dim = launch_dim();
        const bool valid =
            valid_axis<std::uint32_t>(global_dim.x, dim.x, inner_steps) and
            valid_axis<std::uint32_t>(global_dim.y, dim.y, inner_steps) and
            valid_axis<std::uint32_t>(global_dim.z, dim.z, inner_steps);
        if (not valid)
        {
            throw std::invalid_argument("Domain, tile dimensions and --inner-steps do not match.");
        }
        return {
            tile_count<std::uint32_t>(global_dim.x, dim.x, inner_steps),
            tile_count<std::uint32_t>(global_dim.y, dim.y, inner_steps),
            tile_count<std::uint32_t>(global_dim.z, dim.z, inner_steps)
        };
    }
}
