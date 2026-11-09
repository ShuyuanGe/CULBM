#include <concepts>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <cuda/algorithm>
#include <cuda/std/span>
#include "state.hpp"
#include "simulator.hpp"
#include "plain.hpp"
#include "blocking.hpp"

namespace culbm::sim::single
{
    namespace
    {
        void write(const State& state)
        {
            const auto& config = state.config;
            const auto folder = std::filesystem::path{config.dumpFolder};
            std::filesystem::create_directories(folder);

            const auto size = config.domainSize();
            std::unique_ptr<real_t[]> hostBuf;
            for (const auto& [enabled, prefix, devicePtr] : {
                std::tuple{config.dumpRho, "rho", state.rhoBuf.data()},
                std::tuple{config.dumpVx,  "vx",  state.vxBuf.data()},
                std::tuple{config.dumpVy,  "vy",  state.vyBuf.data()},
                std::tuple{config.dumpVz,  "vz",  state.vzBuf.data()}
            })
            {
                if (not enabled) continue;
                if (not hostBuf) hostBuf = std::make_unique_for_overwrite<real_t[]>(size);
                const auto path = folder / std::format("{}_{}.dat", prefix, state.step);
                std::ofstream f(path, std::ios::binary);
                if (not f)
                    throw std::runtime_error(std::format("Could not open the file: {}", path.string()));
                ::cuda::copy_bytes(state.stream, ::cuda::std::span{devicePtr, size}, ::cuda::std::span{hostBuf.get(), size});
                state.stream.sync();
                f.write(reinterpret_cast<const char*>(hostBuf.get()), size * sizeof(real_t));
            }
        }

        template <typename T>
        concept Strategy = requires(State& state, T& strategy, const T& constStrategy) {
            T{state};
            { strategy.advance() } -> std::same_as<std::uint32_t>;
            { strategy.dump() } -> std::same_as<void>;
            { constStrategy.cellsPerStep() } -> std::same_as<std::size_t>;
        };

        template <Strategy T>
        void execute(const Config& config)
        {
            State state(config); // init rho, v_xyz
            T strategy(state); // init ddf, flags
            while (state.step < config.targetStep)
            {
                state.startEvent.record(state.stream);
                const auto steps = strategy.advance();
                state.endEvent.record(state.stream);
                state.endEvent.sync();
                const float ms = (state.endEvent - state.startEvent).count() * 1e-6f;
                const float mlups = strategy.cellsPerStep() * 1e-6f / (ms / 1000) * steps;
                std::println("speed = {:.4f} (MLUPS)", mlups);
                state.step += steps;
                strategy.dump();
                state.stream.sync();
                write(state);
            }
        }
    }

    Simulator::Simulator(int argc, char** argv) : config(Config::parse(argc, argv)) {}

    void Simulator::run()
    {
        switch (config.mode)
        {
            case Mode::Pull:        execute<strategy::Plain<PullKernels>>(config); break;
            case Mode::Twist:       execute<strategy::Plain<TwistKernels>>(config); break;
            case Mode::BlockDouble: execute<strategy::Blocking<Mode::BlockDouble>>(config); break;
            case Mode::BlockRing:   execute<strategy::Blocking<Mode::BlockRing>>(config); break;
        }
    }
}
