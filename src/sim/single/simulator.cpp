#include <concepts>
#include <cuda/std/chrono>
#include <print>
#include <utility>
#include "state.hpp"
#include "simulator.hpp"
#include "plain.hpp"
#include "blocking.hpp"

namespace culbm::sim::single
{
    namespace
    {
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
                const float seconds = ::cuda::std::chrono::duration<float>{state.endEvent - state.startEvent}.count();
                const float mlups = strategy.cellsPerStep() * 1e-6f / seconds * steps;
                std::println("speed = {:.4f} (MLUPS)", mlups);
                state.step += steps;
                strategy.dump();
                state.stream.sync();
                state.write();
            }
        }
    }

    Simulator::Simulator(Config config) : config(std::move(config)) {}

    void Simulator::run()
    {
        switch (config.mode)
        {
            case Mode::Pull:        execute<strategy::Plain<PullKernels>>(config);          break;
            case Mode::Twist:       execute<strategy::Plain<TwistKernels>>(config);         break;
            case Mode::BlockDouble: execute<strategy::Blocking<Mode::BlockDouble>>(config); break;
            case Mode::BlockRing:   execute<strategy::Blocking<Mode::BlockRing>>(config);   break;
        }
    }
}
