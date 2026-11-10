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
        concept Strategy = requires(State& state, T& strategy, const T& const_strategy) {
            T{state};
            { strategy.advance() } -> std::same_as<std::uint32_t>;
            { strategy.dump() } -> std::same_as<void>;
            { const_strategy.cells_per_step() } -> std::same_as<std::size_t>;
        };

        template <Strategy T>
        void execute(const Config& config)
        {
            State state(config); // init rho, v_xyz
            T strategy(state); // init ddf, flags
            while (state.step < config.target_step)
            {
                state.start_event.record(state.stream);
                const auto steps = strategy.advance();
                state.end_event.record(state.stream);
                state.end_event.sync();
                const float seconds = ::cuda::std::chrono::duration<float>{state.end_event - state.start_event}.count();
                const float mlups = strategy.cells_per_step() * 1e-6f / seconds * steps;
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
            case Mode::pull:         execute<strategy::Plain<PullKernels>>(config);           break;
            case Mode::twist:        execute<strategy::Plain<TwistKernels>>(config);          break;
            case Mode::block_double: execute<strategy::Blocking<Mode::block_double>>(config); break;
            case Mode::block_ring:   execute<strategy::Blocking<Mode::block_ring>>(config);   break;
        }
    }
}
