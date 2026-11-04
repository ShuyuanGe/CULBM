#include <utility>
#include "simulator.hpp"

namespace culbm::simulator::single_dev
{
    namespace
    {
        SimulationConfig makeConfig(int argc, char** argv)
        {
            ParsedOptions options(argc, argv);
            options.validate();
            options.config.strategy = options.selectStrategy();
            return std::move(options.config);
        }
    }

    Simulator::Simulator(int argc, char** argv) :
        config(makeConfig(argc, argv)),
        kernels(makeKernels(config.q, config.collision, config.strategy, config.cacheHint == 1)),
        state(config, std::visit([](const auto& kernels) { return kernels.equilibrium; }, kernels))
    {
    }

    void Simulator::run()
    {
        simulate(state, kernels);
    }
}
