#include <algorithm>
#include <memory>
#include "state.hpp"

namespace culbm::sim::multi
{
    Simulator::Simulator(int argc, char** argv) :
        state(std::make_unique<State>(Config::parse(argc, argv))),
        workers(state->config.deviceCount())
    {
        state->config.log();
        workers.submit([this](std::uint32_t id, util::Logger& logger, std::barrier<>& barrier) {
            state->init(id, logger, barrier);
        });
        workers.wait();
    }

    void Simulator::run()
    {
        const auto& config = state->config;
        while (state->step < config.targetStep)
        {
            const auto endStep = state->step + std::min(config.targetStep - state->step, config.stepsPerBatch);
            workers.submit([this, endStep](std::uint32_t id, util::Logger& logger, std::barrier<>& barrier) {
                state->advance(id, endStep, logger, barrier);
            });
            workers.wait();
            state->step = endStep;
        }
    }

    Simulator::~Simulator()
    {
        // Workers drain this task before State is destroyed.
        workers.submit([this](std::uint32_t id, util::Logger& logger, std::barrier<>&) {
            state->clear(id, logger);
        });
    }
}
