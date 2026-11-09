#include <algorithm>
#include <memory>
#include <print>
#include <utility>
#include "simulator.hpp"
#include "state.hpp"
#include "util/workers.hpp"

namespace culbm::sim::multi
{
    struct Simulator::Impl
    {
        State state;
        util::Workers workers;

        explicit Impl(Config config) :
            state(std::move(config)), workers(state.config.deviceCount())
        {
            const auto& c = state.config;
            const auto localDim = c.localDim();
            std::println(
                "InvTau: {}\nTotal Time Step: {}\nDevice Dimension: [{},{},{}]\nSingle Device Domain Dimension: [{},{},{}]\nAll Device Domain Dimension: [{},{},{}]\n",
                c.invTau, c.targetStep,
                c.devDim.x, c.devDim.y, c.devDim.z,
                localDim.x, localDim.y, localDim.z,
                c.devDim.x * localDim.x, c.devDim.y * localDim.y, c.devDim.z * localDim.z
            );
            workers.submit([this](std::uint32_t id, util::Logger& logger, std::barrier<>& barrier) {
                state.init(id, logger, barrier);
            });
            workers.wait();
        }

        void run()
        {
            const auto& config = state.config;
            while (state.step < config.targetStep)
            {
                const auto endStep = state.step + std::min(config.targetStep - state.step, config.stepsPerBatch);
                workers.submit([this, endStep](std::uint32_t id, util::Logger& logger, std::barrier<>& barrier) {
                    state.advance(id, endStep, logger, barrier);
                });
                workers.wait();
                state.step = endStep;
            }
        }

        ~Impl()
        {
            // Workers drain this task before State is destroyed.
            workers.submit([this](std::uint32_t id, util::Logger& logger, std::barrier<>&) {
                state.clear(id, logger);
            });
        }
    };

    Simulator::Simulator(Config config) : impl(std::make_unique<Impl>(std::move(config))) {}

    void Simulator::run() { impl->run(); }

    Simulator::~Simulator() = default;
}
