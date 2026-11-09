#pragma once

#include <memory>
#include "util/workers.hpp"

namespace culbm::sim::multi
{
    class Simulator
    {
        struct State;
        std::unique_ptr<State> state;
        util::Workers workers;

    public:
        Simulator(int argc, char** argv);
        void run();
        ~Simulator();
    };
}
