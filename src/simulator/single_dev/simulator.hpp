#pragma once

#include "simulation.hpp"

namespace culbm::simulator::single_dev
{
    struct Simulator
    {
        SimulationConfig config;
        KernelSelection kernels;
        SimulationState state;
        Simulator(int argc, char** argv);
        Simulator(const Simulator&) = delete;
        Simulator& operator=(const Simulator&) = delete;
        void run();
    };
}
