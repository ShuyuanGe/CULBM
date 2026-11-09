#pragma once

#include "config.hpp"

namespace culbm::sim::single
{
    struct Simulator
    {
        Config config;
        Simulator(int argc, char** argv);
        Simulator(const Simulator&) = delete;
        Simulator& operator=(const Simulator&) = delete;
        void run();
    };
}
