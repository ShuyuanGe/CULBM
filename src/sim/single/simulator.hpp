#pragma once

#include "config.hpp"

namespace culbm::sim::single
{
    class Simulator
    {
        Config config;

    public:
        explicit Simulator(Config config);
        Simulator(const Simulator&) = delete;
        Simulator& operator=(const Simulator&) = delete;
        void run();
    };
}
