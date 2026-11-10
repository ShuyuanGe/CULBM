#pragma once

#include <memory>

namespace culbm::sim::multi
{
    struct Config;

    class Simulator
    {
        struct Impl;
        std::unique_ptr<Impl> impl;

    public:
        explicit Simulator(Config config);
        void run();
        ~Simulator();
    };
}
