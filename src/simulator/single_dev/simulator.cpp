#include "simulator.hpp"

namespace culbm::simulator::single_dev
{
    Simulator::Simulator(int argc, char** argv) :
        config(SimulationConfig::parse(argc, argv))
    {
    }

    void Simulator::run()
    {
        simulate(config);
    }
}
