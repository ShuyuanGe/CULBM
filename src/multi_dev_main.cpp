#include <exception>
#include <iostream>
#include <print>
#include "sim/multi/simulator.hpp"

int main(int argc, char** argv)
{
    try
    {
        culbm::sim::multi::Simulator simulator(argc, argv);
        simulator.run();
    }
    catch (const std::exception& e)
    {
        std::println(std::cerr, "{}", e.what());
        return 1;
    }
    return 0;
}
