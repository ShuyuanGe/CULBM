#include <exception>
#include <iostream>
#include <print>
#include "sim/single/simulator.hpp"

int main(int argc, char** argv)
{
    try
    {
        culbm::sim::single::Simulator simulator(culbm::sim::single::Config::parse(argc, argv));
        simulator.run();
    }
    catch (const std::exception& e)
    {
        std::println(std::cerr, "{}", e.what());
        return 1;
    }
    return 0;
}
