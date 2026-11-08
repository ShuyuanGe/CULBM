#include <exception>
#include <iostream>
#include <print>
#include "simulator.hpp"

int main(int argc, char** argv)
{
    try
    {
        culbm::simulator::single_dev::Simulator simulator(argc, argv);
        simulator.run();
    }
    catch (const std::exception& e)
    {
        std::println(std::cerr, "{}", e.what());
        return 1;
    }
    return 0;
}
