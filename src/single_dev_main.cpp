#include <iostream>
#include <stdexcept>
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
        std::cerr << e.what() << std::endl;
        exit(1);
    }
    return 0;
}
