#include <format>
#include <iostream>
#include "options.hpp"
#include "simulator.hpp"
#include "device_function.hpp"

int main(int argc, const char** argv)
{
    try
    {
        using namespace gf::basic;
        using namespace gf::literal;
        using namespace gf::simulator::single_dev;

        Options::getInstance()->cmdInit(argc, argv);
        Simulator simulator(Options::getInstance());
        simulator.run(100);
        simulator.run(100);
        simulator.run(100);
        constexpr i32 max = std::numeric_limits<i32>::max();
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }    
    return 0;
}