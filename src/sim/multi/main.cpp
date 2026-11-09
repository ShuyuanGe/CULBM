#include <exception>
#include <iostream>
#include <print>
#include "sim/exit.hpp"
#include "config.hpp"
#include "simulator.hpp"

int main(int argc, char** argv) try
{
    culbm::sim::multi::Simulator simulator(culbm::sim::multi::Config::parse(argc, argv));
    simulator.run();
}
catch (const culbm::sim::cli::Exit& e)
{
    return e.code;
}
catch (const std::exception& e)
{
    std::println(std::cerr, "{}", e.what());
    return 1;
}
