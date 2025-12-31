#pragma once

#include <memory>

namespace culbm::simulator::single_dev_expt
{
    class Simulator
    {
        private:
            class Data;
            std::unique_ptr<Data> _data;
        public:
            Simulator(int argc, char** argv);
            void run();
            ~Simulator();
    };
}