#pragma once

#include <memory>
#include "thread_pool.hpp"

namespace culbm::simulator::multi_dev
{
    class Simulator
    {
        private:
            class Data;
            std::unique_ptr<Data> data;
            culbm::basic::ThreadPool pool;
        public:
            Simulator(int argc, char** argv);
            void run();
            ~Simulator();
    };
}