#pragma once

#include <memory>
#include "config.hpp"
#include "options.hpp"

namespace gf::simulator::single_dev
{
    using gf::basic::Options;
    
    class Simulator
    {
        private:
            class Data;

            std::unique_ptr<Data> _data;
        public:
            Simulator(std::shared_ptr<Options> opts);

            void run(idx_t batch_step);

            ~Simulator();
    };
}