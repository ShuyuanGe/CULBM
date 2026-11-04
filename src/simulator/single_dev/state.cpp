#include <format>
#include <fstream>
#include <functional>
#include <print>
#include <string_view>
#include <vector>
#include "simulation.hpp"

namespace culbm::simulator::single_dev
{
    namespace error = culbm::basic::error;

    SimulationState::SimulationState(const SimulationConfig& config) : config(config)
    {
        error::check(cudaSetDevice(config.deviceId));
        stream.create();
        startEvent.create();
        endEvent.create();
        init();
    }

    void SimulationState::init()
    {
        const std::size_t domSize = config.domainSize();
        for (auto& buf : {std::ref(rhoBuf), std::ref(vxBuf), std::ref(vyBuf), std::ref(vzBuf)})
            buf.get() = basic::cuda::allocate<real_t>(domSize);

        auto initField = [this, domSize](std::string_view desc, real_t dftv, real_t* devBuf)
        {
            if (std::ifstream f (std::format("{}/{}.dat", config.initStateFolder, desc), std::ios::binary) ; f)
            {
                std::vector<real_t> hostBuf(domSize);
                f.read(reinterpret_cast<char*>(hostBuf.data()), domSize * sizeof(real_t));
                error::check(cudaMemcpy(devBuf, hostBuf.data(), domSize * sizeof(real_t), cudaMemcpyHostToDevice));
                std::println("Init {} successfully, from file: {}/{}.dat.", desc, config.initStateFolder, desc);
            }
            else
            {
                basic::cuda::fill(devBuf, domSize, dftv, stream);
                std::println("Init {} successfully, default init with value {}.", desc, dftv);
            }
        };

        initField("rho", 1, rhoBuf.get());
        initField("vx" , 0, vxBuf.get());
        initField("vy" , 0, vyBuf.get());
        initField("vz" , 0, vzBuf.get());
    }

}
