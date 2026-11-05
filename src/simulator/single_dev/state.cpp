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
        const std::size_t size = config.domainSize();
        for (auto& buf : {std::ref(rhoBuf), std::ref(vxBuf), std::ref(vyBuf), std::ref(vzBuf)})
            buf.get() = basic::cuda::allocate<real_t>(size);

        auto initField = [this, size](std::string_view name, real_t defaultValue, real_t* deviceBuf)
        {
            if (std::ifstream f (std::format("{}/{}.dat", this->config.initFolder, name), std::ios::binary) ; f)
            {
                std::vector<real_t> hostBuf(size);
                f.read(reinterpret_cast<char*>(hostBuf.data()), size * sizeof(real_t));
                error::check(cudaMemcpy(deviceBuf, hostBuf.data(), size * sizeof(real_t), cudaMemcpyHostToDevice));
                std::println("Init {} successfully, from file: {}/{}.dat.", name, this->config.initFolder, name);
            }
            else
            {
                basic::cuda::fill(deviceBuf, size, defaultValue, stream);
                std::println("Init {} successfully, default init with value {}.", name, defaultValue);
            }
        };

        initField("rho", 1, rhoBuf.get());
        initField("vx" , 0, vxBuf.get());
        initField("vy" , 0, vyBuf.get());
        initField("vz" , 0, vzBuf.get());
    }

}
