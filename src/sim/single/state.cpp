#include <format>
#include <memory>
#include <tuple>
#include <fstream>
#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/std/span>
#include <print>
#include <string_view>
#include <vector>
#include "gpu/error.hpp"
#include "gpu/fill.hpp"
#include "state.hpp"
#include "sim/output.hpp"

namespace culbm::sim::single
{

    State::State(const Config& config) :
        config(config), device(::cuda::devices[config.deviceId]), stream(device), pool(device),
        startEvent(device), endEvent(device),
        rhoBuf(stream, pool.as_ref(), config.globalSize(), ::cuda::no_init),
        vxBuf(stream, pool.as_ref(), config.globalSize(), ::cuda::no_init),
        vyBuf(stream, pool.as_ref(), config.globalSize(), ::cuda::no_init),
        vzBuf(stream, pool.as_ref(), config.globalSize(), ::cuda::no_init)
    {
        gpu::check(cudaSetDevice(config.deviceId));
        const std::size_t size = config.globalSize();

        auto initField = [this, &config, size](std::string_view name, real_t defaultValue, real_t* deviceBuf)
        {
            const auto path = std::format("{}/{}.dat", config.initFolder, name);
            if (std::ifstream f(path, std::ios::binary); f)
            {
                std::vector<real_t> hostBuf(size);
                f.read(reinterpret_cast<char*>(hostBuf.data()), size * sizeof(real_t));
                ::cuda::copy_bytes(stream, ::cuda::std::span{hostBuf.data(), size}, ::cuda::std::span{deviceBuf, size});
                stream.sync();
                std::println("Init {} successfully, from file: {}.", name, path);
            }
            else
            {
                gpu::fill(deviceBuf, size, defaultValue, stream);
                std::println("Init {} successfully, default init with value {}.", name, defaultValue);
            }
        };

        initField("rho", 1, rhoBuf.data());
        initField("vx" , 0, vxBuf.data());
        initField("vy" , 0, vyBuf.data());
        initField("vz" , 0, vzBuf.data());
    }

    void State::write() const
    {
        if (not (config.dumpRho or config.dumpVx or config.dumpVy or config.dumpVz)) return;

        const auto folder = std::filesystem::path{config.dumpFolder};
        std::filesystem::create_directories(folder);

        const auto size = config.globalSize();
        std::unique_ptr<real_t[]> hostBuf;
        for (const auto& [enabled, prefix, devicePtr] : {
            std::tuple{config.dumpRho, "rho", rhoBuf.data()},
            std::tuple{config.dumpVx,  "vx",  vxBuf.data()},
            std::tuple{config.dumpVy,  "vy",  vyBuf.data()},
            std::tuple{config.dumpVz,  "vz",  vzBuf.data()}
        })
        {
            if (not enabled) continue;
            if (not hostBuf) hostBuf = std::make_unique_for_overwrite<real_t[]>(size);
            const auto path = folder / std::format("{}_{}.dat", prefix, step);
            writeField(stream, devicePtr, {hostBuf.get(), size}, path);
        }
    }

}
