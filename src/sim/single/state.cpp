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
        config(config), device(::cuda::devices[config.device_id]), stream(device), pool(device),
        start_event(device), end_event(device),
        rho_buf(stream, pool.as_ref(), config.global_size(), ::cuda::no_init),
        vx_buf(stream, pool.as_ref(), config.global_size(), ::cuda::no_init),
        vy_buf(stream, pool.as_ref(), config.global_size(), ::cuda::no_init),
        vz_buf(stream, pool.as_ref(), config.global_size(), ::cuda::no_init),
        ddf_buf(stream, pool.as_ref(), config.dump_ddf ? config.q * config.global_size() : 0, ::cuda::no_init)
    {
        gpu::check(cudaSetDevice(config.device_id));
        const std::size_t size = config.global_size();

        auto init_field = [this, &config, size](std::string_view name, real_t default_value, real_t* device_buf)
        {
            const auto path = std::format("{}/{}.dat", config.init_folder, name);
            if (std::ifstream f(path, std::ios::binary); f)
            {
                std::vector<real_t> host_buf(size);
                f.read(reinterpret_cast<char*>(host_buf.data()), size * sizeof(real_t));
                ::cuda::copy_bytes(stream, ::cuda::std::span{host_buf.data(), size}, ::cuda::std::span{device_buf, size});
                stream.sync();
                std::println("Init {} successfully, from file: {}.", name, path);
            }
            else
            {
                gpu::fill(device_buf, size, default_value, stream);
                std::println("Init {} successfully, default init with value {}.", name, default_value);
            }
        };

        init_field("rho", 1, rho_buf.data());
        init_field("vx" , 0, vx_buf.data());
        init_field("vy" , 0, vy_buf.data());
        init_field("vz" , 0, vz_buf.data());
    }

    void State::write() const
    {
        if (not (config.dump_rho or config.dump_vx or config.dump_vy or config.dump_vz or config.dump_ddf)) return;

        const auto folder = std::filesystem::path{config.dump_folder};
        std::filesystem::create_directories(folder);

        const auto size = config.global_size();
        std::unique_ptr<real_t[]> host_buf;
        for (const auto& [enabled, prefix, device_ptr] : {
            std::tuple{config.dump_rho, "rho", rho_buf.data()},
            std::tuple{config.dump_vx,  "vx",  vx_buf.data()},
            std::tuple{config.dump_vy,  "vy",  vy_buf.data()},
            std::tuple{config.dump_vz,  "vz",  vz_buf.data()}
        })
        {
            if (not enabled) continue;
            if (not host_buf) host_buf = std::make_unique_for_overwrite<real_t[]>(size);
            const auto path = folder / std::format("{}_{}.dat", prefix, step);
            write_field(stream, device_ptr, {host_buf.get(), size}, path);
        }
        if (config.dump_ddf)
        {
            const auto ddf_size = config.q * size;
            auto ddf_host = std::make_unique_for_overwrite<ddf_t[]>(ddf_size);
            write_field(stream, ddf_buf.data(), {ddf_host.get(), ddf_size}, folder / std::format("ddf_{}.dat", step));
        }
    }

}
