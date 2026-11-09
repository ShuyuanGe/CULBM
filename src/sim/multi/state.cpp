#include <filesystem>
#include <format>
#include <memory>
#include <utility>
#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/std/span>
#include <cuda_runtime.h>
#include "core/stencil.hpp"
#include "gpu/error.hpp"
#include "state.hpp"
#include "sim/output.hpp"
#include <tuple>

namespace culbm::sim::multi
{
    State::Device::Device(int id, std::size_t size, int q) :
        stream(::cuda::device_ref{id}), pool(::cuda::device_ref{id}),
        start_event(::cuda::device_ref{id}), end_event(::cuda::device_ref{id}),
        flag_buf(stream, pool.as_ref(), size, ::cuda::no_init),
        rho_buf(stream, pool.as_ref(), size, ::cuda::no_init),
        vx_buf(stream, pool.as_ref(), size, ::cuda::no_init),
        vy_buf(stream, pool.as_ref(), size, ::cuda::no_init),
        vz_buf(stream, pool.as_ref(), size, ::cuda::no_init),
        ddf0(stream, pool.as_ref(), q * size, ::cuda::no_init),
        ddf1(stream, pool.as_ref(), q * size, ::cuda::no_init) {}

    State::State(Config config) :
        config(std::move(config)), devices(this->config.device_count()) {}

    void State::init(std::uint32_t dev_id, void (*init_kernel)(InitKernelParam), util::Logger& logger, std::barrier<>& barrier)
    {
        gpu::check(cudaSetDevice(dev_id));
        const auto dev_dim = config.dev_dim;
        const auto position = config.coordinates(dev_id);
        const idx_t dev_idx_x = position.x, dev_idx_y = position.y, dev_idx_z = position.z;
        auto& device = devices[dev_id].emplace(dev_id, config.local_size(), config.q);
        using PeerDirections = culbm::core::Stencil<27>;
        for (dir_t dir = 0 ; dir < 27 ; ++dir)
        {
            const auto d = PeerDirections::directions[dir];
            if (d.x == 0 and d.y == 0 and d.z == 0) continue;
            const std::int32_t nbr_idx_x = dev_idx_x + d.x;
            const std::int32_t nbr_idx_y = dev_idx_y + d.y;
            const std::int32_t nbr_idx_z = dev_idx_z + d.z;
            if (
                0 <= nbr_idx_x and nbr_idx_x < dev_dim.x and
                0 <= nbr_idx_y and nbr_idx_y < dev_dim.y and
                0 <= nbr_idx_z and nbr_idx_z < dev_dim.z
            )
            {
                const std::int32_t nbr_idx = nbr_idx_x + dev_dim.x * (nbr_idx_y + dev_dim.y * nbr_idx_z);
                device.pool.enable_access_from(::cuda::device_ref{nbr_idx});
            }
        }

        logger.info("Initialize cuda environment successfully!");

        barrier.arrive_and_wait();

        logger.info("Allocate device memory successfully!");

        barrier.arrive_and_wait();

        InitKernelParam param;
        param.dev_dim   = config.dev_dim;
        param.dev_pos.x = dev_idx_x;
        param.dev_pos.y = dev_idx_y;
        param.dev_pos.z = dev_idx_z;
        param.flag_buf  = device.flag_buf.data();
        param.rho_buf   = device.rho_buf.data();
        param.vx_buf    = device.vx_buf.data();
        param.vy_buf    = device.vy_buf.data();
        param.vz_buf    = device.vz_buf.data();
        param.src_ddf   = device.ddf0.data();
        param.dst_ddf   = device.ddf1.data();

        const auto launch = ::cuda::make_config(::cuda::grid_dims(config.grid_dim), ::cuda::block_dims(config.block_dim));
        ::cuda::launch(device.stream, launch, init_kernel, param);
        device.stream.sync();

        logger.info("Set device data successfully!");
    }

    void State::write(std::uint32_t dev_id, std::uint32_t end_step, uint3 position) const
    {
        if (not (config.dump_rho || config.dump_vx || config.dump_vy || config.dump_vz)) return;

        const auto dump_folder = std::filesystem::path{config.dump_folder};
        std::filesystem::create_directories(dump_folder);

        const auto size = config.local_size();
        auto host_buf = std::make_unique_for_overwrite<real_t[]>(size);
        const auto& device = *devices[dev_id];

        for (const auto& [enabled, prefix, device_ptr] : {
            std::tuple{config.dump_rho, "rho", device.rho_buf.data()},
            std::tuple{config.dump_vx,  "vx",  device.vx_buf.data()},
            std::tuple{config.dump_vy,  "vy",  device.vy_buf.data()},
            std::tuple{config.dump_vz,  "vz",  device.vz_buf.data()}
        })
        {
            if (not enabled) continue;
            const auto path = dump_folder / std::format("{}_{}_dev_{}_{}_{}.dat", prefix, end_step, position.x, position.y, position.z);
            write_field(device.stream, device_ptr, {host_buf.get(), size}, path);
        }
    }

    void State::clear(std::uint32_t dev_id, util::Logger& logger)
    {
        gpu::check(cudaSetDevice(dev_id));
        auto& device = *devices[dev_id];
        device.ddf1.destroy();
        device.ddf0.destroy();
        device.vz_buf.destroy();
        device.vy_buf.destroy();
        device.vx_buf.destroy();
        device.rho_buf.destroy();
        device.flag_buf.destroy();
        device.stream.sync();
        devices[dev_id].reset();
        logger.info("Deallocate device memory successfully!");
    }
}
