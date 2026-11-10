#include <algorithm>
#include <format>
#include <memory>
#include <print>
#include <utility>
#include <cuda/launch>
#include <cuda/std/chrono>
#include "core/stencil.hpp"
#include "kernel.cuh"
#include "simulator.hpp"
#include "state.hpp"
#include "util/workers.hpp"

namespace culbm::sim::multi
{
    struct Simulator::Impl
    {
        State state;
        Kernels kernels;
        std::uint32_t step = 0;
        util::Workers workers;

        explicit Impl(Config config) :
            state(std::move(config)), kernels(Kernels::make(state.config.q, state.config.collision)),
            workers(state.config.device_count())
        {
            const auto& c = state.config;
            const auto local_dim = c.local_dim();
            std::println(
                "InvTau: {}\nTotal Time Step: {}\nDevice Dimension: [{},{},{}]\nSingle Device Domain Dimension: [{},{},{}]\nAll Device Domain Dimension: [{},{},{}]\n",
                c.inv_tau, c.target_step,
                c.dev_dim.x, c.dev_dim.y, c.dev_dim.z,
                local_dim.x, local_dim.y, local_dim.z,
                c.dev_dim.x * local_dim.x, c.dev_dim.y * local_dim.y, c.dev_dim.z * local_dim.z
            );
            workers.submit([this](std::uint32_t id, util::Logger& logger, std::barrier<>& barrier) {
                state.init(id, kernels.init, logger, barrier);
            });
            workers.wait();
        }

        void advance(std::uint32_t dev_id, std::uint32_t end_step, util::Logger& logger, std::barrier<>& barrier)
        {
            const auto& config = state.config;
            const auto dev_dim = config.dev_dim;
            auto& device = *state.devices[dev_id];
            KernelParam even_param{
                .inv_tau = config.inv_tau,
                .flag_buf = device.flag_buf.data(),
                .rho_buf = device.rho_buf.data(),
                .vx_buf = device.vx_buf.data(),
                .vy_buf = device.vy_buf.data(),
                .vz_buf = device.vz_buf.data(),
                .dst_ddf = device.ddf1.data()
            };
            auto odd_param = even_param;
            odd_param.dst_ddf = device.ddf0.data();
            const auto launch = ::cuda::make_config(::cuda::grid_dims(config.grid_dim), ::cuda::block_dims(config.block_dim));

            const auto position = config.coordinates(dev_id);
            const idx_t dev_idx_x = position.x, dev_idx_y = position.y, dev_idx_z = position.z;

            for (dir_t dir = 0 ; dir < 27 ; ++dir)
            {
                const auto d = culbm::core::Stencil<27>::directions[dir];
                const auto nbr_idx_x = std::clamp<std::int32_t>(dev_idx_x + d.x, 0, dev_dim.x - 1);
                const auto nbr_idx_y = std::clamp<std::int32_t>(dev_idx_y + d.y, 0, dev_dim.y - 1);
                const auto nbr_idx_z = std::clamp<std::int32_t>(dev_idx_z + d.z, 0, dev_dim.z - 1);
                const std::int32_t nbr_idx = nbr_idx_x + dev_dim.x * (nbr_idx_y + dev_dim.y * nbr_idx_z);
                even_param.src_ddf[dir] = state.devices[nbr_idx]->ddf0.data();
                odd_param.src_ddf[dir]  = state.devices[nbr_idx]->ddf1.data();
            }

            barrier.arrive_and_wait();

            device.start_event.record(device.stream);

            for (std::uint32_t current = step ; current < end_step ; ++current)
            {
                ::cuda::launch(device.stream, launch, kernels.step, (current % 2) == 0 ? even_param : odd_param);
                device.stream.sync();

                barrier.arrive_and_wait();
            }

            device.end_event.record(device.stream);
            device.end_event.sync();
            const float seconds = ::cuda::std::chrono::duration<float>{device.end_event - device.start_event}.count();
            const float mlups = config.local_size() * 1e-6f / seconds * (end_step - step);
            logger.info(std::format("speed: {:.4f} (MLUPS)", mlups));

            state.write(dev_id, end_step, position);
        }

        void run()
        {
            const auto& config = state.config;
            while (step < config.target_step)
            {
                const auto end_step = step + std::min(config.target_step - step, config.steps_per_batch);
                workers.submit([this, end_step](std::uint32_t id, util::Logger& logger, std::barrier<>& barrier) {
                    advance(id, end_step, logger, barrier);
                });
                workers.wait();
                step = end_step;
            }
        }

        ~Impl()
        {
            // Workers drain this task before State is destroyed.
            workers.submit([this](std::uint32_t id, util::Logger& logger, std::barrier<>&) {
                state.clear(id, logger);
            });
        }
    };

    Simulator::Simulator(Config config) : impl(std::make_unique<Impl>(std::move(config))) {}

    void Simulator::run() { impl->run(); }

    Simulator::~Simulator() = default;
}
