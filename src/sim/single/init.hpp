#pragma once

#include <algorithm>
#include <format>
#include <fstream>
#include <print>
#include <stdexcept>
#include <vector>
#include "gpu/fill.hpp"
#include "kernel.cuh"
#include "state.hpp"

namespace culbm::sim::single::strategy
{
    namespace detail
    {
        inline std::vector<flag_t> load_flags(const Config& config)
        {
            std::vector<flag_t> flag(config.global_size());
            if (config.init_folder.empty())
            {
                std::ranges::fill(flag, LOAD_DDF_BIT | COLLIDE_BIT | STORE_DDF_BIT | DUMP_RHO_BIT | DUMP_VX_BIT | DUMP_VY_BIT | DUMP_VZ_BIT);
            }
            else if (std::ifstream f (std::format("{}/flag.dat", config.init_folder), std::ios::binary); f)
            {
                f.read(reinterpret_cast<char*>(flag.data()), flag.size() * sizeof(flag_t));
            }
            else
            {
                throw std::runtime_error(std::format("File: {}/flag.dat doesn't exist!", config.init_folder));
            }
            return flag;
        }

        inline void init_ddf(State& state, Equilibrium equilibrium, std::size_t size, bool double_buffer,
                             ::cuda::device_buffer<ddf_t>& src, ::cuda::device_buffer<ddf_t>& dst)
        {
            const auto q = state.config.q;
            src = ::cuda::device_buffer<ddf_t>(state.stream, state.pool.as_ref(), q * size, ::cuda::no_init);
            if (double_buffer) dst = ::cuda::device_buffer<ddf_t>(state.stream, state.pool.as_ref(), q * size, ::cuda::no_init);
            std::vector<ddf_t> feq(q);
            equilibrium(1, 0, 0, 0, feq.data());
            for (int dir = 0; dir < q; ++dir)
            {
                gpu::fill(src.data() + dir * size, size, feq[dir], state.stream);
                if (double_buffer) gpu::fill(dst.data() + dir * size, size, feq[dir], state.stream);
            }
            std::println("Init ddf buffer successfully.");
        }
    }
}
