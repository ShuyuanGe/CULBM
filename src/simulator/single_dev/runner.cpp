#include <concepts>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <memory>
#include <stdexcept>
#include <tuple>
#include "exception.cuh"
#include "simulation.hpp"
#include "strategy.hpp"

namespace culbm::simulator::single_dev
{
    namespace error = culbm::basic::error;

    namespace
    {
        void write(const SimulationState& state)
        {
            const auto& config = state.config;
            const auto dumpFolder = std::filesystem::path{config.dumpFolder};
            std::filesystem::create_directories(dumpFolder);

            const auto nbytes = sizeof(real_t) * config.domainSize();
            std::unique_ptr<char[]> hostBuf;
            for (const auto& [enabled, prefix, devicePtr] : {
                std::tuple{config.dumpRho, "rho", state.rhoBuf.get()},
                std::tuple{config.dumpVx,  "vx",  state.vxBuf.get()},
                std::tuple{config.dumpVy,  "vy",  state.vyBuf.get()},
                std::tuple{config.dumpVz,  "vz",  state.vzBuf.get()}
            })
            {
                if (not enabled) continue;
                if (not hostBuf) hostBuf = std::make_unique_for_overwrite<char[]>(nbytes);
                const auto filePath = dumpFolder / std::format("{}_{}.dat", prefix, state.step);
                std::ofstream f(filePath, std::ios::binary);
                if (not f)
                    throw std::runtime_error(std::format("Could not open the file: {}", filePath.string()));
                error::check(cudaMemcpy(hostBuf.get(), devicePtr, nbytes, cudaMemcpyDeviceToHost));
                f.write(hostBuf.get(), nbytes);
            }
        }

        template <typename T>
        concept Strategy = requires(SimulationState& state, T& strategy, const T& constStrategy) {
            T{state};
            { strategy.advance() } -> std::same_as<std::uint32_t>;
            { strategy.computeMacros() } -> std::same_as<void>;
            { constStrategy.cellsPerStep() } -> std::same_as<std::size_t>;
        };

        template <Strategy T>
        void run(const SimulationConfig& config)
        {
            SimulationState state(config); // init rho, v_xyz
            T strategy(state); // init ddf, flags
            while (state.step < config.targetStep)
            {
                error::check(cudaEventRecord(state.startEvent, state.stream));
                const auto steps = strategy.advance();
                error::check(cudaEventRecord(state.endEvent, state.stream));
                error::check(cudaEventSynchronize(state.endEvent));
                float ms;
                error::check(cudaEventElapsedTime(&ms, state.startEvent, state.endEvent));
                const float mlups = strategy.cellsPerStep() * 1e-6f / (ms / 1000) * steps;
                std::println("speed = {:.4f} (MLUPS)", mlups);
                std::fflush(stdout);
                state.step += steps;
                strategy.computeMacros();
                error::check(cudaStreamSynchronize(state.stream));
                write(state);
            }
        }
    }

    void simulate(const SimulationConfig& config)
    {
        if (const auto* plain = std::get_if<PlainScheme>(&config.strategy))
        {
            switch (*plain)
            {
                case PlainScheme::TwoFieldPull: run<strategy::Plain<PlainPullKernels>>(config); break;
                case PlainScheme::EsoTwist:     run<strategy::Plain<PlainEsoKernels>>(config); break;
            }
        }
        else if (const auto* blocking = std::get_if<BlockingGlobalBuffer>(&config.strategy))
        {
            switch (*blocking)
            {
                case BlockingGlobalBuffer::Double: run<strategy::Blocking<BlockingGlobalBuffer::Double>>(config); break;
                case BlockingGlobalBuffer::Ring:   run<strategy::Blocking<BlockingGlobalBuffer::Ring>>(config); break;
            }
        }
    }
}
