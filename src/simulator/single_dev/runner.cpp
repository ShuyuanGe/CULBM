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
                std::tuple{config.dumpRho, "rho", state.rhoBuf},
                std::tuple{config.dumpVx,  "vx",  state.vxBuf},
                std::tuple{config.dumpVy,  "vy",  state.vyBuf},
                std::tuple{config.dumpVz,  "vz",  state.vzBuf}
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

        template <typename Strategy, typename Kernels>
        void run(SimulationState& state, const Kernels& kernels)
        {
            Strategy strategy(state, kernels);
            while (state.step < state.config.targetStep)
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

    void simulate(SimulationState& state, const KernelSelection& kernels)
    {
        if (const auto* plain = std::get_if<PlainPullKernels>(&kernels))
            run<strategy::Plain<PlainPullKernels>>(state, *plain);
        else if (const auto* eso = std::get_if<PlainEsoKernels>(&kernels))
            run<strategy::Plain<PlainEsoKernels>>(state, *eso);
        else if (const auto* blocking = std::get_if<BlockingKernels>(&kernels))
        {
            switch (std::get<BlockingGlobalBuffer>(state.config.strategy))
            {
                case BlockingGlobalBuffer::Double: run<strategy::Blocking<BlockingGlobalBuffer::Double>>(state, *blocking); break;
                case BlockingGlobalBuffer::Ring:   run<strategy::Blocking<BlockingGlobalBuffer::Ring>>(state, *blocking); break;
            }
        }
    }
}
