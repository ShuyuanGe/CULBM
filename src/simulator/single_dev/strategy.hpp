#pragma once

#include <cstdint>
#include <utility>
#include "blocking.hpp"
#include "kernel.cuh"
#include "exception.cuh"
#include "simulation.hpp"

namespace culbm::simulator::single_dev::strategy
{
    namespace error = culbm::basic::error;

    template <typename Kernels>
    struct Plain
    {
        SimulationState& state;
        Kernels kernels;
        basic::cuda::DeviceBuffer<flag_t> flags;
        basic::cuda::DeviceBuffer<ddf_t> src, dst;
        PlainParam param;

        static Kernels makeKernels(const SimulationConfig& config)
        {
            if constexpr (Kernels::scheme == PlainScheme::TwoFieldPull)
                return makePlainPullKernels(config.q, config.collision);
            else if constexpr (Kernels::scheme == PlainScheme::EsoTwist)
                return makePlainEsoKernels(config.q, config.collision);
        }

        Plain(SimulationState& state, const Kernels& kernels);

        std::uint32_t advance()
        {
            const auto& config = state.config;
            void* args[] = {&param};
            for (std::uint32_t step = 0; step < config.stepsPerBatch; ++step)
            {
                if constexpr (Kernels::scheme == PlainScheme::EsoTwist)
                {
                    const bool even = ((state.step + step) % 2) == 0;
                    error::check(cudaLaunchKernel(even ? kernels.stepEven : kernels.stepOdd, config.gridDim, config.blockDim, args, 0, state.stream));
                }
                else if constexpr (Kernels::scheme == PlainScheme::TwoFieldPull)
                {
                    error::check(cudaLaunchKernel(kernels.step, config.gridDim, config.blockDim, args, 0, state.stream));
                    std::swap(param.glbSrcDDFBuf, param.glbDstDDFBuf);
                }
            }
            return config.stepsPerBatch;
        }

        void computeMacros()
        {
            const auto& config = state.config;
            void* args[] = {&param};
            if constexpr (Kernels::scheme == PlainScheme::EsoTwist)
            {
                const bool even = (state.step % 2) == 0;
                error::check(cudaLaunchKernel(even ? kernels.dumpEven : kernels.dumpOdd, config.gridDim, config.blockDim, args, 0, state.stream));
            }
            else if constexpr (Kernels::scheme == PlainScheme::TwoFieldPull)
                error::check(cudaLaunchKernel(kernels.dump, config.gridDim, config.blockDim, args, 0, state.stream));
        }

        std::size_t cellsPerStep() const
        {
            return state.config.domainSize();
        }
    };

    template <BlockingGlobalBuffer Buffer>
    struct Blocking
    {
        SimulationState& state;
        BlockingKernels kernels;
        basic::cuda::DeviceBuffer<flag_t> flags;
        basic::cuda::DeviceBuffer<ddf_t> src, dst, tile;
        BlockingParam param;
        uint3 tileDim, tileCounts;
        std::uint32_t tileCount;
        idx_t ringOffsetStep;

        static BlockingKernels makeKernels(const SimulationConfig& config)
        {
            return makeBlockingKernels(config.q, config.collision, Buffer, config.cacheHint == 1);
        }

        Blocking(SimulationState& state, const BlockingKernels& kernels);
        ~Blocking();

        std::uint32_t advance()
        {
            const auto& config = state.config;
            void* args[] = {&param};
            std::uint32_t step = 0;
            for (; step < config.stepsPerBatch; step += config.innerLoop)
            {
                for (std::uint32_t index = 0; index < tileCount; ++index)
                {
                    const auto tile = coordinates(index);
                    if (config.interiorTilesOnly and
                        (tile.x == 0 or tile.x == tileCounts.x - 1 or
                         tile.y == 0 or tile.y == tileCounts.y - 1 or
                         tile.z == 0 or tile.z == tileCounts.z - 1))
                        continue;
                    setTile(index, tile);
                    for (std::int32_t loop = 0; loop < config.innerLoop; ++loop)
                    {
                        BlockingKernel kernel;
                        if (loop == 0)
                        {
                            if (config.innerLoop == 1) kernel = kernels.single;
                            else                       kernel = kernels.first;
                        }
                        else if (loop < config.innerLoop - 1)
                        {
                            if (loop % 2 == 0) kernel = kernels.middleEven;
                            else               kernel = kernels.middleOdd;
                        }
                        else
                        {
                            if (loop % 2 == 0) kernel = kernels.lastEven;
                            else               kernel = kernels.lastOdd;
                        }
                        error::check(cudaLaunchKernel(kernel, config.gridDim, config.blockDim, args, 0, state.stream));
                    }
                }
                if constexpr (Buffer == BlockingGlobalBuffer::Ring)
                {
                    param.glbSrcDDFOffz = param.glbDstDDFOffz;
                    param.glbDstDDFOffz -= ringOffsetStep;
                    if (param.glbDstDDFOffz < 0) param.glbDstDDFOffz += param.glbDDFRingNz;
                }
                else if constexpr (Buffer == BlockingGlobalBuffer::Double)
                    std::swap(param.glbSrcDDFBuf, param.glbDstDDFBuf);
            }
            return step;
        }

        void computeMacros()
        {
            const auto& config = state.config;
            void* args[] = {&param};
            for (std::uint32_t index = 0; index < tileCount; ++index)
            {
                setTile(index, coordinates(index));
                error::check(cudaLaunchKernel(kernels.dump, config.gridDim, config.blockDim, args, 0, state.stream));
            }
        }

        std::size_t cellsPerStep() const
        {
            const auto& config = state.config;
            if (config.interiorTilesOnly)
                return static_cast<std::size_t>(tileCounts.x - 2) * (tileCounts.y - 2) * (tileCounts.z - 2) *
                    (tileDim.x - 2 * (config.innerLoop - 1)) *
                    (tileDim.y - 2 * (config.innerLoop - 1)) *
                    (tileDim.z - 2 * (config.innerLoop - 1));
            return config.domainSize();
        }

        uint3 coordinates(std::uint32_t index) const
        {
            return { index % tileCounts.x,
                     (index / tileCounts.x) % tileCounts.y,
                     index / (tileCounts.x * tileCounts.y) };
        }

        void setTile(std::uint32_t index, uint3 tile)
        {
            const auto& config = state.config;
            using culbm::core::blocking::tileBounds;
            param.offx = tileBounds<idx_t>(tile.x, tileDim.x, tileCounts.x, config.innerLoop, config.domainDim.x).loadBegin;
            param.offy = tileBounds<idx_t>(tile.y, tileDim.y, tileCounts.y, config.innerLoop, config.domainDim.y).loadBegin;
            param.offz = tileBounds<idx_t>(tile.z, tileDim.z, tileCounts.z, config.innerLoop, config.domainDim.z).loadBegin;
            param.flagBuf = flags.get() + index * tileDim.x * tileDim.y * tileDim.z;
        }

    };

}
