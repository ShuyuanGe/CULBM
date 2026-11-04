#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include "algorithm.hpp"
#include "exception.cuh"
#include "simulation.hpp"

namespace culbm::simulator::single_dev::strategy
{
    namespace error = culbm::basic::error;

    template <typename Kernels>
    struct Plain
    {
        SimulationState& state;
        const Kernels& kernels;
        PlainParam param;

        Plain(SimulationState& state, const Kernels& kernels) : state(state), kernels(kernels), param{
            .invTau = state.config.invTau,
            .flagBuf = state.flagBuf,
            .glbRhoBuf = state.rhoBuf,
            .glbVxBuf = state.vxBuf,
            .glbVyBuf = state.vyBuf,
            .glbVzBuf = state.vzBuf,
            .glbSrcDDFBuf = state.srcDDFBuf,
            .glbDstDDFBuf = state.dstDDFBuf
        }
        {
            if constexpr (Kernels::scheme == PlainScheme::EsoTwist) param.glbDstDDFBuf = state.srcDDFBuf;
            const auto launchExtent = state.config.launchExtent();
            const auto domainDim = state.config.domainDim;
            if (launchExtent.x != domainDim.x or launchExtent.y != domainDim.y or launchExtent.z != domainDim.z)
                throw std::runtime_error("Dimension of domain doesn't match computation grid!");
        }

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

        std::uint32_t cellsPerStep() const
        {
            const auto dim = state.config.domainDim;
            return dim.x * dim.y * dim.z;
        }
    };

    template <BlockingGlobalBuffer Buffer>
    struct Blocking
    {
        SimulationState& state;
        const BlockingKernels& kernels;
        BlockingParam param;
        uint3 tileDim, tileCounts;
        std::uint32_t tileCount;
        idx_t ringOffsetStep;

        uint3 tileCoordinates(std::uint32_t index) const
        {
            return { index % tileCounts.x, 
                     (index / tileCounts.x) % tileCounts.y,
                     index / (tileCounts.x * tileCounts.y) };
        }

        void setTileParams(std::uint32_t index, uint3 tile)
        {
            const auto& config = state.config;
            using culbm::blocking_core::calcValidPrev;
            param.offx = std::max<idx_t>(calcValidPrev<idx_t>(tile.x, tileDim.x, tileCounts.x, config.innerLoop, config.domainDim.x) - (config.innerLoop - 1), 0);
            param.offy = std::max<idx_t>(calcValidPrev<idx_t>(tile.y, tileDim.y, tileCounts.y, config.innerLoop, config.domainDim.y) - (config.innerLoop - 1), 0);
            param.offz = std::max<idx_t>(calcValidPrev<idx_t>(tile.z, tileDim.z, tileCounts.z, config.innerLoop, config.domainDim.z) - (config.innerLoop - 1), 0);
            param.flagBuf = state.flagBuf + index * tileDim.x * tileDim.y * tileDim.z;
        }

        void launchTileSteps(void** args) const
        {
            const auto& config = state.config;
            for (std::int32_t loop = 0; loop < config.innerLoop; ++loop)
            {
                const void* kernel;
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

        Blocking(SimulationState& state, const BlockingKernels& kernels) : state(state), kernels(kernels), param{
            .invTau = state.config.invTau,
            .offx = 0, .offy = 0, .offz = 0,
            .glbnx = static_cast<idx_t>(state.config.domainDim.x),
            .glbny = static_cast<idx_t>(state.config.domainDim.y),
            .glbnz = static_cast<idx_t>(state.config.domainDim.z),
            .flagBuf = nullptr,
            .glbRhoBuf = state.rhoBuf,
            .glbVxBuf = state.vxBuf,
            .glbVyBuf = state.vyBuf,
            .glbVzBuf = state.vzBuf,
            .glbSrcDDFBuf = state.srcDDFBuf,
            .glbDstDDFBuf = state.dstDDFBuf,
            .tileDDFBuf = state.tileDDFBuf,
            .glbDDFRingNz = 0,
            .glbSrcDDFOffz = 0,
            .glbDstDDFOffz = 0
        }, tileDim(state.config.launchExtent()),
           tileCounts(state.config.tileCounts()),
           tileCount(tileCounts.x * tileCounts.y * tileCounts.z),
           ringOffsetStep(static_cast<idx_t>(state.config.gridDim.z * state.config.blockDim.z) + 1)
        {
            if constexpr (Buffer == BlockingGlobalBuffer::Ring)
            {
                param.glbDstDDFBuf = state.srcDDFBuf;
                param.glbDDFRingNz = static_cast<idx_t>(state.config.domainDim.z) + ringOffsetStep;
                param.glbDstDDFOffz = static_cast<idx_t>(state.config.domainDim.z);
            }
        }

        std::uint32_t advance()
        {
            const auto& config = state.config;
            void* args[] = {&param};
            std::uint32_t step = 0;
            for (; step < config.stepsPerBatch; step += config.innerLoop)
            {
                for (std::uint32_t index = 0; index < tileCount; ++index)
                {
                    const auto tile = tileCoordinates(index);
                    if (config.interiorTilesOnly and
                        (tile.x == 0 or tile.x == tileCounts.x - 1 or
                         tile.y == 0 or tile.y == tileCounts.y - 1 or
                         tile.z == 0 or tile.z == tileCounts.z - 1))
                        continue;
                    setTileParams(index, tile);
                    launchTileSteps(args);
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
                setTileParams(index, tileCoordinates(index));
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
    };

}
