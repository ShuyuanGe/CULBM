#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <format>
#include <fstream>
#include <print>
#include <stdexcept>
#include <utility>
#include <vector>
#include "blocking.hpp"
#include "kernel.cuh"
#include "exception.cuh"
#include "simulation.hpp"

namespace culbm::simulator::single_dev::strategy
{
    namespace error = culbm::basic::error;

    namespace detail
    {
        inline std::vector<flag_t> loadFlags(const SimulationConfig& config)
        {
            std::vector<flag_t> flag(config.domainSize());
            if (config.initFolder.empty())
            {
                std::ranges::fill(flag, LOAD_DDF_BIT | COLLIDE_BIT | STORE_DDF_BIT | DUMP_RHO_BIT | DUMP_VX_BIT | DUMP_VY_BIT | DUMP_VZ_BIT);
            }
            else if (std::ifstream f (std::format("{}/flag.dat", config.initFolder), std::ios::binary); f)
            {
                f.read(reinterpret_cast<char*>(flag.data()), flag.size() * sizeof(flag_t));
            }
            else
            {
                throw std::runtime_error(std::format("File: {}/flag.dat doesn't exist!", config.initFolder));
            }
            return flag;
        }

        inline void initDDF(SimulationState& state, Equilibrium equilibrium, std::size_t size, bool doubleBuffer,
                            basic::cuda::DeviceBuffer<ddf_t>& src, basic::cuda::DeviceBuffer<ddf_t>& dst)
        {
            const auto q = state.config.q;
            src = basic::cuda::allocate<ddf_t>(q * size);
            if (doubleBuffer) dst = basic::cuda::allocate<ddf_t>(q * size);
            std::vector<ddf_t> feq(q);
            equilibrium(1, 0, 0, 0, feq.data());
            for (int dir = 0; dir < q; ++dir)
            {
                basic::cuda::fill(src.get() + dir * size, size, feq[dir], state.stream);
                if (doubleBuffer) basic::cuda::fill(dst.get() + dir * size, size, feq[dir], state.stream);
            }
            std::println("Init ddf buffer successfully.");
        }
    }

    template <typename Kernels>
    struct Plain
    {
        SimulationState& state;
        Kernels kernels;
        basic::cuda::DeviceBuffer<flag_t> flags;
        basic::cuda::DeviceBuffer<ddf_t> src, dst;
        PlainParam param;

        Plain(SimulationState& state) : state(state), kernels(Kernels::make(state.config.q, state.config.collision))
        {
            constexpr auto scheme = Kernels::scheme;
            const auto& config = state.config;
            const auto domainFlags = detail::loadFlags(config);
            if (scheme == PlainScheme::EsoTwist and
                std::ranges::any_of(domainFlags, [](flag_t f) { return not (f & STORE_DDF_BIT); }))
                throw std::invalid_argument("--ddfBuffer esotwist requires STORE_DDF_BIT for every domain cell.");
            flags = basic::cuda::allocate<flag_t>(domainFlags.size());
            if (not config.initFolder.empty())
                std::println("Init flag successfully, from file: {}/flag.dat.", config.initFolder);
            error::check(cudaMemcpy(flags.get(), domainFlags.data(), domainFlags.size() * sizeof(flag_t), cudaMemcpyHostToDevice));
            detail::initDDF(state, kernels.equilibrium, config.domainSize(), scheme == PlainScheme::TwoFieldPull, src, dst);
            param = PlainParam{
                .invTau = config.invTau,
                .flagBuf = flags.get(),
                .rhoBuf = state.rhoBuf.get(),
                .vxBuf = state.vxBuf.get(),
                .vyBuf = state.vyBuf.get(),
                .vzBuf = state.vzBuf.get(),
                .glbSrcDDFBuf = src.get(),
                .glbDstDDFBuf = dst.get()
            };
            if (scheme == PlainScheme::EsoTwist) param.glbDstDDFBuf = src.get();
            const auto launchExtent = config.launchExtent();
            const auto domainDim = config.domainDim;
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

        void dump()
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

        Blocking(SimulationState& state) :
            state(state), kernels(BlockingKernels::make(state.config.q, state.config.collision, Buffer, state.config.cacheHint == 1)),
            tileDim(state.config.launchExtent()), tileCounts(state.config.tileCounts()),
            tileCount(tileCounts.x * tileCounts.y * tileCounts.z),
            ringOffsetStep(tileDim.z + 1)
        {
            const auto& config = state.config;
            const std::uint32_t tileSize = tileDim.x * tileDim.y * tileDim.z;
            flags = basic::cuda::allocate<flag_t>(tileCount * tileSize);
            const auto domainFlags = detail::loadFlags(config);

            std::vector<flag_t> tileFlags(tileCount * tileSize);
            std::int32_t tileBase = 0;
            for (std::int32_t tileZ = 0 ; tileZ < tileCounts.z ; ++tileZ)
            {
                const auto boundsZ = culbm::core::blocking::tileBounds<std::int32_t>(tileZ, tileDim.z, tileCounts.z, config.innerLoop, config.domainDim.z);
                for (std::int32_t tileY = 0 ; tileY < tileCounts.y ; ++tileY)
                {
                    const auto boundsY = culbm::core::blocking::tileBounds<std::int32_t>(tileY, tileDim.y, tileCounts.y, config.innerLoop, config.domainDim.y);
                    for (std::int32_t tileX = 0 ; tileX < tileCounts.x ; ++tileX)
                    {
                        const auto boundsX = culbm::core::blocking::tileBounds<std::int32_t>(tileX, tileDim.x, tileCounts.x, config.innerLoop, config.domainDim.x);

                        for (std::int32_t localZ = 0, globalZ = boundsZ.loadBegin ; globalZ < boundsZ.loadEnd ; ++localZ, ++globalZ)
                        {
                            for (std::int32_t localY = 0, globalY = boundsY.loadBegin ; globalY < boundsY.loadEnd ; ++localY, ++globalY)
                            {
                                for (std::int32_t localX = 0, globalX = boundsX.loadBegin ; globalX < boundsX.loadEnd ; ++localX, ++globalX)
                                {
                                    const std::int32_t domainIndex = globalX + config.domainDim.x * (globalY + config.domainDim.y * globalZ);
                                    const std::int32_t tileIndex = localX + tileDim.x * (localY + tileDim.y * localZ);
                                    if (
                                        boundsZ.storeBegin <= globalZ and globalZ < boundsZ.storeEnd and
                                        boundsY.storeBegin <= globalY and globalY < boundsY.storeEnd and
                                        boundsX.storeBegin <= globalX and globalX < boundsX.storeEnd
                                    )
                                    {
                                        tileFlags[tileBase + tileIndex] = domainFlags[domainIndex] | CORRECT_BIT;
                                    }
                                    else
                                    {
                                        tileFlags[tileBase + tileIndex] = domainFlags[domainIndex];
                                    }
                                }
                            }
                        }

                        tileBase += tileSize;
                    }
                }
            }

            error::check(cudaMemcpy(flags.get(), tileFlags.data(), sizeof(flag_t) * tileCount * tileSize, cudaMemcpyHostToDevice));
            std::println("Init flag successfully, from file: {}/flag.dat.", config.initFolder);
            auto globalSize = config.domainSize();
            if constexpr (Buffer == BlockingGlobalBuffer::Ring)
            {
                assert(tileDim.z >= config.innerLoop);
                globalSize = static_cast<std::size_t>(config.domainDim.x) * config.domainDim.y * (config.domainDim.z + ringOffsetStep);
            }
            detail::initDDF(state, kernels.equilibrium, globalSize, Buffer == BlockingGlobalBuffer::Double, src, dst);
            tile = basic::cuda::allocate<ddf_t>(static_cast<std::size_t>(config.q) * config.tileSize());
            if (config.disablePersistentL2)
            {
                std::println("Persisting L2 configuration skipped.");
            }
            else
            {
                auto numBytes = static_cast<std::size_t>(config.q) * config.tileSize() * sizeof(ddf_t);
                cudaDeviceProp prop;
                error::check(cudaGetDeviceProperties(&prop, config.deviceId));
                if (numBytes > prop.persistingL2CacheMaxSize) {
                    std::println(
                        "Warning: Requested persisting L2 cache size {} exceeds the maximum allowed size {}. Capping to maximum.",
                        numBytes, prop.persistingL2CacheMaxSize
                    );
                    numBytes = prop.persistingL2CacheMaxSize;
                }
                error::check(cudaDeviceSetLimit(cudaLimit::cudaLimitPersistingL2CacheSize, numBytes));
                cudaStreamAttrValue streamAttr;
                streamAttr.accessPolicyWindow.base_ptr = tile.get();
                streamAttr.accessPolicyWindow.num_bytes = numBytes;
                streamAttr.accessPolicyWindow.hitRatio = 1.0;
                streamAttr.accessPolicyWindow.hitProp = cudaAccessProperty::cudaAccessPropertyPersisting;
                streamAttr.accessPolicyWindow.missProp = cudaAccessProperty::cudaAccessPropertyStreaming;
                error::check(cudaStreamSetAttribute(state.stream, cudaLaunchAttributeID::cudaStreamAttributeAccessPolicyWindow, &streamAttr));
            }
            param = BlockingParam{
                .invTau = config.invTau,
                .offx = 0, .offy = 0, .offz = 0,
                .glbnx = static_cast<idx_t>(config.domainDim.x),
                .glbny = static_cast<idx_t>(config.domainDim.y),
                .glbnz = static_cast<idx_t>(config.domainDim.z),
                .flagBuf = nullptr,
                .rhoBuf = state.rhoBuf.get(),
                .vxBuf = state.vxBuf.get(),
                .vyBuf = state.vyBuf.get(),
                .vzBuf = state.vzBuf.get(),
                .glbSrcDDFBuf = src.get(),
                .glbDstDDFBuf = dst.get(),
                .tileDDFBuf = tile.get(),
                .glbDDFRingNz = 0,
                .glbSrcDDFOffz = 0,
                .glbDstDDFOffz = 0
            };
            if constexpr (Buffer == BlockingGlobalBuffer::Ring)
            {
                param.glbDstDDFBuf = src.get();
                param.glbDstDDFOffz = config.domainDim.z;
                param.glbDDFRingNz = param.glbDstDDFOffz + ringOffsetStep;
            }
        }

        ~Blocking()
        {
            cudaStreamAttrValue attr{};
            cudaStreamSetAttribute(state.stream, cudaStreamAttributeAccessPolicyWindow, &attr);
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

        void dump()
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
