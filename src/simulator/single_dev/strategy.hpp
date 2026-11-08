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
#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/std/span>
#include "blocking.hpp"
#include "kernel.cuh"
#include "exception.cuh"
#include "cuda.hpp"
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
                            ::cuda::device_buffer<ddf_t>& src, ::cuda::device_buffer<ddf_t>& dst)
        {
            const auto q = state.config.q;
            src = ::cuda::device_buffer<ddf_t>(state.stream, state.pool.as_ref(), q * size, ::cuda::no_init);
            if (doubleBuffer) dst = ::cuda::device_buffer<ddf_t>(state.stream, state.pool.as_ref(), q * size, ::cuda::no_init);
            std::vector<ddf_t> feq(q);
            equilibrium(1, 0, 0, 0, feq.data());
            for (int dir = 0; dir < q; ++dir)
            {
                basic::cuda::fill(src.data() + dir * size, size, feq[dir], state.stream);
                if (doubleBuffer) basic::cuda::fill(dst.data() + dir * size, size, feq[dir], state.stream);
            }
            std::println("Init ddf buffer successfully.");
        }
    }

    template <typename Kernels>
    struct Plain
    {
        SimulationState& state;
        Kernels kernels;
        ::cuda::device_buffer<flag_t> flags;
        ::cuda::device_buffer<ddf_t> src, dst;
        PlainParam param;

        Plain(SimulationState& state) :
            state(state), kernels(Kernels::make(state.config.q, state.config.collision)),
            flags(state.stream, state.pool.as_ref()), src(state.stream, state.pool.as_ref()), dst(state.stream, state.pool.as_ref())
        {
            constexpr auto scheme = Kernels::scheme;
            const auto& config = state.config;
            const auto domainFlags = detail::loadFlags(config);
            if (scheme == PlainScheme::EsoTwist and
                std::ranges::any_of(domainFlags, [](flag_t f) { return not (f & STORE_DDF_BIT); }))
                throw std::invalid_argument("--ddfBuffer esotwist requires STORE_DDF_BIT for every domain cell.");
            flags = ::cuda::device_buffer<flag_t>(state.stream, state.pool.as_ref(), domainFlags.size(), ::cuda::no_init);
            if (not config.initFolder.empty())
                std::println("Init flag successfully, from file: {}/flag.dat.", config.initFolder);
            ::cuda::copy_bytes(state.stream, ::cuda::std::span{domainFlags.data(), domainFlags.size()}, flags);
            state.stream.sync();
            detail::initDDF(state, kernels.equilibrium, config.domainSize(), scheme == PlainScheme::TwoFieldPull, src, dst);
            param = PlainParam{
                .invTau = config.invTau,
                .flagBuf = flags.data(),
                .rhoBuf = state.rhoBuf.data(),
                .vxBuf = state.vxBuf.data(),
                .vyBuf = state.vyBuf.data(),
                .vzBuf = state.vzBuf.data(),
                .glbSrcDDFBuf = src.data(),
                .glbDstDDFBuf = dst.data()
            };
            if (scheme == PlainScheme::EsoTwist) param.glbDstDDFBuf = src.data();
            const auto launchExtent = config.launchExtent();
            const auto domainDim = config.domainDim;
            if (launchExtent.x != domainDim.x or launchExtent.y != domainDim.y or launchExtent.z != domainDim.z)
                throw std::runtime_error("Dimension of domain doesn't match computation grid!");
        }

        std::uint32_t advance()
        {
            const auto& config = state.config;
            const auto launch = ::cuda::make_config(::cuda::grid_dims(config.gridDim), ::cuda::block_dims(config.blockDim));
            for (std::uint32_t step = 0; step < config.stepsPerBatch; ++step)
            {
                if constexpr (Kernels::scheme == PlainScheme::EsoTwist)
                {
                    const bool even = ((state.step + step) % 2) == 0;
                    ::cuda::launch(state.stream, launch, even ? kernels.stepEven : kernels.stepOdd, param);
                }
                else if constexpr (Kernels::scheme == PlainScheme::TwoFieldPull)
                {
                    ::cuda::launch(state.stream, launch, kernels.step, param);
                    std::swap(param.glbSrcDDFBuf, param.glbDstDDFBuf);
                }
            }
            return config.stepsPerBatch;
        }

        void dump()
        {
            const auto& config = state.config;
            const auto launch = ::cuda::make_config(::cuda::grid_dims(config.gridDim), ::cuda::block_dims(config.blockDim));
            if constexpr (Kernels::scheme == PlainScheme::EsoTwist)
            {
                const bool even = (state.step % 2) == 0;
                ::cuda::launch(state.stream, launch, even ? kernels.dumpEven : kernels.dumpOdd, param);
            }
            else if constexpr (Kernels::scheme == PlainScheme::TwoFieldPull)
                ::cuda::launch(state.stream, launch, kernels.dump, param);
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
        ::cuda::device_buffer<flag_t> flags;
        ::cuda::device_buffer<ddf_t> tile, src, dst;
        BlockingParam param;
        dim3 tileDim, tileCounts;
        std::uint32_t tileCount;
        idx_t ringOffsetStep;

        Blocking(SimulationState& state) :
            state(state), kernels(BlockingKernels::make(state.config.q, state.config.collision, Buffer, state.config.cacheHint == 1)),
            flags(state.stream, state.pool.as_ref()),
            tile(state.stream, state.pool.as_ref(), static_cast<std::size_t>(state.config.q) * state.config.tileSize(), ::cuda::no_init),
            src(state.stream, state.pool.as_ref()), dst(state.stream, state.pool.as_ref()),
            tileDim(state.config.launchExtent()), tileCounts(state.config.tileCounts()),
            tileCount(tileCounts.x * tileCounts.y * tileCounts.z),
            ringOffsetStep(tileDim.z + 1)
        {
            const auto& config = state.config;
            const std::uint32_t tileSize = tileDim.x * tileDim.y * tileDim.z;
            flags = ::cuda::device_buffer<flag_t>(state.stream, state.pool.as_ref(), tileCount * tileSize, ::cuda::no_init);
            const auto domainFlags = detail::loadFlags(config);

            std::vector<flag_t> tileFlags(tileCount * tileSize);
            idx_t tileBase = 0;
            for (idx_t tileZ = 0 ; tileZ < tileCounts.z ; ++tileZ)
            {
                const auto boundsZ = culbm::core::blocking::tileBounds<idx_t>(tileZ, tileDim.z, tileCounts.z, config.innerLoop, config.domainDim.z);
                for (idx_t tileY = 0 ; tileY < tileCounts.y ; ++tileY)
                {
                    const auto boundsY = culbm::core::blocking::tileBounds<idx_t>(tileY, tileDim.y, tileCounts.y, config.innerLoop, config.domainDim.y);
                    for (idx_t tileX = 0 ; tileX < tileCounts.x ; ++tileX)
                    {
                        const auto boundsX = culbm::core::blocking::tileBounds<idx_t>(tileX, tileDim.x, tileCounts.x, config.innerLoop, config.domainDim.x);

                        for (idx_t localZ = 0, globalZ = boundsZ.loadBegin ; globalZ < boundsZ.loadEnd ; ++localZ, ++globalZ)
                        {
                            for (idx_t localY = 0, globalY = boundsY.loadBegin ; globalY < boundsY.loadEnd ; ++localY, ++globalY)
                            {
                                for (idx_t localX = 0, globalX = boundsX.loadBegin ; globalX < boundsX.loadEnd ; ++localX, ++globalX)
                                {
                                    const idx_t domainIndex = globalX + config.domainDim.x * (globalY + config.domainDim.y * globalZ);
                                    const idx_t tileIndex = localX + tileDim.x * (localY + tileDim.y * localZ);
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

            ::cuda::copy_bytes(state.stream, ::cuda::std::span{tileFlags.data(), tileFlags.size()}, flags);
            state.stream.sync();
            std::println("Init flag successfully, from file: {}/flag.dat.", config.initFolder);
            auto globalSize = config.domainSize();
            if constexpr (Buffer == BlockingGlobalBuffer::Ring)
            {
                assert(tileDim.z >= config.innerLoop);
                globalSize = static_cast<std::size_t>(config.domainDim.x) * config.domainDim.y * (config.domainDim.z + ringOffsetStep);
            }
            detail::initDDF(state, kernels.equilibrium, globalSize, Buffer == BlockingGlobalBuffer::Double, src, dst);
            if (config.disablePersistentL2)
            {
                std::println("Persisting L2 configuration skipped.");
            }
            else
            {
                auto numBytes = static_cast<std::size_t>(config.q) * config.tileSize() * sizeof(ddf_t);
                const auto maxSize = state.device.attribute(::cuda::device_attributes::max_persisting_l2_cache_size);
                if (numBytes > maxSize) {
                    std::println(
                        "Warning: Requested persisting L2 cache size {} exceeds the maximum allowed size {}. Capping to maximum.",
                        numBytes, maxSize
                    );
                    numBytes = maxSize;
                }
                error::check(cudaDeviceSetLimit(cudaLimit::cudaLimitPersistingL2CacheSize, numBytes));
                cudaStreamAttrValue streamAttr;
                streamAttr.accessPolicyWindow.base_ptr = tile.data();
                streamAttr.accessPolicyWindow.num_bytes = numBytes;
                streamAttr.accessPolicyWindow.hitRatio = 1.0;
                streamAttr.accessPolicyWindow.hitProp = cudaAccessProperty::cudaAccessPropertyPersisting;
                streamAttr.accessPolicyWindow.missProp = cudaAccessProperty::cudaAccessPropertyStreaming;
                error::check(cudaStreamSetAttribute(state.stream.get(), cudaLaunchAttributeID::cudaStreamAttributeAccessPolicyWindow, &streamAttr));
            }
            param = BlockingParam{
                .invTau = config.invTau,
                .glbnx = static_cast<idx_t>(config.domainDim.x),
                .glbny = static_cast<idx_t>(config.domainDim.y),
                .glbnz = static_cast<idx_t>(config.domainDim.z),
                .rhoBuf = state.rhoBuf.data(),
                .vxBuf = state.vxBuf.data(),
                .vyBuf = state.vyBuf.data(),
                .vzBuf = state.vzBuf.data(),
                .glbSrcDDFBuf = src.data(),
                .glbDstDDFBuf = dst.data(),
                .tileDDFBuf = tile.data()
            };
            if constexpr (Buffer == BlockingGlobalBuffer::Ring)
            {
                param.glbDstDDFBuf = src.data();
                param.glbDstDDFOffz = config.domainDim.z;
                param.glbDDFRingNz = param.glbDstDDFOffz + ringOffsetStep;
            }
        }

        ~Blocking()
        {
            cudaStreamAttrValue attr{};
            cudaStreamSetAttribute(state.stream.get(), cudaStreamAttributeAccessPolicyWindow, &attr);
        }

        std::uint32_t advance()
        {
            const auto& config = state.config;
            const auto launch = ::cuda::make_config(::cuda::grid_dims(config.gridDim), ::cuda::block_dims(config.blockDim));
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
                        ::cuda::launch(state.stream, launch, kernel, param);
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
            const auto launch = ::cuda::make_config(::cuda::grid_dims(config.gridDim), ::cuda::block_dims(config.blockDim));
            for (std::uint32_t index = 0; index < tileCount; ++index)
            {
                setTile(index, coordinates(index));
                ::cuda::launch(state.stream, launch, kernels.dump, param);
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
            param.flagBuf = flags.data() + index * tileDim.x * tileDim.y * tileDim.z;
        }

    };

}
