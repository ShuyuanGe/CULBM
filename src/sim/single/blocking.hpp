#pragma once

#include <cassert>
#include <cuda/algorithm>
#include <cuda/devices>
#include <cuda/launch>
#include <cuda/std/span>
#include "core/blocking.hpp"
#include "gpu/error.hpp"
#include "physical/resource.hpp"
#include "init.hpp"

namespace culbm::sim::single::strategy
{
    template <Mode Buffer>
    struct Blocking
    {
        State& state;
        BlockingKernels kernels;
        ::cuda::device_buffer<flag_t> flags;
        ::cuda::device_buffer<ddf_t> tile, src, dst;
        BlockingParam param;
        dim3 tileDim, tileCounts;
        std::uint32_t tileCount;
        idx_t ringOffsetStep;

        Blocking(State& state) :
            state(state), kernels(BlockingKernels::make(state.config.q, state.config.collision, Buffer, state.config.cacheHint == 1)),
            flags(state.stream, state.pool.as_ref()),
            tile(state.stream, ::cuda::mr::synchronous_resource_adapter{physical::Resource{state.device}},
                 static_cast<std::size_t>(state.config.q) * state.config.tileSize(), ::cuda::no_init),
            src(state.stream, state.pool.as_ref()), dst(state.stream, state.pool.as_ref()),
            tileDim(state.config.launchExtent()), tileCounts(state.config.tileCounts()),
            tileCount(tileCounts.x * tileCounts.y * tileCounts.z),
            ringOffsetStep(tileDim.z + 1)
        {
            const auto& config = state.config;
            initFlags();
            auto globalSize = config.domainSize();
            if constexpr (Buffer == Mode::BlockRing)
            {
                assert(tileDim.z >= config.innerLoop);
                globalSize = static_cast<std::size_t>(config.domainDim.x) * config.domainDim.y * (config.domainDim.z + ringOffsetStep);
            }
            detail::initDdf(state, kernels.equilibrium, globalSize, Buffer == Mode::BlockDouble, src, dst);
            configureCache();
            param = BlockingParam{
                .invTau = config.invTau,
                .globalNx = static_cast<idx_t>(config.domainDim.x),
                .globalNy = static_cast<idx_t>(config.domainDim.y),
                .globalNz = static_cast<idx_t>(config.domainDim.z),
                .rhoBuf = state.rhoBuf.data(),
                .vxBuf = state.vxBuf.data(),
                .vyBuf = state.vyBuf.data(),
                .vzBuf = state.vzBuf.data(),
                .srcDdf = src.data(),
                .dstDdf = dst.data(),
                .tileDdf = tile.data()
            };
            if constexpr (Buffer == Mode::BlockRing)
            {
                param.dstDdf = src.data();
                param.dstZOffset = config.domainDim.z;
                param.ringNz = param.dstZOffset + ringOffsetStep;
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
                if constexpr (Buffer == Mode::BlockRing)
                {
                    param.srcZOffset = param.dstZOffset;
                    param.dstZOffset -= ringOffsetStep;
                    if (param.dstZOffset < 0) param.dstZOffset += param.ringNz;
                }
                else if constexpr (Buffer == Mode::BlockDouble)
                    std::swap(param.srcDdf, param.dstDdf);
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

    private:
        void initFlags()
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
                                    const bool owned =
                                        boundsZ.storeBegin <= globalZ and globalZ < boundsZ.storeEnd and
                                        boundsY.storeBegin <= globalY and globalY < boundsY.storeEnd and
                                        boundsX.storeBegin <= globalX and globalX < boundsX.storeEnd;
                                    tileFlags[tileBase + tileIndex] = domainFlags[domainIndex] | (owned ? CORRECT_BIT : 0);
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
        }

        void configureCache()
        {
            const auto& config = state.config;
            if (config.disablePersistentL2)
            {
                std::println("Persisting L2 configuration skipped.");
            }
            else
            {
                auto numBytes = tile.size() * sizeof(ddf_t);
                const auto maxSize = state.device.attribute(::cuda::device_attributes::max_persisting_l2_cache_size);
                if (numBytes > maxSize) {
                    std::println(
                        "Warning: Requested persisting L2 cache size {} exceeds the maximum allowed size {}. Capping to maximum.",
                        numBytes, maxSize
                    );
                    numBytes = maxSize;
                }
                gpu::check(cudaDeviceSetLimit(cudaLimit::cudaLimitPersistingL2CacheSize, numBytes));
                cudaStreamAttrValue streamAttr;
                streamAttr.accessPolicyWindow.base_ptr = tile.data();
                streamAttr.accessPolicyWindow.num_bytes = numBytes;
                streamAttr.accessPolicyWindow.hitRatio = 1.0;
                streamAttr.accessPolicyWindow.hitProp = cudaAccessProperty::cudaAccessPropertyPersisting;
                streamAttr.accessPolicyWindow.missProp = cudaAccessProperty::cudaAccessPropertyStreaming;
                gpu::check(cudaStreamSetAttribute(state.stream.get(), cudaLaunchAttributeID::cudaStreamAttributeAccessPolicyWindow, &streamAttr));
            }
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
            param.originX = tileBounds<idx_t>(tile.x, tileDim.x, tileCounts.x, config.innerLoop, config.domainDim.x).loadBegin;
            param.originY = tileBounds<idx_t>(tile.y, tileDim.y, tileCounts.y, config.innerLoop, config.domainDim.y).loadBegin;
            param.originZ = tileBounds<idx_t>(tile.z, tileDim.z, tileCounts.z, config.innerLoop, config.domainDim.z).loadBegin;
            param.flagBuf = flags.data() + index * tileDim.x * tileDim.y * tileDim.z;
        }

    };
}
