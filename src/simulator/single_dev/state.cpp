#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <format>
#include <fstream>
#include <print>
#include <stdexcept>
#include <string_view>
#include <vector>
#include "algorithm.hpp"
#include "exception.cuh"
#include "simulation.hpp"

namespace culbm::simulator::single_dev
{
    namespace error = culbm::basic::error;

    namespace
    {
        template <typename T>
        void fill(T* data, std::size_t size, T value, cudaStream_t stream)
        {
            static_assert(sizeof(T) == sizeof(std::uint32_t));
            const auto result = cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(data), std::bit_cast<std::uint32_t>(value), size, stream);
            if (result != CUDA_SUCCESS)
            {
                const char* description = nullptr;
                cuGetErrorString(result, &description);
                throw std::runtime_error(std::format("cuMemsetD32Async failed: {} ({})", description ? description : "unknown error", static_cast<int>(result)));
            }
        }
    }

    SimulationState::SimulationState(const SimulationConfig& config, Equilibrium equilibrium) :
        config(config)
    {
        initResources(equilibrium);
    }

    void SimulationState::initResources(Equilibrium equilibrium)
    {
        error::check(cudaSetDevice(config.deviceId));
        error::check(cudaStreamCreate(&stream));
        error::check(cudaEventCreate(&startEvent));
        error::check(cudaEventCreate(&endEvent));

        if (std::holds_alternative<BlockingGlobalBuffer>(config.strategy))
            initTileFlags();
        else if (std::holds_alternative<PlainScheme>(config.strategy))
            initFlags();

        initMacros();
        initDDF(equilibrium);

        if (std::holds_alternative<BlockingGlobalBuffer>(config.strategy))
        {
            const std::uint32_t size = config.q * config.tileSize();
            error::check(cudaMalloc(&tileDDFBuf, sizeof(ddf_t) * size));
            configureL2(reinterpret_cast<void*>(tileDDFBuf), sizeof(ddf_t) * size);
        }
    }

    std::vector<flag_t> SimulationState::loadFlags() const
    {
        std::vector<flag_t> flag(config.domainSize());
        if (config.initStateFolder.empty())
        {
            std::ranges::fill(flag, LOAD_DDF_BIT | COLLIDE_BIT | STORE_DDF_BIT | DUMP_RHO_BIT | DUMP_VX_BIT | DUMP_VY_BIT | DUMP_VZ_BIT);
        }
        else if (std::ifstream f (std::format("{}/flag.dat", config.initStateFolder), std::ios::binary); f)
        {
            f.read(reinterpret_cast<char*>(flag.data()), flag.size() * sizeof(flag_t));
        }
        else
        {
            throw std::runtime_error(std::format("File: {}/flag.dat doesn't exist!", config.initStateFolder));
        }
        return flag;
    }

    void SimulationState::initFlags()
    {
        const auto flag = loadFlags();
        if (std::get<PlainScheme>(config.strategy) == PlainScheme::EsoTwist and
            std::ranges::any_of(flag, [](flag_t f) { return not (f & STORE_DDF_BIT); }))
            throw std::invalid_argument("--ddfBuffer esotwist requires STORE_DDF_BIT for every domain cell.");
        error::check(cudaMalloc(&flagBuf, sizeof(flag_t) * flag.size()));
        if (not config.initStateFolder.empty())
            std::println("Init flag successfully, from file: {}/flag.dat.", config.initStateFolder);
        error::check(cudaMemcpy(flagBuf, flag.data(), flag.size() * sizeof(flag_t), cudaMemcpyHostToDevice));
    }

    void SimulationState::initTileFlags()
    {
        const auto launchExtent = config.launchExtent();
        const auto tileCounts = config.tileCounts();
        const std::uint32_t tileSize = launchExtent.x * launchExtent.y * launchExtent.z;
        const std::uint32_t tileCount = tileCounts.x * tileCounts.y * tileCounts.z;
        error::check(cudaMalloc(&flagBuf, sizeof(flag_t) * tileCount * tileSize));
        const auto orgFlag = loadFlags();

        std::vector<flag_t> tileFlags(tileCount * tileSize);
        std::int32_t istIdx = 0;
        for (std::int32_t tileIdxZ = 0 ; tileIdxZ < tileCounts.z ; ++tileIdxZ)
        {
            const auto tileZStBegin = culbm::blocking_core::calcValidPrev<std::int32_t>(tileIdxZ  , launchExtent.z, tileCounts.z, config.innerLoop, config.domainDim.z);
            const auto tileZStEnd   = culbm::blocking_core::calcValidPrev<std::int32_t>(tileIdxZ + 1, launchExtent.z, tileCounts.z, config.innerLoop, config.domainDim.z);
            const auto tileZLdBegin  = std::max<std::int32_t>(tileZStBegin - (config.innerLoop - 1), 0);
            const auto tileZLdEnd    = std::min<std::int32_t>(tileZStEnd + (config.innerLoop - 1), config.domainDim.z);
            for (std::int32_t tileIdxY = 0 ; tileIdxY < tileCounts.y ; ++tileIdxY)
            {
                const auto tileYStBegin = culbm::blocking_core::calcValidPrev<std::int32_t>(tileIdxY  , launchExtent.y, tileCounts.y, config.innerLoop, config.domainDim.y);
                const auto tileYStEnd   = culbm::blocking_core::calcValidPrev<std::int32_t>(tileIdxY + 1, launchExtent.y, tileCounts.y, config.innerLoop, config.domainDim.y);
                const auto tileYLdBegin  = std::max<std::int32_t>(tileYStBegin - (config.innerLoop - 1), 0);
                const auto tileYLdEnd    = std::min<std::int32_t>(tileYStEnd + (config.innerLoop - 1), config.domainDim.y);
                for (std::int32_t tileIdxX = 0 ; tileIdxX < tileCounts.x ; ++tileIdxX)
                {
                    const auto tileXStBegin = culbm::blocking_core::calcValidPrev<std::int32_t>(tileIdxX  , launchExtent.x, tileCounts.x, config.innerLoop, config.domainDim.x);
                    const auto tileXStEnd   = culbm::blocking_core::calcValidPrev<std::int32_t>(tileIdxX + 1, launchExtent.x, tileCounts.x, config.innerLoop, config.domainDim.x);
                    const auto tileXLdBegin  = std::max<std::int32_t>(tileXStBegin - (config.innerLoop - 1), 0);
                    const auto tileXLdEnd    = std::min<std::int32_t>(tileXStEnd + (config.innerLoop - 1), config.domainDim.x);

                    for (std::int32_t tileOffZ = 0, glbOffZ = tileZLdBegin ; glbOffZ < tileZLdEnd ; ++tileOffZ, ++glbOffZ)
                    {
                        for (std::int32_t tileOffY = 0, glbOffY = tileYLdBegin ; glbOffY < tileYLdEnd ; ++tileOffY, ++glbOffY)
                        {
                            for (std::int32_t tileOffX = 0, glbOffX = tileXLdBegin ; glbOffX < tileXLdEnd ; ++tileOffX, ++glbOffX)
                            {
                                const std::int32_t glbOff = glbOffX + config.domainDim.x * (glbOffY + config.domainDim.y * glbOffZ);
                                const std::int32_t tileOff = tileOffX + launchExtent.x * (tileOffY + launchExtent.y * tileOffZ);
                                if (
                                    tileZStBegin <= glbOffZ and glbOffZ < tileZStEnd and
                                    tileYStBegin <= glbOffY and glbOffY < tileYStEnd and
                                    tileXStBegin <= glbOffX and glbOffX < tileXStEnd
                                )
                                {
                                    tileFlags[istIdx + tileOff] = orgFlag[glbOff] | CORRECT_BIT;
                                }
                                else
                                {
                                    tileFlags[istIdx + tileOff] = orgFlag[glbOff];
                                }
                            }
                        }
                    }

                    istIdx += tileSize;
                }
            }
        }

        error::check(cudaMemcpy(flagBuf, tileFlags.data(), sizeof(flag_t) * tileCount * tileSize, cudaMemcpyHostToDevice));
        std::println("Init flag successfully, from file: {}/flag.dat.", config.initStateFolder);
    }

    void SimulationState::initMacros()
    {
        const std::size_t domSize = config.domainSize();
        for (auto* buf : {&rhoBuf, &vxBuf, &vyBuf, &vzBuf})
            error::check(cudaMalloc(buf, domSize * sizeof(real_t)));

        auto init = [this, domSize](std::string_view desc, real_t dftv, real_t* devBuf)
        {
            if (std::ifstream f (std::format("{}/{}.dat", config.initStateFolder, desc), std::ios::binary) ; f)
            {
                std::vector<real_t> hostBuf(domSize);
                f.read(reinterpret_cast<char*>(hostBuf.data()), domSize * sizeof(real_t));
                error::check(cudaMemcpy(devBuf, hostBuf.data(), domSize * sizeof(real_t), cudaMemcpyHostToDevice));
                std::println("Init {} successfully, from file: {}/{}.dat.", desc, config.initStateFolder, desc);
            }
            else
            {
                fill(devBuf, domSize, dftv, stream);
                std::println("Init {} successfully, default init with value {}.", desc, dftv);
            }
        };

        init("rho", 1, rhoBuf);
        init("vx" , 0, vxBuf);
        init("vy" , 0, vyBuf);
        init("vz" , 0, vzBuf);
    }

    void SimulationState::initDDF(Equilibrium equilibrium)
    {
        const auto* plain = std::get_if<PlainScheme>(&config.strategy);
        const auto* blocking = std::get_if<BlockingGlobalBuffer>(&config.strategy);
        const bool doubleBuffer = (plain and *plain == PlainScheme::TwoFieldPull) or
                                  (blocking and *blocking == BlockingGlobalBuffer::Double);
        auto size = config.domainSize();
        if (blocking and *blocking == BlockingGlobalBuffer::Ring)
        {
            const auto launchExtent = config.launchExtent();
            assert(launchExtent.z >= config.innerLoop);
            const std::size_t extraNz = launchExtent.z + 1;
            size = static_cast<std::size_t>(config.domainDim.x) * config.domainDim.y * (config.domainDim.z + extraNz);
        }
        error::check(cudaMalloc(&srcDDFBuf, config.q * sizeof(ddf_t) * size));
        if (doubleBuffer)
            error::check(cudaMalloc(&dstDDFBuf, config.q * sizeof(ddf_t) * size));
        std::vector<ddf_t> feq(config.q);
        equilibrium(1, 0, 0, 0, feq.data());
        for (int dir = 0; dir < config.q; ++dir)
        {
            fill(srcDDFBuf + dir * size, size, feq[dir], stream);
            if (doubleBuffer)
                fill(dstDDFBuf + dir * size, size, feq[dir], stream);
        }
        std::println("Init ddf buffer successfully.");
    }

    void SimulationState::configureL2(void* basePtr, std::size_t numBytes)
    {
        if (config.disablePersistentL2)
        {
            std::println("Persisting L2 configuration skipped.");
            return;
        }
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        if (numBytes > prop.persistingL2CacheMaxSize) {
            std::println(
                "Warning: Requested persisting L2 cache size {} exceeds the maximum allowed size {}. Capping to maximum.",
                numBytes, prop.persistingL2CacheMaxSize
            );
            numBytes = prop.persistingL2CacheMaxSize;
        }

        error::check(cudaDeviceSetLimit(cudaLimit::cudaLimitPersistingL2CacheSize, numBytes));
        cudaStreamAttrValue streamAttr;
        streamAttr.accessPolicyWindow.base_ptr = basePtr;
        streamAttr.accessPolicyWindow.num_bytes = numBytes;
        streamAttr.accessPolicyWindow.hitRatio = 1.0;
        streamAttr.accessPolicyWindow.hitProp = cudaAccessProperty::cudaAccessPropertyPersisting;
        streamAttr.accessPolicyWindow.missProp = cudaAccessProperty::cudaAccessPropertyStreaming;
        error::check(cudaStreamSetAttribute(stream, cudaLaunchAttributeID::cudaStreamAttributeAccessPolicyWindow, &streamAttr));
    }

    SimulationState::~SimulationState()
    {
        if (std::holds_alternative<BlockingGlobalBuffer>(config.strategy))
            error::check(cudaFree(tileDDFBuf));

        error::check(cudaFree(srcDDFBuf));
        if (dstDDFBuf != nullptr)
            error::check(cudaFree(dstDDFBuf));
        error::check(cudaFree(flagBuf));

        for (auto* buf : {rhoBuf, vxBuf, vyBuf, vzBuf})
            error::check(cudaFree(buf));

        error::check(cudaEventDestroy(endEvent));
        error::check(cudaEventDestroy(startEvent));
        error::check(cudaStreamDestroy(stream));
    }
}
