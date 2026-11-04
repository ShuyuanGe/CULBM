#include <algorithm>
#include <cassert>
#include <cstdint>
#include <format>
#include <fstream>
#include <print>
#include <stdexcept>
#include <vector>
#include "blocking.hpp"
#include "strategy.hpp"

namespace culbm::simulator::single_dev::strategy
{
    namespace
    {
        std::vector<flag_t> loadFlags(const SimulationConfig& config)
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

        void initDDF(SimulationState& state, Equilibrium equilibrium, std::size_t size, bool doubleBuffer,
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
    Plain<Kernels>::Plain(SimulationState& state, const Kernels& kernels) : simulation(state), kernels(kernels)
    {
        constexpr auto scheme = Kernels::scheme;
        const auto equilibrium = kernels.equilibrium;
        const auto& config = state.config;
        const auto flag = loadFlags(config);
        if (scheme == PlainScheme::EsoTwist and
            std::ranges::any_of(flag, [](flag_t f) { return not (f & STORE_DDF_BIT); }))
            throw std::invalid_argument("--ddfBuffer esotwist requires STORE_DDF_BIT for every domain cell.");
        flags = basic::cuda::allocate<flag_t>(flag.size());
        if (not config.initStateFolder.empty())
            std::println("Init flag successfully, from file: {}/flag.dat.", config.initStateFolder);
        error::check(cudaMemcpy(flags.get(), flag.data(), flag.size() * sizeof(flag_t), cudaMemcpyHostToDevice));
        initDDF(state, equilibrium, config.domainSize(), scheme == PlainScheme::TwoFieldPull, src, dst);
        param = PlainParam{
            .invTau = state.config.invTau,
            .flagBuf = flags.get(),
            .glbRhoBuf = state.rhoBuf.get(),
            .glbVxBuf = state.vxBuf.get(),
            .glbVyBuf = state.vyBuf.get(),
            .glbVzBuf = state.vzBuf.get(),
            .glbSrcDDFBuf = src.get(),
            .glbDstDDFBuf = dst.get()
        };
        if (scheme == PlainScheme::EsoTwist) param.glbDstDDFBuf = src.get();
        const auto launchExtent = state.config.launchExtent();
        const auto domainDim = state.config.domainDim;
        if (launchExtent.x != domainDim.x or launchExtent.y != domainDim.y or launchExtent.z != domainDim.z)
            throw std::runtime_error("Dimension of domain doesn't match computation grid!");
    }

    void configureL2(SimulationState& state, void* basePtr, std::size_t numBytes)
    {
        if (state.config.disablePersistentL2)
        {
            std::println("Persisting L2 configuration skipped.");
            return;
        }
        cudaDeviceProp prop;
        error::check(cudaGetDeviceProperties(&prop, state.config.deviceId));
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
        error::check(cudaStreamSetAttribute(state.stream, cudaLaunchAttributeID::cudaStreamAttributeAccessPolicyWindow, &streamAttr));
    }

    template <BlockingGlobalBuffer Buffer>
    Blocking<Buffer>::Blocking(SimulationState& state, const BlockingKernels& kernels) :
        simulation(state), kernels(kernels), tileDim(state.config.launchExtent()), tileCounts(state.config.tileCounts()),
        tileCount(tileCounts.x * tileCounts.y * tileCounts.z),
        ringOffsetStep(static_cast<idx_t>(tileDim.z) + 1)
    {
        constexpr auto buffer = Buffer;
        const auto equilibrium = kernels.equilibrium;
        const auto& config = state.config;
        const auto launchExtent = tileDim;
        const std::uint32_t tileSize = launchExtent.x * launchExtent.y * launchExtent.z;
        flags = basic::cuda::allocate<flag_t>(tileCount * tileSize);
        const auto orgFlag = loadFlags(config);

        std::vector<flag_t> tileFlags(tileCount * tileSize);
        std::int32_t istIdx = 0;
        for (std::int32_t tileIdxZ = 0 ; tileIdxZ < tileCounts.z ; ++tileIdxZ)
        {
            const auto boundsZ = culbm::core::blocking::calcTileBounds<std::int32_t>(tileIdxZ, launchExtent.z, tileCounts.z, config.innerLoop, config.domainDim.z);
            for (std::int32_t tileIdxY = 0 ; tileIdxY < tileCounts.y ; ++tileIdxY)
            {
                const auto boundsY = culbm::core::blocking::calcTileBounds<std::int32_t>(tileIdxY, launchExtent.y, tileCounts.y, config.innerLoop, config.domainDim.y);
                for (std::int32_t tileIdxX = 0 ; tileIdxX < tileCounts.x ; ++tileIdxX)
                {
                    const auto boundsX = culbm::core::blocking::calcTileBounds<std::int32_t>(tileIdxX, launchExtent.x, tileCounts.x, config.innerLoop, config.domainDim.x);

                    for (std::int32_t tileOffZ = 0, glbOffZ = boundsZ.loadBegin ; glbOffZ < boundsZ.loadEnd ; ++tileOffZ, ++glbOffZ)
                    {
                        for (std::int32_t tileOffY = 0, glbOffY = boundsY.loadBegin ; glbOffY < boundsY.loadEnd ; ++tileOffY, ++glbOffY)
                        {
                            for (std::int32_t tileOffX = 0, glbOffX = boundsX.loadBegin ; glbOffX < boundsX.loadEnd ; ++tileOffX, ++glbOffX)
                            {
                                const std::int32_t glbOff = glbOffX + config.domainDim.x * (glbOffY + config.domainDim.y * glbOffZ);
                                const std::int32_t tileOff = tileOffX + launchExtent.x * (tileOffY + launchExtent.y * tileOffZ);
                                if (
                                    boundsZ.storeBegin <= glbOffZ and glbOffZ < boundsZ.storeEnd and
                                    boundsY.storeBegin <= glbOffY and glbOffY < boundsY.storeEnd and
                                    boundsX.storeBegin <= glbOffX and glbOffX < boundsX.storeEnd
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

        error::check(cudaMemcpy(flags.get(), tileFlags.data(), sizeof(flag_t) * tileCount * tileSize, cudaMemcpyHostToDevice));
        std::println("Init flag successfully, from file: {}/flag.dat.", config.initStateFolder);
        auto size = config.domainSize();
        if (buffer == BlockingGlobalBuffer::Ring)
        {
            assert(tileDim.z >= config.innerLoop);
            size = static_cast<std::size_t>(config.domainDim.x) * config.domainDim.y * (config.domainDim.z + ringOffsetStep);
        }
        initDDF(state, equilibrium, size, buffer == BlockingGlobalBuffer::Double, src, dst);
        tile = basic::cuda::allocate<ddf_t>(static_cast<std::size_t>(config.q) * config.tileSize());
        configureL2(state, tile.get(), static_cast<std::size_t>(config.q) * config.tileSize() * sizeof(ddf_t));
        param = BlockingParam{
            .invTau = state.config.invTau,
            .offx = 0, .offy = 0, .offz = 0,
            .glbnx = static_cast<idx_t>(state.config.domainDim.x),
            .glbny = static_cast<idx_t>(state.config.domainDim.y),
            .glbnz = static_cast<idx_t>(state.config.domainDim.z),
            .flagBuf = nullptr,
            .glbRhoBuf = state.rhoBuf.get(),
            .glbVxBuf = state.vxBuf.get(),
            .glbVyBuf = state.vyBuf.get(),
            .glbVzBuf = state.vzBuf.get(),
            .glbSrcDDFBuf = src.get(),
            .glbDstDDFBuf = dst.get(),
            .tileDDFBuf = tile.get(),
            .glbDDFRingNz = 0,
            .glbSrcDDFOffz = 0,
            .glbDstDDFOffz = 0
        };
        if (buffer == BlockingGlobalBuffer::Ring)
        {
            param.glbDstDDFBuf = src.get();
            param.glbDstDDFOffz = state.config.domainDim.z;
            param.glbDDFRingNz = param.glbDstDDFOffz + ringOffsetStep;
        }
    }

    template <BlockingGlobalBuffer Buffer>
    Blocking<Buffer>::~Blocking()
    {
        cudaStreamAttrValue attr{};
        cudaStreamSetAttribute(simulation.stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    }
    template struct Plain<PlainPullKernels>;
    template struct Plain<PlainEsoKernels>;
    template struct Blocking<BlockingGlobalBuffer::Double>;
    template struct Blocking<BlockingGlobalBuffer::Ring>;
}
