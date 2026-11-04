#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "kernel.cuh"

namespace culbm::simulator::single_dev
{
    struct SimulationConfig
    {
        int deviceId = 0;
        std::uint32_t stepsPerBatch = 0, targetStep = 0;
        real_t invTau = 0;
        uint3 blockDim{}, gridDim{}, domainDim{};
        StrategySelection strategy = PlainScheme::TwoFieldPull;
        int q = 27;
        std::string collision = "bgk";
        std::string initStateFolder, dumpFolder;
        bool dumpRho = false, dumpVx = false, dumpVy = false, dumpVz = false;
        bool interiorTilesOnly = false;
        int cacheHint = 1;
        bool disablePersistentL2 = false;
        std::int32_t innerLoop = 0;

        std::size_t domainSize() const noexcept;
        uint3 launchExtent() const noexcept;
        std::uint32_t tileSize() const noexcept;
        uint3 tileCounts() const;
    };

    struct ParsedOptions
    {
        enum class StreamPolicy { PullStream, InplaceStream };
        enum class OptPolicy { Plain, OverlappedBlocking = 3 };
        enum class BufferOption { Double, Ring, EsoTwist };

        SimulationConfig config;
        StreamPolicy stream = StreamPolicy::PullStream;
        OptPolicy opt = OptPolicy::Plain;
        BufferOption buffer = BufferOption::Double;

        ParsedOptions(int argc, char** argv);
        void validate() const;
        StrategySelection selectStrategy() const;
    };

    struct SimulationState
    {
        const SimulationConfig& config;
        std::uint32_t step = 0;
        cudaStream_t stream;
        cudaEvent_t startEvent, endEvent;
        flag_t* flagBuf = nullptr;
        real_t* rhoBuf = nullptr;
        real_t* vxBuf = nullptr;
        real_t* vyBuf = nullptr;
        real_t* vzBuf = nullptr;
        ddf_t* srcDDFBuf = nullptr;
        ddf_t* dstDDFBuf = nullptr;
        ddf_t* tileDDFBuf = nullptr;

        SimulationState(const SimulationConfig& config, Equilibrium equilibrium);
        ~SimulationState();

        void initResources(Equilibrium equilibrium);
        std::vector<flag_t> loadFlags() const;
        void initFlags();
        void initTileFlags();
        void initMacros();
        void initDDF(Equilibrium equilibrium);
        void configureL2(void* basePtr, std::size_t numBytes);
    };

    void simulate(SimulationState& state, const KernelSelection& kernels);
}
