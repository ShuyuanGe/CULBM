#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <format>
#include <string>
#include <vector>
#include <cassert>
#include "types.hpp"
#include "CLI11.hpp"
#include "kernel.cuh"
#include "method/dispatch.hpp"
#include "exception.cuh"
#include "flags.hpp"
#include "algorithm.hpp"
#include "simulator.hpp"

namespace culbm::simulator::single_dev
{
    namespace error = culbm::basic::error;

    class Simulator::Data
    {
        friend class Simulator;

        private:
            int _devIdx = 0;
            std::uint32_t _step = 0;
            std::uint32_t _dStep = 0;
            std::uint32_t _nStep = 0;
            real_t _invTau = 0;
            uint3 _blockDim{};
            uint3 _gridDim{};
            uint3 _domDim{};

            enum struct StreamPolicy
            {
                PullStream,
                InplaceStream
            };

            enum struct OptPolicy
            {
                Plain,
                OverlappedBlocking = 3
            };

            enum struct DDFBuffer
            {
                Double,
                Ring,
                EsoTwist
            };

            StreamPolicy    _streamPolicy = StreamPolicy::PullStream;
            OptPolicy       _optPolicy    = OptPolicy::Plain;
            DDFBuffer       _ddfBuffer    = DDFBuffer::Double;
            int             _q            = 27;
            std::string     _collision    = "bgk";
            Kernels         _kernels;
            std::string     _initStateFolder;
            std::string     _dumpFolder;
            bool _dumpRho = false, _dumpVx = false, _dumpVy = false, _dumpVz = false;
            bool _interiorTilesOnly = false;
            int _cacheHint = 1;
            bool _disablePersistentL2 = false;

            cudaStream_t _stream;
            cudaEvent_t _start, _end;

            flag_t* _flagBuf = nullptr;
            real_t* _rhoBuf = nullptr;
            real_t* _vxBuf = nullptr;
            real_t* _vyBuf = nullptr;
            real_t* _vzBuf = nullptr;

            //If we use pull or push stream,
            //the following parameters are valid.
            ddf_t* _srcDDFBuf = nullptr;
            ddf_t* _dstDDFBuf = nullptr;

            //If we use halo-blocking method,
            //the following parameters are valid.
            std::int32_t _innerLoop = 0;

            //If we use l2 cache based halo-blocking method,
            //the following parameters are valid.
            ddf_t* _l2DDFBuf0 = nullptr;

            template <typename T>
            static void fill(T* data, std::size_t size, T value, cudaStream_t stream)
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

            std::vector<flag_t> readFlag() const
            {
                std::vector<flag_t> flag(getDomainSize(), 0);
                if (_initStateFolder.empty())
                {
                    std::fill_n(flag.data(), flag.size(), LOAD_DDF_BIT | COLLIDE_BIT | STORE_DDF_BIT | DUMP_RHO_BIT | DUMP_VX_BIT | DUMP_VY_BIT | DUMP_VZ_BIT);
                }
                else if (std::ifstream f (_initStateFolder +"/flag.dat", std::ios::binary); f)
                {
                    f.read((char*)flag.data(), flag.size() * sizeof(flag_t));
                }
                else
                {
                    throw std::runtime_error(std::format("File: {}/flag.dat doesn't exist!", _initStateFolder));
                }
                return flag;
            }

            void initFlag()
            {
                const auto flag = readFlag();
                if (_ddfBuffer == DDFBuffer::EsoTwist and
                    std::any_of(flag.begin(), flag.end(), [](flag_t f) { return not (f & STORE_DDF_BIT); }))
                    throw std::invalid_argument("--ddfBuffer esotwist requires STORE_DDF_BIT for every domain cell.");
                error::check(cudaMalloc(&_flagBuf, sizeof(flag_t) * getDomainSize()));
                if (not _initStateFolder.empty())
                    std::cout << std::format("Init flag successfully, from file: {}/flag.dat.", _initStateFolder) << std::endl;
                error::check(cudaMemcpy(_flagBuf, flag.data(), flag.size() * sizeof(flag_t), cudaMemcpyHostToDevice));
            }

            void initBlockingFlag()
            {
                const auto blockingDim = getBlockingDim();
                const auto blockingNumDim = getBlockingNumDim();
                const std::uint32_t blockingSize = blockingDim.x * blockingDim.y * blockingDim.z;
                const std::uint32_t blockingNum = blockingNumDim.x * blockingNumDim.y * blockingNumDim.z;
                error::check(cudaMalloc(&_flagBuf, sizeof(flag_t) * blockingNum * blockingSize));
                const auto orgFlag = readFlag();

                std::vector<flag_t> blockingFlag (blockingNum * blockingSize, 0);
                std::int32_t istIdx = 0;
                for (std::int32_t blkIdxZ = 0 ; blkIdxZ < blockingNumDim.z ; ++blkIdxZ)
                {
                    const auto blkZStBegin = culbm::blocking_core::calcValidPrev<std::int32_t>(blkIdxZ  , blockingDim.z, blockingNumDim.z, _innerLoop, _domDim.z);
                    const auto blkZStEnd   = culbm::blocking_core::calcValidPrev<std::int32_t>(blkIdxZ + 1, blockingDim.z, blockingNumDim.z, _innerLoop, _domDim.z);
                    const auto blkZLdBegin  = std::max<std::int32_t>(blkZStBegin - (_innerLoop - 1), 0);
                    const auto blkZLdEnd    = std::min<std::int32_t>(blkZStEnd + (_innerLoop - 1), _domDim.z);
                    for (std::int32_t blkIdxY = 0 ; blkIdxY < blockingNumDim.y ; ++blkIdxY)
                    {
                        const auto blkYStBegin = culbm::blocking_core::calcValidPrev<std::int32_t>(blkIdxY  , blockingDim.y, blockingNumDim.y, _innerLoop, _domDim.y);
                        const auto blkYStEnd   = culbm::blocking_core::calcValidPrev<std::int32_t>(blkIdxY + 1, blockingDim.y, blockingNumDim.y, _innerLoop, _domDim.y);
                        const auto blkYLdBegin  = std::max<std::int32_t>(blkYStBegin - (_innerLoop - 1), 0);
                        const auto blkYLdEnd    = std::min<std::int32_t>(blkYStEnd + (_innerLoop - 1), _domDim.y);
                        for (std::int32_t blkIdxX = 0 ; blkIdxX < blockingNumDim.x ; ++blkIdxX)
                        {
                            const auto blkXStBegin = culbm::blocking_core::calcValidPrev<std::int32_t>(blkIdxX  , blockingDim.x, blockingNumDim.x, _innerLoop, _domDim.x);
                            const auto blkXStEnd   = culbm::blocking_core::calcValidPrev<std::int32_t>(blkIdxX + 1, blockingDim.x, blockingNumDim.x, _innerLoop, _domDim.x);
                            const auto blkXLdBegin  = std::max<std::int32_t>(blkXStBegin - (_innerLoop - 1), 0);
                            const auto blkXLdEnd    = std::min<std::int32_t>(blkXStEnd + (_innerLoop - 1), _domDim.x);

                            for (std::int32_t blkOffZ = 0, glbOffZ = blkZLdBegin ; glbOffZ < blkZLdEnd ; ++blkOffZ, ++glbOffZ)
                            {
                                for (std::int32_t blkOffY = 0, glbOffY = blkYLdBegin ; glbOffY < blkYLdEnd ; ++blkOffY, ++glbOffY)
                                {
                                    for (std::int32_t blkOffX = 0, glbOffX = blkXLdBegin ; glbOffX < blkXLdEnd ; ++blkOffX, ++glbOffX)
                                    {
                                        const std::int32_t glbOff = glbOffX + _domDim.x * (glbOffY + _domDim.y * glbOffZ);
                                        const std::int32_t blkOff = blkOffX + blockingDim.x * (blkOffY + blockingDim.y * blkOffZ);
                                        if (
                                            blkZStBegin <= glbOffZ and glbOffZ < blkZStEnd and
                                            blkYStBegin <= glbOffY and glbOffY < blkYStEnd and
                                            blkXStBegin <= glbOffX and glbOffX < blkXStEnd
                                        )
                                        {
                                            blockingFlag[istIdx + blkOff] = orgFlag[glbOff] | CORRECT_BIT;
                                        }
                                        else
                                        {
                                            blockingFlag[istIdx + blkOff] = orgFlag[glbOff];
                                        }
                                    }
                                }
                            }

                            istIdx += blockingSize;
                        }
                    }
                }

                error::check(cudaMemcpy(_flagBuf, blockingFlag.data(), sizeof(flag_t) * blockingNum * blockingSize, cudaMemcpyHostToDevice));
                std::cout << std::format("Init flag successfully, from file: {}/flag.dat.", _initStateFolder) << std::endl;
            }

            void initRhoU()
            {
                const std::size_t domSize = getDomainSize();
                error::check(cudaMalloc(&_rhoBuf, domSize * sizeof(real_t)));
                error::check(cudaMalloc(&_vxBuf , domSize * sizeof(real_t)));
                error::check(cudaMalloc(&_vyBuf , domSize * sizeof(real_t)));
                error::check(cudaMalloc(&_vzBuf , domSize * sizeof(real_t)));

                auto initHelper = [=, this](std::string_view desc, real_t dftv, real_t* devBuf) -> void
                {
                    if (std::ifstream f (_initStateFolder + std::format("/{}.dat", desc), std::ios::binary) ; f)
                    {
                        std::vector<real_t> hostBuf (domSize, 0);
                        f.read((char*)hostBuf.data(), domSize * sizeof(real_t));
                        error::check(cudaMemcpy(devBuf, hostBuf.data(), domSize * sizeof(real_t), cudaMemcpyHostToDevice));
                        std::cout << std::format("Init {} successfully, from file: {}/{}.dat.", desc, _initStateFolder, desc) << std::endl;
                    }
                    else
                    {
                        fill(devBuf, domSize, dftv, _stream);
                        std::cout << std::format("Init {} successfully, default init with value {}.", desc, dftv) << std::endl;
                    }
                };

                initHelper("rho", 1, _rhoBuf);
                initHelper("vx" , 0, _vxBuf);
                initHelper("vy" , 0, _vyBuf);
                initHelper("vz" , 0, _vzBuf);
            }

            void initDDFBuf()
            {
                auto size = getDomainSize();
                if (_ddfBuffer == DDFBuffer::Ring)
                {
                    const auto blockingDim = getBlockingDim();
                    assert(blockingDim.z >= _innerLoop);
                    const std::size_t extraNz = blockingDim.z + 1;
                    size = static_cast<std::size_t>(_domDim.x) * _domDim.y * (_domDim.z + extraNz);
                }
                error::check(cudaMalloc(&_srcDDFBuf, _q * sizeof(ddf_t) * size));
                if (_ddfBuffer == DDFBuffer::Double)
                    error::check(cudaMalloc(&_dstDDFBuf, _q * sizeof(ddf_t) * size));
                std::vector<ddf_t> feq(_q);
                _kernels.equilibrium(1, 0, 0, 0, feq.data());
                for (int dir = 0; dir < _q; ++dir)
                {
                    fill(_srcDDFBuf + dir * size, size, feq[dir], _stream);
                    if (_ddfBuffer == DDFBuffer::Double)
                        fill(_dstDDFBuf + dir * size, size, feq[dir], _stream);
                }
                std::cout << "Init ddf buffer successfully." << std::endl;
            }

            void mapGlobalMem2PersistL2(void* basePtr, std::size_t numBytes)
            {
                if (_disablePersistentL2)
                {
                    std::cout << "Persisting L2 configuration skipped." << std::endl;
                    return;
                }
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, 0);
                if (numBytes > prop.persistingL2CacheMaxSize) {
                    std::cout << std::format(
                        "Warning: Requested persisting L2 cache size {} exceeds the maximum allowed size {}. Capping to maximum.",
                        numBytes, prop.persistingL2CacheMaxSize
                    ) << std::endl;
                    numBytes = prop.persistingL2CacheMaxSize;
                }

                error::check(cudaDeviceSetLimit(cudaLimit::cudaLimitPersistingL2CacheSize, numBytes));
                cudaStreamAttrValue streamAttr;
                streamAttr.accessPolicyWindow.base_ptr = basePtr;
                streamAttr.accessPolicyWindow.num_bytes = numBytes;
                streamAttr.accessPolicyWindow.hitRatio = 1.0;
                streamAttr.accessPolicyWindow.hitProp = cudaAccessProperty::cudaAccessPropertyPersisting;
                streamAttr.accessPolicyWindow.missProp = cudaAccessProperty::cudaAccessPropertyStreaming;
                error::check(cudaStreamSetAttribute(_stream, cudaLaunchAttributeID::cudaStreamAttributeAccessPolicyWindow, &streamAttr));
            }

        public:
            Data(int argc, char** argv)
            {
                CLI::App app {"Single Device Fluid Solver"};
                using DimValues = std::array<decltype(uint3::x), 3>;
                const auto addDimOption = [&](const char* name, uint3& dim, DimValues defaultValue, const char* description) {
                    return app.add_option_function<DimValues>(name, [&dim](const auto& values) {
                        dim = {values[0], values[1], values[2]};
                    }, description)->run_callback_for_default()->default_val(defaultValue);
                };
                app.add_option("--devId", _devIdx, "Device index of the GPU")
                    ->default_val(0);

                app.add_option("--dstep", _dStep, "Delta time steps of the simulation.")
                    ->default_val(100);

                app.add_option("--nstep", _nStep, "Total time steps of the simulation.")
                    ->default_val(1000);

                app.add_option("--invTau", _invTau, "Reciprocal of Tau.")
                    ->default_val(0.5);

                addDimOption("--blockDim", _blockDim, {32, 16, 2}, "Dimension of blocks per grid.");

                auto* gridOption = addDimOption("--gridDim", _gridDim, {2, 3, 19}, "Dimension of grids per kernel.");

                addDimOption("--domainDim", _domDim, {256, 256, 256}, "Dimension of the domain.");

                app.add_option("--streamPolicy", _streamPolicy, "The stream policy of the solver. {[0]:PullStream, [1]:InplaceStream}.")
                    ->default_val(StreamPolicy::PullStream)->check(CLI::IsMember({0, 1}));

                app.add_option("--optPolicy", _optPolicy, "Optimization: 0 = plain, 3 = overlapped blocking.")
                    ->default_val(OptPolicy::Plain)->check(CLI::IsMember({0, 3}));

                app.add_option("--cacheHint", _cacheHint, "Cache hints for in-place L2 kernels: 0 = normal, 1 = hinted.")
                    ->default_val(1)->check(CLI::Range(0, 1));
                app.add_flag("--no-persistent-l2", _disablePersistentL2, "Do not configure the persisting L2 limit or access-policy window.");

                app.add_option("--velSet", _q, "Velocity set: 15, 19, or 27.")
                    ->default_val(27)->check(CLI::IsMember({15, 19, 27}));
                app.add_option("--collision", _collision, "Collision operator: bgk, mrt, or cumulant.")
                    ->default_val("bgk")->check(CLI::IsMember({"bgk", "mrt", "cumulant"}));

                app.add_option("--initStateFolder", _initStateFolder, "The folder contains initial state of the simulation.");

                app.add_option("--dumpFolder", _dumpFolder, "The folder contains final output.")
                    ->default_val("data/left_inlet_right_outlet_256_256_256_output");

                app.add_flag("--dumpRho", _dumpRho, "Enable dump the density every `dstep` time steps in the simulation.");
                app.add_flag("--dumpVx", _dumpVx, "Enable dumping the x-axis velocity every `dstep` time steps in the simulation.");
                app.add_flag("--dumpVy", _dumpVy, "Enable dumping the y-axis velocity every `dstep` time steps in the simulation.");
                app.add_flag("--dumpVz", _dumpVz, "Enable dumping the z-axis velocity every `dstep` time steps in the simulation.");
                app.add_option("--ddfBuffer", _ddfBuffer, "DDF buffer mode: double, ring, or esotwist.")
                    ->transform(CLI::CheckedTransformer({{"double", "0"}, {"ring", "1"}, {"esotwist", "2"}}))
                    ->default_val(DDFBuffer::Double);
                app.add_flag("--interior-tiles-only", _interiorTilesOnly, "Run only tiles that are non-boundary along all three axes.");

                app.add_option("--innerLoop", _innerLoop, "# of iterations in halo-blocking.")
                    ->default_val(5);

                try
                {
                    app.parse(argc, argv);
                }
                catch (const CLI::ParseError& e)
                {
                    std::exit(app.exit(e));
                }

                if (_optPolicy == OptPolicy::Plain and gridOption->count() == 0 and
                   _domDim.x % _blockDim.x == 0 and _domDim.y % _blockDim.y == 0 and _domDim.z % _blockDim.z == 0)
                {
                    _gridDim = {_domDim.x / _blockDim.x, _domDim.y / _blockDim.y, _domDim.z / _blockDim.z};
                }

                _kernels = culbm::lbm_core::method::dispatch(_q, _collision, [this]<int NDir, culbm::lbm_core::method::Kind Kind>() {
                    return kernels<NDir, Kind>(_cacheHint == 1, _ddfBuffer == DDFBuffer::Ring);
                });

                if ((_optPolicy == OptPolicy::Plain and _streamPolicy != StreamPolicy::PullStream) or
                   (_optPolicy == OptPolicy::OverlappedBlocking and _streamPolicy != StreamPolicy::InplaceStream))
                {
                    throw std::invalid_argument("Supported combinations are --streamPolicy 0 --optPolicy 0 and --streamPolicy 1 --optPolicy 3.");
                }

                if (_ddfBuffer == DDFBuffer::Ring and (_streamPolicy != StreamPolicy::InplaceStream or _optPolicy != OptPolicy::OverlappedBlocking))
                {
                    throw std::runtime_error("--ddfBuffer ring requires --streamPolicy 1 --optPolicy 3.");
                }

                if (_ddfBuffer == DDFBuffer::EsoTwist and _optPolicy != OptPolicy::Plain)
                    throw std::runtime_error("--ddfBuffer esotwist requires plain CUDA.");

                if (_interiorTilesOnly and (_streamPolicy != StreamPolicy::InplaceStream or _optPolicy != OptPolicy::OverlappedBlocking))
                {
                    throw std::runtime_error("--interior-tiles-only requires --streamPolicy 1 --optPolicy 3.");
                }

                if (_interiorTilesOnly)
                {
                    const auto blockingNumDim = getBlockingNumDim();
                    if (blockingNumDim.x < 3 or blockingNumDim.y < 3 or blockingNumDim.z < 3)
                    {
                        throw std::runtime_error("--interior-tiles-only requires at least three tiles along each axis.");
                    }
                }

                if (_interiorTilesOnly and (_dumpRho or _dumpVx or _dumpVy or _dumpVz))
                {
                    std::cout << "Warning: --interior-tiles-only is intended for profiling; dumped results are incorrect." << std::endl;
                }

                error::check(cudaSetDevice(_devIdx));
                error::check(cudaStreamCreate(&_stream));
                error::check(cudaEventCreate(&_start));
                error::check(cudaEventCreate(&_end));

                if (_optPolicy == OptPolicy::OverlappedBlocking)
                    initBlockingFlag();
                else if (_optPolicy == OptPolicy::Plain)
                    initFlag();

                initRhoU();
                initDDFBuf();

                if (_optPolicy == OptPolicy::OverlappedBlocking)
                {
                    const std::uint32_t size = _q * getBlockingSize();
                    error::check(cudaMalloc(&_l2DDFBuf0, sizeof(ddf_t) * size));
                    mapGlobalMem2PersistL2(reinterpret_cast<void*>(_l2DDFBuf0), sizeof(ddf_t) * size);
                }
            }

            ~Data()
            {
                if (_optPolicy == OptPolicy::OverlappedBlocking)
                    error::check(cudaFree(_l2DDFBuf0));

                error::check(cudaFree(_srcDDFBuf));
                if (_dstDDFBuf != nullptr)
                    error::check(cudaFree(_dstDDFBuf));
                error::check(cudaFree(_flagBuf));

                error::check(cudaFree(_rhoBuf));
                error::check(cudaFree(_vxBuf));
                error::check(cudaFree(_vyBuf));
                error::check(cudaFree(_vzBuf));

                error::check(cudaEventDestroy(_end));
                error::check(cudaEventDestroy(_start));
                error::check(cudaStreamDestroy(_stream));
            }

            std::size_t getDomainSize() const noexcept
            {
                return static_cast<std::size_t>(_domDim.x) * _domDim.y * _domDim.z;
            }

            uint3 getBlockingDim() const noexcept
            {
                return {_gridDim.x * _blockDim.x, _gridDim.y * _blockDim.y, _gridDim.z * _blockDim.z};
            }

            std::uint32_t getBlockingSize() const noexcept
            {
                const auto blockingDim = getBlockingDim();
                return blockingDim.x * blockingDim.y * blockingDim.z;
            }

            /**
             * @brief This function is valid only for L2 blocking policies
             */
            uint3 getBlockingNumDim() const
            {
                using namespace culbm::blocking_core;
                const auto blockingDim = getBlockingDim();
                const bool valid =
                    validBlkAxisConfig<std::uint32_t>(_domDim.x, blockingDim.x, _innerLoop) and
                    validBlkAxisConfig<std::uint32_t>(_domDim.y, blockingDim.y, _innerLoop) and
                    validBlkAxisConfig<std::uint32_t>(_domDim.z, blockingDim.z, _innerLoop);
                if (not valid)
                {
                    throw std::invalid_argument("<Domain Dimension, Blocking Dimension and InnerLoop> don't match!");
                }
                return {
                    calcBlkNum<std::uint32_t>(_domDim.x, blockingDim.x, _innerLoop),
                    calcBlkNum<std::uint32_t>(_domDim.y, blockingDim.y, _innerLoop),
                    calcBlkNum<std::uint32_t>(_domDim.z, blockingDim.z, _innerLoop)
                };
            }

    };

    Simulator::Simulator(int argc, char** argv) :
        _data(std::make_unique<Data>(argc, argv))
    {
    }

    void Simulator::run()
    {
        auto dumpRes = [this]() -> void
        {
            // Create dump folder if not exists
            const auto dumpFolder = std::filesystem::path { _data->_dumpFolder };
            std::filesystem::create_directories(dumpFolder);

            if (not (_data->_dumpRho or _data->_dumpVx or _data->_dumpVy or _data->_dumpVz)) return;

            // Allocate host buffer
            const auto nbytes = sizeof(real_t) * _data->getDomainSize();
            auto hostBuf = std::make_unique<char[]>(nbytes);

            auto dumpField = [&](bool enabled, std::string_view prefix, const void *devicePtr)
            {
                if (not enabled) return;

                const auto filePath = dumpFolder / std::format("{}_{}.dat", prefix, _data->_step);

                if (std::ofstream f(filePath, std::ios::binary) ; f)
                {
                    error::check(cudaMemcpy(hostBuf.get(), devicePtr, nbytes, cudaMemcpyDeviceToHost));
                    f.write(hostBuf.get(), nbytes);
                }
                else
                {
                    throw std::runtime_error(std::format("Could not open the file: {}", filePath.string()));
                }

            };

            dumpField(_data->_dumpRho, "rho", _data->_rhoBuf);
            dumpField(_data->_dumpVx, "vx", _data->_vxBuf);
            dumpField(_data->_dumpVy, "vy", _data->_vyBuf);
            dumpField(_data->_dumpVz, "vz", _data->_vzBuf);
        };

        auto blockingRun = [dumpRes, this]() -> void
        {
            const idx_t ringDDFOffzStep =
                static_cast<idx_t>(_data->_gridDim.z * _data->_blockDim.z) + 1;
            BlockingParam param
            {
                .invTau = _data->_invTau,
                .offx = 0, .offy = 0, .offz = 0,
                .glbnx = static_cast<idx_t>(_data->_domDim.x), .glbny = static_cast<idx_t>(_data->_domDim.y), .glbnz = static_cast<idx_t>(_data->_domDim.z),
                .flagBuf = nullptr,
                .glbRhoBuf = _data->_rhoBuf,
                .glbVxBuf = _data->_vxBuf,
                .glbVyBuf = _data->_vyBuf,
                .glbVzBuf = _data->_vzBuf,
                .glbSrcDDFBuf = _data->_srcDDFBuf,
                .glbDstDDFBuf = _data->_ddfBuffer == Data::DDFBuffer::Ring ? _data->_srcDDFBuf : _data->_dstDDFBuf,
                .blkDDFBuf = _data->_l2DDFBuf0,
                .glbDDFRingNz = _data->_ddfBuffer == Data::DDFBuffer::Ring ? static_cast<idx_t>(_data->_domDim.z) + ringDDFOffzStep : 0,
                .glbSrcDDFOffz = 0,
                .glbDstDDFOffz = _data->_ddfBuffer == Data::DDFBuffer::Ring ? static_cast<idx_t>(_data->_domDim.z) : 0
            };

            const dim3 gridDim = _data->_gridDim;
            const dim3 blockDim = _data->_blockDim;
            void* kernelArgs[] = {&param};

            const auto& k = _data->_kernels.blocking;

            const auto blockingDim = _data->getBlockingDim();
            const auto blockingNumDim = _data->getBlockingNumDim();
            const std::uint32_t blockingNum = blockingNumDim.x * blockingNumDim.y * blockingNumDim.z;


            while (_data->_step < _data->_nStep)
            {
                std::uint32_t locStep = 0;
                error::check(cudaEventRecord(_data->_start, _data->_stream));
                for ( ; locStep < _data->_dStep ; locStep += _data->_innerLoop)
                {
                    for (std::uint32_t blkIdx = 0 ; blkIdx < blockingNum ; ++blkIdx)
                    {
                        const idx_t blkIdxX = blkIdx % blockingNumDim.x;
                        const idx_t blkIdxY = (blkIdx / blockingNumDim.x) % blockingNumDim.y;
                        const idx_t blkIdxZ = blkIdx / (blockingNumDim.x * blockingNumDim.y);
                        if (
                            _data->_interiorTilesOnly and
                            (blkIdxX == 0 or blkIdxX == blockingNumDim.x - 1 or
                             blkIdxY == 0 or blkIdxY == blockingNumDim.y - 1 or
                             blkIdxZ == 0 or blkIdxZ == blockingNumDim.z - 1)
                        )
                        {
                            continue;
                        }
                        param.offx = std::max<idx_t>(culbm::blocking_core::calcValidPrev<idx_t>(blkIdxX, blockingDim.x, blockingNumDim.x, _data->_innerLoop, _data->_domDim.x) - (_data->_innerLoop - 1), 0);
                        param.offy = std::max<idx_t>(culbm::blocking_core::calcValidPrev<idx_t>(blkIdxY, blockingDim.y, blockingNumDim.y, _data->_innerLoop, _data->_domDim.y) - (_data->_innerLoop - 1), 0);
                        param.offz = std::max<idx_t>(culbm::blocking_core::calcValidPrev<idx_t>(blkIdxZ, blockingDim.z, blockingNumDim.z, _data->_innerLoop, _data->_domDim.z) - (_data->_innerLoop - 1), 0);
                        param.flagBuf = _data->_flagBuf + blkIdx * blockingDim.x * blockingDim.y * blockingDim.z;

                        for (std::int32_t loop = 0 ; loop < _data->_innerLoop ; ++loop)
                        {
                            if (loop == 0)
                            {
                                if (_data->_innerLoop == 1)
                                {
                                    error::check(cudaLaunchKernel(k.single, gridDim, blockDim, kernelArgs, 0, _data->_stream));
                                }
                                else
                                {
                                    error::check(cudaLaunchKernel(k.first, gridDim, blockDim, kernelArgs, 0, _data->_stream));
                                }
                            }
                            else if (loop < (_data->_innerLoop - 1))
                            {
                                if ((loop % 2) == 0)
                                {
                                    error::check(cudaLaunchKernel(k.middleEven, gridDim, blockDim, kernelArgs, 0, _data->_stream));
                                }
                                else
                                {
                                    error::check(cudaLaunchKernel(k.middleOdd, gridDim, blockDim, kernelArgs, 0, _data->_stream));
                                }
                            }
                            else
                            {
                                if ((loop % 2) == 0)
                                {
                                    error::check(cudaLaunchKernel(k.lastEven, gridDim, blockDim, kernelArgs, 0, _data->_stream));
                                }
                                else
                                {
                                    error::check(cudaLaunchKernel(k.lastOdd, gridDim, blockDim, kernelArgs, 0, _data->_stream));
                                }
                            }
                        }
                    }

                    if (_data->_ddfBuffer == Data::DDFBuffer::Ring)
                    {
                        param.glbSrcDDFOffz = param.glbDstDDFOffz;
                        param.glbDstDDFOffz -= ringDDFOffzStep;
                        if (param.glbDstDDFOffz < 0)
                        {
                            param.glbDstDDFOffz += param.glbDDFRingNz;
                        }
                    }
                    else
                    {
                        std::swap(param.glbSrcDDFBuf, param.glbDstDDFBuf);
                    }
                }
                error::check(cudaEventRecord(_data->_end, _data->_stream));
                error::check(cudaEventSynchronize(_data->_end));
                float ms;
                error::check(cudaEventElapsedTime(&ms, _data->_start, _data->_end));
                const std::size_t updateNum = _data->_interiorTilesOnly ?
                    static_cast<std::size_t>(blockingNumDim.x - 2) * (blockingNumDim.y - 2) * (blockingNumDim.z - 2) *
                    (blockingDim.x - 2 * (_data->_innerLoop - 1)) *
                    (blockingDim.y - 2 * (_data->_innerLoop - 1)) *
                    (blockingDim.z - 2 * (_data->_innerLoop - 1)) :
                    static_cast<std::size_t>(_data->_domDim.x) * _data->_domDim.y * _data->_domDim.z;
                const float mlups = updateNum * 1e-6f / (ms / 1000) * locStep;
                std::cout << std::format("speed = {:.4f} (MLUPS)", mlups) << std::endl;
                _data->_step += locStep;

                for (std::uint32_t blkIdx = 0 ; blkIdx < blockingNum ; ++blkIdx)
                {
                    const idx_t blkIdxX = blkIdx % blockingNumDim.x;
                    const idx_t blkIdxY = (blkIdx / blockingNumDim.x) % blockingNumDim.y;
                    const idx_t blkIdxZ = blkIdx / (blockingNumDim.x * blockingNumDim.y);
                    param.offx = std::max<idx_t>(culbm::blocking_core::calcValidPrev<idx_t>(blkIdxX, blockingDim.x, blockingNumDim.x, _data->_innerLoop, _data->_domDim.x) - (_data->_innerLoop - 1), 0);
                    param.offy = std::max<idx_t>(culbm::blocking_core::calcValidPrev<idx_t>(blkIdxY, blockingDim.y, blockingNumDim.y, _data->_innerLoop, _data->_domDim.y) - (_data->_innerLoop - 1), 0);
                    param.offz = std::max<idx_t>(culbm::blocking_core::calcValidPrev<idx_t>(blkIdxZ, blockingDim.z, blockingNumDim.z, _data->_innerLoop, _data->_domDim.z) - (_data->_innerLoop - 1), 0);
                    param.flagBuf = _data->_flagBuf + blkIdx * blockingDim.x * blockingDim.y * blockingDim.z;
                    error::check(cudaLaunchKernel(k.dump, gridDim, blockDim, kernelArgs, 0, _data->_stream));
                }

                error::check(cudaStreamSynchronize(_data->_stream));
                dumpRes();
            }
        };

        auto plainRun = [dumpRes, this]() -> void
        {
            PlainParam param
            {
                .invTau = _data->_invTau,
                .flagBuf = _data->_flagBuf,
                .glbRhoBuf  = _data->_rhoBuf,
                .glbVxBuf   = _data->_vxBuf,
                .glbVyBuf   = _data->_vyBuf,
                .glbVzBuf   = _data->_vzBuf,
                .glbSrcDDFBuf = _data->_srcDDFBuf,
                .glbDstDDFBuf = _data->_ddfBuffer == Data::DDFBuffer::EsoTwist ? _data->_srcDDFBuf : _data->_dstDDFBuf
            };

            const dim3 gridDim = _data->_gridDim;
            const dim3 blockDim = _data->_blockDim;
            const auto blockingDim = _data->getBlockingDim();

            if (blockingDim.x != _data->_domDim.x or blockingDim.y != _data->_domDim.y or blockingDim.z != _data->_domDim.z)
            {
                throw std::runtime_error(
                    std::format("Dimension of domain doesn't match computation grid!")
                );
            }

            void* kernelArgs[] = {&param};
            const auto& k = _data->_kernels;

            while (_data->_step < _data->_nStep)
            {
                error::check(cudaEventRecord(_data->_start, _data->_stream));
                for (std::uint32_t locStep = 0 ; locStep < _data->_dStep ; ++locStep)
                {
                    if (_data->_ddfBuffer == Data::DDFBuffer::EsoTwist)
                    {
                        const bool even = ((_data->_step + locStep) % 2) == 0;
                        error::check(cudaLaunchKernel(even ? k.plainEso.stepEven : k.plainEso.stepOdd, gridDim, blockDim, kernelArgs, 0, _data->_stream));
                    }
                    else
                    {
                        error::check(cudaLaunchKernel(k.plain.step, gridDim, blockDim, kernelArgs, 0, _data->_stream));
                        std::swap(param.glbSrcDDFBuf, param.glbDstDDFBuf);
                    }
                }
                error::check(cudaEventRecord(_data->_end, _data->_stream));
                error::check(cudaEventSynchronize(_data->_end));
                float ms;
                error::check(cudaEventElapsedTime(&ms, _data->_start, _data->_end));
                const float mlups = (_data->_domDim.x * _data->_domDim.y * _data->_domDim.z) * 1e-6f / (ms / 1000) * _data->_dStep;
                std::cout << std::format("speed = {:.4f} (MLUPS)", mlups) << std::endl;
                _data->_step += _data->_dStep;

                if (_data->_ddfBuffer == Data::DDFBuffer::EsoTwist)
                {
                    const bool even = (_data->_step % 2) == 0;
                    error::check(cudaLaunchKernel(even ? k.plainEso.dumpEven : k.plainEso.dumpOdd, gridDim, blockDim, kernelArgs, 0, _data->_stream));
                }
                else
                    error::check(cudaLaunchKernel(k.plain.dump, gridDim, blockDim, kernelArgs, 0, _data->_stream));
                error::check(cudaStreamSynchronize(_data->_stream));
                dumpRes();
            }
        };

        if (
            _data->_streamPolicy == Data::StreamPolicy::InplaceStream and
            _data->_optPolicy == Data::OptPolicy::OverlappedBlocking
        )
        {
            blockingRun();
        }

        if (
            _data->_streamPolicy == Data::StreamPolicy::PullStream and
            _data->_optPolicy == Data::OptPolicy::Plain
        )
        {
            plainRun();
        }
    }

    Simulator::~Simulator() = default;
}
