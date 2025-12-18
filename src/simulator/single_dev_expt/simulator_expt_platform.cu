#include <cmath>
#include <format>
#include <string>
#include <vector>
#include <cassert>
#include "vec3.hpp"
#include "CLI11.hpp"
#include "kernel.cuh"
#include <thrust/fill.h>
#include "cu_exception.cuh"
#include "local_config.hpp"
#include "device_function.hpp"
#include <thrust/device_ptr.h>
#include "blocking_alogrithm.hpp"
#include "simulator_expt_platform.hpp"

namespace gf::simulator::single_dev_expt
{
    class Simulator::Data
    {
        friend class Simulator;

        private:
            int _devIdx = 0;
            std::uint32_t _step = 0;
            std::uint32_t _dStep = 0;
            std::uint32_t _nStep = 0;
            real_t _invTau = 0;
            gf::basic::Vec3<std::uint32_t> _blockDim;
            gf::basic::Vec3<std::uint32_t> _gridDim;
            gf::basic::Vec3<std::uint32_t> _domDim;

            enum struct StreamPolicy
            {
                PULL_STREAM, 
                INPLACE_STREAM
            };

            enum struct OptPolicy
            {
                NONE,
                HALO_BLOCKING_STATIC_L2,
                HALO_BLOCKING_L1L2,
                HALO_BLOCKING_DYNAMIC_L2
            };

            enum struct VelSet
            {
                D3Q27
            };

            StreamPolicy    _streamPolicy = StreamPolicy::PULL_STREAM;
            OptPolicy       _optPolicy    = OptPolicy::NONE;
            VelSet          _velSet       = VelSet::D3Q27;
            std::string     _initStateFolder;
            std::string     _dumpFolder;
            bool _dumpRho = false, _dumpVx = false, _dumpVy = false, _dumpVz = false;

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

            //If we use inplace-stream, 
            //the following parameters are valid.
            ddf_t* DDFBuf = nullptr;

            //If we use halo-blocking method, 
            //the following parameters are valid.
            std::int32_t _innerLoop = 0;

            //If we use l2 cache based halo-blocking method, 
            //the following parameters are valid.
            real_t* _l2DDFBuf0 = nullptr;
            real_t* _l2DDFBuf1 = nullptr;

            //If we use l1-l2 mix cache based halo-blocking method,
            //the following parameters are valid.
            real_t* _swapDDFBuf = nullptr;

            void initFlag()
            {
                CU_CHECK(cudaMalloc(&_flagBuf, sizeof(flag_t)*getDomainSize()));
                if(std::ifstream f (_initStateFolder+"/flag.dat", std::ios::binary) ; f)
                {
                    std::vector<flag_t> flag (getDomainSize(), 0);
                    f.read((char*)flag.data(), flag.size()*sizeof(flag_t));
                    CU_CHECK(cudaMemcpy((void*)_flagBuf, (const void*)flag.data(), flag.size()*sizeof(flag_t), cudaMemcpyHostToDevice));
                    std::cout << std::format("Init flag successfully, from file: {}/flag.dat.", _initStateFolder) << std::endl;
                }
                else
                {
                    throw std::runtime_error(
                        std::format("File: {}/flag.dat doesn't exist!", _initStateFolder)
                    );
                }
            }

            void initBlockingFlag()
            {
                const auto blockingDim = getBlockingDim();
                const auto blockingNumDim = getBlockingNumDim();
                const std::uint32_t blockingSize = blockingDim.x * blockingDim.y * blockingDim.z;
                const std::uint32_t blockingNum = blockingNumDim.x * blockingNumDim.y * blockingNumDim.z;
                CU_CHECK(cudaMalloc(&_flagBuf, sizeof(flag_t)*blockingNum * blockingSize));

                if(std::ifstream f (_initStateFolder+"/flag.dat", std::ios::binary) ; f)
                {
                    std::vector<flag_t> orgFlag (getDomainSize(), 0);
                    f.read((char*)orgFlag.data(), orgFlag.size()*sizeof(flag_t));
                    std::vector<flag_t> blockingFlag (blockingNum * blockingSize, 0);
                    std::int32_t istIdx = 0;
                    for(std::int32_t blkIdxZ=0 ; blkIdxZ<blockingNumDim.z ; ++blkIdxZ)
                    {
                        const std::int32_t blkZStBegin = gf::blocking_core::calcValidPrev<std::int32_t>(blkIdxZ  , blockingDim.z, blockingNumDim.z, _innerLoop, _domDim.z);
                        const std::int32_t blkZStEnd   = gf::blocking_core::calcValidPrev<std::int32_t>(blkIdxZ+1, blockingDim.z, blockingNumDim.z, _innerLoop, _domDim.z);
                        const std::int32_t blkZLdBegin  = std::max<std::int32_t>(blkZStBegin-(_innerLoop-1), 0);
                        const std::int32_t blkZLdEnd    = std::min<std::int32_t>(blkZStEnd+(_innerLoop-1), _domDim.z);
                        for(std::int32_t blkIdxY=0 ; blkIdxY<blockingNumDim.y ; ++blkIdxY)
                        {
                            const std::int32_t blkYStBegin = gf::blocking_core::calcValidPrev<std::int32_t>(blkIdxY  , blockingDim.y, blockingNumDim.y, _innerLoop, _domDim.y);
                            const std::int32_t blkYStEnd   = gf::blocking_core::calcValidPrev<std::int32_t>(blkIdxY+1, blockingDim.y, blockingNumDim.y, _innerLoop, _domDim.y);
                            const std::int32_t blkYLdBegin  = std::max<std::int32_t>(blkYStBegin-(_innerLoop-1), 0);
                            const std::int32_t blkYLdEnd    = std::min<std::int32_t>(blkYStEnd+(_innerLoop-1), _domDim.y);
                            for(std::int32_t blkIdxX=0 ; blkIdxX<blockingNumDim.x ; ++blkIdxX)
                            {
                                const std::int32_t blkXStBegin = gf::blocking_core::calcValidPrev<std::int32_t>(blkIdxX  , blockingDim.x, blockingNumDim.x, _innerLoop, _domDim.x);
                                const std::int32_t blkXStEnd   = gf::blocking_core::calcValidPrev<std::int32_t>(blkIdxX+1, blockingDim.x, blockingNumDim.x, _innerLoop, _domDim.x);
                                const std::int32_t blkXLdBegin  = std::max<std::int32_t>(blkXStBegin-(_innerLoop-1), 0);
                                const std::int32_t blkXLdEnd    = std::min<std::int32_t>(blkXStEnd+(_innerLoop-1), _domDim.x);

                                for(std::int32_t blkOffZ=0, glbOffZ=blkZLdBegin ; glbOffZ<blkZLdEnd ; ++blkOffZ, ++glbOffZ)
                                {
                                    for(std::int32_t blkOffY=0, glbOffY=blkYLdBegin ; glbOffY<blkYLdEnd ; ++blkOffY, ++glbOffY)
                                    {
                                        for(std::int32_t blkOffX=0, glbOffX=blkXLdBegin ; glbOffX<blkXLdEnd ; ++blkOffX, ++glbOffX)
                                        {
                                            const std::int32_t glbOff = glbOffX + _domDim.x * (glbOffY + _domDim.y * glbOffZ);
                                            const std::int32_t blkOff = blkOffX + blockingDim.x * (blkOffY + blockingDim.y * blkOffZ);
                                            if(
                                                blkZStBegin<=glbOffZ and glbOffZ<blkZStEnd and
                                                blkYStBegin<=glbOffY and glbOffY<blkYStEnd and 
                                                blkXStBegin<=glbOffX and glbOffX<blkXStEnd
                                            )
                                            {
                                                blockingFlag[istIdx+blkOff] = orgFlag[glbOff] | CORRECT_BIT;
                                            }
                                            else
                                            {
                                                blockingFlag[istIdx+blkOff] = orgFlag[glbOff];
                                            }
                                        }
                                    }
                                }

                                istIdx += blockingSize;
                            }
                        }
                    }

                    CU_CHECK(cudaMemcpy(_flagBuf, blockingFlag.data(), sizeof(flag_t)*blockingNum*blockingSize, cudaMemcpyHostToDevice));
                    std::cout << std::format("Init flag successfully, from file: {}/flag.dat.", _initStateFolder) << std::endl;
                }
                else
                {
                    throw std::runtime_error(
                        std::format("File: {}/flag.dat doesn't exist!", _initStateFolder)
                    );
                }
            }

            void initRhoU()
            {
                const std::int32_t domSize = getDomainSize();
                CU_CHECK(cudaMalloc(&_rhoBuf, domSize*sizeof(real_t)));
                CU_CHECK(cudaMalloc(&_vxBuf , domSize*sizeof(real_t)));
                CU_CHECK(cudaMalloc(&_vyBuf , domSize*sizeof(real_t)));
                CU_CHECK(cudaMalloc(&_vzBuf , domSize*sizeof(real_t)));

                auto initHelper = [=,this](std::string_view desc, real_t dftv, real_t* devBuf) -> void
                {
                    if(std::ifstream f (_initStateFolder+std::format("/{}.dat", desc), std::ios::binary) ; f)
                    {
                        std::vector<real_t> hostBuf (domSize, 0);
                        f.read((char*)hostBuf.data(), domSize*sizeof(real_t));
                        CU_CHECK(cudaMemcpy(devBuf, hostBuf.data(), domSize*sizeof(real_t), cudaMemcpyHostToDevice));
                        std::cout << std::format("Init {} successfully, from file: {}/{}.dat.", desc, _initStateFolder, desc) << std::endl;
                    }
                    else
                    {
                        thrust::fill_n(thrust::device_pointer_cast(devBuf), domSize, dftv);
                        std::cout << std::format("Init {} successfully, default init with value {}.", desc, dftv) << std::endl;
                    }
                };

                initHelper("rho", 1, _rhoBuf);
                initHelper("vx" , 0, _vxBuf);
                initHelper("vy" , 0, _vyBuf);
                initHelper("vz" , 0, _vzBuf);
            }

            void initDoubleDDFBuf()
            {
                const std::int32_t domSize = getDomainSize();
                switch(_velSet)
                {
                    case VelSet::D3Q27:
                    {
                        CU_CHECK(cudaMalloc(&_srcDDFBuf, 27*sizeof(ddf_t)*domSize));
                        CU_CHECK(cudaMalloc(&_dstDDFBuf, 27*sizeof(ddf_t)*domSize));
                        ddf_t feq[27];
                        gf::lbm_core::bgk::calcEqu<27>(1,0,0,0, std::begin(feq));
                        for(std::int32_t dir=0 ; dir<27 ; ++dir)
                        {
                            thrust::fill_n(thrust::device_pointer_cast(_srcDDFBuf+dir*domSize), domSize, feq[dir]);
                            thrust::fill_n(thrust::device_pointer_cast(_dstDDFBuf+dir*domSize), domSize, feq[dir]);
                        }
                        break;
                    }
                }
                std::cout << std::format("Init double ddf buffer successfully.") << std::endl;
            }

        public:
            Data(int argc, char** argv)
            {
                CLI::App app {"Single Device Expt Fluid Solver"};

                app.add_option("--devId", _devIdx, "Device index of the GPU")
                    ->default_val(0);

                app.add_option("--dstep", _dStep, "Delta time steps of the simulation.")
                    ->default_val(100);
                
                app.add_option("--nstep", _nStep, "Total time steps of the simulation.")
                    ->default_val(1000);
                
                app.add_option("--invTau", _invTau, "Reciprocal of Tau.")
                    ->default_val(0.5);

                app.add_option("--blockDim", _blockDim.data, "Dimension of blocks per grid.")
                    ->default_val(std::array<std::uint32_t,3>{32,16,2});

                app.add_option("--gridDim", _gridDim.data, "Dimension of grids per kernel.")
                    ->default_val(std::array<std::uint32_t,3>{2,3,19});

                app.add_option("--domainDim", _domDim.data, "Dimension of the domain.")
                    ->default_val(std::array<std::uint32_t,3>{256,256,256});

                app.add_option("--streamPolicy", _streamPolicy, "The stream policy of the solver. {[0]:PULL_STREAM, [1]:INPLACE_STREAM}.")
                    ->default_val(StreamPolicy::PULL_STREAM);

                app.add_option("--optPolicy", _optPolicy, "The optimization policy of the solver. {[0]:NONE, [1]:HALO_BLOCKING_STATIC_L2, [2]:HALO_BLOCKING_L1L2, [3]:HALO_BLOCKING_DYNAMIC_L2}.")
                    ->default_val(OptPolicy::NONE);

                app.add_option("--velSet", _velSet, "The velocity set of the solver.")
                    ->default_val(VelSet::D3Q27);
                
                app.add_option("--initStateFolder", _initStateFolder, "The folder contains initial state of the simulation.")
                    ->default_val("data/left_inlet_right_outlet_256_256_256_init_state");

                app.add_option("--dumpFolder", _dumpFolder, "The folder contains final output.")
                    ->default_val("data/left_inlet_right_outlet_256_256_256_output");

                app.add_flag("--dumpRho", _dumpRho, "Enable dump the density every `dstep` time steps in the simulation.");
                app.add_flag("--dumpVx", _dumpVx, "Enable dumping the x-axis velocity every `dstep` time steps in the simulation.");
                app.add_flag("--dumpVy", _dumpVy, "Enable dumping the y-axis velocity every `dstep` time steps in the simulation.");
                app.add_flag("--dumpVz", _dumpVz, "Enable dumping the z-axis velocity every `dstep` time steps in the simulation.");

                app.add_option("--innerLoop", _innerLoop, "# of iterations in halo-blocking.")
                    ->default_val(5);

                try
                {
                    app.parse(argc, argv);
                }
                catch(const CLI::ParseError& e)
                {
                    std::exit(app.exit(e));
                }

                CU_CHECK(cudaSetDevice(_devIdx));
                CU_CHECK(cudaStreamCreate(&_stream));
                CU_CHECK(cudaEventCreate(&_start));
                CU_CHECK(cudaEventCreate(&_end));

                const std::uint32_t domSize = getDomainSize();
                const std::uint32_t q = getQ();

                if(
                    _optPolicy==OptPolicy::HALO_BLOCKING_STATIC_L2 or 
                    _optPolicy==OptPolicy::HALO_BLOCKING_L1L2 or
                    _optPolicy==OptPolicy::HALO_BLOCKING_DYNAMIC_L2
                )
                {
                    initBlockingFlag();
                }

                if(_optPolicy==OptPolicy::NONE)
                {
                    initFlag();
                }

                initRhoU();

                if(
                    _streamPolicy==StreamPolicy::PULL_STREAM or
                    (_streamPolicy==StreamPolicy::INPLACE_STREAM and _optPolicy==OptPolicy::HALO_BLOCKING_DYNAMIC_L2) or
                    (_streamPolicy==StreamPolicy::INPLACE_STREAM and _optPolicy==OptPolicy::HALO_BLOCKING_STATIC_L2)
                )
                {
                    initDoubleDDFBuf();
                }

                if(_optPolicy==OptPolicy::HALO_BLOCKING_STATIC_L2 or _optPolicy==OptPolicy::HALO_BLOCKING_DYNAMIC_L2)
                {
                    const std::uint32_t blockingSize = getBlockingSize();
                    if(_streamPolicy==StreamPolicy::PULL_STREAM)
                    {
                        CU_CHECK(cudaMalloc(&_l2DDFBuf0, sizeof(real_t)*q*blockingSize));
                        CU_CHECK(cudaMalloc(&_l2DDFBuf1, sizeof(real_t)*q*blockingSize));
                    }
                    else if (_streamPolicy==StreamPolicy::INPLACE_STREAM)
                    {
                        CU_CHECK(cudaMalloc(&_l2DDFBuf0, sizeof(real_t)*q*blockingSize));
                    }
                }

                if(_optPolicy==OptPolicy::HALO_BLOCKING_L1L2)
                {
                    const std::uint32_t xBufSize = (_gridDim.x+1)*(_gridDim.z*_blockDim.z)*(_gridDim.y*_blockDim.y);
                    const std::uint32_t yBufSize = (_gridDim.y+1)*(_gridDim.z*_blockDim.z)*(_gridDim.x*_blockDim.x);
                    const std::uint32_t zBufSize = (_gridDim.z+1)*(_gridDim.y*_blockDim.y)*(_gridDim.x*_blockDim.x);
                    std::uint32_t swapDDFBufSize = xBufSize+yBufSize+zBufSize;
                    switch(_velSet)
                    {
                        case VelSet::D3Q27:
                            swapDDFBufSize *= 18;
                            break;
                        default:
                            throw std::invalid_argument(
                                std::format("Unsupport Velocity Set: {}", static_cast<std::uint32_t>(_velSet))
                            );
                    }
                    CU_CHECK(cudaMalloc(&_swapDDFBuf, swapDDFBufSize*sizeof(real_t)));
                }
            }

            ~Data()
            {
                if(_optPolicy==OptPolicy::HALO_BLOCKING_L1L2)
                {
                    CU_CHECK(cudaFree(_swapDDFBuf));
                }

                if(_optPolicy==OptPolicy::HALO_BLOCKING_STATIC_L2 or _optPolicy==OptPolicy::HALO_BLOCKING_DYNAMIC_L2)
                {
                    if(_streamPolicy==StreamPolicy::PULL_STREAM)
                    {
                        CU_CHECK(cudaFree(_l2DDFBuf0));
                        CU_CHECK(cudaFree(_l2DDFBuf1));
                    }
                    else if(_streamPolicy==StreamPolicy::INPLACE_STREAM)
                    {
                        CU_CHECK(cudaFree(_l2DDFBuf0));
                    }
                }

                if(
                    _streamPolicy==StreamPolicy::PULL_STREAM or
                    (_streamPolicy==StreamPolicy::INPLACE_STREAM and _optPolicy==OptPolicy::HALO_BLOCKING_DYNAMIC_L2) or
                    (_streamPolicy==StreamPolicy::INPLACE_STREAM and _optPolicy==OptPolicy::HALO_BLOCKING_STATIC_L2)
                )
                {
                    CU_CHECK(cudaFree(_srcDDFBuf));
                    CU_CHECK(cudaFree(_dstDDFBuf));
                }

                if(
                    _optPolicy==OptPolicy::HALO_BLOCKING_STATIC_L2 or
                    _optPolicy==OptPolicy::HALO_BLOCKING_L1L2 or
                    _optPolicy==OptPolicy::HALO_BLOCKING_DYNAMIC_L2
                )
                {
                    CU_CHECK(cudaFree(_flagBuf));
                }

                if(_optPolicy==OptPolicy::NONE)
                {
                    CU_CHECK(cudaFree(_flagBuf));
                }

                CU_CHECK(cudaFree(_rhoBuf));
                CU_CHECK(cudaFree(_vxBuf));
                CU_CHECK(cudaFree(_vyBuf));
                CU_CHECK(cudaFree(_vzBuf));

                CU_CHECK(cudaEventDestroy(_end));
                CU_CHECK(cudaEventDestroy(_start));
                CU_CHECK(cudaStreamDestroy(_stream));
            }

            std::uint32_t getDomainSize() const noexcept
            {
                return _domDim.x * _domDim.y * _domDim.z;
            }

            gf::basic::Vec3<std::uint32_t> getBlockingDim() const noexcept
            {
                return {_gridDim.x*_blockDim.x, _gridDim.y*_blockDim.y, _gridDim.z*_blockDim.z};
            }

            std::uint32_t getBlockingSize() const noexcept
            {
                const auto blockingDim = getBlockingDim();
                return blockingDim.x * blockingDim.y * blockingDim.z;
            }

            /**
             * @brief This function is valid only if _optPolicy is `OptPolicy::HALO_BLOCKING_L2` or `OptPolicy::HALO_BLOCKING_L1L2`
             */
            gf::basic::Vec3<std::uint32_t> getBlockingNumDim() const
            {
                using namespace gf::blocking_core;                const auto blockingDim = getBlockingDim();
                const bool valid = 
                    validBlkAxisConfig<std::uint32_t>(_domDim.x, blockingDim.x, _innerLoop) and
                    validBlkAxisConfig<std::uint32_t>(_domDim.y, blockingDim.y, _innerLoop) and
                    validBlkAxisConfig<std::uint32_t>(_domDim.z, blockingDim.z, _innerLoop);
                if(not valid)
                {
                    throw std::invalid_argument("<Domain Dimension, Blocking Dimension and InnerLoop> don't match!");
                }
                return {
                    calcBlkNum<std::uint32_t>(_domDim.x, blockingDim.x, _innerLoop), 
                    calcBlkNum<std::uint32_t>(_domDim.y, blockingDim.y, _innerLoop), 
                    calcBlkNum<std::uint32_t>(_domDim.z, blockingDim.z, _innerLoop)
                };
            }

            std::uint32_t getQ() const noexcept
            {
                switch(_velSet)
                {
                    case VelSet::D3Q27:
                        return 27;
                    default:
                        throw std::invalid_argument(
                            std::format("Unsupport Velocity Set: {}", static_cast<std::uint32_t>(_velSet))
                        );
                }
            }
    };

    Simulator::Simulator(int argc, char** argv) :
        _data(std::make_unique<Data>(argc, argv))
    {
        if(_data->_streamPolicy==Data::StreamPolicy::PULL_STREAM and _data->_optPolicy==Data::OptPolicy::HALO_BLOCKING_L1L2)
        {
            CU_CHECK(cudaFuncSetAttribute((const void*)&HaloBlockingL1L2D3Q27PullKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(real_t)*_data->_blockDim.x*_data->_blockDim.y*_data->_blockDim.z*24));
        }

        if(_data->_streamPolicy==Data::StreamPolicy::INPLACE_STREAM and _data->_optPolicy==Data::OptPolicy::HALO_BLOCKING_STATIC_L2)
        {
            CU_CHECK(cudaDeviceSetCacheConfig(cudaFuncCache::cudaFuncCachePreferL1));
        }
    }

    void Simulator::run()
    {
        auto dumpRes = [this]()->void
        {
            std::vector<real_t> buf;

            if(_data->_dumpRho or _data->_dumpVx or _data->_dumpVy or _data->_dumpVz)
            {
                buf.resize(_data->getDomainSize(), 0);
            }

            if(_data->_dumpRho)
            {
                if(std::ofstream f (_data->_dumpFolder+std::format("/rho_{}.dat", _data->_step), std::ios::binary) ; f)
                {
                    CU_CHECK(cudaMemcpy(buf.data(), _data->_rhoBuf, sizeof(real_t)*buf.size(), cudaMemcpyDeviceToHost));
                    f.write((const char*)buf.data(), sizeof(real_t)*buf.size());
                }
                else
                {
                    throw std::runtime_error(std::format("Could not open the file: {}/rho_{}.dat", _data->_dumpFolder, _data->_step));
                }
            }

            if(_data->_dumpVx)
            {
                if(std::ofstream f (_data->_dumpFolder+std::format("/vx_{}.dat", _data->_step), std::ios::binary) ; f)
                {
                    CU_CHECK(cudaMemcpy(buf.data(), _data->_vxBuf, sizeof(real_t)*buf.size(), cudaMemcpyDeviceToHost));
                    f.write((const char*)buf.data(), sizeof(real_t)*buf.size());
                }
                else
                {
                    throw std::runtime_error(std::format("Could not open the file: {}/vx_{}.dat", _data->_dumpFolder, _data->_step));
                }
            }

            if(_data->_dumpVy)
            {
                if(std::ofstream f (_data->_dumpFolder+std::format("/vy_{}.dat", _data->_step), std::ios::binary) ; f)
                {
                    CU_CHECK(cudaMemcpy(buf.data(), _data->_vyBuf, sizeof(real_t)*buf.size(), cudaMemcpyDeviceToHost));
                    f.write((const char*)buf.data(), sizeof(real_t)*buf.size());
                }
                else
                {
                    throw std::runtime_error(std::format("Could not open the file: {}/vy_{}.dat", _data->_dumpFolder, _data->_step));
                }
            }

            if(_data->_dumpVz)
            {
                if(std::ofstream f (_data->_dumpFolder+std::format("/vz_{}.dat", _data->_step), std::ios::binary) ; f)
                {
                    CU_CHECK(cudaMemcpy(buf.data(), _data->_vzBuf, sizeof(real_t)*buf.size(), cudaMemcpyDeviceToHost));
                    f.write((const char*)buf.data(), sizeof(real_t)*buf.size());
                }
                else
                {
                    throw std::runtime_error(std::format("Could not open the file: {}/vz_{}.dat", _data->_dumpFolder, _data->_step));
                }
            }
        };

        auto haloBlockingL1L2PullRun = [dumpRes, this]()->void
        {
            HaloBlockingL1L2Param param
            {
                .invTau = _data->_invTau, 
                .nloop = _data->_innerLoop, 
                .offx = 0, .offy = 0, .offz = 0, 
                .glbnx = static_cast<idx_t>(_data->_domDim.x), .glbny = static_cast<idx_t>(_data->_domDim.y), .glbnz = static_cast<idx_t>(_data->_domDim.z), 
                .blkFlagBuf = nullptr, 
                .glbRhoBuf = _data->_rhoBuf, 
                .glbVxBuf = _data->_vxBuf, 
                .glbVyBuf = _data->_vyBuf, 
                .glbVzBuf = _data->_vzBuf, 
                .glbSrcDDFBuf = _data->_srcDDFBuf, 
                .glbDstDDFBuf = _data->_dstDDFBuf, 
                .glbSwapDDFBuf = _data->_swapDDFBuf
            };

            const dim3 gridDim {_data->_gridDim.x, _data->_gridDim.y, _data->_gridDim.z};
            const dim3 blockDim {_data->_blockDim.x, _data->_blockDim.y, _data->_blockDim.z};
            void* kernelArgs[1] = { (void*)&param };

            const idx_t blkDDFBufSize = sizeof(real_t)*blockDim.x*blockDim.y*blockDim.z*24;
            const auto blockingDim = _data->getBlockingDim();
            const auto blockingNumDim = _data->getBlockingNumDim();
            const std::uint32_t blockingNum = blockingNumDim.x * blockingNumDim.y * blockingNumDim.z;

            while(_data->_step < _data->_nStep)
            {
                std::uint32_t locStep = 0;
                CU_CHECK(cudaEventRecord(_data->_start, _data->_stream));
                for( ; locStep<_data->_dStep ;locStep+=_data->_innerLoop)
                {
                    for(std::uint32_t blkIdx=0 ; blkIdx<blockingNum ; ++blkIdx)
                    {
                        const idx_t blkIdxX = blkIdx % blockingNumDim.x;
                        const idx_t blkIdxY = (blkIdx / blockingNumDim.x) % blockingNumDim.y;
                        const idx_t blkIdxZ = blkIdx / (blockingNumDim.x * blockingNumDim.y);
                        param.offx = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxX, blockingDim.x, blockingNumDim.x, _data->_innerLoop, _data->_domDim.x)-(_data->_innerLoop-1), 0);
                        param.offy = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxY, blockingDim.y, blockingNumDim.y, _data->_innerLoop, _data->_domDim.y)-(_data->_innerLoop-1), 0);
                        param.offz = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxZ, blockingDim.z, blockingNumDim.z, _data->_innerLoop, _data->_domDim.z)-(_data->_innerLoop-1), 0);
                        param.blkFlagBuf = _data->_flagBuf + blkIdx * blockingDim.x * blockingDim.y * blockingDim.z;
                        CU_CHECK(cudaLaunchCooperativeKernel((const void*)&HaloBlockingL1L2D3Q27PullKernel, gridDim, blockDim, std::begin(kernelArgs), blkDDFBufSize, _data->_stream));
                    }
                    std::swap(param.glbSrcDDFBuf, param.glbDstDDFBuf);
                }
                CU_CHECK(cudaEventRecord(_data->_end, _data->_stream));
                CU_CHECK(cudaEventSynchronize(_data->_end));
                float ms;
                CU_CHECK(cudaEventElapsedTime(&ms, _data->_start, _data->_end));
                const float mlups = static_cast<float>(_data->_domDim.x*_data->_domDim.y*_data->_domDim.z) / (1024*1024) / (ms / 1000) * locStep;
                std::cout << std::format("speed = {:.4f} (MLUPS)", mlups) << std::endl;
                _data->_step += locStep;

                for(std::uint32_t blkIdx=0 ; blkIdx<blockingNum ; ++blkIdx)
                {
                    const idx_t blkIdxX = blkIdx % blockingNumDim.x;
                    const idx_t blkIdxY = (blkIdx / blockingNumDim.x) % blockingNumDim.y;
                    const idx_t blkIdxZ = blkIdx / (blockingNumDim.x * blockingNumDim.y);
                    param.offx = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxX, blockingDim.x, blockingNumDim.x, _data->_innerLoop, _data->_domDim.x)-(_data->_innerLoop-1), 0);
                    param.offy = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxY, blockingDim.y, blockingNumDim.y, _data->_innerLoop, _data->_domDim.y)-(_data->_innerLoop-1), 0);
                    param.offz = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxZ, blockingDim.z, blockingNumDim.z, _data->_innerLoop, _data->_domDim.z)-(_data->_innerLoop-1), 0);
                    param.blkFlagBuf = _data->_flagBuf + blkIdx * blockingDim.x * blockingDim.y * blockingDim.z;
                    CU_CHECK(cudaLaunchKernel((const void*)&HaloBlockingL1L2D3Q27DumpKernel, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                }
                CU_CHECK(cudaStreamSynchronize(_data->_stream));
                dumpRes();
            }
        };

        auto haloBlockingStaticL2PullRun = [dumpRes, this]()->void
        {
            HaloBlockingStaticL2Param param
            {
                .invTau = _data->_invTau, 
                .nloop  = _data->_innerLoop, 
                .offx = 0, .offy = 0, .offz = 0, 
                .glbnx = static_cast<idx_t>(_data->_domDim.x), .glbny = static_cast<idx_t>(_data->_domDim.y), .glbnz = static_cast<idx_t>(_data->_domDim.z),
                .blkFlagBuf = nullptr, 
                .glbRhoBuf = _data->_rhoBuf, 
                .glbVxBuf = _data->_vxBuf, 
                .glbVyBuf = _data->_vyBuf, 
                .glbVzBuf = _data->_vzBuf, 
                .glbSrcDDFBuf = _data->_srcDDFBuf, 
                .glbDstDDFBuf = _data->_dstDDFBuf, 
                .blkDDFBuf0 = _data->_l2DDFBuf0, 
                .blkDDFBuf1 = _data->_l2DDFBuf1
            };

            const dim3 gridDim {_data->_gridDim.x, _data->_gridDim.y, _data->_gridDim.z};
            const dim3 blockDim {_data->_blockDim.x, _data->_blockDim.y, _data->_blockDim.z};
            void* kernelArgs[1] = { (void*)&param };

            const auto blockingDim = _data->getBlockingDim();
            const auto blockingNumDim = _data->getBlockingNumDim();
            const std::uint32_t blockingNum = blockingNumDim.x * blockingNumDim.y * blockingNumDim.z;

            while(_data->_step < _data->_nStep)
            {
                std::uint32_t locStep = 0;
                CU_CHECK(cudaEventRecord(_data->_start, _data->_stream));
                for( ; locStep<_data->_dStep ; locStep += _data->_innerLoop )
                {
                    for(std::uint32_t blkIdx=0 ; blkIdx<blockingNum ; ++blkIdx)
                    {
                        const idx_t blkIdxX = blkIdx % blockingNumDim.x;
                        const idx_t blkIdxY = (blkIdx / blockingNumDim.x) % blockingNumDim.y;
                        const idx_t blkIdxZ = blkIdx / (blockingNumDim.x * blockingNumDim.y);
                        param.offx = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxX, blockingDim.x, blockingNumDim.x, _data->_innerLoop, _data->_domDim.x)-(_data->_innerLoop-1), 0);
                        param.offy = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxY, blockingDim.y, blockingNumDim.y, _data->_innerLoop, _data->_domDim.y)-(_data->_innerLoop-1), 0);
                        param.offz = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxZ, blockingDim.z, blockingNumDim.z, _data->_innerLoop, _data->_domDim.z)-(_data->_innerLoop-1), 0);
                        param.blkFlagBuf = _data->_flagBuf + blkIdx * blockingDim.x * blockingDim.y * blockingDim.z;
                        CU_CHECK(cudaLaunchCooperativeKernel((const void*)&HaloBlockingStaticL2D3Q27PullKernel, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                    }
                    std::swap(param.glbSrcDDFBuf, param.glbDstDDFBuf);
                }
                CU_CHECK(cudaEventRecord(_data->_end, _data->_stream));
                CU_CHECK(cudaEventSynchronize(_data->_end));
                float ms;
                CU_CHECK(cudaEventElapsedTime(&ms, _data->_start, _data->_end));
                const float mlups = static_cast<float>(_data->_domDim.x * _data->_domDim.y * _data->_domDim.z) / (1024*1024) / (ms / 1000) * locStep;
                std::cout << std::format("speed = {:.4f} (MLUPS)", mlups) << std::endl;
                _data->_step += locStep;

                for(std::uint32_t blkIdx=0 ; blkIdx<blockingNum ; ++blkIdx)
                {
                    const idx_t blkIdxX = blkIdx % blockingNumDim.x;
                    const idx_t blkIdxY = (blkIdx / blockingNumDim.x) % blockingNumDim.y;
                    const idx_t blkIdxZ = blkIdx / (blockingNumDim.x * blockingNumDim.y);
                    param.offx = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxX, blockingDim.x, blockingNumDim.x, _data->_innerLoop, _data->_domDim.x)-(_data->_innerLoop-1), 0);
                    param.offy = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxY, blockingDim.y, blockingNumDim.y, _data->_innerLoop, _data->_domDim.y)-(_data->_innerLoop-1), 0);
                    param.offz = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxZ, blockingDim.z, blockingNumDim.z, _data->_innerLoop, _data->_domDim.z)-(_data->_innerLoop-1), 0);
                    param.blkFlagBuf = _data->_flagBuf + blkIdx * blockingDim.x * blockingDim.y * blockingDim.z;
                    CU_CHECK(cudaLaunchKernel((const void*)&HaloBlockingStaticL2D3Q27DumpKernel, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                }

                CU_CHECK(cudaStreamSynchronize(_data->_stream));
                dumpRes();
            }
        };

        auto haloBlockingDynamicL2InplaceRun = [dumpRes, this]()->void
        {
            HaloBlockingDynamicL2InplaceParam param
            {
                .invTau = _data->_invTau, 
                .offx = 0, .offy = 0, .offz = 0, 
                .glbnx = static_cast<idx_t>(_data->_domDim.x), .glbny = static_cast<idx_t>(_data->_domDim.y), .glbnz = static_cast<idx_t>(_data->_domDim.z), 
                .blkFlagBuf = nullptr, 
                .glbRhoBuf = _data->_rhoBuf, 
                .glbVxBuf = _data->_vxBuf, 
                .glbVyBuf = _data->_vyBuf, 
                .glbVzBuf = _data->_vzBuf, 
                .glbSrcDDFBuf = _data->_srcDDFBuf, 
                .glbDstDDFBuf = _data->_dstDDFBuf, 
                .blkDDFBuf = _data->_l2DDFBuf0
            };

            const dim3 gridDim {_data->_gridDim.x, _data->_gridDim.y, _data->_gridDim.z};
            const dim3 blockDim {_data->_blockDim.x, _data->_blockDim.y, _data->_blockDim.z};
            void* kernelArgs[1] = { (void*)&param };

            const auto blockingDim = _data->getBlockingDim();
            const auto blockingNumDim = _data->getBlockingNumDim();
            const std::uint32_t blockingNum = blockingNumDim.x * blockingNumDim.y * blockingNumDim.z;
            std::vector<real_t> hostBlkDDFBuf (27*blockingDim.x*blockingDim.y*blockingDim.z, 0);


            while(_data->_step < _data->_nStep)
            {
                std::uint32_t locStep = 0;
                CU_CHECK(cudaEventRecord(_data->_start, _data->_stream));
                for( ; locStep < _data->_dStep ; locStep += _data->_innerLoop)
                {
                    for(std::uint32_t blkIdx=0 ; blkIdx<blockingNum ; ++blkIdx)
                    {
                        const idx_t blkIdxX = blkIdx % blockingNumDim.x;
                        const idx_t blkIdxY = (blkIdx / blockingNumDim.x) % blockingNumDim.y;
                        const idx_t blkIdxZ = blkIdx / (blockingNumDim.x * blockingNumDim.y);
                        param.offx = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxX, blockingDim.x, blockingNumDim.x, _data->_innerLoop, _data->_domDim.x)-(_data->_innerLoop-1), 0);
                        param.offy = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxY, blockingDim.y, blockingNumDim.y, _data->_innerLoop, _data->_domDim.y)-(_data->_innerLoop-1), 0);
                        param.offz = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxZ, blockingDim.z, blockingNumDim.z, _data->_innerLoop, _data->_domDim.z)-(_data->_innerLoop-1), 0);
                        param.blkFlagBuf = _data->_flagBuf + blkIdx * blockingDim.x * blockingDim.y * blockingDim.z;
                        
                        for(std::int32_t loop=0 ; loop < _data->_innerLoop ; ++loop)
                        {
                            if(loop==0)
                            {
                                if(_data->_innerLoop==1)
                                {
                                    CU_CHECK(cudaLaunchKernel((const void*)&HaloBlockingDynamicL2D3Q27InplaceFirstKernel<true>, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                                }
                                else
                                {
                                    CU_CHECK(cudaLaunchKernel((const void*)&HaloBlockingDynamicL2D3Q27InplaceFirstKernel<false>, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                                }
                            }
                            else if(loop<(_data->_innerLoop-1))
                            {
                                if((loop%2)==0)
                                {
                                    CU_CHECK(cudaLaunchKernel((const void*)&HaloBlockingDynamicL2D3Q27InplaceMiddleKernel<true>, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                                }
                                else
                                {
                                    CU_CHECK(cudaLaunchKernel((const void*)&HaloBlockingDynamicL2D3Q27InplaceMiddleKernel<false>, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                                }
                            }
                            else
                            {
                                if((loop%2)==0)
                                {
                                    CU_CHECK(cudaLaunchKernel((const void*)&HaloBlockingDynamicL2D3Q27InplaceLastKernel<true>, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                                }
                                else
                                {
                                    CU_CHECK(cudaLaunchKernel((const void*)&HaloBlockingDynamicL2D3Q27InplaceLastKernel<false>, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                                }
                            }
                        }
                    }

                    std::swap(param.glbSrcDDFBuf, param.glbDstDDFBuf);
                }
                CU_CHECK(cudaEventRecord(_data->_end, _data->_stream));
                CU_CHECK(cudaEventSynchronize(_data->_end));
                float ms;
                CU_CHECK(cudaEventElapsedTime(&ms, _data->_start, _data->_end));
                const float mlups = static_cast<float>(_data->_domDim.x * _data->_domDim.y * _data->_domDim.z) / (1024*1024) / (ms / 1000) * locStep;
                std::cout << std::format("speed = {:.4f} (MLUPS)", mlups) << std::endl;
                _data->_step += locStep;

                for(std::uint32_t blkIdx=0 ; blkIdx<blockingNum ; ++blkIdx)
                {
                    const idx_t blkIdxX = blkIdx % blockingNumDim.x;
                    const idx_t blkIdxY = (blkIdx / blockingNumDim.x) % blockingNumDim.y;
                    const idx_t blkIdxZ = blkIdx / (blockingNumDim.x * blockingNumDim.y);
                    param.offx = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxX, blockingDim.x, blockingNumDim.x, _data->_innerLoop, _data->_domDim.x)-(_data->_innerLoop-1), 0);
                    param.offy = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxY, blockingDim.y, blockingNumDim.y, _data->_innerLoop, _data->_domDim.y)-(_data->_innerLoop-1), 0);
                    param.offz = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxZ, blockingDim.z, blockingNumDim.z, _data->_innerLoop, _data->_domDim.z)-(_data->_innerLoop-1), 0);
                    param.blkFlagBuf = _data->_flagBuf + blkIdx * blockingDim.x * blockingDim.y * blockingDim.z;
                    CU_CHECK(cudaLaunchKernel((const void*)&HaloBlockingDynamicL2D3Q27InplaceDumpKernel, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                }

                CU_CHECK(cudaStreamSynchronize(_data->_stream));
                dumpRes();
            }
        };

        auto haloBlockingStaticL2InplaceRun = [dumpRes, this]()->void
        {
            HaloBlockingStaticL2InplaceParam param
            {
                .invTau = _data->_invTau, 
                .nloop = _data->_innerLoop, 
                .offx = 0, .offy = 0, .offz = 0, 
                .glbnx = static_cast<idx_t>(_data->_domDim.x), .glbny = static_cast<idx_t>(_data->_domDim.y), .glbnz = static_cast<idx_t>(_data->_domDim.z), 
                .blkFlagBuf = nullptr, 
                .glbRhoBuf = _data->_rhoBuf, 
                .glbVxBuf = _data->_vxBuf, 
                .glbVyBuf = _data->_vyBuf, 
                .glbVzBuf = _data->_vzBuf, 
                .glbSrcDDFBuf = _data->_srcDDFBuf, 
                .glbDstDDFBuf = _data->_dstDDFBuf, 
                .blkDDFBuf = _data->_l2DDFBuf0
            };

            const dim3 gridDim {_data->_gridDim.x, _data->_gridDim.y, _data->_gridDim.z};
            const dim3 blockDim {_data->_blockDim.x, _data->_blockDim.y, _data->_blockDim.z};
            void* kernelArgs[1] = { (void*)&param };

            const auto blockingDim = _data->getBlockingDim();
            const auto blockingNumDim = _data->getBlockingNumDim();
            const std::uint32_t blockingNum = blockingNumDim.x * blockingNumDim.y * blockingNumDim.z;

            while(_data->_step < _data->_nStep)
            {
                std::uint32_t locStep = 0;
                CU_CHECK(cudaEventRecord(_data->_start, _data->_stream));
                for( ; locStep<_data->_dStep ; locStep += _data->_innerLoop )
                {
                    for(std::uint32_t blkIdx=0 ; blkIdx<blockingNum ; ++blkIdx)
                    {
                        const idx_t blkIdxX = blkIdx % blockingNumDim.x;
                        const idx_t blkIdxY = (blkIdx / blockingNumDim.x) % blockingNumDim.y;
                        const idx_t blkIdxZ = blkIdx / (blockingNumDim.x * blockingNumDim.y);
                        param.offx = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxX, blockingDim.x, blockingNumDim.x, _data->_innerLoop, _data->_domDim.x)-(_data->_innerLoop-1), 0);
                        param.offy = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxY, blockingDim.y, blockingNumDim.y, _data->_innerLoop, _data->_domDim.y)-(_data->_innerLoop-1), 0);
                        param.offz = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxZ, blockingDim.z, blockingNumDim.z, _data->_innerLoop, _data->_domDim.z)-(_data->_innerLoop-1), 0);
                        param.blkFlagBuf = _data->_flagBuf + blkIdx * blockingDim.x * blockingDim.y * blockingDim.z;
                        CU_CHECK(cudaLaunchCooperativeKernel((const void*)&HaloBlockingStaticL2D3Q27InplaceKernel, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                    }
                    std::swap(param.glbSrcDDFBuf, param.glbDstDDFBuf);
                }
                CU_CHECK(cudaEventRecord(_data->_end, _data->_stream));
                CU_CHECK(cudaEventSynchronize(_data->_end));
                float ms;
                CU_CHECK(cudaEventElapsedTime(&ms, _data->_start, _data->_end));
                const float mlups = static_cast<float>(_data->_domDim.x * _data->_domDim.y * _data->_domDim.z) / (1024*1024) / (ms / 1000) * locStep;
                std::cout << std::format("speed = {:.4f} (MLUPS)", mlups) << std::endl;
                _data->_step += locStep;

                for(std::uint32_t blkIdx=0 ; blkIdx<blockingNum ; ++blkIdx)
                {
                    const idx_t blkIdxX = blkIdx % blockingNumDim.x;
                    const idx_t blkIdxY = (blkIdx / blockingNumDim.x) % blockingNumDim.y;
                    const idx_t blkIdxZ = blkIdx / (blockingNumDim.x * blockingNumDim.y);
                    param.offx = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxX, blockingDim.x, blockingNumDim.x, _data->_innerLoop, _data->_domDim.x)-(_data->_innerLoop-1), 0);
                    param.offy = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxY, blockingDim.y, blockingNumDim.y, _data->_innerLoop, _data->_domDim.y)-(_data->_innerLoop-1), 0);
                    param.offz = std::max<idx_t>(gf::blocking_core::calcValidPrev<idx_t>(blkIdxZ, blockingDim.z, blockingNumDim.z, _data->_innerLoop, _data->_domDim.z)-(_data->_innerLoop-1), 0);
                    param.blkFlagBuf = _data->_flagBuf + blkIdx * blockingDim.x * blockingDim.y * blockingDim.z;
                    CU_CHECK(cudaLaunchKernel((const void*)&HaloBlockingStaticL2D3Q27InplaceDumpKernel, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                }

                CU_CHECK(cudaStreamSynchronize(_data->_stream));
                dumpRes();
            }
        };

        auto staticPullRun = [dumpRes, this]()->void
        {
            StaticPullParam param
            {
                .invTau = _data->_invTau, 
                .glbFlagBuf = _data->_flagBuf, 
                .glbRhoBuf  = _data->_rhoBuf, 
                .glbVxBuf   = _data->_vxBuf, 
                .glbVyBuf   = _data->_vyBuf, 
                .glbVzBuf   = _data->_vzBuf, 
                .glbSrcDDFBuf = _data->_srcDDFBuf, 
                .glbDstDDFBuf = _data->_dstDDFBuf
            };

            const dim3 gridDim = {_data->_gridDim.x, _data->_gridDim.y, _data->_gridDim.z};
            const dim3 blockDim = {_data->_blockDim.x, _data->_blockDim.y, _data->_blockDim.z};
            const auto blockingDim = _data->getBlockingDim();

            if(blockingDim.x!=_data->_domDim.x or blockingDim.y!=_data->_domDim.y or blockingDim.z!=_data->_domDim.z)
            {
                throw std::runtime_error(
                    std::format("Dimension of domain doesn't match computation grid!")
                );
            }

            void *kernelArgs[1] = { (void*)&param };

            while(_data->_step < _data->_nStep)
            {
                CU_CHECK(cudaEventRecord(_data->_start, _data->_stream));
                for(std::uint32_t locStep=0 ; locStep<_data->_dStep ; ++locStep)
                {
                    CU_CHECK(cudaLaunchKernel((const void*)&StaticD3Q27PullKernel, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                    std::swap(param.glbSrcDDFBuf, param.glbDstDDFBuf);
                }
                CU_CHECK(cudaEventRecord(_data->_end, _data->_stream));
                CU_CHECK(cudaEventSynchronize(_data->_end));
                float ms;
                CU_CHECK(cudaEventElapsedTime(&ms, _data->_start, _data->_end));
                const float mlups = static_cast<float>(_data->_domDim.x * _data->_domDim.y * _data->_domDim.z) / (1024*1024) / (ms / 1000) * _data->_dStep;
                std::cout << std::format("speed = {:.4f} (MLUPS)", mlups) << std::endl;;
                _data->_step += _data->_dStep;

                CU_CHECK(cudaLaunchKernel((const void*)&StaticD3Q27PullDumpKernel, gridDim, blockDim, std::begin(kernelArgs), 0, _data->_stream));
                CU_CHECK(cudaStreamSynchronize(_data->_stream));
                dumpRes();
            }
        };

        if(
            _data->_streamPolicy==Data::StreamPolicy::PULL_STREAM and
            _data->_optPolicy==Data::OptPolicy::HALO_BLOCKING_L1L2
        )
        {
            haloBlockingL1L2PullRun();
        }

        if(
            _data->_streamPolicy==Data::StreamPolicy::PULL_STREAM and 
            _data->_optPolicy==Data::OptPolicy::HALO_BLOCKING_STATIC_L2
        )
        {
            haloBlockingStaticL2PullRun();
        }

        if(
            _data->_streamPolicy==Data::StreamPolicy::INPLACE_STREAM and
            _data->_optPolicy==Data::OptPolicy::HALO_BLOCKING_DYNAMIC_L2
        )
        {
            haloBlockingDynamicL2InplaceRun();
        }

        if(
            _data->_streamPolicy==Data::StreamPolicy::INPLACE_STREAM and 
            _data->_optPolicy==Data::OptPolicy::HALO_BLOCKING_STATIC_L2
        )
        {
            haloBlockingStaticL2InplaceRun();
        }

        if(
            _data->_streamPolicy==Data::StreamPolicy::PULL_STREAM and
            _data->_optPolicy==Data::OptPolicy::NONE
        )
        {
            staticPullRun();
        }
    }

    Simulator::~Simulator()
    {

    }
}