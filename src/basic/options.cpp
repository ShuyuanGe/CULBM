#include "CLI11.hpp"
#include "config.hpp"
#include "options.hpp"


namespace culbm::basic
{
    std::shared_ptr<Options> Options::getInstance()
    {
        static std::shared_ptr<Options> instance (new Options());
        return instance;
    }

    void Options::cmdInit(int argc, const char** argv)
    {
        using namespace block_based_config;
        auto pInstance = getInstance();

        CLI::App app { "cuda LBM Fluid Simulator" };

        std::set<std::string> allowKernelType = {"Static", "Dynamic"};
        std::string kernelType;
        app.add_option("-K,--Kernel", kernelType, "The kernel type of the solver.")->check(CLI::IsMember(allowKernelType))
            ->default_val("Static");

        DyKernelParam param;
        std::tuple<u32, u32, u32> blkDim;
        std::tuple<u32, u32, u32> gridDim;
        app.add_option("--BlockingIteration", param.iter, "# of iterations within a block.")
            ->default_val(BLOCKING_ITER);
        app.add_option("--BlockDim", blkDim, "Dimension of a thread block")
            ->default_val(std::make_tuple(block_based_config::BLOCK_DIM.x, block_based_config::BLOCK_DIM.y, block_based_config::BLOCK_DIM.z));
        app.add_option("--GridDim", gridDim, "Dimension of a grid")
            ->default_val(std::make_tuple(block_based_config::GRID_DIM.x, block_based_config::GRID_DIM.y, block_based_config::GRID_DIM.z));

#if !USE_STATIC_CONFIG
        std::tuple<u32, u32, u32> domDim;
        app.add_option("--InvTau", pInstance->invTau, "The reciprocal of tau.")->required();
        app.add_option("--DomDim", domDim, "Dimension of the domain")->required();
        pInstance->domDim.x = std::get<0>(domDim);
        pInstance->domDim.y = std::get<1>(domDim);
        pInstance->domDim.z = std::get<2>(domDim);
#endif //USE_STATIC_CONFIG
        app.add_option("--BndCondFile", pInstance->bndCondFile, "The path for recording boundary conditions.");
        app.add_option("--DeviceId", pInstance->deviceId, "GPU index.")->default_val(0);
        try
        {
            app.parse(argc, argv);
        }
        catch(const CLI::ParseError& e)
        {
            std::exit(app.exit(e));
        }

        if(kernelType == "Static")
        {
            pInstance->kernelType = KernelType::staticKernel;
        }

        if(kernelType == "Dynamic")
        {
            pInstance->kernelType = KernelType::dynamicKernel;
            param.blkDim.x = std::get<0>(blkDim);
            param.blkDim.y = std::get<1>(blkDim);
            param.blkDim.z = std::get<2>(blkDim);
            param.gridDim.x = std::get<0>(gridDim);
            param.gridDim.y = std::get<1>(gridDim);
            param.gridDim.z = std::get<2>(gridDim);
            pInstance->dyKernelParam = param;
        } 
    }
}