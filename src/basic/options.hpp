#pragma once

#include <memory>
#include <string>
#include <optional>
#include "types.hpp"
#include "config.hpp"

namespace gf::basic
{
    class Options
    {

        protected:
            constexpr Options() = default;
        public:
            enum struct KernelType
            {
                staticKernel, 
                dynamicKernel
            } kernelType;

            struct DyKernelParam
            {
                u32 iter;
                Vec3<u32> blkDim;
                Vec3<u32> gridDim;
            };

            std::optional<DyKernelParam> dyKernelParam;

            int deviceId;
            real_t invTau;
            Vec3<u32> domDim = Vec3<u32>{DOM_NX, DOM_NY, DOM_NZ};
            std::string bndCondFile;

            static std::shared_ptr<Options> getInstance();

            static void cmdInit(int argc, const char** argv);
    };
}