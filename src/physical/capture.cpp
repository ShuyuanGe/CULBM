#include "capture.hpp"
#include "gpu/error.hpp"
#include <cerrno>
#include <cstdarg>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cuda_runtime_api.h>

namespace culbm::physical::detail
{
    namespace
    {
        // Linux x86-64 private RM/UVM ABI; decoding depends on the command numbers,
        // field offsets, widths and meanings below.
        // These structs are read-only views of driver-owned ioctl buffers.
        static_assert(sizeof(void*) == 8);
        namespace abi
        {
            // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/550.40.53/src/nvidia/arch/nvalloc/unix/include/nv-ioctl-numbers.h#L25-L33
            constexpr unsigned NV_IOCTL_MAGIC = 'F', NV_ESC_IOCTL_XFER_CMD = 211;
            // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/550.40.53/src/nvidia/arch/nvalloc/unix/include/nv_escape.h#L24-L28
            constexpr unsigned NV_ESC_RM_ALLOC = 0x2b, NV_ESC_RM_FREE = 0x29;

            // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/550.40.53/src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h#L35-L47
            struct nv_ioctl_xfer_t { std::uint32_t cmd, size; std::uint64_t ptr; };

            // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/550.40.53/src/common/sdk/nvidia/inc/class/cl0040.h#L29-L30
            constexpr std::uint32_t NV01_MEMORY_LOCAL_USER = 0x40;

            // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/550.40.53/src/common/sdk/nvidia/inc/nvos.h
            constexpr std::uint32_t NVOS64_FLAGS_FINN_SERIALIZED = 1;
            constexpr std::uint32_t NVOS32_ALLOC_FLAGS_VIRTUAL = 0x80000;
            constexpr std::uint32_t NVOS32_ATTR_PHYSICALITY_MASK = 3u << 27, NVOS32_ATTR_PHYSICALITY_CONTIGUOUS = 2;

            // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/550.40.53/src/common/sdk/nvidia/inc/nvos.h#L412-L440
            struct NVOS21_PARAMETERS
            {
                std::uint32_t hRoot, hObjectParent, hObjectNew, hClass;
                std::uint64_t pAllocParms;
                std::uint32_t paramsSize, status;
            };
            struct NVOS64_PARAMETERS
            {
                std::uint32_t hRoot, hObjectParent, hObjectNew, hClass;
                std::uint64_t pAllocParms, pRightsRequested;
                std::uint32_t paramsSize, flags, status;
            };

            // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/550.40.53/src/common/sdk/nvidia/inc/nvos.h#L147-L156
            struct NVOS00_PARAMETERS_PREFIX { std::uint32_t hRoot, hObjectParent, hObjectOld; };

            /*
             * offset is the first physical address.
             * https://github.com/NVIDIA/open-gpu-kernel-modules/blob/550.40.53/src/common/sdk/nvidia/inc/nvos.h#L1498-L1530
             */
            struct NV_MEMORY_ALLOCATION_PARAMS_PREFIX
            {
                std::uint32_t owner, type, flags, width, height;
                std::int32_t pitch;
                std::uint32_t attr, attr2, format, comprCovg, zcullCovg;
                std::uint64_t rangeLo, rangeHi, size, alignment, offset;
            };

            // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/550.40.53/kernel-open/nvidia-uvm/uvm_ioctl.h
            constexpr unsigned UVM_MAP_EXTERNAL_ALLOCATION = 33, UVM_FREE = 34, UVM_UNMAP_EXTERNAL = 66;

            /*
             * Oldest published version tag with 256 GPU entries: 550.40.53.
             * Earlier tags use 32 entries.
             * https://github.com/NVIDIA/open-gpu-kernel-modules/blob/550.40.53/kernel-open/nvidia-uvm/uvm_ioctl.h#L449-L463
             */
            struct UVM_MAP_EXTERNAL_ALLOCATION_PARAMS
            {
                std::uint64_t base, length, offset;
                // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/550.40.53/kernel-open/nvidia-uvm/uvm_types.h#L147-L157
                std::byte perGpuAttributes[256 * 36]; // 256 UvmGpuMappingAttributes, 36 bytes each.
                std::uint64_t gpuAttributesCount;
                std::int32_t rmCtrlFd;
                std::uint32_t hClient, hMemory, rmStatus;
            };

            // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/550.40.53/kernel-open/nvidia-uvm/uvm_ioctl.h#L895-L904
            struct UVM_UNMAP_EXTERNAL_PARAMS_PREFIX { std::uint64_t base, length; };

            static_assert(sizeof(nv_ioctl_xfer_t) == 16 && offsetof(nv_ioctl_xfer_t, ptr) == 8);
            static_assert(sizeof(NVOS21_PARAMETERS) == 32 && offsetof(NVOS21_PARAMETERS, hObjectNew) == 8 &&
                          offsetof(NVOS21_PARAMETERS, hClass) == 12 && offsetof(NVOS21_PARAMETERS, pAllocParms) == 16 &&
                          offsetof(NVOS21_PARAMETERS, paramsSize) == 24 && offsetof(NVOS21_PARAMETERS, status) == 28);
            static_assert(sizeof(NVOS64_PARAMETERS) == 48 && offsetof(NVOS64_PARAMETERS, hObjectNew) == 8 &&
                          offsetof(NVOS64_PARAMETERS, hClass) == 12 && offsetof(NVOS64_PARAMETERS, pAllocParms) == 16 &&
                          offsetof(NVOS64_PARAMETERS, paramsSize) == 32 && offsetof(NVOS64_PARAMETERS, flags) == 36 &&
                          offsetof(NVOS64_PARAMETERS, status) == 40);
            static_assert(sizeof(NVOS00_PARAMETERS_PREFIX) == 12 && offsetof(NVOS00_PARAMETERS_PREFIX, hObjectOld) == 8);
            static_assert(sizeof(NV_MEMORY_ALLOCATION_PARAMS_PREFIX) == 88 && offsetof(NV_MEMORY_ALLOCATION_PARAMS_PREFIX, flags) == 8 &&
                          offsetof(NV_MEMORY_ALLOCATION_PARAMS_PREFIX, attr) == 24 && offsetof(NV_MEMORY_ALLOCATION_PARAMS_PREFIX, size) == 64 &&
                          offsetof(NV_MEMORY_ALLOCATION_PARAMS_PREFIX, offset) == 80);
            static_assert(sizeof(UVM_MAP_EXTERNAL_ALLOCATION_PARAMS) == 9264 && offsetof(UVM_MAP_EXTERNAL_ALLOCATION_PARAMS, gpuAttributesCount) == 9240 &&
                          offsetof(UVM_MAP_EXTERNAL_ALLOCATION_PARAMS, hClient) == 9252 && offsetof(UVM_MAP_EXTERNAL_ALLOCATION_PARAMS, hMemory) == 9256 &&
                          offsetof(UVM_MAP_EXTERNAL_ALLOCATION_PARAMS, rmStatus) == 9260);
            static_assert(sizeof(UVM_UNMAP_EXTERNAL_PARAMS_PREFIX) == 16);
        }

        struct Memory { std::uint32_t client, object; std::uint64_t physical, bytes; bool contiguous; };
        struct Mapping { std::uint64_t base, bytes, offset; std::uint32_t client, object; };
        // Discard records on free/unmap even if payload status reports failure.
        struct Capture
        {
            std::vector<Memory> allocations;
            std::vector<Mapping> mappings;
            bool failed = false;
        };

        void observe_rm(Capture& capture, unsigned command, unsigned packet_bytes, const void* params)
        {
            if (command == abi::NV_ESC_IOCTL_XFER_CMD)
            {
                auto& transfer = *reinterpret_cast<const abi::nv_ioctl_xfer_t*>(params);
                command = transfer.cmd;
                packet_bytes = transfer.size;
                params = reinterpret_cast<const void*>(transfer.ptr);
            }
            if (command == abi::NV_ESC_RM_ALLOC)
            {
                const auto record = [&](const auto& allocation, bool serialized) {
                    if (allocation.status != 0 || allocation.hClass != abi::NV01_MEMORY_LOCAL_USER) return;
                    // FINN-serialized parameters cannot be read as a native struct.
                    if (serialized) { capture.failed = true; return; }
                    // A zero paramsSize is unspecified; otherwise it must cover the prefix.
                    if (allocation.pAllocParms == 0 ||
                        (allocation.paramsSize != 0 && allocation.paramsSize < sizeof(abi::NV_MEMORY_ALLOCATION_PARAMS_PREFIX))) return;
                    auto& memory = *reinterpret_cast<const abi::NV_MEMORY_ALLOCATION_PARAMS_PREFIX*>(allocation.pAllocParms);
                    if (memory.flags & abi::NVOS32_ALLOC_FLAGS_VIRTUAL) return;
                    capture.allocations.push_back({
                        .client = allocation.hRoot,
                        .object = allocation.hObjectNew,
                        .physical = memory.offset,
                        .bytes = memory.size,
                        .contiguous = (memory.attr & abi::NVOS32_ATTR_PHYSICALITY_MASK) == (abi::NVOS32_ATTR_PHYSICALITY_CONTIGUOUS << 27)
                    });
                };
                if (packet_bytes == sizeof(abi::NVOS21_PARAMETERS))
                    record(*reinterpret_cast<const abi::NVOS21_PARAMETERS*>(params), false);
                else if (packet_bytes == sizeof(abi::NVOS64_PARAMETERS))
                {
                    auto& allocation = *reinterpret_cast<const abi::NVOS64_PARAMETERS*>(params);
                    record(allocation, allocation.flags & abi::NVOS64_FLAGS_FINN_SERIALIZED);
                }
            }
            else if (command == abi::NV_ESC_RM_FREE && packet_bytes >= sizeof(abi::NVOS00_PARAMETERS_PREFIX))
            {
                auto& free = *reinterpret_cast<const abi::NVOS00_PARAMETERS_PREFIX*>(params);
                const auto matches = [&](const auto& record) { return record.client == free.hRoot && record.object == free.hObjectOld; };
                std::erase_if(capture.allocations, matches);
                std::erase_if(capture.mappings, matches);
            }
        }

        void observe_uvm(Capture& capture, unsigned long request, const void* argument)
        {
            if (request == abi::UVM_MAP_EXTERNAL_ALLOCATION)
            {
                auto& map = *reinterpret_cast<const abi::UVM_MAP_EXTERNAL_ALLOCATION_PARAMS*>(argument);
                if (map.rmStatus == 0 && map.gpuAttributesCount == 1)
                    capture.mappings.push_back({
                        .base = map.base,
                        .bytes = map.length,
                        .offset = map.offset,
                        .client = map.hClient,
                        .object = map.hMemory
                    });
            }
            else if (request == abi::UVM_UNMAP_EXTERNAL)
            {
                auto& range = *reinterpret_cast<const abi::UVM_UNMAP_EXTERNAL_PARAMS_PREFIX*>(argument);
                if (range.length == 0) return;
                std::erase_if(capture.mappings, [&](const auto& mapping) {
                    // Compare differences rather than endpoints to avoid address overflow.
                    if (mapping.base <= range.base)
                        return range.base - mapping.base < mapping.bytes;
                    else
                        return mapping.base - range.base < range.length;
                });
            }
            else if (request == abi::UVM_FREE)
            {
                const auto base = *reinterpret_cast<const std::uint64_t*>(argument);
                std::erase_if(capture.mappings, [&](const auto& m) { return m.base == base; });
            }
        }

        void observe(Capture& capture, unsigned long request, const void* argument)
        {
            if (_IOC_TYPE(request) == abi::NV_IOCTL_MAGIC)
                observe_rm(capture, _IOC_NR(request), _IOC_SIZE(request), argument);
            else
                observe_uvm(capture, request, argument);
        }

        // Only the allocation scope and ioctl hook need thread-local state.
        thread_local Capture* active = nullptr;
    }

    Allocation allocate(std::size_t bytes)
    {
        // Capture records only for this cudaMalloc.
        Capture capture;
        if (active != nullptr) throw std::logic_error("Nested physical allocation capture");
        active = &capture;
        void* base = nullptr;
        const auto status = cudaMalloc(&base, bytes);
        active = nullptr;
        gpu::check(status);
        try
        {
            if (capture.failed) throw std::runtime_error("Physical allocation capture failed");
            const auto va = reinterpret_cast<std::uintptr_t>(base);
            for (const auto& mapping : capture.mappings)
            {
                if (va < mapping.base || va - mapping.base >= mapping.bytes) continue;
                // First PA + an arbitrary object offset is unsafe for paged backing.
                if (mapping.offset != 0 || va != mapping.base)
                    throw std::runtime_error("Physical allocation has nonzero object offset");
                for (const auto& memory : capture.allocations)
                {
                    if (memory.client != mapping.client || memory.object != mapping.object) continue;
                    return {base, memory.physical, memory.bytes, mapping.bytes, memory.contiguous};
                }
            }
            throw std::runtime_error("No RM allocation matched the physical buffer's UVM mapping");
        }
        catch (...)
        {
            gpu::check(cudaFree(base));
            throw;
        }
    }
}

// Normal shared-library linkage loads this hook before libcuda, without LD_PRELOAD.
// Forward requests unchanged, preserve errno, and never throw through the C ABI.
extern "C" int ioctl(int fd, unsigned long request, ...) noexcept
{
    va_list args;
    va_start(args, request);
    void* argument = va_arg(args, void*);
    va_end(args);
    const int result = syscall(SYS_ioctl, fd, request, argument);
    const int saved_errno = errno;
    using namespace culbm::physical::detail;
    if (result == 0 && active != nullptr)
    {
        try { observe(*active, request, argument); }
        catch (...) { active->failed = true; }
    }
    errno = saved_errno;
    return result;
}
