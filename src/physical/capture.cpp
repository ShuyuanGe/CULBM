#include "capture.hpp"
#include "gpu/error.hpp"
#include <cerrno>
#include <cstdarg>
#include <cstring>
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
        static_assert(sizeof(void*) == 8);
        constexpr unsigned rmType = 'F';

        // nv_ioctl_xfer_t (16 bytes): cmd(u32)@0, size(u32)@4, ptr(u64)@8.
        // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/610.43.02/src/nvidia/arch/nvalloc/unix/include/nv-ioctl.h#L49-L54
        namespace rm_xfer
        {
            constexpr unsigned code = 211;
            constexpr std::size_t command = 0, bytes = 4, params = 8;
        }

        // Packet size selects NVOS21/NVOS64.
        // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/610.43.02/src/common/sdk/nvidia/inc/nvos.h#L464-L490
        namespace rm_alloc
        {
            constexpr unsigned code = 0x2b, localMemoryClass = 0x40;
            constexpr std::size_t client = 0, object = 8, classId = 12, params = 16;
            struct Layout { unsigned packetBytes; std::size_t paramsSizeOffset, statusOffset; };
            constexpr Layout nvos21{32, 24, 28}, nvos64{48, 32, 40};
        }

        // Only the client/object prefix is needed to forget RM_FREE observations.
        // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/610.43.02/src/common/sdk/nvidia/inc/nvos.h#L162-L168
        namespace rm_free
        {
            constexpr unsigned code = 0x29;
            constexpr std::size_t prefixBytes = 12, client = 0, object = 8;
        }

        // NV_MEMORY_ALLOCATION_PARAMS.offset returns the first PA; attr[28:27] == 2
        // reports physical contiguity.
        // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/610.43.02/src/common/sdk/nvidia/inc/nvos.h#L1600-L1634
        namespace memory_alloc
        {
            constexpr std::size_t flags = 8, attr = 24, bytes = 64, physical = 80;
            constexpr std::size_t requiredBytes = physical + sizeof(std::uint64_t);
            constexpr std::uint32_t virtualFlag = 0x00080000, physicalityMask = 3u << 27, contiguous = 2u << 27;
        }

        // Skip the unused GPU attribute array to read the mapping's RM object fields.
        // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/610.43.02/kernel-open/nvidia-uvm/uvm_ioctl.h#L366-L378
        // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/610.43.02/kernel-open/nvidia-uvm/uvm_types.h#L49-L96
        namespace uvm_map
        {
            constexpr unsigned code = 33;
            constexpr std::size_t base = 0, bytes = 8, offset = 16;
            constexpr std::size_t count = 9240, client = 9252, object = 9256, status = 9260;
        }

        // Only the base/length prefix is needed to forget overlapping mappings.
        // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/610.43.02/kernel-open/nvidia-uvm/uvm_ioctl.h#L736-L742
        namespace uvm_unmap
        {
            constexpr unsigned code = 66;
            constexpr std::size_t base = 0, bytes = 8;
        }

        // Only base is needed; length/status layouts may differ.
        // https://github.com/NVIDIA/open-gpu-kernel-modules/blob/610.43.02/kernel-open/nvidia-uvm/uvm_ioctl.h#L384-L388
        namespace uvm_free
        {
            constexpr unsigned code = 34;
            constexpr std::size_t base = 0;
        }

        template <typename T>
        T read(const void* data, std::size_t offset)
        {
            T value;
            std::memcpy(&value, static_cast<const char*>(data) + offset, sizeof(T));
            return value;
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

        void observe_rm(Capture& capture, unsigned command, unsigned packetBytes, const void* params)
        {
            if (command == rm_xfer::code)
            {
                command = read<std::uint32_t>(params, rm_xfer::command);
                packetBytes = read<std::uint32_t>(params, rm_xfer::bytes);
                params = reinterpret_cast<const void*>(read<std::uint64_t>(params, rm_xfer::params));
            }
            if (command == rm_alloc::code)
            {
                const rm_alloc::Layout* layout;
                if (packetBytes == rm_alloc::nvos21.packetBytes)      layout = &rm_alloc::nvos21;
                else if (packetBytes == rm_alloc::nvos64.packetBytes) layout = &rm_alloc::nvos64;
                else return;

                if (read<std::uint32_t>(params, layout->statusOffset) != 0 ||
                    read<std::uint32_t>(params, rm_alloc::classId) != rm_alloc::localMemoryClass) return;
                const auto paramsBytes = read<std::uint32_t>(params, layout->paramsSizeOffset);
                const auto* memoryParams = reinterpret_cast<const void*>(read<std::uint64_t>(params, rm_alloc::params));
                // Zero means unspecified; a supplied length must cover our last field.
                if (memoryParams == nullptr || (paramsBytes != 0 && paramsBytes < memory_alloc::requiredBytes)) return;
                if (read<std::uint32_t>(memoryParams, memory_alloc::flags) & memory_alloc::virtualFlag) return;
                const auto attr = read<std::uint32_t>(memoryParams, memory_alloc::attr);
                capture.allocations.push_back({
                    .client = read<std::uint32_t>(params, rm_alloc::client),
                    .object = read<std::uint32_t>(params, rm_alloc::object),
                    .physical = read<std::uint64_t>(memoryParams, memory_alloc::physical),
                    .bytes = read<std::uint64_t>(memoryParams, memory_alloc::bytes),
                    .contiguous = (attr & memory_alloc::physicalityMask) == memory_alloc::contiguous
                });
            }
            else if (command == rm_free::code && packetBytes >= rm_free::prefixBytes)
            {
                const auto client = read<std::uint32_t>(params, rm_free::client);
                const auto object = read<std::uint32_t>(params, rm_free::object);
                std::erase_if(capture.allocations, [&](const auto& a) { return a.client == client && a.object == object; });
                std::erase_if(capture.mappings, [&](const auto& m) { return m.client == client && m.object == object; });
            }
        }

        void observe_uvm(Capture& capture, unsigned long request, const void* argument)
        {
            if (request == uvm_map::code)
            {
                if (read<std::uint32_t>(argument, uvm_map::status) == 0 && read<std::uint64_t>(argument, uvm_map::count) == 1)
                    capture.mappings.push_back({read<std::uint64_t>(argument, uvm_map::base),
                        read<std::uint64_t>(argument, uvm_map::bytes), read<std::uint64_t>(argument, uvm_map::offset),
                        read<std::uint32_t>(argument, uvm_map::client), read<std::uint32_t>(argument, uvm_map::object)});
            }
            else if (request == uvm_unmap::code)
            {
                const auto base = read<std::uint64_t>(argument, uvm_unmap::base);
                const auto bytes = read<std::uint64_t>(argument, uvm_unmap::bytes);
                std::erase_if(capture.mappings, [&](const auto& m) {
                    return bytes != 0 && (m.base <= base ? base - m.base < m.bytes : m.base - base < bytes);
                });
            }
            else if (request == uvm_free::code)
            {
                const auto base = read<std::uint64_t>(argument, uvm_free::base);
                std::erase_if(capture.mappings, [&](const auto& m) { return m.base == base; });
            }
        }

        void observe(Capture& capture, unsigned long request, const void* argument)
        {
            if (_IOC_TYPE(request) == rmType)
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
    const int savedErrno = errno;
    using namespace culbm::physical::detail;
    if (result == 0 && active != nullptr)
    {
        try { observe(*active, request, argument); }
        catch (...) { active->failed = true; }
    }
    errno = savedErrno;
    return result;
}
