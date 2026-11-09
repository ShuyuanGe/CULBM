#include "resource.hpp"
#include "capture.hpp"
#include "gpu/error.hpp"
#include <algorithm>
#include <bit>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <cuda_runtime_api.h>

namespace culbm::physical
{
    namespace
    {
        std::size_t aligned_offset(const detail::Allocation& allocation, std::size_t bytes, std::size_t allocated_bytes,
                                   std::size_t alignment, std::size_t physical_alignment)
        {
            if (!allocation.contiguous)
                throw std::runtime_error("Physical allocation is not contiguous");
            const auto va = reinterpret_cast<std::uintptr_t>(allocation.base);
            const auto shift = (physical_alignment - allocation.physical % physical_alignment) % physical_alignment;
            const auto capacity = std::min({static_cast<std::uint64_t>(allocated_bytes), allocation.mapped_bytes, allocation.bytes});
            if (shift > capacity || bytes > capacity - shift ||
                shift > std::numeric_limits<std::uintptr_t>::max() - va ||
                (va + shift) % alignment != 0 ||
                shift > std::numeric_limits<std::uint64_t>::max() - allocation.physical ||
                bytes > std::numeric_limits<std::uint64_t>::max() - (allocation.physical + shift))
                throw std::runtime_error("Physical allocation does not cover aligned interval");
            return shift;
        }

        struct DeviceScope
        {
            int previous;
            explicit DeviceScope(int device)
            {
                gpu::check(cudaGetDevice(&previous));
                gpu::check(cudaSetDevice(device));
            }
            ~DeviceScope() { if (cudaSetDevice(previous) != cudaSuccess) std::terminate(); }
        };
    }

    struct Resource::State
    {
        int device;
        std::size_t physical_alignment;
        std::mutex mutex;
        std::unordered_map<void*, void*> bases;
        State(int device, std::size_t physical_alignment) : device(device), physical_alignment(physical_alignment) {}
    };

    Resource::Resource(::cuda::device_ref device, std::size_t physical_alignment) :
        state(std::make_shared<State>(device.get(), physical_alignment))
    {
        if (physical_alignment == 0) throw std::invalid_argument("Physical alignment must be positive");
    }

    void* Resource::allocate_sync(std::size_t bytes, std::size_t alignment)
    {
        if (!std::has_single_bit(alignment) || alignment > 256 || state->physical_alignment % alignment != 0)
            throw std::invalid_argument("Unsupported native alignment for physical allocation");
        if (bytes > std::numeric_limits<std::size_t>::max() - (state->physical_alignment - 1))
            throw std::bad_alloc();
        DeviceScope device(state->device);
        // Only the interior buffer is exposed; the persisting window excludes this slack.
        const auto allocated_bytes = bytes + state->physical_alignment - 1;
        const auto allocation = detail::allocate(allocated_bytes);
        try
        {
            const auto va = reinterpret_cast<std::uintptr_t>(allocation.base);
            const auto shift = aligned_offset(allocation, bytes, allocated_bytes, alignment, state->physical_alignment);
            auto* ptr = reinterpret_cast<void*>(va + shift);
            std::lock_guard lock(state->mutex);
            state->bases.emplace(ptr, allocation.base);
            return ptr;
        }
        catch (...)
        {
            gpu::check(cudaFree(allocation.base));
            throw;
        }
    }

    void Resource::deallocate_sync(void* ptr, std::size_t, std::size_t) noexcept
    {
        DeviceScope device(state->device);
        void* base;
        {
            std::lock_guard lock(state->mutex);
            const auto entry = state->bases.find(ptr);
            if (entry == state->bases.end()) std::terminate();
            base = entry->second;
            state->bases.erase(entry);
        }
        // Free the original CUDA allocation, not its physically aligned interior pointer.
        if (cudaFree(base) != cudaSuccess) std::terminate();
    }
}
