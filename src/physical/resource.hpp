#pragma once

#include <cstddef>
#include <memory>
#include <cuda/devices>
#include <cuda/memory_resource>

namespace culbm::physical
{
    // Copies share the allocation records; adapt with cuda::mr::synchronous_resource_adapter
    // so buffer destruction waits for its stream. The stream must outlive the buffer.
    class Resource
    {
        struct State;
        std::shared_ptr<State> state;

    public:
        // Physical alignment is separate from CCCL's native power-of-two alignment.
        explicit Resource(::cuda::device_ref device, std::size_t physicalAlignment = 6 * 1024 * 1024);
        void* allocate_sync(std::size_t bytes, std::size_t alignment);
        void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept;
        bool operator==(const Resource& other) const noexcept { return state == other.state; }
        friend void get_property(const Resource&, ::cuda::mr::device_accessible) noexcept {}
    };

    static_assert(::cuda::mr::synchronous_resource_with<Resource, ::cuda::mr::device_accessible>);
}
