#pragma once

#include <cstddef>
#include <cstdint>

namespace culbm::physical::detail
{
    struct Allocation
    {
        void* base;
        std::uint64_t physical, bytes, mappedBytes;
        bool contiguous;
    };

    Allocation allocate(std::size_t bytes);
}
