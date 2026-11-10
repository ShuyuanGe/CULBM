#pragma once

#include <filesystem>
#include <format>
#include <fstream>
#include <span>
#include <stdexcept>
#include <cuda/algorithm>
#include <cuda/std/span>
#include "core/types.hpp"

namespace culbm::sim
{
    inline void write_field(::cuda::stream_ref stream, const real_t* data,
                            std::span<real_t> host, const std::filesystem::path& path)
    {
        std::ofstream file(path, std::ios::binary);
        if (not file)
            throw std::runtime_error(std::format("Could not open the file: {}", path.string()));
        ::cuda::copy_bytes(stream, ::cuda::std::span{data, host.size()},
                           ::cuda::std::span{host.data(), host.size()});
        stream.sync();
        file.write(reinterpret_cast<const char*>(host.data()), host.size_bytes());
    }
}
