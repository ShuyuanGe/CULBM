#pragma once

#include <concepts>

namespace gf::basic
{
    template<std::unsigned_integral T>
    constexpr T divCeil(T num, T deno)
    {
        return (num+deno-1) / deno;
    }
}