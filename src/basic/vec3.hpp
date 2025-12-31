#pragma once

#include <array>
#include <cstdint>

namespace culbm::basic
{
    template<typename T>
    struct Vec3
    {
        union
        {
            std::array<T, 3> data;
            struct
            {
                T x, y, z;
            };
        };

        constexpr Vec3() : x(0), y(0), z(0) {}
        constexpr Vec3(T x, T y, T z) : x(x), y(y), z(z) {}
        constexpr Vec3(const std::array<T, 3> data) : data(data) {}

        constexpr const T& operator[](std::uint32_t n) const { return data[n]; }
        constexpr T& operator[](std::uint32_t n) { return data[n]; }
    };
}