#pragma once

#include <cstdint>

using u32 = std::uint32_t;
using u64 = std::uint64_t;

using i32 = std::int32_t;
using i64 = std::int64_t;

using f32 = float;
using f64 = double;

template<typename T>
struct Vec3
{
    T x, y, z;
    
    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(T x, T y, T z) : x(x), y(y), z(z) {}
    constexpr Vec3(T v) : x(v), y(v), z(v) {}
};