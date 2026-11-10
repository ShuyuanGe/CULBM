#pragma once

#include <array>
#include <cstddef>

namespace culbm::core
{
    struct Direction { int x, y, z; };

    template <int n_dir>
        requires (n_dir == 15 || n_dir == 19 || n_dir == 27)
    struct Stencil
    {
        static constexpr auto directions = [] {
            std::array<Direction, n_dir> result{};
            std::size_t i = 0;
            for (int z = -1; z <= 1; ++z)
                for (int y = -1; y <= 1; ++y)
                    for (int x = -1; x <= 1; ++x)
                        if (const int n = (x != 0) + (y != 0) + (z != 0);
                           n_dir == 27 || (n_dir == 19 && n <= 2) || (n_dir == 15 && n != 2))
                            result[i++] = {x, y, z};
            return result;
        }();

        static constexpr int opposite(std::size_t q) { return n_dir - 1 - q; }

        static_assert([] {
            for (int q = 0; q < n_dir; ++q) {
                const auto a = directions[q];
                const auto b = directions[opposite(q)];
                if (a.x != -b.x || a.y != -b.y || a.z != -b.z) return false;
            }
            return true;
        }(), "Stencil directions must be ordered in opposite pairs");

    };
}
