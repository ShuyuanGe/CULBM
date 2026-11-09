#pragma once

#include <cstddef>
#include <utility>

namespace culbm::core::ddf
{
    template <int NDir, typename Accessor, typename IO, typename Value, typename View>
    constexpr void load(Value* f, const View& view)
    {
        [f, &view]<std::size_t... Q>(std::index_sequence<Q...>) constexpr {
            ([f, &view]() constexpr {
                constexpr auto a = Accessor::template read<NDir, Q>();
                f[Q] = IO::load(view.template readAddress<a.direction, a.x, a.y, a.z>());
            }(), ...);
        }(std::make_index_sequence<NDir>{});
    }

    template <int NDir, typename Accessor, typename IO, typename Value, typename View>
    constexpr void store(const Value* f, const View& view)
    {
        [f, &view]<std::size_t... Q>(std::index_sequence<Q...>) constexpr {
            ([f, &view]() constexpr {
                constexpr auto a = Accessor::template write<NDir, Q>();
                IO::store(view.template writeAddress<a.direction, a.x, a.y, a.z>(), f[Q]);
            }(), ...);
        }(std::make_index_sequence<NDir>{});
    }
}
