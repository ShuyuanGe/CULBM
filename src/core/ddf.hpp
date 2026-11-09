#pragma once

#include <concepts>
#include <cstddef>
#include <utility>
#include "accessor.hpp"
#include "types.hpp"

namespace culbm::core::ddf
{
    template <typename T, int NDir>
    concept Accessor = requires {
        { T::template read<NDir, 0>() } -> std::same_as<accessor::Access>;
        { T::template write<NDir, 0>() } -> std::same_as<accessor::Access>;
    };

    template <typename T>
    concept IO = requires(const ddf_t* src, ddf_t* dst, ddf_t value) {
        { T::load(src) } -> std::same_as<ddf_t>;
        { T::store(dst, value) } -> std::same_as<void>;
    };

    template <typename T>
    concept View = requires(const T& view) {
        { view.template readAddress<0, 0, 0, 0>() } -> std::same_as<const ddf_t*>;
        { view.template writeAddress<0, 0, 0, 0>() } -> std::same_as<ddf_t*>;
    };

    template <int NDir, Accessor<NDir> AccessorType, IO IOType, typename Value, View ViewType>
    constexpr void load(Value (&f)[NDir], const ViewType& view)
    {
        [&]<std::size_t... Q>(std::index_sequence<Q...>) constexpr {
            ([&]() constexpr {
                constexpr auto a = AccessorType::template read<NDir, Q>();
                f[Q] = IOType::load(view.template readAddress<a.direction, a.x, a.y, a.z>());
            }(), ...);
        }(std::make_index_sequence<NDir>{});
    }

    template <int NDir, Accessor<NDir> AccessorType, IO IOType, typename Value, View ViewType>
    constexpr void store(const Value (&f)[NDir], const ViewType& view)
    {
        [&]<std::size_t... Q>(std::index_sequence<Q...>) constexpr {
            ([&]() constexpr {
                constexpr auto a = AccessorType::template write<NDir, Q>();
                IOType::store(view.template writeAddress<a.direction, a.x, a.y, a.z>(), f[Q]);
            }(), ...);
        }(std::make_index_sequence<NDir>{});
    }
}
