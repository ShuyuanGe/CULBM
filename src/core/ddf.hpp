#pragma once

#include <concepts>
#include <cstddef>
#include <utility>
#include "accessor.hpp"
#include "types.hpp"

namespace culbm::core::ddf
{
    template <typename T, int n_dir>
    concept Accessor = requires {
        { T::template read<n_dir, 0>() } -> std::same_as<accessor::Access>;
        { T::template write<n_dir, 0>() } -> std::same_as<accessor::Access>;
    };

    template <typename T>
    concept IO = requires(const ddf_t* src, ddf_t* dst, ddf_t value) {
        { T::load(src) } -> std::same_as<ddf_t>;
        { T::store(dst, value) } -> std::same_as<void>;
    };

    template <typename T>
    concept View = requires(const T& view) {
        { view.template read_address<0, 0, 0, 0>() } -> std::same_as<const ddf_t*>;
        { view.template write_address<0, 0, 0, 0>() } -> std::same_as<ddf_t*>;
    };

    template <int n_dir, Accessor<n_dir> AccessorType, IO IOType, typename Value, View ViewType>
    constexpr void load(Value (&f)[n_dir], const ViewType& view)
    {
        [&]<std::size_t... Q>(std::index_sequence<Q...>) constexpr {
            ([&]() constexpr {
                constexpr auto a = AccessorType::template read<n_dir, Q>();
                f[Q] = IOType::load(view.template read_address<a.direction, a.x, a.y, a.z>());
            }(), ...);
        }(std::make_index_sequence<n_dir>{});
    }

    template <int n_dir, Accessor<n_dir> AccessorType, IO IOType, typename Value, View ViewType>
    constexpr void store(const Value (&f)[n_dir], const ViewType& view)
    {
        [&]<std::size_t... Q>(std::index_sequence<Q...>) constexpr {
            ([&]() constexpr {
                constexpr auto a = AccessorType::template write<n_dir, Q>();
                IOType::store(view.template write_address<a.direction, a.x, a.y, a.z>(), f[Q]);
            }(), ...);
        }(std::make_index_sequence<n_dir>{});
    }
}
