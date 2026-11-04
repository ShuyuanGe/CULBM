#pragma once

#include <cstddef>
#include <utility>

namespace culbm::lbm_core::ddf::detail
{
    template <int NDir, typename Accessor, typename IO, std::size_t Q, typename Value, typename View>
    constexpr void loadOne(Value* f, const View& view)
    {
        constexpr auto a = Accessor::template read<NDir, Q>();
        f[Q] = IO::load(view.template readAddress<a.direction, a.x, a.y, a.z>());
    }

    template <int NDir, typename Accessor, typename IO, std::size_t Q, typename Value, typename View>
    constexpr void storeOne(const Value* f, const View& view)
    {
        constexpr auto a = Accessor::template write<NDir, Q>();
        IO::store(view.template writeAddress<a.direction, a.x, a.y, a.z>(), f[Q]);
    }

    template <int NDir, typename Accessor, typename IO, typename Value, typename View, std::size_t... Q>
    constexpr void loadImpl(Value* f, const View& view, std::index_sequence<Q...>)
    {
        (loadOne<NDir, Accessor, IO, Q>(f, view), ...);
    }

    template <int NDir, typename Accessor, typename IO, typename Value, typename View, std::size_t... Q>
    constexpr void storeImpl(const Value* f, const View& view, std::index_sequence<Q...>)
    {
        (storeOne<NDir, Accessor, IO, Q>(f, view), ...);
    }
}

namespace culbm::lbm_core::ddf
{
    template <int NDir, typename Accessor, typename IO, typename Value, typename View>
    constexpr void load(Value* f, const View& view)
    {
        detail::loadImpl<NDir, Accessor, IO>(f, view, std::make_index_sequence<NDir>{});
    }

    template <int NDir, typename Accessor, typename IO, typename Value, typename View>
    constexpr void store(const Value* f, const View& view)
    {
        detail::storeImpl<NDir, Accessor, IO>(f, view, std::make_index_sequence<NDir>{});
    }
}
