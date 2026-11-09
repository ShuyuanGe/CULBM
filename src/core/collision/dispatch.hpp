#pragma once

#include "kind.hpp"
#include <format>
#include <stdexcept>
#include <string_view>

namespace culbm::core::collision
{
    template <typename Factory>
    auto dispatch(int q, std::string_view kind, Factory&& make)
    {
        if (q == 15 && kind == "bgk")      return make.template operator()<15, Kind::Bgk>();
        if (q == 15 && kind == "mrt")      return make.template operator()<15, Kind::Mrt>();
        if (q == 19 && kind == "bgk")      return make.template operator()<19, Kind::Bgk>();
        if (q == 19 && kind == "mrt")      return make.template operator()<19, Kind::Mrt>();
        if (q == 19 && kind == "cumulant") return make.template operator()<19, Kind::Cumulant>();
        if (q == 27 && kind == "bgk")      return make.template operator()<27, Kind::Bgk>();
        if (q == 27 && kind == "mrt")      return make.template operator()<27, Kind::Mrt>();
        if (q == 27 && kind == "cumulant") return make.template operator()<27, Kind::Cumulant>();
        throw std::invalid_argument(std::format("Unsupported combination: D3Q{} / {}", q, kind));
    }
}
