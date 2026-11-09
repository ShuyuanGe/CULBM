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
        if (q == 15 && kind == "bgk")      return make.template operator()<15, Kind::bgk>();
        if (q == 15 && kind == "mrt")      return make.template operator()<15, Kind::mrt>();
        if (q == 19 && kind == "bgk")      return make.template operator()<19, Kind::bgk>();
        if (q == 19 && kind == "mrt")      return make.template operator()<19, Kind::mrt>();
        if (q == 19 && kind == "cumulant") return make.template operator()<19, Kind::cumulant>();
        if (q == 27 && kind == "bgk")      return make.template operator()<27, Kind::bgk>();
        if (q == 27 && kind == "mrt")      return make.template operator()<27, Kind::mrt>();
        if (q == 27 && kind == "cumulant") return make.template operator()<27, Kind::cumulant>();
        throw std::invalid_argument(std::format("Unsupported combination: D3Q{} / {}", q, kind));
    }
}
