#pragma once

#include <variant>

namespace culbm::simulator::single_dev
{
    enum class PlainScheme { TwoFieldPull, EsoTwist };
    enum class BlockingGlobalBuffer { Double, Ring };
    using StrategySelection = std::variant<PlainScheme, BlockingGlobalBuffer>;
}
