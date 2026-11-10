#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

namespace culbm::tune
{
    inline constexpr std::string_view model_id = "calibrated_service_cost_v1";

    struct Selection
    {
        int bx, by, bz, i;
        double score;
        std::uint64_t candidate_count;
    };

    double predicted_score(int bx, int by, int bz, int i, double lambda, double gamma);
    Selection select(int q, double capacity_bytes, double lambda, double gamma,
                     int i_min, int i_max, std::optional<int> b_max);
}
