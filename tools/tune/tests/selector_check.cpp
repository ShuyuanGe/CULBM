#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include "selector.hpp"

namespace
{
    struct Expected
    {
        int bx, by, bz, i;
        std::uint64_t candidates;
    };

    struct Case
    {
        double lambda, gamma, capacity_mib;
        std::array<Expected, 3> expected;
    };

    void require(bool condition, const char* message)
    {
        if (not condition) throw std::runtime_error(message);
    }
}

int main() try
{
    // Calibrations are the unmodified values in data/reference/fig7/* target records.
    constexpr std::array<int, 3> q_values{15, 19, 27};
    constexpr std::array cases{
        Case{.20792, 1.0088, 49.5, {{{96,99,91,6,2259952}, {96,90,79,6,1632253}, {96,71,70,5,998178}}}},
        Case{.18668, 10.3660, 60, {{{96,105,104,9,2937434}, {96,99,87,9,2126670}, {96,82,74,9,1307805}}}},
        Case{.15744, 9.4191, 66, {{{96,110,109,9,3339537}, {96,103,92,9,2423340}, {96,82,81,9,1494509}}}},
        Case{.19061, 10.1504, 80, {{{128,105,104,9,4320794}, {96,107,107,9,3145422}, {96,94,86,9,1948562}}}},
        Case{.11381, 6.0663, 66, {{{96,110,109,9,3339537}, {96,103,92,9,2423340}, {96,82,81,9,1494509}}}},
        Case{.11180, 6.3724, 66, {{{96,110,109,9,3339537}, {96,103,92,9,2423340}, {96,82,81,9,1494509}}}},
    };
    std::uint64_t total = 0;
    for (const auto& sample : cases)
        for (std::size_t index = 0; index < q_values.size(); ++index)
        {
            const auto& expected = sample.expected[index];
            const auto selected = culbm::tune::select(q_values[index], sample.capacity_mib * 1048576.0,
                                                     sample.lambda, sample.gamma, 2, 20, std::nullopt);
            require(selected.bx == expected.bx and selected.by == expected.by and selected.bz == expected.bz and
                    selected.i == expected.i, "calibrated service-cost selection differs");
            require(selected.candidate_count == expected.candidates, "complete feasible-set count differs");
            require(std::isfinite(selected.score) and selected.score > 1, "invalid selected score");
            require(4.0 * q_values[index] * selected.bx * selected.by * selected.bz <= sample.capacity_mib * 1048576.0,
                    "selected tile exceeds persisting capacity");
            total += selected.candidate_count;
        }
    require(total == 42449228, "18-case candidate total differs");
    const double prediction = culbm::tune::predicted_score(96, 71, 70, 5, .20792, 1.0088);
    require(std::abs(prediction - 2.0879221358116813) < 1e-12, "nonzero-edge service cost differs");
    require(culbm::tune::predicted_score(96, 71, 70, 5, .20792, 2.0088) < prediction,
            "increased partial-sector cost did not reduce the score");
    require(culbm::tune::predicted_score(96, 82, 74, 9, .18668, 1.0) ==
            culbm::tune::predicted_score(96, 82, 74, 9, .18668, 20.0),
            "zero-edge configuration depends on partial-sector cost");
    const auto restricted = culbm::tune::select(27, 49.5 * 1048576.0, .20792, 1.0088, 2, 20, 160);
    require(restricted.candidate_count == 394793, "explicit exploration bound is not applied");
    const auto exact = culbm::tune::select(27, 108.0 * 32 * 3 * 3, .2, 1.0, 2, 20, std::nullopt);
    require(exact.bx == 32 and exact.by == 3 and exact.bz == 3 and exact.i == 2 and exact.candidate_count == 1,
            "exact capacity-boundary candidate is not retained");
    require(culbm::tune::model_id == "calibrated_service_cost_v1", "model identifier differs");
    std::cout << "PASS: 18 calibrated selections, 42449228 candidates, direct service-cost score, "
                 "partial-sector response, capacity boundary, explicit exploration bound\n";
}
catch (const std::exception& error)
{
    std::cerr << error.what() << '\n';
    return 1;
}
