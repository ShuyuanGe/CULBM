#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include "selector.hpp"

namespace
{
    std::pair<double, double> beta(int bx, int i)
    {
        const int r = i - 1;
        const int length = bx - 2 * r;
        const int gcd = std::gcd(length, 8);
        const int phases = 8 / gcd;
        int sectors = 0, edges = 0;
        for (int phase = 0; phase < 8; phase += gcd)
        {
            for (int warp = 0; warp < bx; warp += 32)
            {
                const int lo = std::max(r, warp);
                const int hi = std::min(bx - r, warp + 32);
                if (hi > lo) sectors += (phase + hi - 1) / 8 + 1 - (phase + lo) / 8;
            }
            edges += (phase + r) % 8 != 0;
            edges += (phase + bx - r) % 8 != 0;
        }
        return {8.0 * sectors / (phases * length), 4.0 * edges / (phases * length)};
    }

    double score(int bx, int by, int bz, int i, double lambda, double gamma,
                 double sectors, double edges)
    {
        const int halo = 2 * (i - 1);
        const double h = (1.0 - static_cast<double>(halo) / bx) *
                         (1.0 - static_cast<double>(halo) / by) *
                         (1.0 - static_cast<double>(halo) / bz);
        const double alpha = 1.0 + lambda * (sectors - 1.0) + (2.0 * gamma - (1.0 + lambda)) * edges;
        return h / ((1.0 + alpha * h) / (2.0 * i) + lambda * (1.0 - 1.0 / i));
    }
}

double culbm::tune::predicted_score(int bx, int by, int bz, int i, double lambda, double gamma)
{
    const auto [sectors, edges] = beta(bx, i);
    return score(bx, by, bz, i, lambda, gamma, sectors, edges);
}

culbm::tune::Selection culbm::tune::select(int q, double capacity_bytes, double lambda, double gamma,
                                         int i_min, int i_max, std::optional<int> b_max)
{
    if (not std::isfinite(capacity_bytes) or capacity_bytes <= 0)
        throw std::invalid_argument("Persisting capacity must be finite and positive.");
    Selection best{0, 0, 0, 0, 0.0, 0};
    double best_score = -std::numeric_limits<double>::infinity();
    std::uint64_t candidate_count = 0;
    const auto capacity_cells = static_cast<std::uint64_t>(capacity_bytes / (q * 4));
    const int bx_max = b_max ? std::min(1024, *b_max) : 1024;
    for (int i = i_min; i <= i_max; ++i)
    {
        const int halo = 2 * (i - 1);
        const int minimum = halo + 1;
        for (int bx = 32; bx <= bx_max; bx += 32)
        {
            if (bx <= halo) continue;
            const auto [sectors, edges] = beta(bx, i);
            auto by_max = capacity_cells / (static_cast<std::uint64_t>(bx) * minimum);
            if (b_max) by_max = std::min(by_max, static_cast<std::uint64_t>(*b_max));
            for (std::uint64_t by = minimum; by <= by_max; ++by)
            {
                const auto bz_max = std::min(by, capacity_cells / (bx * by));
                for (std::uint64_t bz = minimum; bz <= bz_max; ++bz)
                {
                    ++candidate_count;
                    const double prediction = score(bx, by, bz, i, lambda, gamma, sectors, edges);
                    if (prediction > best_score)
                    {
                        best = {bx, static_cast<int>(by), static_cast<int>(bz), i, prediction, 0};
                        best_score = prediction;
                    }
                }
            }
        }
    }
    if (best.bx == 0) throw std::invalid_argument("No tile fits the capacity and search bounds.");
    best.candidate_count = candidate_count;
    return best;
}
