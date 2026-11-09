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
}

culbm::tune::Selection culbm::tune::select(int q, double capacityBytes, double lambda, double gamma, int iMin, int iMax, int bMax)
{
    Selection best{0, 0, 0, 0};
    double bestScore = -std::numeric_limits<double>::infinity();
    for (int i = iMin; i <= iMax; ++i)
    {
        const int halo = 2 * (i - 1);
        for (int bx = 32; bx <= bMax; bx += 32)
        {
            if (bx <= halo) continue;
            const auto [sectors, edges] = beta(bx, i);
            for (int by = halo + 1; by <= bMax; ++by)
                for (int bz = halo + 1; bz <= by; ++bz)
                {
                    const double volume = static_cast<double>(q * 4) * bx * by * bz;
                    if (volume > capacityBytes) continue;
                    const double h = (1.0 - static_cast<double>(halo) / bx) *
                                     (1.0 - static_cast<double>(halo) / by) *
                                     (1.0 - static_cast<double>(halo) / bz);
                    const double p = volume == capacityBytes ? 1.0 : 1.0 - std::exp(-volume / (capacityBytes - volume));
                    const double alpha = 1.0 + lambda * (sectors - 1.0) + p * (2.0 * gamma - (1.0 + lambda)) * edges;
                    const double score = h / ((1.0 + alpha * h) / (2.0 * i) + lambda * (1.0 - 1.0 / i));
                    if (score > bestScore)
                    {
                        best = {bx, by, bz, i};
                        bestScore = score;
                    }
                }
        }
    }
    if (best.bx == 0) throw std::invalid_argument("No tile fits the capacity and search bounds.");
    return best;
}
