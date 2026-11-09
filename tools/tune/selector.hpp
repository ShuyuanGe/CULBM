#pragma once

namespace culbm::tune
{
    struct Selection { int bx, by, bz, i; };

    Selection select(int q, double capacity_bytes, double lambda, double gamma, int i_min, int i_max, int b_max);
}
