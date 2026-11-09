#pragma once

namespace culbm::tune
{
    struct Selection { int bx, by, bz, i; };

    Selection select(int q, double capacityBytes, double lambda, double gamma, int iMin, int iMax, int bMax);
}
