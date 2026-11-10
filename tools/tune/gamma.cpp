#include "report.hpp"

int main()
{
    culbm::tune::report(culbm::tune::measure_gamma(cuda::device_ref{0}));
}
