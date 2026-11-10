#include "report.hpp"

int main()
{
    culbm::tune::report(culbm::tune::measure_lambda(cuda::device_ref{0}));
}
