#include "report.hpp"

int main()
{
    culbm::tune::report(culbm::tune::measureGamma(cuda::device_ref{0}));
}
