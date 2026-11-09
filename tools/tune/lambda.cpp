#include "report.hpp"

int main()
{
    culbm::tune::report(culbm::tune::measureLambda(cuda::device_ref{0}));
}
