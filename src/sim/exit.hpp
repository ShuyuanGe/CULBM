#pragma once

namespace culbm::sim::cli
{
    // CLI output is emitted before throwing; main returns the code after unwinding.
    struct Exit { int code; };
}
