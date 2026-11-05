#pragma once

#include <utility>

#ifndef __cpp_lib_unreachable
namespace std
{
    [[noreturn]] static inline void unreachable()
    {
        __builtin_unreachable();
    }
}
#endif
