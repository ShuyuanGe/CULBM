#pragma once

#include "math.hpp"

namespace gf::blocking_core
{
    template<std::integral T>
    constexpr bool validBlkAxisConfig(T domLen, T blkLen, T blkIter)
    {
        if(blkLen<domLen and blkLen+1<=blkIter) return false;
        if(2*(blkLen+1-blkIter)<domLen and blkLen+2<=blkIter) return false;
        return true;
    }

    template<std::integral T>
    constexpr T calcBlkNum(T domLen, T blkLen, T blkIter)
    {
        if(domLen <= blkLen)
        {
            return 1;
        }
        if(domLen<=2*(blkLen+1-blkIter))
        {
            return 2;
        }
        return 2+gf::basic::divCeil<T>(domLen-2*(blkLen+1-blkIter), blkLen+2-2*blkIter);
    }

    template<std::integral T>
    constexpr T calcValidPrev(T blkIdx, T blkLen, T blkNum, T blkIter, T domLen)
    {
        if(blkIdx==blkNum)
        {
            return domLen;
        }
        if(blkIdx==0)
        {
            return 0;
        }
        return (blkLen+1-blkIter)+(blkIdx-1)*(blkLen+2-2*blkIter);
    }
}