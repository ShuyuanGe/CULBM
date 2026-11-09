#pragma once

#include "core/types.hpp"

namespace culbm::sim::multi::mapping
{
    template <bool CrossDevice>
    struct DeviceView
    {
        ddf_t* const* src;
        ddf_t* dst;
        idx_t idx, n, ndx, pdx, ndy, pdy, ndz, pdz;
        nbr_t nbrndx, nbrpdx, nbrndy, nbrpdy, nbrndz, nbrpdz;

        template <int Dir, int DX, int DY, int DZ>
        constexpr const ddf_t* readAddress() const
        {
            nbr_t peer = 13;
            if constexpr (CrossDevice)
                peer += (DX < 0 ? nbrndx : DX > 0 ? nbrpdx : 0)
                      + (DY < 0 ? nbrndy : DY > 0 ? nbrpdy : 0)
                      + (DZ < 0 ? nbrndz : DZ > 0 ? nbrpdz : 0);

            const idx_t cell = idx
                + (DX < 0 ? ndx : DX > 0 ? pdx : 0)
                + (DY < 0 ? ndy : DY > 0 ? pdy : 0)
                + (DZ < 0 ? ndz : DZ > 0 ? pdz : 0);
            return src[peer] + static_cast<ddf_offset_t>(Dir) * n + cell;
        }

        template <int Dir, int DX, int DY, int DZ>
        constexpr ddf_t* writeAddress() const
        {
            return dst + (static_cast<ddf_offset_t>(Dir) * n + idx);
        }
    };
}
