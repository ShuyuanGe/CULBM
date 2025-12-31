#pragma once

#include <cmath>
#include "config.hpp"
#include "defines.hpp"
#include "literal.hpp"
#include "velocity_set.hpp"

namespace culbm::lbm_core
{
    namespace detail
    {
        template<
            u32 NDIR, 
            u32 DIR,
            i32 DX = culbm::basic::detail::VelSet3D<NDIR>::template getDx<DIR>(), 
            i32 DY = culbm::basic::detail::VelSet3D<NDIR>::template getDy<DIR>(), 
            i32 DZ = culbm::basic::detail::VelSet3D<NDIR>::template getDz<DIR>()
        >
        constexpr i64 calcDirOff(i32 n, i32 ndx, i32 pdx, i32 ndy, i32 pdy, i32 ndz, i32 pdz, i32 nCell)
        {
            const i32 xOff = (DX==-1) ? ndx : (DX==1) ? pdx : 0;
            const i32 yOff = (DY==-1) ? ndy : (DY==1) ? pdy : 0;
            const i32 zOff = (DZ==-1) ? ndz : (DZ==1) ? pdz : 0;
            return DIR * static_cast<i64>(nCell) + n + xOff + yOff + zOff;
        }

        template<u32 NDIR, u32 DIR=0>
        HOST_DEV_CONSTEXPR void pullLoad(bool revLoad, i32 n, i32 ndx, i32 pdx, i32 ndy, i32 pdy, i32 ndz, i32 pdz, i32 nCell, real_t* fn, const ddf_t* srcBuf)
        {
            if constexpr (DIR < NDIR)
            {
                fn[revLoad ? NDIR-1-DIR : DIR] = static_cast<real_t>(srcBuf[calcDirOff<NDIR, DIR>(n, pdx, ndx, pdy, ndy, pdz, ndz, nCell)]);
            }
            
            if constexpr (DIR+1 < NDIR)
            {
                pullLoad<NDIR, DIR+1>(revLoad, n, ndx, pdx, ndy, pdy, ndz, pdz, nCell, fn, srcBuf);
            }
        }

        template<u32 NDIR, u32 DIR=0>
        HOST_DEV_CONSTEXPR void pushStore(bool revStore, i32 n, i32 ndx, i32 pdx, i32 ndy, i32 pdy, i32 ndz, i32 pdz, i32 nCell, const real_t* fn, ddf_t* dstBuf)
        {

            if constexpr (DIR < NDIR)
            {
                dstBuf[calcDirOff<NDIR, DIR>(n, ndx, pdx, ndy, pdy, ndz, pdz, nCell)] = static_cast<ddf_t>(fn[revStore ? NDIR-1-DIR : DIR]);
            }

            if constexpr (DIR+1 < NDIR)
            {
                pushStore<NDIR, DIR+1>(revStore, n, ndx, pdx, ndy, pdy, ndz, pdz, nCell, fn, dstBuf);
            }
        }

        template<u32 NDIR, u32 DIR=0>
        HOST_DEV_CONSTEXPR void load(bool revLoad, i32 n, i32 nCell, real_t* fn, const ddf_t* srcBuf)
        {
            if constexpr (DIR < NDIR)
            {
                fn[revLoad ? NDIR-1-DIR : DIR] = static_cast<real_t>(srcBuf[DIR * static_cast<idx_t>(nCell) + n]);
            }

            if constexpr (DIR+1 < NDIR)
            {
                load<NDIR, DIR+1>(revLoad, n, nCell, fn, srcBuf);
            }
        }

        template<u32 NDIR, u32 DIR=0>
        HOST_DEV_CONSTEXPR void store(bool revStore, i32 n, i32 nCell, const real_t* fn, ddf_t* dstBuf)
        {
            if constexpr (DIR < NDIR)
            {
                dstBuf[DIR * static_cast<idx_t>(nCell) + n] = static_cast<ddf_t>(fn[revStore ? NDIR-1-DIR : DIR]);
            }

            if constexpr (DIR+1 < NDIR)
            {
                store<NDIR, DIR+1>(revStore, n, nCell, fn, dstBuf);
            }
        }

        template<u32 NDIR>
        HOST_DEV_CONSTEXPR void pullLoadExpand(bool revLoad, i32 n, i32 ndx, i32 pdx, i32 ndy, i32 pdy, i32 ndz, i32 pdz, i32 nCell, real_t* fn, const ddf_t* srcBuf);

        template<>
        HOST_DEV_CONSTEXPR void pullLoadExpand<27>(bool revLoad, i32 n, i32 ndx, i32 pdx, i32 ndy, i32 pdy, i32 ndz, i32 pdz, i32 nCell, real_t* fn, const ddf_t* srcBuf)
        {
            fn[revLoad ? 26 : 0] = srcBuf[ 0*nCell+n+pdx+pdy+pdz];
            fn[revLoad ? 25 : 1] = srcBuf[ 1*nCell+n    +pdy+pdz];
            fn[revLoad ? 24 : 2] = srcBuf[ 2*nCell+n+ndx+pdy+pdz];
            fn[revLoad ? 23 : 3] = srcBuf[ 3*nCell+n+pdx    +pdz];
            fn[revLoad ? 22 : 4] = srcBuf[ 4*nCell+n        +pdz];
            fn[revLoad ? 21 : 5] = srcBuf[ 5*nCell+n+ndx    +pdz];
            fn[revLoad ? 20 : 6] = srcBuf[ 6*nCell+n+pdx+ndy+pdz];
            fn[revLoad ? 19 : 7] = srcBuf[ 7*nCell+n    +ndy+pdz];
            fn[revLoad ? 18 : 8] = srcBuf[ 8*nCell+n+ndx+ndy+pdz];

            fn[revLoad ? 17 : 9] = srcBuf[ 9*nCell+n+pdx+pdy    ];
            fn[revLoad ? 16 :10] = srcBuf[10*nCell+n    +pdy    ];
            fn[revLoad ? 15 :11] = srcBuf[11*nCell+n+ndx+pdy    ];
            fn[revLoad ? 14 :12] = srcBuf[12*nCell+n+pdx        ];
            fn[13]               = srcBuf[13*nCell+n            ];
            fn[revLoad ? 12 :14] = srcBuf[14*nCell+n+ndx        ];
            fn[revLoad ? 11 :15] = srcBuf[15*nCell+n+pdx+ndy    ];
            fn[revLoad ? 10 :16] = srcBuf[16*nCell+n    +ndy    ];
            fn[revLoad ?  9 :17] = srcBuf[17*nCell+n+ndx+ndy    ];

            fn[revLoad ?  8 :18] = srcBuf[18*nCell+n+pdx+pdy+ndz];
            fn[revLoad ?  7 :19] = srcBuf[19*nCell+n    +pdy+ndz];
            fn[revLoad ?  6 :20] = srcBuf[20*nCell+n+ndx+pdy+ndz];
            fn[revLoad ?  5 :21] = srcBuf[21*nCell+n+pdx    +ndz];
            fn[revLoad ?  4 :22] = srcBuf[22*nCell+n        +ndz];
            fn[revLoad ?  3 :23] = srcBuf[23*nCell+n+ndx    +ndz];
            fn[revLoad ?  2 :24] = srcBuf[24*nCell+n+pdx+ndy+ndz];
            fn[revLoad ?  1 :25] = srcBuf[25*nCell+n    +ndy+ndz];
            fn[revLoad ?  0 :26] = srcBuf[26*nCell+n+ndx+ndy+ndz];
        }
    }

    template<u32 NDIR>
    HOST_DEV_CONSTEXPR void pullLoad(bool revLoad, i32 n, i32 ndx, i32 pdx, i32 ndy, i32 pdy, i32 ndz, i32 pdz, i32 nCell, real_t* fn, const ddf_t* srcBuf)
    {
        detail::pullLoad<NDIR, 0>(revLoad, n, ndx, pdx, ndy, pdy, ndz, pdz, nCell, fn, srcBuf);
        //detail::pullLoadExpand<NDIR>(revLoad, n, ndx, pdx, ndy, pdy, ndz, pdz, nCell, fn, srcBuf);
    }

    template<u32 NDIR>
    HOST_DEV_CONSTEXPR void pushStore(bool revStore, i32 n, i32 ndx, i32 pdx, i32 ndy, i32 pdy, i32 ndz, i32 pdz, i32 nCell, const real_t* fn, ddf_t* dstBuf)
    {
        detail::pushStore<NDIR, 0>(revStore, n, ndx, pdx, ndy, pdy, ndz, pdz, nCell, fn, dstBuf);
    }

    template<u32 NDIR>
    HOST_DEV_CONSTEXPR void load(bool revLoad, i32 n, i32 nCell, real_t* fn, const ddf_t* srcBuf)
    {
        detail::load<NDIR, 0>(revLoad, n, nCell, fn, srcBuf);
    }

    template<u32 NDIR>
    HOST_DEV_CONSTEXPR void store(bool revStore, i32 n, i32 nCell, const real_t* fn, ddf_t* dstBuf)
    {
        detail::store<NDIR, 0>(revStore, n, nCell, fn, dstBuf);
    }

    namespace bgk
    {
        namespace detail
        {
            template<u32 NDIR>
            HOST_DEV_CONSTEXPR void calcEqu(real_t rho, real_t vx, real_t vy, real_t vz, real_t* f);

            template<u32 NDIR>
            HOST_DEV_CONSTEXPR void calcRhoU(real_t& rho, real_t& vx, real_t& vy, real_t& vz, const real_t* fn);

            template<u32 NDIR>
            HOST_DEV_CONSTEXPR void collision(real_t invTau, real_t& rho, real_t& vx, real_t& vy, real_t& vz, real_t* f);

            template<u32 NDIR>
            HOST_DEV_CONSTEXPR void collision2(real_t invTau, real_t* f);

            template<>
            HOST_DEV_CONSTEXPR void calcEqu<15>(real_t rho, real_t vx, real_t vy, real_t vz, real_t* f)
            {
                using std::fma;
                using namespace culbm::literal;

                constexpr real_t W0 = 2._r / 9._r;  //center
                constexpr real_t WS = 1._r / 9._r;  //straight
                constexpr real_t WC = 1._r / 72._r; //corner

                const real_t rhom1 = rho - 1._r;
                const real_t c3    = -3._r * (vx * vx + vy * vy + vz * vz);
                vx *= 3._r;
                vy *= 3._r;
                vz *= 3._r;

                /*
                    const float u0=ux+uy+uz, u1=ux+uy-uz, u2=ux-uy+uz, u3=-ux+uy+uz;
                    const float rhos=def_ws*rho, rhoc=def_wc*rho, rhom1s=def_ws*rhom1, rhom1c=def_wc*rhom1;
                    feq[ 1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[ 2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
                    feq[ 3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[ 4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
                    feq[ 5] = fma(rhos, fma(0.5f, fma(uz, uz, c3), uz), rhom1s); feq[ 6] = fma(rhos, fma(0.5f, fma(uz, uz, c3), -uz), rhom1s); // 00+ 00-
                    feq[ 7] = fma(rhoc, fma(0.5f, fma(u0, u0, c3), u0), rhom1c); feq[ 8] = fma(rhoc, fma(0.5f, fma(u0, u0, c3), -u0), rhom1c); // +++ ---
                    feq[ 9] = fma(rhoc, fma(0.5f, fma(u1, u1, c3), u1), rhom1c); feq[10] = fma(rhoc, fma(0.5f, fma(u1, u1, c3), -u1), rhom1c); // ++- --+
                    feq[11] = fma(rhoc, fma(0.5f, fma(u2, u2, c3), u2), rhom1c); feq[12] = fma(rhoc, fma(0.5f, fma(u2, u2, c3), -u2), rhom1c); // +-+ -+-
                    feq[13] = fma(rhoc, fma(0.5f, fma(u3, u3, c3), u3), rhom1c); feq[14] = fma(rhoc, fma(0.5f, fma(u3, u3, c3), -u3), rhom1c); // -++ +--
                */

                const real_t v0 = vx+vy+vz, v1 = vx+vy-vz, v2 = vx-vy+vz, v3 = vy-vx+vz;
                const real_t rhos = WS*rho, rhoc = WC*rho, rhom1s = WS*rhom1, rhom1c = WC*rhom1;

                //x:-,y:-,z:-
                f[ 0] = fma(rhoc, fma(0.5_r, fma(v0, v0, c3), -v0), rhom1c);
                //x:+,y:-,z:-
                f[ 1] = fma(rhoc, fma(0.5_r, fma(v3, v3, c3), -v3), rhom1c);
                //x:0,y:0,z:-
                f[ 2] = fma(rhos, fma(0.5_r, fma(vz, vz, c3), -vz), rhom1s);
                //x:-,y:+,z:-
                f[ 3] = fma(rhoc, fma(0.5_r, fma(v2, v2, c3), -v2), rhom1c); 
                //x:+,y:+,z:-
                f[ 4] = fma(rhoc, fma(0.5_r, fma(v1, v1, c3), v1), rhom1c);

                //x:0,y:-,z:0
                f[ 5] = fma(rhos, fma(0.5_r, fma(vy, vy, c3), -vy), rhom1s);
                //x:-,y:0,z:0
                f[ 6] = fma(rhos, fma(0.5_r, fma(vx, vx, c3), -vx), rhom1s); 
                //x:0,y:0,z:0
                f[ 7] = W0*fma(rho, 0.5_r*c3, rhom1);
                //x:+,y:0,z:0
                f[ 8] = fma(rhos, fma(0.5_r, fma(vx, vx, c3), vx), rhom1s);
                //x:0,y:+,z:0
                f[ 9] = fma(rhos, fma(0.5_r, fma(vy, vy, c3), vy), rhom1s);

                //x:-,y:-,z:+
                f[10] = fma(rhoc, fma(0.5_r, fma(v1, v1, c3), -v1), rhom1c);
                //x:+,y:-,z:+
                f[11] = fma(rhoc, fma(0.5_r, fma(v2, v2, c3), v2), rhom1c);
                //x:0,y:0,z:+
                f[12] = fma(rhos, fma(0.5_r, fma(vz, vz, c3), vz), rhom1s);
                //x:-,y:+,z:+
                f[13] = fma(rhoc, fma(0.5_r, fma(v3, v3, c3), v3), rhom1c);
                //x:+,y:+,z:+
                f[14] = fma(rhoc, fma(0.5_r, fma(v0, v0, c3), v0), rhom1c);
            }

            template<>
            HOST_DEV_CONSTEXPR void calcEqu<19>(real_t rho, real_t vx, real_t vy, real_t vz, real_t* f)
            {
                using std::fma;
                using namespace culbm::literal;

                constexpr real_t W0 = 1._r / 3._r;  //center
                constexpr real_t WS = 1._r / 18._r; //straight
                constexpr real_t WE = 1._r / 36._r; //edge

                const real_t rhom1 = rho - 1._r;
                const real_t c3    = -3._r * (vx * vx + vy * vy + vz * vz);
                vx *= 3._r;
                vy *= 3._r;
                vz *= 3._r;

                /*
                    const float u0=ux+uy, u1=ux+uz, u2=uy+uz, u3=ux-uy, u4=ux-uz, u5=uy-uz;
                    const float rhos=def_ws*rho, rhoe=def_we*rho, rhom1s=def_ws*rhom1, rhom1e=def_we*rhom1;
                    feq[ 1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[ 2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
                    feq[ 3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[ 4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
                    feq[ 5] = fma(rhos, fma(0.5f, fma(uz, uz, c3), uz), rhom1s); feq[ 6] = fma(rhos, fma(0.5f, fma(uz, uz, c3), -uz), rhom1s); // 00+ 00-
                    feq[ 7] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), u0), rhom1e); feq[ 8] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), -u0), rhom1e); // ++0 --0
                    feq[ 9] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), u1), rhom1e); feq[10] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), -u1), rhom1e); // +0+ -0-
                    feq[11] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), u2), rhom1e); feq[12] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), -u2), rhom1e); // 0++ 0--
                    feq[13] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), u3), rhom1e); feq[14] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), -u3), rhom1e); // +-0 -+0
                    feq[15] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), u4), rhom1e); feq[16] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), -u4), rhom1e); // +0- -0+
                    feq[17] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), u5), rhom1e); feq[18] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), -u5), rhom1e); // 0+- 0-+
                */

                const real_t v0 = vx+vy, v1 = vx+vz, v2 =vy+vz, v3 = vx-vy, v4 = vx-vz, v5 = vy-vz;
                const real_t rhos = WS*rho, rhoe = WE*rho, rhom1s = WS*rhom1, rhom1e = WE*rhom1;

                //x:0,y:-,z:-
                f[ 0] = fma(rhoe, fma(0.5_r, fma(v2, v2, c3), -v2), rhom1e);
                //x:-,y:0,z:-
                f[ 1] = fma(rhoe, fma(0.5_r, fma(v1, v1, c3), -v1), rhom1e);
                //x:0,y:0,z:-
                f[ 2] = fma(rhos, fma(0.5_r, fma(vz, vz, c3), -vz), rhom1s);
                //x:+,y:0,z:-
                f[ 3] = fma(rhoe, fma(0.5_r, fma(v4, v4, c3), v4), rhom1e);
                //x:0,y:+,z:-
                f[ 4] = fma(rhoe, fma(0.5_r, fma(v5, v5, c3), v5), rhom1e);

                //x:-,y:-,z:0
                f[ 5] = fma(rhoe, fma(0.5_r, fma(v0, v0, c3), -v0), rhom1e);
                //x:0,y:-,z:0
                f[ 6] = fma(rhos, fma(0.5_r, fma(vy, vy, c3), -vy), rhom1s);
                //x:+,y:-,z:0
                f[ 7] = fma(rhoe, fma(0.5_r, fma(v3, v3, c3), v3), rhom1e);
                //x:-,y:0,z:0
                f[ 8] = fma(rhos, fma(0.5_r, fma(vx, vx, c3), -vx), rhom1s);
                //x:0,y:0,z:0
                f[ 9] = W0*fma(rho, 0.5_r*c3, rhom1);
                //x:+,y:0,z:0
                f[10] = fma(rhos, fma(0.5_r, fma(vx, vx, c3), vx), rhom1s);
                //x:-,y:+,z:0
                f[11] = fma(rhoe, fma(0.5_r, fma(v3, v3, c3), -v3), rhom1e);
                //x:0,y:+,z:0
                f[12] = fma(rhos, fma(0.5_r, fma(vy, vy, c3), vy), rhom1s);
                //x:+,y:+,z:0
                f[13] = fma(rhoe, fma(0.5_r, fma(v0, v0, c3), v0), rhom1e);

                //x:0,y:-,z:+
                f[14] = fma(rhoe, fma(0.5_r, fma(v5, v5, c3), -v5), rhom1e);
                //x:-,y:0,z:+
                f[15] = fma(rhoe, fma(0.5_r, fma(v4, v4, c3), -v4), rhom1e);
                //x:0,y:0,z:+
                f[16] = fma(rhos, fma(0.5_r, fma(vz, vz, c3), vz), rhom1s);
                //x:+,y:0,z:+
                f[17] = fma(rhoe, fma(0.5_r, fma(v1, v1, c3), v1), rhom1e);
                //x:0,y:+,z:+
                f[18] = fma(rhoe, fma(0.5_r, fma(v2, v2, c3), v2), rhom1e);
            }

            template<>
            HOST_DEV_CONSTEXPR void calcEqu<27>(real_t rho, real_t vx, real_t vy, real_t vz, real_t* f)
            {
                using std::fma;
                using namespace culbm::literal;

                constexpr real_t W0 = 8._r / 27._r; //center
                constexpr real_t WS = 2._r / 27._r; //straight
                constexpr real_t WE = 1._r / 54._r; //edge
                constexpr real_t WC = 1._r / 216._r;//corner

                const real_t rhom1 = rho - 1._r;
                const real_t c3    = -3._r * (vx * vx + vy * vy + vz * vz);
                vx *= 3._r;
                vy *= 3._r;
                vz *= 3._r;

                /*
                    const float u0=ux+uy, u1=ux+uz, u2=uy+uz, u3=ux-uy, u4=ux-uz, u5=uy-uz, u6=ux+uy+uz, u7=ux+uy-uz, u8=ux-uy+uz, u9=-ux+uy+uz;
                    const float rhos=def_ws*rho, rhoe=def_we*rho, rhoc=def_wc*rho, rhom1s=def_ws*rhom1, rhom1e=def_we*rhom1, rhom1c=def_wc*rhom1;
                    feq[ 1] = fma(rhos, fma(0.5f, fma(ux, ux, c3), ux), rhom1s); feq[ 2] = fma(rhos, fma(0.5f, fma(ux, ux, c3), -ux), rhom1s); // +00 -00
                    feq[ 3] = fma(rhos, fma(0.5f, fma(uy, uy, c3), uy), rhom1s); feq[ 4] = fma(rhos, fma(0.5f, fma(uy, uy, c3), -uy), rhom1s); // 0+0 0-0
                    feq[ 5] = fma(rhos, fma(0.5f, fma(uz, uz, c3), uz), rhom1s); feq[ 6] = fma(rhos, fma(0.5f, fma(uz, uz, c3), -uz), rhom1s); // 00+ 00-
                    feq[ 7] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), u0), rhom1e); feq[ 8] = fma(rhoe, fma(0.5f, fma(u0, u0, c3), -u0), rhom1e); // ++0 --0
                    feq[ 9] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), u1), rhom1e); feq[10] = fma(rhoe, fma(0.5f, fma(u1, u1, c3), -u1), rhom1e); // +0+ -0-
                    feq[11] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), u2), rhom1e); feq[12] = fma(rhoe, fma(0.5f, fma(u2, u2, c3), -u2), rhom1e); // 0++ 0--
                    feq[13] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), u3), rhom1e); feq[14] = fma(rhoe, fma(0.5f, fma(u3, u3, c3), -u3), rhom1e); // +-0 -+0
                    feq[15] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), u4), rhom1e); feq[16] = fma(rhoe, fma(0.5f, fma(u4, u4, c3), -u4), rhom1e); // +0- -0+
                    feq[17] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), u5), rhom1e); feq[18] = fma(rhoe, fma(0.5f, fma(u5, u5, c3), -u5), rhom1e); // 0+- 0-+
                    feq[19] = fma(rhoc, fma(0.5f, fma(u6, u6, c3), u6), rhom1c); feq[20] = fma(rhoc, fma(0.5f, fma(u6, u6, c3), -u6), rhom1c); // +++ ---
                    feq[21] = fma(rhoc, fma(0.5f, fma(u7, u7, c3), u7), rhom1c); feq[22] = fma(rhoc, fma(0.5f, fma(u7, u7, c3), -u7), rhom1c); // ++- --+
                    feq[23] = fma(rhoc, fma(0.5f, fma(u8, u8, c3), u8), rhom1c); feq[24] = fma(rhoc, fma(0.5f, fma(u8, u8, c3), -u8), rhom1c); // +-+ -+-
                    feq[25] = fma(rhoc, fma(0.5f, fma(u9, u9, c3), u9), rhom1c); feq[26] = fma(rhoc, fma(0.5f, fma(u9, u9, c3), -u9), rhom1c); // -++ +--
                */

                const real_t v0=vx+vy, v1=vx+vz, v2=vy+vz, v3=vx-vy, v4=vx-vz, v5=vy-vz, v6=vx+vy+vz, v7=vx+vy-vz, v8=vx-vy+vz, v9=vy-vx+vz;
                const real_t rhos=WS*rho, rhoe=WE*rho, rhoc=WC*rho, rhom1s=WS*rhom1, rhom1e=WE*rhom1, rhom1c=WC*rhom1;

                //x:-,y:-,z:-
                f[ 0] = fma(rhoc, fma(0.5_r, fma(v6, v6, c3), -v6), rhom1c);
                //x:0,y:-,z:-
                f[ 1] = fma(rhoe, fma(0.5_r, fma(v2, v2, c3), -v2), rhom1e);
                //x:+,y:-,z:-
                f[ 2] = fma(rhoc, fma(0.5_r, fma(v9, v9, c3), -v9), rhom1c);
                //x:-,y:0,z:-
                f[ 3] = fma(rhoe, fma(0.5_r, fma(v1, v1, c3), -v1), rhom1e);
                //x:0,y:0,z:-
                f[ 4] = fma(rhos, fma(0.5_r, fma(vz, vz, c3), -vz), rhom1s);
                //x:+,y:0,z:-
                f[ 5] = fma(rhoe, fma(0.5_r, fma(v4, v4, c3), v4), rhom1e);
                //x:-,y:+,z:-
                f[ 6] = fma(rhoc, fma(0.5_r, fma(v8, v8, c3), -v8), rhom1c);
                //x:0,y:+,z:-
                f[ 7] = fma(rhoe, fma(0.5_r, fma(v5, v5, c3), v5), rhom1e);
                //x:+,y:+,z:-
                f[ 8] = fma(rhoc, fma(0.5_r, fma(v7, v7, c3), v7), rhom1c);

                //x:-,y:-,z:0
                f[ 9] = fma(rhoe, fma(0.5_r, fma(v0, v0, c3), -v0), rhom1e);
                //x:0,y:-,z:0
                f[10] = fma(rhos, fma(0.5_r, fma(vy, vy, c3), -vy), rhom1s);
                //x:+,y:-,z:0
                f[11] = fma(rhoe, fma(0.5_r, fma(v3, v3, c3), v3), rhom1e);
                //x:-,y:0,z:0
                f[12] = fma(rhos, fma(0.5_r, fma(vx, vx, c3), -vx), rhom1s);
                //x:0,y:0,z:0
                f[13] = W0*fma(rho, 0.5_r*c3, rhom1);
                //x:+,y:0,z:0
                f[14] = fma(rhos, fma(0.5_r, fma(vx, vx, c3), vx), rhom1s);
                //x:-,y:+,z:0
                f[15] = fma(rhoe, fma(0.5_r, fma(v3, v3, c3), -v3), rhom1e);
                //x:0,y:+,z:0
                f[16] = fma(rhos, fma(0.5_r, fma(vy, vy, c3), vy), rhom1s);
                //x:+,y:+,z:0
                f[17] = fma(rhoe, fma(0.5_r, fma(v0, v0, c3), v0), rhom1e);

                //x:-,y:-,z:+
                f[18] = fma(rhoc, fma(0.5f, fma(v7, v7, c3), -v7), rhom1c);
                //x:0,y:-,z:+
                f[19] = fma(rhoe, fma(0.5f, fma(v5, v5, c3), -v5), rhom1e);
                //x:+,y:-,z:+
                f[20] = fma(rhoc, fma(0.5f, fma(v8, v8, c3), v8), rhom1c);
                //x:-,y:0,z:+
                f[21] = fma(rhoe, fma(0.5f, fma(v4, v4, c3), -v4), rhom1e);
                //x:0,y:0,z:+
                f[22] = fma(rhos, fma(0.5f, fma(vz, vz, c3), vz), rhom1s);
                //x:+,y:0,z:+
                f[23] = fma(rhoe, fma(0.5f, fma(v1, v1, c3), v1), rhom1e);
                //x:-,y:+,z:+
                f[24] = fma(rhoc, fma(0.5f, fma(v9, v9, c3), v9), rhom1c);
                //x:0,y:+,z:+
                f[25] = fma(rhoe, fma(0.5f, fma(v2, v2, c3), v2), rhom1e);
                //x:+,y:+,z:+
                f[26] = fma(rhoc, fma(0.5f, fma(v6, v6, c3), v6), rhom1c);
            }

            template<>
            HOST_DEV_CONSTEXPR void calcRhoU<15>(real_t& rho, real_t& vx, real_t& vy, real_t& vz, const real_t* f)
            {
                using namespace culbm::literal;

                rho = (1._r + f[ 0]) + (f[ 1] + f[ 2]) + (f[ 3] + f[ 4]) + (f[ 5] + f[ 6]) +
                     (f[ 7] + f[ 8]) + (f[ 9] + f[10]) + (f[11] + f[12]) + (f[13] + f[14]);

                const real_t invRho = 1._r / rho;

                vx  = (f[ 1] - f[ 0]) + (f[ 4] - f[ 3]) + (f[ 8] - f[ 6]) + (f[11] - f[10]) + (f[14] - f[13]);
                vy  = (f[ 3] - f[ 0]) + (f[ 4] - f[ 1]) + (f[ 9] - f[ 5]) + (f[13] - f[10]) + (f[14] - f[11]);
                vz  = (f[10] - f[ 0]) + (f[11] - f[ 1]) + (f[12] - f[ 2]) + (f[13] - f[ 3]) + (f[14] - f[ 4]);

                vx *= invRho;
                vy *= invRho;
                vz *= invRho;
            }

            template<>
            HOST_DEV_CONSTEXPR void calcRhoU<19>(real_t& rho, real_t& vx, real_t& vy, real_t& vz, const real_t* f)
            {
                using namespace culbm::literal;

                rho = (1._r + f[ 0]) + (f[ 1] + f[ 2]) + (f[ 3] + f[ 4]) + (f[ 5] + f[ 6]) + (f[ 7] + f[ 8]) + 
                     (f[ 9] + f[10]) + (f[11] + f[12]) + (f[13] + f[14]) + (f[15] + f[16]) + (f[17] + f[18]);

                const real_t invRho = 1._r / rho;

                vx = (f[ 3] - f[ 1]) + (f[ 7] - f[ 5]) + (f[10] - f[ 8]) + (f[13] - f[11]) + (f[17] - f[15]);
                vy = (f[ 4] - f[ 0]) + (f[11] - f[ 5]) + (f[12] - f[ 6]) + (f[13] - f[ 7]) + (f[18] - f[14]);
                vz = (f[14] - f[ 0]) + (f[15] - f[ 1]) + (f[16] - f[ 2]) + (f[17] - f[ 3]) + (f[18] - f[ 4]);

                vx *= invRho;
                vy *= invRho;
                vz *= invRho;
            }

            template<>
            HOST_DEV_CONSTEXPR void calcRhoU<27>(real_t& rho, real_t& vx, real_t& vy, real_t& vz, const real_t* f)
            {
                using namespace culbm::literal;

                rho = (1._r + f[ 0]) + (f[ 1] + f[ 2]) + (f[ 3] + f[ 4]) + (f[ 5] + f[ 6]) + (f[ 7] + f[ 8]) + (f[ 9] + f[10]) + (f[11] + f[12]) +
                     (f[13] + f[14]) + (f[15] + f[16]) + (f[17] + f[18]) + (f[19] + f[20]) + (f[21] + f[22]) + (f[23] + f[24]) + (f[25] + f[26]);

                const real_t invRho = 1._r / rho;

                vx = (f[ 2] - f[ 0]) + (f[ 5] - f[ 3]) + (f[ 8] - f[ 6]) +
                     (f[11] - f[ 9]) + (f[14] - f[12]) + (f[17] - f[15]) +
                     (f[20] - f[18]) + (f[23] - f[21]) + (f[26] - f[24]);

                vy = (f[ 6] - f[ 0]) + (f[ 7] - f[ 1]) + (f[ 8] - f[ 2]) +
                     (f[15] - f[ 9]) + (f[16] - f[10]) + (f[17] - f[11]) +
                     (f[24] - f[18]) + (f[25] - f[19]) + (f[26] - f[20]);

                vz = (f[18] - f[ 0]) + (f[19] - f[ 1]) + (f[20] - f[ 2]) +
                     (f[21] - f[ 3]) + (f[22] - f[ 4]) + (f[23] - f[ 5]) +
                     (f[24] - f[ 6]) + (f[25] - f[ 7]) + (f[26] - f[ 8]);

                vx *= invRho;
                vy *= invRho;
                vz *= invRho;
            }

            template<>
            HOST_DEV_CONSTEXPR void collision<15>(real_t invTau, real_t& rho, real_t& vx, real_t& vy, real_t& vz, real_t* f)
            {
                using std::fma;
                using namespace culbm::literal;

                real_t vx_, vy_, vz_;
                calcRhoU<15>(rho, vx_, vy_, vz_, f);
                vx = vx_;
                vy = vy_;
                vz = vz_;

                constexpr real_t W0 = 2._r / 9._r;  //center
                constexpr real_t WS = 1._r / 9._r;  //straight
                constexpr real_t WC = 1._r / 72._r; //corner

                const real_t rhom1 = rho - 1._r;
                const real_t c3    = -3._r * (vx_ * vx_ + vy_ * vy_ + vz_ * vz_);
                vx_ *= 3._r;
                vy_ *= 3._r;
                vz_ *= 3._r;

                const real_t v0 = vx_+vy_+vz_, v1 = vx_+vy_-vz_, v2 = vx_-vy_+vz_, v3 = vy_-vx_+vz_;
                const real_t rhos = WS*rho, rhoc = WC*rho, rhom1s = WS*rhom1, rhom1c = WC*rhom1;

                //x:-,y:-,z:-
                f[ 0] = invTau * fma(rhoc, fma(0.5_r, fma(v0, v0, c3), -v0), rhom1c) + (1._r - invTau) * f[ 0];
                //x:+,y:-,z:-
                f[ 1] = invTau * fma(rhoc, fma(0.5_r, fma(v3, v3, c3), -v3), rhom1c) + (1._r - invTau) * f[ 1];
                //x:0,y:0,z:-
                f[ 2] = invTau * fma(rhos, fma(0.5_r, fma(vz_, vz_, c3), -vz_), rhom1s) + (1._r - invTau) * f[ 2];
                //x:-,y:+,z:-
                f[ 3] = invTau * fma(rhoc, fma(0.5_r, fma(v2, v2, c3), -v2), rhom1c) + (1._r - invTau) * f[ 3]; 
                //x:+,y:+,z:-
                f[ 4] = invTau * fma(rhoc, fma(0.5_r, fma(v1, v1, c3), v1), rhom1c) + (1._r - invTau) * f[ 4];

                //x:0,y:-,z:0
                f[ 5] = invTau * fma(rhos, fma(0.5_r, fma(vy_, vy_, c3), -vy_), rhom1s) + (1._r - invTau) * f[ 5];
                //x:-,y:0,z:0
                f[ 6] = invTau * fma(rhos, fma(0.5_r, fma(vx_, vx_, c3), -vx_), rhom1s) + (1._r - invTau) * f[ 6]; 
                //x:0,y:0,z:0
                f[ 7] = invTau * W0*fma(rho, 0.5_r*c3, rhom1) + (1._r - invTau) * f[ 7];
                //x:+,y:0,z:0
                f[ 8] = invTau * fma(rhos, fma(0.5_r, fma(vx_, vx_, c3), vx_), rhom1s) + (1._r - invTau) * f[ 8];
                //x:0,y:+,z:0
                f[ 9] = invTau * fma(rhos, fma(0.5_r, fma(vy_, vy_, c3), vy_), rhom1s) + (1._r - invTau) * f[ 9];

                //x:-,y:-,z:+
                f[10] = invTau * fma(rhoc, fma(0.5_r, fma(v1, v1, c3), -v1), rhom1c) + (1._r - invTau) * f[10];
                //x:+,y:-,z:+
                f[11] = invTau * fma(rhoc, fma(0.5_r, fma(v2, v2, c3), v2), rhom1c) + (1._r - invTau) * f[11];
                //x:0,y:0,z:+
                f[12] = invTau * fma(rhos, fma(0.5_r, fma(vz_, vz_, c3), vz_), rhom1s) + (1._r - invTau) * f[12];
                //x:-,y:+,z:+
                f[13] = invTau * fma(rhoc, fma(0.5_r, fma(v3, v3, c3), v3), rhom1c) + (1._r - invTau) * f[13];
                //x:+,y:+,z:+
                f[14] = invTau * fma(rhoc, fma(0.5_r, fma(v0, v0, c3), v0), rhom1c) + (1._r - invTau) * f[14];
            }

            template<>
            HOST_DEV_CONSTEXPR void collision<19>(real_t invTau, real_t& rho, real_t& vx, real_t& vy, real_t& vz, real_t* f)
            {
                using std::fma;
                using namespace culbm::literal;

                real_t vx_, vy_, vz_;
                calcRhoU<19>(rho, vx_, vy_, vz_, f);
                vx = vx_;
                vy = vy_;
                vz = vz_;

                constexpr real_t W0 = 1._r / 3._r;  //center
                constexpr real_t WS = 1._r / 18._r; //straight
                constexpr real_t WE = 1._r / 36._r; //edge

                const real_t rhom1 = rho - 1._r;
                const real_t c3    = -3._r * (vx_ * vx_ + vy_ * vy_ + vz_ * vz_);
                vx_ *= 3._r;
                vy_ *= 3._r;
                vz_ *= 3._r;

                const real_t v0 = vx_+vy_, v1 = vx_+vz_, v2 =vy_+vz_, v3 = vx_-vy_, v4 = vx_-vz_, v5 = vy_-vz_;
                const real_t rhos = WS*rho, rhoe = WE*rho, rhom1s = WS*rhom1, rhom1e = WE*rhom1;

                //x:0,y:-,z:-
                f[ 0] = invTau * fma(rhoe, fma(0.5_r, fma(v2, v2, c3), -v2), rhom1e) + (1._r - invTau) * f[ 0];
                //x:-,y:0,z:-
                f[ 1] = invTau * fma(rhoe, fma(0.5_r, fma(v1, v1, c3), -v1), rhom1e) + (1._r - invTau) * f[ 1];
                //x:0,y:0,z:-
                f[ 2] = invTau * fma(rhos, fma(0.5_r, fma(vz_, vz_, c3), -vz_), rhom1s) + (1._r - invTau) * f[ 2];
                //x:+,y:0,z:-
                f[ 3] = invTau * fma(rhoe, fma(0.5_r, fma(v4, v4, c3), v4), rhom1e) + (1._r - invTau) * f[ 3];
                //x:0,y:+,z:-
                f[ 4] = invTau * fma(rhoe, fma(0.5_r, fma(v5, v5, c3), v5), rhom1e) + (1._r - invTau) * f[ 4];

                //x:-,y:-,z:0
                f[ 5] = invTau * fma(rhoe, fma(0.5_r, fma(v0, v0, c3), -v0), rhom1e) + (1._r - invTau) * f[ 5];
                //x:0,y:-,z:0
                f[ 6] = invTau * fma(rhos, fma(0.5_r, fma(vy_, vy_, c3), -vy_), rhom1s) + (1._r - invTau) * f[ 6];
                //x:+,y:-,z:0
                f[ 7] = invTau * fma(rhoe, fma(0.5_r, fma(v3, v3, c3), v3), rhom1e) + (1._r - invTau) * f[ 7];
                //x:-,y:0,z:0
                f[ 8] = invTau * fma(rhos, fma(0.5_r, fma(vx_, vx_, c3), -vx_), rhom1s) + (1._r - invTau) * f[ 8];
                //x:0,y:0,z:0
                f[ 9] = invTau * W0*fma(rho, 0.5_r*c3, rhom1) + (1._r - invTau) * f[ 9];
                //x:+,y:0,z:0
                f[10] = invTau * fma(rhos, fma(0.5_r, fma(vx_, vx_, c3), vx_), rhom1s) + (1._r - invTau) * f[10];
                //x:-,y:+,z:0
                f[11] = invTau * fma(rhoe, fma(0.5_r, fma(v3, v3, c3), -v3), rhom1e) + (1._r - invTau) * f[11];
                //x:0,y:+,z:0
                f[12] = invTau * fma(rhos, fma(0.5_r, fma(vy_, vy_, c3), vy_), rhom1s) + (1._r - invTau) * f[12];
                //x:+,y:+,z:0
                f[13] = invTau * fma(rhoe, fma(0.5_r, fma(v0, v0, c3), v0), rhom1e) + (1._r - invTau) * f[13];

                //x:0,y:-,z:+
                f[14] = invTau * fma(rhoe, fma(0.5_r, fma(v5, v5, c3), -v5), rhom1e) + (1._r - invTau)* f[14];
                //x:-,y:0,z:+
                f[15] = invTau * fma(rhoe, fma(0.5_r, fma(v4, v4, c3), -v4), rhom1e) + (1._r - invTau) * f[15];
                //x:0,y:0,z:+
                f[16] = invTau * fma(rhos, fma(0.5_r, fma(vz_, vz_, c3), vz_), rhom1s) + (1._r - invTau) * f[16];
                //x:+,y:0,z:+
                f[17] = invTau * fma(rhoe, fma(0.5_r, fma(v1, v1, c3), v1), rhom1e) + (1._r - invTau) * f[17];
                //x:0,y:+,z:+
                f[18] = invTau * fma(rhoe, fma(0.5_r, fma(v2, v2, c3), v2), rhom1e) + (1._r - invTau) * f[18];                
            }

            template<>
            HOST_DEV_CONSTEXPR void collision<27>(real_t invTau, real_t& rho, real_t& vx, real_t& vy, real_t& vz, real_t* f)
            {
                using std::fma;
                using namespace culbm::literal;

                real_t vx_, vy_, vz_;
                calcRhoU<27>(rho, vx_, vy_, vz_, f);
                vx = vx_;
                vy = vy_;
                vz = vz_;

                constexpr real_t W0 = 8._r / 27._r; //center
                constexpr real_t WS = 2._r / 27._r; //straight
                constexpr real_t WE = 1._r / 54._r; //edge
                constexpr real_t WC = 1._r / 216._r;//corner

                const real_t rhom1 = rho - 1._r;
                const real_t c3    = -3._r * (vx_ * vx_ + vy_ * vy_ + vz_ * vz_);
                vx_ *= 3._r;
                vy_ *= 3._r;
                vz_ *= 3._r;

                const real_t v0=vx_+vy_, v1=vx_+vz_, v2=vy_+vz_, v3=vx_-vy_, v4=vx_-vz_, v5=vy_-vz_, v6=vx_+vy_+vz_, v7=vx_+vy_-vz_, v8=vx_-vy_+vz_, v9=vy_-vx_+vz_;
                const real_t rhos=WS*rho, rhoe=WE*rho, rhoc=WC*rho, rhom1s=WS*rhom1, rhom1e=WE*rhom1, rhom1c=WC*rhom1;

                //x:-,y:-,z:-
                f[ 0] = invTau * fma(rhoc, fma(0.5_r, fma(v6, v6, c3), -v6), rhom1c) + (1._r - invTau) * f[ 0];
                //x:0,y:-,z:-
                f[ 1] = invTau * fma(rhoe, fma(0.5_r, fma(v2, v2, c3), -v2), rhom1e) + (1._r - invTau) * f[ 1];
                //x:+,y:-,z:-
                f[ 2] = invTau * fma(rhoc, fma(0.5_r, fma(v9, v9, c3), -v9), rhom1c) + (1._r - invTau) * f[ 2];
                //x:-,y:0,z:-
                f[ 3] = invTau * fma(rhoe, fma(0.5_r, fma(v1, v1, c3), -v1), rhom1e) + (1._r - invTau) * f[ 3];
                //x:0,y:0,z:-
                f[ 4] = invTau * fma(rhos, fma(0.5_r, fma(vz_, vz_, c3), -vz_), rhom1s) + (1._r - invTau) * f[ 4];
                //x:+,y:0,z:-
                f[ 5] = invTau * fma(rhoe, fma(0.5_r, fma(v4, v4, c3), v4), rhom1e) + (1._r - invTau) * f[ 5];
                //x:-,y:+,z:-
                f[ 6] = invTau * fma(rhoc, fma(0.5_r, fma(v8, v8, c3), -v8), rhom1c) + (1._r - invTau) * f[ 6];
                //x:0,y:+,z:-
                f[ 7] = invTau * fma(rhoe, fma(0.5_r, fma(v5, v5, c3), v5), rhom1e) + (1._r - invTau) * f[ 7];
                //x:+,y:+,z:-
                f[ 8] = invTau * fma(rhoc, fma(0.5_r, fma(v7, v7, c3), v7), rhom1c) + (1._r - invTau) * f[ 8];

                //x:-,y:-,z:0
                f[ 9] = invTau * fma(rhoe, fma(0.5_r, fma(v0, v0, c3), -v0), rhom1e) + (1._r - invTau) * f[ 9];
                //x:0,y:-,z:0
                f[10] = invTau * fma(rhos, fma(0.5_r, fma(vy_, vy_, c3), -vy_), rhom1s) + (1._r - invTau) * f[10];
                //x:+,y:-,z:0
                f[11] = invTau * fma(rhoe, fma(0.5_r, fma(v3, v3, c3), v3), rhom1e) + (1._r - invTau) * f[11];
                //x:-,y:0,z:0
                f[12] = invTau * fma(rhos, fma(0.5_r, fma(vx_, vx_, c3), -vx_), rhom1s) + (1._r - invTau) * f[12];
                //x:0,y:0,z:0
                f[13] = invTau * W0*fma(rho, 0.5_r*c3, rhom1) + (1._r - invTau) * f[13];
                //x:+,y:0,z:0
                f[14] = invTau * fma(rhos, fma(0.5_r, fma(vx_, vx_, c3), vx_), rhom1s) + (1._r - invTau) * f[14];
                //x:-,y:+,z:0
                f[15] = invTau * fma(rhoe, fma(0.5_r, fma(v3, v3, c3), -v3), rhom1e) + (1._r - invTau) * f[15];
                //x:0,y:+,z:0
                f[16] = invTau * fma(rhos, fma(0.5_r, fma(vy_, vy_, c3), vy_), rhom1s) + (1._r - invTau) * f[16];
                //x:+,y:+,z:0
                f[17] = invTau * fma(rhoe, fma(0.5_r, fma(v0, v0, c3), v0), rhom1e) + (1._r - invTau) * f[17];

                //x:-,y:-,z:+
                f[18] = invTau * fma(rhoc, fma(0.5f, fma(v7, v7, c3), -v7), rhom1c) + (1._r - invTau) * f[18];
                //x:0,y:-,z:+
                f[19] = invTau * fma(rhoe, fma(0.5f, fma(v5, v5, c3), -v5), rhom1e) + (1._r - invTau) * f[19];
                //x:+,y:-,z:+
                f[20] = invTau * fma(rhoc, fma(0.5f, fma(v8, v8, c3), v8), rhom1c) + (1._r - invTau) * f[20];
                //x:-,y:0,z:+
                f[21] = invTau * fma(rhoe, fma(0.5f, fma(v4, v4, c3), -v4), rhom1e) + (1._r - invTau) * f[21];
                //x:0,y:0,z:+
                f[22] = invTau * fma(rhos, fma(0.5f, fma(vz_, vz_, c3), vz_), rhom1s) + (1._r - invTau) * f[22];
                //x:+,y:0,z:+
                f[23] = invTau * fma(rhoe, fma(0.5f, fma(v1, v1, c3), v1), rhom1e) + (1._r - invTau) * f[23];
                //x:-,y:+,z:+
                f[24] = invTau * fma(rhoc, fma(0.5f, fma(v9, v9, c3), v9), rhom1c) + (1._r - invTau) * f[24];
                //x:0,y:+,z:+
                f[25] = invTau * fma(rhoe, fma(0.5f, fma(v2, v2, c3), v2), rhom1e) + (1._r - invTau)* f[25];
                //x:+,y:+,z:+
                f[26] = invTau * fma(rhoc, fma(0.5f, fma(v6, v6, c3), v6), rhom1c) + (1._r - invTau) * f[26];
            }

            template<>
            HOST_DEV_CONSTEXPR void collision2<27>(real_t invTau, real_t* f)
            {
                using std::fma;
                using namespace culbm::literal;

                real_t rho_, vx_, vy_, vz_;
                calcRhoU<27>(rho_, vx_, vy_, vz_, f);

                constexpr real_t W0 = 8._r / 27._r; //center
                constexpr real_t WS = 2._r / 27._r; //straight
                constexpr real_t WE = 1._r / 54._r; //edge
                constexpr real_t WC = 1._r / 216._r;//corner

                const real_t rhom1 = rho_ - 1._r;
                const real_t c3    = -3._r * (vx_ * vx_ + vy_ * vy_ + vz_ * vz_);
                vx_ *= 3._r;
                vy_ *= 3._r;
                vz_ *= 3._r;

                const real_t v0=vx_+vy_, v1=vx_+vz_, v2=vy_+vz_, v3=vx_-vy_, v4=vx_-vz_, v5=vy_-vz_, v6=vx_+vy_+vz_, v7=vx_+vy_-vz_, v8=vx_-vy_+vz_, v9=vy_-vx_+vz_;
                const real_t rhos=WS*rho_, rhoe=WE*rho_, rhoc=WC*rho_, rhom1s=WS*rhom1, rhom1e=WE*rhom1, rhom1c=WC*rhom1;

                //x:-,y:-,z:-
                f[ 0] = invTau * fma(rhoc, fma(0.5_r, fma(v6, v6, c3), -v6), rhom1c) + (1._r - invTau) * f[ 0];
                //x:0,y:-,z:-
                f[ 1] = invTau * fma(rhoe, fma(0.5_r, fma(v2, v2, c3), -v2), rhom1e) + (1._r - invTau) * f[ 1];
                //x:+,y:-,z:-
                f[ 2] = invTau * fma(rhoc, fma(0.5_r, fma(v9, v9, c3), -v9), rhom1c) + (1._r - invTau) * f[ 2];
                //x:-,y:0,z:-
                f[ 3] = invTau * fma(rhoe, fma(0.5_r, fma(v1, v1, c3), -v1), rhom1e) + (1._r - invTau) * f[ 3];
                //x:0,y:0,z:-
                f[ 4] = invTau * fma(rhos, fma(0.5_r, fma(vz_, vz_, c3), -vz_), rhom1s) + (1._r - invTau) * f[ 4];
                //x:+,y:0,z:-
                f[ 5] = invTau * fma(rhoe, fma(0.5_r, fma(v4, v4, c3), v4), rhom1e) + (1._r - invTau) * f[ 5];
                //x:-,y:+,z:-
                f[ 6] = invTau * fma(rhoc, fma(0.5_r, fma(v8, v8, c3), -v8), rhom1c) + (1._r - invTau) * f[ 6];
                //x:0,y:+,z:-
                f[ 7] = invTau * fma(rhoe, fma(0.5_r, fma(v5, v5, c3), v5), rhom1e) + (1._r - invTau) * f[ 7];
                //x:+,y:+,z:-
                f[ 8] = invTau * fma(rhoc, fma(0.5_r, fma(v7, v7, c3), v7), rhom1c) + (1._r - invTau) * f[ 8];

                //x:-,y:-,z:0
                f[ 9] = invTau * fma(rhoe, fma(0.5_r, fma(v0, v0, c3), -v0), rhom1e) + (1._r - invTau) * f[ 9];
                //x:0,y:-,z:0
                f[10] = invTau * fma(rhos, fma(0.5_r, fma(vy_, vy_, c3), -vy_), rhom1s) + (1._r - invTau) * f[10];
                //x:+,y:-,z:0
                f[11] = invTau * fma(rhoe, fma(0.5_r, fma(v3, v3, c3), v3), rhom1e) + (1._r - invTau) * f[11];
                //x:-,y:0,z:0
                f[12] = invTau * fma(rhos, fma(0.5_r, fma(vx_, vx_, c3), -vx_), rhom1s) + (1._r - invTau) * f[12];
                //x:0,y:0,z:0
                f[13] = invTau * W0*fma(rho_, 0.5_r*c3, rhom1) + (1._r - invTau) * f[13];
                //x:+,y:0,z:0
                f[14] = invTau * fma(rhos, fma(0.5_r, fma(vx_, vx_, c3), vx_), rhom1s) + (1._r - invTau) * f[14];
                //x:-,y:+,z:0
                f[15] = invTau * fma(rhoe, fma(0.5_r, fma(v3, v3, c3), -v3), rhom1e) + (1._r - invTau) * f[15];
                //x:0,y:+,z:0
                f[16] = invTau * fma(rhos, fma(0.5_r, fma(vy_, vy_, c3), vy_), rhom1s) + (1._r - invTau) * f[16];
                //x:+,y:+,z:0
                f[17] = invTau * fma(rhoe, fma(0.5_r, fma(v0, v0, c3), v0), rhom1e) + (1._r - invTau) * f[17];

                //x:-,y:-,z:+
                f[18] = invTau * fma(rhoc, fma(0.5f, fma(v7, v7, c3), -v7), rhom1c) + (1._r - invTau) * f[18];
                //x:0,y:-,z:+
                f[19] = invTau * fma(rhoe, fma(0.5f, fma(v5, v5, c3), -v5), rhom1e) + (1._r - invTau) * f[19];
                //x:+,y:-,z:+
                f[20] = invTau * fma(rhoc, fma(0.5f, fma(v8, v8, c3), v8), rhom1c) + (1._r - invTau) * f[20];
                //x:-,y:0,z:+
                f[21] = invTau * fma(rhoe, fma(0.5f, fma(v4, v4, c3), -v4), rhom1e) + (1._r - invTau) * f[21];
                //x:0,y:0,z:+
                f[22] = invTau * fma(rhos, fma(0.5f, fma(vz_, vz_, c3), vz_), rhom1s) + (1._r - invTau) * f[22];
                //x:+,y:0,z:+
                f[23] = invTau * fma(rhoe, fma(0.5f, fma(v1, v1, c3), v1), rhom1e) + (1._r - invTau) * f[23];
                //x:-,y:+,z:+
                f[24] = invTau * fma(rhoc, fma(0.5f, fma(v9, v9, c3), v9), rhom1c) + (1._r - invTau) * f[24];
                //x:0,y:+,z:+
                f[25] = invTau * fma(rhoe, fma(0.5f, fma(v2, v2, c3), v2), rhom1e) + (1._r - invTau)* f[25];
                //x:+,y:+,z:+
                f[26] = invTau * fma(rhoc, fma(0.5f, fma(v6, v6, c3), v6), rhom1c) + (1._r - invTau) * f[26];
            }
        }

        template<u32 NDIR>
        HOST_DEV_CONSTEXPR void calcEqu(real_t rhon, real_t vxn, real_t vyn, real_t vzn, real_t* fn)
        {
            detail::calcEqu<NDIR>(rhon, vxn, vyn, vzn, fn);
        }

        template<u32 NDIR>
        HOST_DEV_CONSTEXPR void calcRhoU(real_t& rhon, real_t& vxn, real_t& vyn, real_t& vzn, const real_t* fn)
        {
            detail::calcRhoU<NDIR>(rhon, vxn, vyn, vzn, fn);
        }

        template<u32 NDIR>
        HOST_DEV_CONSTEXPR void collision(real_t invTau, real_t& rhon, real_t& vxn, real_t& vyn, real_t& vzn, real_t* fn)
        {
            detail::collision<NDIR>(invTau, rhon, vxn, vyn, vzn, fn);
        }

        template<u32 NDIR>
        HOST_DEV_CONSTEXPR void collision2(real_t invTau, real_t* fn)
        {
            detail::collision2<NDIR>(invTau, fn);
        }
    }
}