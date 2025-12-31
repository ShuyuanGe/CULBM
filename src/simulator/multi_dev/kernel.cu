#include "kernel.cuh"
#include "velocity_set.hpp"
#include "device_function.hpp"

namespace detail
{
    template<std::uint32_t DIR>
    __device__ __forceinline__ void innerDeviceLoadDDF(
        int idx, int n, int pdx, int pdy, int pdz, int ndx, int ndy, int ndz, 
        real_t* fin, 
        const ddf_t* srcDDFBuf
    );

    template<std::uint32_t DIR>
    __device__ __forceinline__ void innerDeviceRevLoadDDF(
        int idx, int n, int pdx, int pdy, int pdz, int ndx, int ndy, int ndz, 
        real_t* fin, 
        const ddf_t* srcDDFBuf
    );

    template<std::uint32_t DIR>
    __device__ __forceinline__ void crossDeviceLoadDDF(
        int idx, int n, 
        int pdx, int pdy, int pdz, int ndx, int ndy, int ndz,
        int nbrpdx, int nbrpdy, int nbrpdz, int nbrndx, int nbrndy, int nbrndz,
        real_t* fin, 
        const std::array<ddf_t*, DIR>& srcDDFBuf 
    );

    template<std::uint32_t DIR>
    __device__ __forceinline__ void crossDeviceRevLoadDDF(
        int idx, int n, 
        int pdx, int pdy, int pdz, int ndx, int ndy, int ndz,
        int nbrpdx, int nbrpdy, int nbrpdz, int nbrndx, int nbrndy, int nbrndz,
        real_t* fin, 
        const std::array<ddf_t*, DIR>& srcDDFBuf 
    );

    template<>
    __device__ __forceinline__ void innerDeviceLoadDDF<27>(
        int idx, int n, int pdx, int pdy, int pdz, int ndx, int ndy, int ndz, 
        real_t* fin, 
        const ddf_t* srcDDFBuf
    )
    {
        //load f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
        fin[ 0] = srcDDFBuf[ 0*n+idx+pdx+pdy+pdz];
        //load f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
        fin[ 1] = srcDDFBuf[ 1*n+idx    +pdy+pdz];
        //load f2 (x:+,y:-,z:-) from neighbor (x:-,y:+,z:+)
        fin[ 2] = srcDDFBuf[ 2*n+idx+ndx+pdy+pdz];
        //load f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
        fin[ 3] = srcDDFBuf[ 3*n+idx+pdx    +pdz];
        //load f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
        fin[ 4] = srcDDFBuf[ 4*n+idx        +pdz];
        //load f5 (x:+,y:0,z:-) from neighbor (x:-,y:0,z:+)
        fin[ 5] = srcDDFBuf[ 5*n+idx+ndx    +pdz];
        //load f6 (x:-,y:+,z:-) from neighbor (x:+,y:-,z:+)
        fin[ 6] = srcDDFBuf[ 6*n+idx+pdx+ndy+pdz];
        //load f7 (x:0,y:+,z:-) from neighbor (x:0,y:-,z:+)
        fin[ 7] = srcDDFBuf[ 7*n+idx    +ndy+pdz];
        //load f8 (x:+,y:+,z:-) from neighbor (x:-,y:-,z:+)
        fin[ 8] = srcDDFBuf[ 8*n+idx+ndx+ndy+pdz];

        //load f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
        fin[ 9] = srcDDFBuf[ 9*n+idx+pdx+pdy    ];
        //load f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
        fin[10] = srcDDFBuf[10*n+idx    +pdy    ];
        //load f11(x:+,y:-,z:0) from neighbor (x:-,y:+,z:0)
        fin[11] = srcDDFBuf[11*n+idx+ndx+pdy    ];
        //load f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
        fin[12] = srcDDFBuf[12*n+idx+pdx        ];
        //load f13(x:0,y:0,z:0) from neighbor (x:0,y:0,z:0)
        fin[13] = srcDDFBuf[13*n+idx            ];
        //load f14(x:+,y:0,z:0) from neighbor (x:-,y:0,z:0)
        fin[14] = srcDDFBuf[14*n+idx+ndx        ];
        //load f15(x:-,y:+,z:0) from neighbor (x:+,y:-,z:0)
        fin[15] = srcDDFBuf[15*n+idx+pdx+ndy    ];
        //load f16(x:0,y:+,z:0) from neighbor (x:0,y:-,z:0)
        fin[16] = srcDDFBuf[16*n+idx    +ndy    ];
        //load f17(x:+,y:+,z:0) from neighbor (x:-,y:-,z:0)
        fin[17] = srcDDFBuf[17*n+idx+ndx+ndy    ];

        //load f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:-)
        fin[18] = srcDDFBuf[18*n+idx+pdx+pdy+ndz];
        //load f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:-)
        fin[19] = srcDDFBuf[19*n+idx    +pdy+ndz];
        //load f20(x:+,y:-,z:+) from neighbor (x:-,y:+,z:-)
        fin[20] = srcDDFBuf[20*n+idx+ndx+pdy+ndz];
        //load f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:-)
        fin[21] = srcDDFBuf[21*n+idx+pdx    +ndz];
        //load f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:-)
        fin[22] = srcDDFBuf[22*n+idx        +ndz];
        //load f23(x:+,y:0,z:+) from neighbor (x:-,y:0,z:-)
        fin[23] = srcDDFBuf[23*n+idx+ndx    +ndz];
        //load f24(x:-,y:+,z:+) from neighbor (x:+,y:-,z:-)
        fin[24] = srcDDFBuf[24*n+idx+pdx+ndy+ndz];
        //load f25(x:0,y:+,z:+) from neighbor (x:0,y:-,z:-)
        fin[25] = srcDDFBuf[25*n+idx    +ndy+ndz];
        //load f26(x:+,y:+,z:+) from neighbor (x:-,y:-,z:-)
        fin[26] = srcDDFBuf[26*n+idx+ndx+ndy+ndz];
    }

    template<>
    __device__ __forceinline__ void innerDeviceRevLoadDDF<27>(
        int idx, int n, int pdx, int pdy, int pdz, int ndx, int ndy, int ndz, 
        real_t* fin, 
        const ddf_t* srcDDFBuf
    )
    {
        //load f0 (x:-,y:-,z:-) from neighbor (x:-,y:-,z:-)
        fin[ 0] = srcDDFBuf[26*n+idx+ndx+ndy+ndz];
        //load f1 (x:0,y:-,z:-) from neighbor (x:0,y:-,z:-)
        fin[ 1] = srcDDFBuf[25*n+idx    +ndy+ndz];
        //load f2 (x:+,y:-,z:-) from neighbor (x:+,y:-,z:-)
        fin[ 2] = srcDDFBuf[24*n+idx+pdx+ndy+ndz];
        //load f3 (x:-,y:0,z:-) from neighbor (x:-,y:0,z:-)
        fin[ 3] = srcDDFBuf[23*n+idx+ndx    +ndz];
        //load f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:-)
        fin[ 4] = srcDDFBuf[22*n+idx        +ndz];
        //load f5 (x:+,y:0,z:-) from neighbor (x:+,y:0,z:-)
        fin[ 5] = srcDDFBuf[21*n+idx+pdx    +ndz];
        //load f6 (x:-,y:+,z:-) from neighbor (x:-,y:+,z:-)
        fin[ 6] = srcDDFBuf[20*n+idx+ndx+pdy+ndz];
        //load f7 (x:0,y:+,z:-) from neighbor (x:0,y:+,z:-)
        fin[ 7] = srcDDFBuf[19*n+idx    +pdy+ndz];
        //load f8 (x:+,y:+,z:-) from neighbor (x:+,y:+,z:-)
        fin[ 8] = srcDDFBuf[18*n+idx+pdx+pdy+ndz];

        //load f9 (x:-,y:-,z:0) from neighbor (x:-,y:-,z:0)
        fin[ 9] = srcDDFBuf[17*n+idx+ndx+ndy    ];
        //load f10(x:0,y:-,z:0) from neighbor (x:0,y:-,z:0)
        fin[10] = srcDDFBuf[16*n+idx    +ndy    ];
        //load f11(x:+,y:-,z:0) from neighbor (x:+,y:-,z:0)
        fin[11] = srcDDFBuf[15*n+idx+pdx+ndy    ];
        //load f12(x:-,y:0,z:0) from neighbor (x:-,y:0,z:0)
        fin[12] = srcDDFBuf[14*n+idx+ndx        ];
        //load f13(x:0,y:0,z:0) from neighbor (x:0,y:0,z:0)
        fin[13] = srcDDFBuf[13*n+idx            ];
        //load f14(x:+,y:0,z:0) from neighbor (x:+,y:0,z:0)
        fin[14] = srcDDFBuf[12*n+idx+pdx        ];
        //load f15(x:-,y:+,z:0) from neighbor (x:-,y:+,z:0)
        fin[15] = srcDDFBuf[11*n+idx+ndx+pdy    ];
        //load f16(x:0,y:+,z:0) from neighbor (x:0,y:+,z:0)
        fin[16] = srcDDFBuf[10*n+idx    +pdy    ];
        //load f17(x:+,y:+,z:0) from neighbor (x:+,y:+,z:0)
        fin[17] = srcDDFBuf[ 9*n+idx+pdx+pdy    ];
        
        //load f18(x:-,y:-,z:+) from neighbor (x:-,y:-,z:+)
        fin[18] = srcDDFBuf[ 8*n+idx+ndx+ndy+pdz];
        //load f19(x:0,y:-,z:+) from neighbor (x:0,y:-,z:+)
        fin[19] = srcDDFBuf[ 7*n+idx    +ndy+pdz];
        //load f20(x:+,y:-,z:+) from neighbor (x:+,y:-,z:+)
        fin[20] = srcDDFBuf[ 6*n+idx+pdx+ndy+pdz];
        //load f21(x:-,y:0,z:+) from neighbor (x:-,y:0,z:+)
        fin[21] = srcDDFBuf[ 5*n+idx+ndx    +pdz];
        //load f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:+)
        fin[22] = srcDDFBuf[ 4*n+idx        +ndz];
        //load f23(x:+,y:0,z:+) from neighbor (x:+,y:0,z:+)
        fin[23] = srcDDFBuf[ 3*n+idx+pdx    +pdz];
        //load f24(x:-,y:+,z:+) from neighbor (x:-,y:+,z:+)
        fin[24] = srcDDFBuf[ 2*n+idx+ndx+pdy+pdz];
        //load f25(x:0,y:+,z:+) from neighbor (x:0,y:+,z:+)
        fin[25] = srcDDFBuf[ 1*n+idx    +pdy+pdz];
        //load f26(x:+,y:+,z:+) from neighbor (x:+,y:+,z:+)
        fin[26] = srcDDFBuf[ 0*n+idx+pdx+pdy+pdz];
    }

    template<>
    __device__ __forceinline__ void crossDeviceLoadDDF<27>(
        int idx, int n, 
        int pdx, int pdy, int pdz, int ndx, int ndy, int ndz, 
        int nbrpdx, int nbrpdy, int nbrpdz, int nbrndx, int nbrndy, int nbrndz,
        real_t* fin, 
        const std::array<ddf_t*, 27>& srcDDFBuf
    )
    {
        using VelSet = culbm::basic::detail::VelSet3D<27>;
        constexpr int centDir = VelSet::getCentIdx();

        //load f0 (x:-,y:-,z:-) from neighbor (x:+,y:+,z:+)
        fin[ 0] = srcDDFBuf[centDir+nbrpdx+nbrpdy+nbrpdz][ 0*n+idx+pdx+pdy+pdz];
        //load f1 (x:0,y:-,z:-) from neighbor (x:0,y:+,z:+)
        fin[ 1] = srcDDFBuf[centDir       +nbrpdy+nbrpdz][ 1*n+idx    +pdy+pdz];
        //load f2 (x:+,y:-,z:-) from neighbor (x:-,y:+,z:+)
        fin[ 2] = srcDDFBuf[centDir+nbrndx+nbrpdy+nbrpdz][ 2*n+idx+ndx+pdy+pdz];
        //load f3 (x:-,y:0,z:-) from neighbor (x:+,y:0,z:+)
        fin[ 3] = srcDDFBuf[centDir+nbrpdx       +nbrpdz][ 3*n+idx+pdx    +pdz];
        //load f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:+)
        fin[ 4] = srcDDFBuf[centDir              +nbrpdz][ 4*n+idx        +pdz];
        //load f5 (x:+,y:0,z:-) from neighbor (x:-,y:0,z:+)
        fin[ 5] = srcDDFBuf[centDir+nbrndx       +nbrpdz][ 5*n+idx+ndx    +pdz];
        //load f6 (x:-,y:+,z:-) from neighbor (x:+,y:-,z:+)
        fin[ 6] = srcDDFBuf[centDir+nbrpdx+nbrndy+nbrpdz][ 6*n+idx+pdx+ndy+pdz];
        //load f7 (x:0,y:+,z:-) from neighbor (x:0,y:-,z:+)
        fin[ 7] = srcDDFBuf[centDir       +nbrndy+nbrpdz][ 7*n+idx    +ndy+pdz];
        //load f8 (x:+,y:+,z:-) from neighbor (x:-,y:-,z:+)
        fin[ 8] = srcDDFBuf[centDir+nbrndx+nbrndy+nbrpdz][ 8*n+idx+ndx+ndy+pdz];

        //load f9 (x:-,y:-,z:0) from neighbor (x:+,y:+,z:0)
        fin[ 9] = srcDDFBuf[centDir+nbrpdx+nbrpdy       ][ 9*n+idx+pdx+pdy    ];
        //load f10(x:0,y:-,z:0) from neighbor (x:0,y:+,z:0)
        fin[10] = srcDDFBuf[centDir       +nbrpdy       ][10*n+idx    +pdy    ];
        //load f11(x:+,y:-,z:0) from neighbor (x:-,y:+,z:0)
        fin[11] = srcDDFBuf[centDir+nbrndx+nbrpdy       ][11*n+idx+ndx+pdy    ];
        //load f12(x:-,y:0,z:0) from neighbor (x:+,y:0,z:0)
        fin[12] = srcDDFBuf[centDir+nbrpdx              ][12*n+idx+pdx        ];
        //load f13(x:0,y:0,z:0) from neighbor (x:0,y:0,z:0)
        fin[13] = srcDDFBuf[centDir                     ][13*n+idx            ];
        //load f14(x:+,y:0,z:0) from neighbor (x:-,y:0,z:0)
        fin[14] = srcDDFBuf[centDir+nbrndx              ][14*n+idx+ndx        ];
        //load f15(x:-,y:+,z:0) from neighbor (x:+,y:-,z:0)
        fin[15] = srcDDFBuf[centDir+nbrpdx+nbrndy       ][15*n+idx+pdx+ndy    ];
        //load f16(x:0,y:+,z:0) from neighbor (x:0,y:-,z:0)
        fin[16] = srcDDFBuf[centDir       +nbrndy       ][16*n+idx    +ndy    ];
        //load f17(x:+,y:+,z:0) from neighbor (x:-,y:-,z:0)
        fin[17] = srcDDFBuf[centDir+nbrndx+nbrndy       ][17*n+idx+ndx+ndy    ];

        //load f18(x:-,y:-,z:+) from neighbor (x:+,y:+,z:-)
        fin[18] = srcDDFBuf[centDir+nbrpdx+nbrpdy+nbrndz][18*n+idx+pdx+pdy+ndz];
        //load f19(x:0,y:-,z:+) from neighbor (x:0,y:+,z:-)
        fin[19] = srcDDFBuf[centDir       +nbrpdy+nbrndz][19*n+idx    +pdy+ndz];
        //load f20(x:+,y:-,z:+) from neighbor (x:-,y:+,z:-)
        fin[20] = srcDDFBuf[centDir+nbrndx+nbrpdy+nbrndz][20*n+idx+ndx+pdy+ndz];
        //load f21(x:-,y:0,z:+) from neighbor (x:+,y:0,z:-)
        fin[21] = srcDDFBuf[centDir+nbrpdx       +nbrndz][21*n+idx+pdx    +ndz];
        //load f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:-)
        fin[22] = srcDDFBuf[centDir              +nbrndz][22*n+idx        +ndz];
        //load f23(x:+,y:0,z:+) from neighbor (x:-,y:0,z:-)
        fin[23] = srcDDFBuf[centDir+nbrndx       +nbrndz][23*n+idx+ndx    +ndz];
        //load f24(x:-,y:+,z:+) from neighbor (x:+,y:-,z:-)
        fin[24] = srcDDFBuf[centDir+nbrpdx+nbrndy+nbrndz][24*n+idx+pdx+ndy+ndz];
        //load f25(x:0,y:+,z:+) from neighbor (x:0,y:-,z:-)
        fin[25] = srcDDFBuf[centDir       +nbrndy+nbrndz][25*n+idx    +ndy+ndz];
        //load f26(x:+,y:+,z:+) from neighbor (x:-,y:-,z:-)
        fin[26] = srcDDFBuf[centDir+nbrndx+nbrndy+nbrndz][26*n+idx+ndx+ndy+ndz];
    }

    template<>
    __device__ __forceinline__ void crossDeviceRevLoadDDF<27>(
        int idx, int n, 
        int pdx, int pdy, int pdz, int ndx, int ndy, int ndz,
        int nbrpdx, int nbrpdy, int nbrpdz, int nbrndx, int nbrndy, int nbrndz,
        real_t* fin, 
        const std::array<ddf_t*, 27>& srcDDFBuf 
    )
    {
        using VelSet = culbm::basic::detail::VelSet3D<27>;
        constexpr int centDir = VelSet::getCentIdx();

        //load f0 (x:-,y:-,z:-) from neighbor (x:-,y:-,z:-)
        fin[ 0] = srcDDFBuf[centDir+nbrndx+nbrndy+nbrndz][26*n+idx+ndx+ndy+ndz];
        //load f1 (x:0,y:-,z:-) from neighbor (x:0,y:-,z:-)
        fin[ 1] = srcDDFBuf[centDir       +nbrndy+nbrndz][25*n+idx    +ndy+ndz];
        //load f2 (x:+,y:-,z:-) from neighbor (x:+,y:-,z:-)
        fin[ 2] = srcDDFBuf[centDir+nbrpdx+nbrndy+nbrndz][24*n+idx+pdx+ndy+ndz];
        //load f3 (x:-,y:0,z:-) from neighbor (x:-,y:0,z:-)
        fin[ 3] = srcDDFBuf[centDir+nbrndx       +nbrndz][23*n+idx+ndx    +ndz];
        //load f4 (x:0,y:0,z:-) from neighbor (x:0,y:0,z:-)
        fin[ 4] = srcDDFBuf[centDir              +nbrndz][22*n+idx        +ndz];
        //load f5 (x:+,y:0,z:-) from neighbor (x:+,y:0,z:-)
        fin[ 5] = srcDDFBuf[centDir+nbrpdx       +nbrndz][21*n+idx+pdx    +ndz];
        //load f6 (x:-,y:+,z:-) from neighbor (x:-,y:+,z:-)
        fin[ 6] = srcDDFBuf[centDir+nbrndx+nbrpdy+nbrndz][20*n+idx+ndx+pdy+ndz];
        //load f7 (x:0,y:+,z:-) from neighbor (x:0,y:+,z:-)
        fin[ 7] = srcDDFBuf[centDir       +nbrpdy+nbrndz][19*n+idx    +pdy+ndz];
        //load f8 (x:+,y:+,z:-) from neighbor (x:+,y:+,z:-)
        fin[ 8] = srcDDFBuf[centDir+nbrpdx+nbrpdy+nbrndz][18*n+idx+pdx+pdy+ndz];

        //load f9 (x:-,y:-,z:0) from neighbor (x:-,y:-,z:0)
        fin[ 9] = srcDDFBuf[centDir+nbrndx+nbrndy       ][17*n+idx+ndx+ndy    ];
        //load f10(x:0,y:-,z:0) from neighbor (x:0,y:-,z:0)
        fin[10] = srcDDFBuf[centDir       +nbrndy       ][16*n+idx    +ndy    ];
        //load f11(x:+,y:-,z:0) from neighbor (x:+,y:-,z:0)
        fin[11] = srcDDFBuf[centDir+nbrpdx+nbrndy       ][15*n+idx+pdx+ndy    ];
        //load f12(x:-,y:0,z:0) from neighbor (x:-,y:0,z:0)
        fin[12] = srcDDFBuf[centDir+nbrndx              ][14*n+idx+ndx        ];
        //load f13(x:0,y:0,z:0) from neighbor (x:0,y:0,z:0)
        fin[13] = srcDDFBuf[centDir                     ][13*n+idx            ];
        //load f14(x:+,y:0,z:0) from neighbor (x:+,y:0,z:0)
        fin[14] = srcDDFBuf[centDir+nbrpdx              ][12*n+idx+pdx        ];
        //load f15(x:-,y:+,z:0) from neighbor (x:-,y:+,z:0)
        fin[15] = srcDDFBuf[centDir+nbrndx+nbrpdy       ][11*n+idx+ndx+pdy    ];
        //load f16(x:0,y:+,z:0) from neighbor (x:0,y:+,z:0)
        fin[16] = srcDDFBuf[centDir       +nbrpdy       ][10*n+idx    +pdy    ];
        //load f17(x:+,y:+,z:0) from neighbor (x:+,y:+,z:0)
        fin[17] = srcDDFBuf[centDir+nbrpdx+nbrpdy       ][ 9*n+idx+pdx+pdy    ];

        //load f18(x:-,y:-,z:+) from neighbor (x:-,y:-,z:+)
        fin[18] = srcDDFBuf[centDir+nbrndx+nbrndy+nbrpdz][ 8*n+idx+ndx+ndy+pdz];
        //load f19(x:0,y:-,z:+) from neighbor (x:0,y:-,z:+)
        fin[19] = srcDDFBuf[centDir       +nbrndy+nbrpdz][ 7*n+idx    +ndy+pdz];
        //load f20(x:+,y:-,z:+) from neighbor (x:+,y:-,z:+)
        fin[20] = srcDDFBuf[centDir+nbrpdx+nbrndy+nbrpdz][ 6*n+idx+pdx+ndy+pdz];
        //load f21(x:-,y:0,z:+) from neighbor (x:-,y:0,z:+)
        fin[21] = srcDDFBuf[centDir+nbrndx       +nbrpdz][ 5*n+idx+ndx    +pdz];
        //load f22(x:0,y:0,z:+) from neighbor (x:0,y:0,z:+)
        fin[22] = srcDDFBuf[centDir              +nbrpdz][ 4*n+idx        +pdz];
        //load f23(x:+,y:0,z:+) from neighbor (x:+,y:0,z:+)
        fin[23] = srcDDFBuf[centDir+nbrpdx       +nbrpdz][ 3*n+idx+pdx    +pdz];
        //load f24(x:-,y:+,z:+) from neighbor (x:-,y:+,z:+)
        fin[24] = srcDDFBuf[centDir+nbrndx+nbrpdy+nbrpdz][ 2*n+idx+ndx+pdy+pdz];
        //load f25(x:0,y:+,z:+) from neighbor (x:0,y:+,z:+)
        fin[25] = srcDDFBuf[centDir       +nbrpdy+nbrpdz][ 1*n+idx    +pdy+pdz];
        //load f26(x:+,y:+,z:+) from neighbor (x:+,y:+,z:+)
        fin[26] = srcDDFBuf[centDir+nbrpdx+nbrpdy+nbrpdz][ 0*n+idx+pdx+pdy+pdz];
    }
}

namespace culbm::simulator::multi_dev
{
    __global__ __launch_bounds__(1024) void
    D3Q27BGKKernel(const KernelParam<27> __grid_constant__ param)
    {
        const std::int32_t nx = gridDim.x*blockDim.x;
        const std::int32_t ny = gridDim.y*blockDim.y;
        const std::int32_t nz = gridDim.z*blockDim.z;
        const std::int32_t  n = nx*ny*nz;
        const std::int32_t  x = blockDim.x*blockIdx.x+threadIdx.x;
        const std::int32_t  y = blockDim.y*blockIdx.y+threadIdx.y;
        const std::int32_t  z = blockDim.z*blockIdx.z+threadIdx.z;
        const std::int32_t  idx = x+nx*(y+ny*z);

        const flag_t flagn = param.flagBuf[idx];
        real_t rhon, vxn, vyn, vzn;
        real_t fin[27];

        const flag_t iFlagn = flagn&(LOAD_DDF_BIT|REV_LOAD_DDF_BIT|DEV_BND_BIT);
        if(iFlagn)
        {
            const int pdx = (x==nx-1) ? 1-nx : 1;
            const int pdy = (y==ny-1) ? nx*(1-ny) : nx;
            const int pdz = (z==nz-1) ? nx*ny*(1-nz) : nx*ny;
            const int ndx = (x==0) ? nx-1 : -1;
            const int ndy = (y==0) ? nx*(ny-1) : -nx;
            const int ndz = (z==0) ? nx*ny*(nz-1) : -nx*ny;

            if(iFlagn==LOAD_DDF_BIT)
            {
                detail::innerDeviceLoadDDF<27>(
                    idx, n, pdx, pdy, pdz, ndx, ndy, ndz, std::begin(fin), 
                    param.srcDDFBuf[culbm::basic::detail::VelSet3D<27>::getCentIdx()]
                );
            }

            if(iFlagn==REV_LOAD_DDF_BIT)
            {
                detail::innerDeviceRevLoadDDF<27>(
                    idx, n, pdx, pdy, pdz, ndx, ndy, ndz, std::begin(fin),
                    param.srcDDFBuf[culbm::basic::detail::VelSet3D<27>::getCentIdx()]
                );
            }

            if((iFlagn&DEV_BND_BIT)==DEV_BND_BIT)
            {
                const int nbrpdx = (x==nx-1) ? 1 : 0;
                const int nbrpdy = (y==ny-1) ? 3 : 0;
                const int nbrpdz = (z==nz-1) ? 9 : 0;
                const int nbrndx = (x==0) ? -1 : 0;
                const int nbrndy = (y==0) ? -3 : 0;
                const int nbrndz = (z==0) ? -9 : 0;

                if(iFlagn==(LOAD_DDF_BIT|DEV_BND_BIT))
                {
                    detail::crossDeviceLoadDDF<27>(
                        idx, n, pdx, pdy, pdz, ndx, ndy, ndz, 
                        nbrpdx, nbrpdy, nbrpdz, nbrndx, nbrndy, nbrndz, 
                        std::begin(fin), 
                        param.srcDDFBuf
                    );
                }

                if(iFlagn==(REV_LOAD_DDF_BIT|DEV_BND_BIT))
                {
                    detail::crossDeviceRevLoadDDF<27>(
                        idx, n, pdx, pdy, pdz, ndx, ndy, ndz, 
                        nbrpdx, nbrpdy, nbrpdz, nbrndx, nbrndy, nbrndz, 
                        std::begin(fin), 
                        param.srcDDFBuf
                    );
                }
            }
        }

        if((flagn&EQU_DDF_BIT)==EQU_DDF_BIT)
        {
            rhon = param.rhoBuf[idx];
            vxn  = param.vxBuf[idx];
            vyn  = param.vyBuf[idx];
            vzn  = param.vzBuf[idx];
            culbm::lbm_core::bgk::calcEqu<27>(rhon, vxn, vyn, vzn, std::begin(fin));
        }

        if((flagn&COLLIDE_BIT)==COLLIDE_BIT)
        {
            culbm::lbm_core::bgk::collision<27>(param.invTau, rhon, vxn, vyn, vzn, std::begin(fin));
        }

        if((flagn&STORE_DDF_BIT)==STORE_DDF_BIT)
        {
            culbm::lbm_core::store<27>(false, idx, n, std::begin(fin), param.dstDDFBuf);
            param.rhoBuf[idx] = rhon;
            param.vxBuf[idx]  = vxn;
            param.vyBuf[idx]  = vyn;
            param.vzBuf[idx]  = vzn;
        }
    }

    constexpr auto spherical_obstacle_params = []{
        struct SphericalObstacleParams {
            bool enabled = true;
            int cx = 256;
            int cy = 256;
            int cz = 128;
            int r = 80;
        } params;
        return params;
    }();

    __global__ __launch_bounds__(1024) void
    D3Q27BGKInitKernel(const InitKernelParam<27> __grid_constant__ param)
    {
        const std::int32_t nx = gridDim.x*blockDim.x;
        const std::int32_t ny = gridDim.y*blockDim.y;
        const std::int32_t nz = gridDim.z*blockDim.z;
        const std::int32_t x = blockDim.x*blockIdx.x+threadIdx.x;
        const std::int32_t y = blockDim.y*blockIdx.y+threadIdx.y;
        const std::int32_t z = blockDim.z*blockIdx.z+threadIdx.z;
        const std::int32_t n = nx*ny*nz;
        const std::int32_t idx = x+nx*(y+ny*z);

        flag_t flagn = LOAD_DDF_BIT | COLLIDE_BIT | STORE_DDF_BIT;
        real_t rhon = 1;
        real_t vxn = 0;
        real_t vyn = 0;
        real_t vzn = 0;
        ddf_t fin[27];

        // spherical obstacle
        if constexpr (spherical_obstacle_params.enabled) {
            constexpr int cx = spherical_obstacle_params.cx;
            constexpr int cy = spherical_obstacle_params.cy;
            constexpr int cz = spherical_obstacle_params.cz;
            constexpr int r = spherical_obstacle_params.r;
            constexpr int r2 = r * r;
            const int gx = param.devIdx.x * nx + x;
            const int gy = param.devIdx.y * ny + y;
            const int gz = param.devIdx.z * nz + z;

            if ((gx - cx) * (gx - cx) + (gy - cy) * (gy - cy) + (gz - cz) * (gz - cz) < r2) {
                flagn = BOUNCE_BACK_FLAG;
                vxn = 0;
                vyn = 0;
                vzn = 0;
            }
        }

        if(
            (param.devIdx.y==0 and y==0) or                     //front face
            (param.devIdx.y==param.devDim.y-1 and y==ny-1) or   //back face
            (param.devIdx.z==0 and z==0) or                     //down face
            (param.devIdx.z==param.devDim.z-1 and z==nz-1)      //up face
        )
        {
            flagn = REV_LOAD_DDF_BIT | STORE_DDF_BIT;
        }
        else if(param.devIdx.x==0 and x==0)
        {
            flagn = EQU_DDF_BIT | STORE_DDF_BIT;                //left face
            vxn = 0.05;
        }
        else if(param.devIdx.x==param.devDim.x-1 and x==nx-1)
        {
            flagn = EQU_DDF_BIT | STORE_DDF_BIT;                //right face
        }

        if(x==0 or x==nx-1 or y==0 or y==ny-1 or z==0 or z==nz-1)
        {
            flagn |= DEV_BND_BIT;                               //device boundary
        }

        culbm::lbm_core::bgk::calcEqu<27>(rhon, vxn, vyn, vzn, std::begin(fin));

        param.flagBuf[idx] = flagn;
        param.rhoBuf[idx] = rhon;
        param.vxBuf[idx] = vxn;
        param.vyBuf[idx] = vyn;
        param.vzBuf[idx] = vzn;

        #pragma unroll
        for(std::uint32_t dir=0 ; dir<27 ; ++dir)
        {
            param.srcDDFBuf[dir*n+idx] = fin[dir];
            param.dstDDFBuf[dir*n+idx] = fin[dir];
        }
    }
}