#pragma once

#include <array>
#include "config.hpp"

namespace gf::basic
{
    namespace detail
    {
        template<u32 NDIR>
        class VelSet3D
        {
            private:
                static consteval std::array<i32, NDIR>      _getDxArr();
                static consteval std::array<i32, NDIR>      _getDyArr();
                static consteval std::array<i32, NDIR>      _getDzArr();
                static consteval std::array<real_t, NDIR>   _getOmegaArr();
            public:
                static consteval u32 getCentIdx()
                {
                    return NDIR / 2;
                }

                template<u32 DIR>
                static consteval i32 getDx()
                {
                    return _getDxArr()[DIR];
                }

                template<u32 DIR>
                static consteval i32 getDy()
                {
                    return _getDyArr()[DIR];
                }

                template<u32 DIR>
                static consteval i32 getDz()
                {
                    return _getDzArr()[DIR];
                }

                template<u32 DIR>
                static consteval real_t getOmega()
                {
                    return _getOmegaArr()[DIR];
                }
        };

        template<>
        consteval std::array<i32, 15> VelSet3D<15>::_getDxArr()
        {
            return
            {
                -1,     1,
                    0,
                -1,     1,
                    0,
                -1, 0,  1, 
                    0, 
                -1,     1, 
                    0,
                -1,     1
            };
        }

        template<>
        consteval std::array<i32, 15> VelSet3D<15>::_getDyArr()
        {
            return
            {
                -1,     -1,
                    0,
                 1,      1,
                    -1,
                 0,  0,  0, 
                     1, 
                -1,     -1, 
                     0,
                 1,      1
            };
        }

        template<>
        consteval std::array<i32, 15> VelSet3D<15>::_getDzArr()
        {
            return
            {
                -1,     -1,
                    -1,
                -1,     -1,
                     0,
                 0,  0,  0, 
                     0, 
                 1,      1, 
                     1,
                 1,      1 
            };
        }

        template<>
        consteval std::array<real_t, 15> VelSet3D<15>::_getOmegaArr()
        {
            return
            {
                1./72,       1./72, 
                       1./9,
                1./72,       1./72, 
                       1./9,
                1./9,  2./9, 1./9, 
                       1./9, 
                1./72,       1./72, 
                       1./9,
                1./72,       1./72
            };            
        }

        template<>
        consteval std::array<i32, 19> VelSet3D<19>::_getDxArr()
        {
            return
            {
                     0,
                -1,  0,  1, 
                     0, 
                -1,  0,  1, 
                -1,  0,  1, 
                -1,  0,  1, 
                     0, 
                -1,  0,  1, 
                     0
            };
        }

        template<>
        consteval std::array<i32, 19> VelSet3D<19>::_getDyArr()
        {
            return
            {
                    -1,
                 0,  0,  0,  
                     1,
                -1, -1, -1,  
                 0,  0,  0,  
                 1,  1,  1,
                    -1, 
                 0,  0,  0,  
                     1
            };
        }

        template<>
        consteval std::array<i32, 19> VelSet3D<19>::_getDzArr()
        {
            return
            {
                    -1,
                -1, -1, -1, 
                    -1,
                 0,  0,  0,  
                 0,  0,  0,  
                 0,  0,  0, 
                     1,
                 1,  1,  1,  
                     1                
            };
        }

        template<>
        consteval std::array<real_t, 19> VelSet3D<19>::_getOmegaArr()
        {
            return
            {
                       1./36, 
                1./36, 1./18, 1./36, 
                       1./36,
                1./36, 1./18, 1./36,
                1./18, 1./3 , 1./18, 
                1./36, 1./18, 1./36, 
                       1./36, 
                1./36, 1./18, 1./36, 
                       1./36
            };
        }

        template<>
        consteval std::array<i32, 27> VelSet3D<27>::_getDxArr()
        {
            return
            {
                -1,  0,  1, 
                -1,  0,  1, 
                -1,  0,  1, 
                -1,  0,  1, 
                -1,  0,  1, 
                -1,  0,  1, 
                -1,  0,  1, 
                -1,  0,  1, 
                -1,  0,  1
            };            
        }

        template<>
        consteval std::array<i32, 27> VelSet3D<27>::_getDyArr()
        {
            return
            {
                -1, -1, -1,  
                 0,  0,  0,  
                 1,  1,  1,
                -1, -1, -1,  
                 0,  0,  0,  
                 1,  1,  1,
                -1, -1, -1,  
                 0,  0,  0,  
                 1,  1,  1
            };            
        }

        template<>
        consteval std::array<i32, 27> VelSet3D<27>::_getDzArr()
        {
            return
            {
                -1, -1, -1, 
                -1, -1, -1, 
                -1, -1, -1, 
                 0,  0,  0,  
                 0,  0,  0,  
                 0,  0,  0, 
                 1,  1,  1,  
                 1,  1,  1,  
                 1,  1,  1
            };            
        }

        template<>
        consteval std::array<real_t, 27> VelSet3D<27>::_getOmegaArr()
        {
            return
            {
                1./216, 1./54 , 1./216, 
                1./54 , 2./27 , 1./54 ,
                1./216, 1./54 , 1./216,
                1./54 , 2./27 , 1./54 ,
                2./27 , 8./27 , 2./27 , 
                1./54 , 2./27 , 1./54 ,
                1./216, 1./54 , 1./216, 
                1./54 , 2./27 , 1./54 ,
                1./216, 1./54 , 1./216
            };            
        }
    }

    using VelSet3D = detail::VelSet3D<NDIR>;
}