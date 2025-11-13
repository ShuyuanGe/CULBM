#pragma once

#ifdef __CUDACC__
    #define HOST_DEV_CONSTEXPR  __host__ __device__ __forceinline__
#else
    #define HOST_DEV_CONSTEXPR  constexpr
#endif


#ifndef DefOptionalMember 
#define DefOptionalMember(Var, var) \
    template<bool ENABLE, typename T> struct OptionalMember##Var {}; \
    template<typename T> struct OptionalMember##Var<true, T> { T var; }
#endif