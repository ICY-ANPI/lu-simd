/*
 * Copyright (C) 2017 
 * Área Académica de Ingeniería en Computadoras, ITCR, Costa Rica
 *
 * This file is part of the numerical analysis lecture CE3102 at TEC
 *
 * @Author: Pablo Alvarado
 * @Date:   28.12.2017
 */

#ifndef ANPI_INTRINSICS_HPP
#define ANPI_INTRINSICS_HPP

#include <cstdint>

/*
 * Include the proper intrinsics headers for the current architecture
 */

#if defined(_MSC_VER)
   /* Microsoft C/C++-compatible compiler */
#  include <intrin.h>
#elif defined(__GNUC__)
#  if (defined(__x86_64__) || defined(__i386__))
     /* GCC-compatible compiler, targeting x86/x86-64 */
#    include <x86intrin.h>
#  elif defined(__ARM_NEON__)
     /* GCC-compatible compiler, targeting ARM with NEON */
#    include <arm_neon.h>
#  elif defined(__IWMMXT__)
     /* GCC-compatible compiler, targeting ARM with WMMX */
#    include <mmintrin.h>
#  elif defined(__xlC__)) && (defined(__VEC__) || defined(__ALTIVEC__))
     /* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#    include <altivec.h>
#  elif defined(__GNUC__) && defined(__SPE__)
     /* GCC-compatible compiler, targeting PowerPC with SPE */
#    include <spe.h>
#  endif
#endif

template <typename T>
struct is_simd_type {
  static constexpr bool value =
    std::is_same<T,double>::value        ||
    std::is_same<T,float>::value         ||
    std::is_same<T,std::int64_t>::value  ||
    std::is_same<T,std::uint64_t>::value ||    
    std::is_same<T,std::int32_t>::value  ||
    std::is_same<T,std::uint32_t>::value ||
    std::is_same<T,std::int16_t>::value  ||
    std::is_same<T,std::uint16_t>::value ||
    std::is_same<T,std::int8_t>::value   ||
    std::is_same<T,std::uint8_t>::value;
};


#ifdef __AVX512F__
template<typename T> struct avx512_traits { };
template<> struct avx512_traits<double> { typedef __m512d reg_type; };
template<> struct avx512_traits<float> { typedef __m512 reg_type; };
template<> struct avx512_traits<int64_t> { typedef __m512i reg_type; };
template<> struct avx512_traits<uint64_t> { typedef __m512i reg_type; };
template<> struct avx512_traits<int32_t> { typedef __m512i reg_type; };
template<> struct avx512_traits<uint32_t> { typedef __m512i reg_type; };
template<> struct avx512_traits<int16_t> { typedef __m512i reg_type; };
template<> struct avx512_traits<uint16_t> { typedef __m512i reg_type; };
template<> struct avx512_traits<int8_t> { typedef __m512i reg_type; };
template<> struct avx512_traits<uint8_t> { typedef __m512i reg_type; };
#endif

#ifdef __AVX__
template<typename T> struct avx_traits { };
template<> struct avx_traits<double> { typedef __m256d reg_type; };
template<> struct avx_traits<float> { typedef __m256 reg_type; };
template<> struct avx_traits<int64_t> { typedef __m256i reg_type; };
template<> struct avx_traits<uint64_t> { typedef __m256i reg_type; };
template<> struct avx_traits<int32_t> { typedef __m256i reg_type; };
template<> struct avx_traits<uint32_t> { typedef __m256i reg_type; };
template<> struct avx_traits<int16_t> { typedef __m256i reg_type; };
template<> struct avx_traits<uint16_t> { typedef __m256i reg_type; };
template<> struct avx_traits<int8_t> { typedef __m256i reg_type; };
template<> struct avx_traits<uint8_t> { typedef __m256i reg_type; };
#endif

#ifdef __SSE2__
template<typename T> struct sse2_traits { };
template<> struct sse2_traits<double> { typedef __m128d reg_type; };
template<> struct sse2_traits<float> { typedef __m128 reg_type; };
template<> struct sse2_traits<int64_t> { typedef __m128i reg_type; };
template<> struct sse2_traits<uint64_t> { typedef __m128i reg_type; };
template<> struct sse2_traits<int32_t> { typedef __m128i reg_type; };
template<> struct sse2_traits<uint32_t> { typedef __m128i reg_type; };
template<> struct sse2_traits<int16_t> { typedef __m128i reg_type; };
template<> struct sse2_traits<uint16_t> { typedef __m128i reg_type; };
template<> struct sse2_traits<int8_t> { typedef __m128i reg_type; };
template<> struct sse2_traits<uint8_t> { typedef __m128i reg_type; };
#endif


#ifdef __SSE3__

#include <mmintrin.h>
#include <pmmintrin.h>
#include "Exception.hpp"

#include <iostream>
template<typename T> struct sse3_traits { };
template<> struct sse3_traits<double> { typedef __m128d reg_type; };
template<> struct sse3_traits<float> { typedef __m128 reg_type; };
template<> struct sse3_traits<int64_t> { typedef __m128i reg_type; };
template<> struct sse3_traits<uint64_t> { typedef __m128i reg_type; };
template<> struct sse3_traits<int32_t> { typedef __m128i reg_type; };
template<> struct sse3_traits<uint32_t> { typedef __m128i reg_type; };
template<> struct sse3_traits<int16_t> { typedef __m128i reg_type; };
template<> struct sse3_traits<uint16_t> { typedef __m128i reg_type; };
template<> struct sse3_traits<int8_t> { typedef __m128i reg_type; };
template<> struct sse3_traits<uint8_t> { typedef __m128i reg_type; };

template<typename T, typename U> void sse3_stream(T* ptr, T value);

template<> inline void __attribute__((__always_inline__))
  sse3_stream<double,__m128d>(double* ptr, double value){
    //std::cout << "in double" << std::endl;
    _mm_stream_pd(ptr,_mm_set1_pd(value));
}
template<> inline void __attribute__((__always_inline__))
  sse3_stream<float,__m128>(float* ptr, float value){
    //std::cout << "in float" << std::endl;
    __m128 a = _mm_set1_ps(value);
    _mm_stream_ps(ptr,a);
}




template<typename T, typename U> U sse3_set1(T value);

template<> inline __m128d __attribute__((__always_inline__))
  sse3_set1<double,__m128d>(double value){
    //std::cout << "in double" << std::endl;
    return _mm_set1_pd(value);
}
template<> inline __m128 __attribute__((__always_inline__))
  sse3_set1<float,__m128>(float value){
    //std::cout << "in float" << std::endl;
    return _mm_set1_ps(value);
}


template<> inline __m128i __attribute__((__always_inline__))
  sse3_set1<int32_t,__m128i>(int32_t value){
    //std::cout << "in int" << std::endl;
    return _mm_set1_epi32(value);
}





template<typename T, typename U> U sse3_set_s(T value);

template<> inline __m128d __attribute__((__always_inline__))
  sse3_set_s<double,__m128d>(double value){
    //std::cout << "in double" << std::endl;
    return _mm_set_sd(value);
}
template<> inline __m128 __attribute__((__always_inline__))
  sse3_set_s<float,__m128>(float value){
    //std::cout << "in float" << std::endl;
    return _mm_set_ss(value);
}


template<typename T, typename U> size_t colw(){
  return sizeof(U)/sizeof(T);
}


template<typename T, typename U> size_t column_correction(size_t col_size){
  size_t b_lim;
  size_t colw = sizeof(U)/sizeof(T);
  if (col_size%colw == 0) b_lim = col_size/colw;
  else b_lim = (col_size + colw - (col_size%colw ))/colw;
  return b_lim;
}


template<typename T, typename U> size_t reg_mul_value(size_t col_size){
  size_t b_lim;
  size_t colw = sizeof(U)/sizeof(T);
  b_lim = (col_size - (col_size%colw ));
  return b_lim;
}





#endif
  
  
#endif
