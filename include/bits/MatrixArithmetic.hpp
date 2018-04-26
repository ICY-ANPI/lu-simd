/*
 * Copyright (C) 2017 
 * Área Académica de Ingeniería en Computadoras, ITCR, Costa Rica
 *
 * This file is part of the numerical analysis lecture CE3102 at TEC
 *
 * @Author: Pablo Alvarado
 * @Date:   28.12.2017
 */

#ifndef ANPI_MATRIX_ARITHMETIC_HPP
#define ANPI_MATRIX_ARITHMETIC_HPP

#include "Intrinsics.hpp"
#include <type_traits>
#include "Exception.hpp"
#include <iostream>

namespace anpi
{
  namespace fallback {
    /*
     * Sum
     */

    // Fallback implementation
    
    // In-copy implementation c=a+b
    template<typename T,class Alloc>
    inline void add(const Matrix<T,Alloc>& a,
                    const Matrix<T,Alloc>& b,
                    Matrix<T,Alloc>& c) {

      assert( (a.rows() == b.rows()) &&
              (a.cols() == b.cols()) );

      const size_t tentries = a.rows()*a.dcols();
      c.allocate(a.rows(),a.cols());
      
      T* here        = c.data();
      T *const end   = here + tentries;
      const T* aptr = a.data();
      const T* bptr = b.data();

      for (;here!=end;) {
        *here++ = *aptr++ + *bptr++;
      }
    }

    // In-place implementation a = a+b
    template<typename T,class Alloc>
    inline void add(Matrix<T,Alloc>& a,
                    const Matrix<T,Alloc>& b) {

      assert( (a.rows() == b.rows()) &&
              (a.cols() == b.cols()) );

      const size_t tentries = a.rows()*a.dcols();
      
      T* here        = a.data();
      T *const end   = here + tentries;
      
      const T* bptr = b.data();

      for (;here!=end;) {
        *here++ += *bptr++;
      }
    }


    /*
     * Subtraction
     */

    // Fall back implementations

    // In-copy implementation c=a-b
    template<typename T,class Alloc>
    inline void subtract(const Matrix<T,Alloc>& a,
                         const Matrix<T,Alloc>& b,
                         Matrix<T,Alloc>& c) {

      assert( (a.rows() == b.rows()) &&
              (a.cols() == b.cols()) );

      const size_t tentries = a.rows()*a.dcols();
      c.allocate(a.rows(),a.cols());
      
      T* here        = c.data();
      T *const end   = here + tentries;
      const T* aptr = a.data();
      const T* bptr = b.data();

      for (;here!=end;) {
        *here++ = *aptr++ - *bptr++;
      }
    }

    // In-place implementation a = a-b
    template<typename T,class Alloc>
    inline void subtract(Matrix<T,Alloc>& a,
                         const Matrix<T,Alloc>& b) {

      assert( (a.rows() == b.rows()) &&
              (a.cols() == b.cols()) );
      
      const size_t tentries = a.rows()*a.dcols();
      
      T* here        = a.data();
      T *const end   = here + tentries;
      
      const T* bptr = b.data();

      for (;here!=end;) {
        *here++ -= *bptr++;
      }
    }





    /*
     * Multiplication Matrix * Matrix
     */

    // Fall back implementations

    // In-copy implementation c=a*b
    template<typename T,class Alloc>
    inline void multiply(const Matrix<T,Alloc>& a,
    		const Matrix<T,Alloc>& b,
			Matrix<T,Alloc>& c) {


    	if (a.cols() != b.rows()) {
    		throw anpi::Exception("Incompatible size of both matrix at multiplication method");
    	}

    	c.allocate(a.rows(),b.cols());
    	for (size_t i = 0; i < a.rows(); i++){
    		for(size_t j = 0; j < b.cols();j++){
    			T val(T(0));
    			for(size_t z = 0; z < b.rows();z++){
        			val += a[i][z]*b[z][j];
    			}
        		c[i][j] = val;
    		}
    	}
    }


    template<typename T,class Alloc>
    inline void multiply(const Matrix<T,Alloc>& a,
    		const std::vector<T>& b,
			Matrix<T,Alloc>& c) {


    	if (a.cols() != b.size()) {
    		throw anpi::Exception("Incompatible size of both matrix at multiplication method");
    	}

    	c.allocate(a.rows(),1);
    	for (size_t i = 0; i < a.rows(); i++){
    		T val(T(0));
    		for(size_t j = 0; j < b.size();j++){
    			val += a[i][j]*b[j];
    		}
    		c[i][0] = val;
    	}
    }






  } // namespace fallback


  namespace simd
  {
    /*
     * Sum
     */

    /*
     * The following code exemplifies how to manually accelerate code using
     * SIMD instructions.  However, for the simple element-wise algorithms
     * like sum or subtraction, modern compilers can automatically vectorize
     * the code, as the benchmarks show.
     */


    /// We wrap the intrinsics methods to be polymorphic versions
    template<typename T,class regType>
    regType mm_add(regType,regType); // We don't implement this to cause, at
                                     // least, a linker error if this version is
                                     // used.
    //{
    // Generic function should never be called.
    // If it is called, then some SIMD chaos is going on...
    
    // A way to cause a compile time error would be better
    // throw std::bad_function_call();
    // return regType();
    //}


    template<typename T,class regType>
      regType mm_mul(regType,regType);

    template<typename T,class regType>
      regType mm_hadd(regType,regType);

    template<typename T,class regType>
      regType mm_add_s(regType,regType);

    template<typename T,class regType>
      T mm_cvts(regType);

    
#ifdef __SSE2__
    template<>
    inline __m128d __attribute__((__always_inline__))
    mm_add<double>(__m128d a,__m128d b) {
      return _mm_add_pd(a,b);
    }
    template<>
    inline __m128 __attribute__((__always_inline__))
    mm_add<float>(__m128 a,__m128 b) {
      return _mm_add_ps(a,b);
    }
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_add<std::uint64_t>(__m128i a,__m128i b) {
      return _mm_add_epi64(a,b);
    }
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_add<std::int64_t>(__m128i a,__m128i b) {
      return _mm_add_epi64(a,b);
    }
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_add<std::uint32_t>(__m128i a,__m128i b) {
      return _mm_add_epi32(a,b);
    }
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_add<std::int32_t>(__m128i a,__m128i b) {
      return _mm_add_epi16(a,b);
    }
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_add<std::uint16_t>(__m128i a,__m128i b) {
      return _mm_add_epi16(a,b);
    }
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_add<std::int16_t>(__m128i a,__m128i b) {
      return _mm_add_epi32(a,b);
    }
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_add<std::uint8_t>(__m128i a,__m128i b) {
      return _mm_add_epi16(a,b);
    }
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_add<std::int8_t>(__m128i a,__m128i b) {
      return _mm_add_epi32(a,b);
    }






    template<>
    inline __m128d __attribute__((__always_inline__))
    mm_hadd<double>(__m128d a,__m128d b) {
      return _mm_hadd_pd(a,b);
    }
    template<>
    inline __m128 __attribute__((__always_inline__))
    mm_hadd<float>(__m128 a,__m128 b) {
      return _mm_hadd_ps(a,b);
    }




    template<>
    inline __m128d __attribute__((__always_inline__))
    mm_add_s<double>(__m128d a,__m128d b) {
      return _mm_add_sd(a,b);
    }
    template<>
    inline __m128 __attribute__((__always_inline__))
    mm_add_s<float>(__m128 a,__m128 b) {
      return _mm_add_ss(a,b);
    }




    //definitions of multiplications

    template<>
    inline __m128d __attribute__((__always_inline__))
    mm_mul<double>(__m128d a,__m128d b) {
      return _mm_mul_pd(a,b);
    }
    template<>
    inline __m128 __attribute__((__always_inline__))
    mm_mul<float>(__m128 a,__m128 b) {
      return _mm_mul_ps(a,b);
    }

    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_mul<std::uint32_t>(__m128i a,__m128i b) {
      return _mm_mul_epi32(a,b);
    }
    
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_mul<std::int16_t>(__m128i a,__m128i b) {
      return _mm_mul_epi32(a,b);
    }
    
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_mul<std::int8_t>(__m128i a,__m128i b) {
      return _mm_mul_epi32(a,b);
    }



    template<>
    inline float __attribute__((__always_inline__))
    mm_cvts<float>(__m128 a) {
      return _mm_cvtss_f32(a);
    }

        template<>
    inline double __attribute__((__always_inline__))
    mm_cvts<double>(__m128d a) {
      return _mm_cvtsd_f64(a);
    }


    //following functions aren't defined in sse2
    /*
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_mul<std::int32_t>(__m128i a,__m128i b) {
      return _mm_mul_epi16(a,b);
    }
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_mul<std::uint16_t>(__m128i a,__m128i b) {
      return _mm_mul_epi16(a,b);
    }

    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_mul<std::uint64_t>(__m128i a,__m128i b) {
      return _mm_mul_epi64(a,b);
    }
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_mul<std::int64_t>(__m128i a,__m128i b) {
      return _mm_mul_epi64(a,b);
    }
    template<>
    inline __m128i __attribute__((__always_inline__))
    mm_mul<std::uint8_t>(__m128i a,__m128i b) {
      return _mm_mul_epi16(a,b);
    }
    */
   
   
#elif defined __AVX__
    template<>
    inline __m256d __attribute__((__always_inline__))
    mm_add<double>(__m256d a,__m256d b) {
      return _mm256_add_pd(a,b);
    }
    template<>
    inline __m256 __attribute__((__always_inline__))
    mm_add<float>(__m256 a,__m256 b) {
      return _mm256_add_ps(a,b);
    }
    template<>
    inline __m256i __attribute__((__always_inline__))
    mm_add<uint64_t>(__m256i a,__m256i b) {
      return _mm256_add_epi64(a,b);
    }
    template<>
    inline __m256i __attribute__((__always_inline__))
    mm_add<int64_t>(__m256i a,__m256i b) {
      return _mm256_add_epi64(a,b);
    }
    template<>
    inline __m256i __attribute__((__always_inline__))
    mm_add<uint32_t>(__m256i a,__m256i b) {
      return _mm256_add_epi32(a,b);
    }
    template<>
    inline __m256i __attribute__((__always_inline__))
    mm_add<int32_t>(__m256i a,__m256i b) {
      return _mm256_add_epi32(a,b);
    }
    template<>
    inline __m256i __attribute__((__always_inline__))
    mm_add<uint16_t>(__m256i a,__m256i b) {
      return _mm256_add_epi16(a,b);
    }
    template<>
    inline __m256i __attribute__((__always_inline__))
    mm_add<int16_t>(__m256i a,__m256i b) {
      return _mm256_add_epi16(a,b);
    }
    template<>
    inline __m256i __attribute__((__always_inline__))
    mm_add<uint8_t>(__m256i a,__m256i b) {
      return _mm256_add_epi8(a,b);
    }
    template<>
    inline __m256i __attribute__((__always_inline__))
    mm_add<int8_t>(__m256i a,__m256i b) {
      return _mm256_add_epi8(a,b);
    }
#elif  defined __AVX512F__
    template<>
    inline __m512d __attribute__((__always_inline__))
    mm_add<double>(__m512d a,__m512d b) {
      return _mm512_add_pd(a,b);
    }
    template<>
    inline __m512 __attribute__((__always_inline__))
    mm_add<float>(__m512 a,__m512 b) {
      return _mm512_add_ps(a,b);
    }
    template<>
    inline __m512i __attribute__((__always_inline__))
    mm_add<uint64_t>(__m512i a,__m512i b) {
      return _mm512_add_epi64(a,b);
    }
    template<>
    inline __m512i __attribute__((__always_inline__))
    mm_add<int64_t>(__m512i a,__m512i b) {
      return _mm512_add_epi64(a,b);
    }
    template<>
    inline __m512i __attribute__((__always_inline__))
    mm_add<uint32_t>(__m512i a,__m512i b) {
      return _mm512_add_epi32(a,b);
    }
    template<>
    inline __m512i __attribute__((__always_inline__))
    mm_add<int32_t>(__m512i a,__m512i b) {
      return _mm512_add_epi32(a,b);
    }
    template<>
    inline __m512i __attribute__((__always_inline__))
    mm_add<uint16_t>(__m512i a,__m512i b) {
      return _mm512_add_epi16(a,b);
    }
    template<>
    inline __m512i __attribute__((__always_inline__))
    mm_add<int16_t>(__m512i a,__m512i b) {
      return _mm512_add_epi16(a,b);
    }
    template<>
    inline __m512i __attribute__((__always_inline__))
    mm_add<uint8_t>(__m512i a,__m512i b) {
      return _mm512_add_epi8(a,b);
    }
    template<>
    inline __m512i __attribute__((__always_inline__))
    mm_add<int8_t>(__m512i a,__m512i b) {
      return _mm512_add_epi8(a,b);
    }
    
#endif
    
    // On-copy implementation c=a+b
    template<typename T,class Alloc,typename regType>
    inline void addSIMD(const Matrix<T,Alloc>& a, 
                        const Matrix<T,Alloc>& b,
                        Matrix<T,Alloc>& c) {

      // This method is instantiated with unaligned allocators.  We
      // allow the instantiation although externally this is never
      // called unaligned
      static_assert(!extract_alignment<Alloc>::aligned ||
		    (extract_alignment<Alloc>::value >= sizeof(regType)),
		    "Insufficient alignment for the registers used");
      
      const size_t tentries = a.rows()*a.dcols();
      c.allocate(a.rows(),a.cols());

      regType* here        = reinterpret_cast<regType*>(c.data());
      const size_t  blocks = ( tentries*sizeof(T) + (sizeof(regType)-1) )/
        sizeof(regType);
      regType *const end   = here + blocks;
      const regType* aptr  = reinterpret_cast<const regType*>(a.data());
      const regType* bptr  = reinterpret_cast<const regType*>(b.data());
      
      for (;here!=end;) {
        *here++ = mm_add<T>(*aptr++,*bptr++);
      }
      
    }


    // On-copy implementation c=a*b
    template<typename T,class Alloc,typename regType>
    inline void mulSIMD(const Matrix<T,Alloc>& a, 
                        const Matrix<T,Alloc>& b,
                        Matrix<T,Alloc>& c) {

      // This method is instantiated with unaligned allocators.  We
      // allow the instantiation although externally this is never
      // called unaligned
      static_assert(!extract_alignment<Alloc>::aligned ||
		    (extract_alignment<Alloc>::value >= sizeof(regType)),
		    "Insufficient alignment for the registers used");
      
      //const size_t tentries = a.rows()*a.dcols();
      c.allocate(a.rows(),a.cols());
      c.fill(T(0));

      regType* here        = reinterpret_cast<regType*>(c.data());
      /*const size_t  blocks = ( tentries*sizeof(T) + (sizeof(regType)-1) )/
        sizeof(regType);
      regType *const end   = here + blocks;
      */
      //const regType* aptr  = reinterpret_cast<const regType*>(a.data());
      //const T * aptr = a.data();
      //const T * a_end   = aptr + tentries;
      
      const regType* bptr  = reinterpret_cast<const regType*>(b.data());
      //const regType* tmp;
      //const T* btmp, *b_end;
      regType val;
      /*
      std::cout << "cols: " << b.cols() << " d cols: " << b.dcols() <<" t:" << b.cols() + b.dcols() << std::endl;

      for(size_t i = 0; i < b.rows();i++){
        std::cout << "{ ";
        for(size_t j = 0; j < b.dcols();j+=colw<T,regType>()){
          for(size_t k = 0; k < colw<T,regType>();k++){
            std::cout << *((T*)bptr + k) << " ";
          }
          //std::cout << std::endl;
          bptr++;
        }
        std::cout << "}" << std::endl;
      }

      std::cout << "======================" << std::endl;
      */
      bptr  = reinterpret_cast<const regType*>(b.data());
      
      size_t b_lim = column_correction<T,regType>(b.dcols());
      //size_t colw = colw<T,regType>();
      //por cada fila
      for (size_t a_row = 0; a_row < a.rows(); a_row++){
        //std::cout << "mul lvl 1" <<std::endl;
        bptr = reinterpret_cast<const regType*>(b.data());
    		for(size_t b_col = 0; b_col < b_lim;b_col++){
          //std::cout << "b_col: " << b_col << " b_lim: " << b_lim << std::endl;
          //std::cout << "mul lvl 2 - iterations: " << b.cols()/(sizeof(regType)/sizeof(T)) <<std::endl;
          //setear el valor inicial de val a cero
          val = sse3_set1<T,regType>(T(0));
          //el valor btmp guarda el puntero en b
          /**
           * btmp = bptr
           * [x] [ ] [ ]
           * [ ] [ ] [ ]
           * [ ] [ ] [ ]
           */
          const regType* btmp  = bptr;
          
          //por cada elemento
          for(size_t b_row = 0; b_row < b.rows();b_row++){
            //std::cout << "b_row: " << b_row << " b.row: " << b.rows() << std::endl;
            //std::cout << "mul lvl 3 - a value is: " << a[a_row][b_row] <<std::endl;
            //quiero obtener el valor en la casilla aij
            regType cell = sse3_set1<T,regType>(a[a_row][b_row]);

            //que quiero multiplicar?
            //multiplicar el valor de aij con el de bji
            regType mul = mm_mul<T,regType>(cell,*btmp);
        		val = mm_add<T,regType>(val,mul);
            /*
            std::cout<< "sz of rT: "<<  sizeof(regType) << " vector value: " << std::endl;
            std::cout << "{ ";
            for(size_t x = 0; x < sizeof(regType)/sizeof(T);x++)
              std::cout << *(reinterpret_cast<const float*>(btmp) + x) << " ";
            std::cout << "}\n";
            */
            //como obtener btmp
            btmp  += b_lim;
    			}
          /**
           * btmp = bptr
           * [ ] [x] [ ]
           * [ ] [ ] [ ]
           * [ ] [ ] [ ]
           */
          bptr++;


        	*here++ = val;

          /*
          std::cout << "========== c ===========" << std::endl;

          for(size_t i = 0; i < c.rows();i++){
            std::cout << "{ ";
            for(size_t j = 0; j < c.cols();j++)
              std::cout << c[i][j] << "\t";
            std::cout << "}" << std::endl;
          }

          */

    		}


    	}
      /*
      std::cout << "========== x ===========" << std::endl;
      std::cout << "========== x ===========" << std::endl;
      std::cout << "========== x ===========" << std::endl;
      std::cout << "========== x ===========" << std::endl;
      //exit(0);      
      */
    }

       
    // On-copy implementation c=a+b for SIMD-capable types
    template<typename T,
	     class Alloc,
	     typename std::enable_if<is_simd_type<T>::value,int>::type=0>
    inline void add(const Matrix<T,Alloc>& a,
                    const Matrix<T,Alloc>& b,
                    Matrix<T,Alloc>& c) {

      assert( (a.rows() == b.rows()) &&
              (a.cols() == b.cols()) );


      if (is_aligned_alloc<Alloc>::value) {        
#ifdef __SSE2__
        addSIMD<T,Alloc,typename sse2_traits<T>::reg_type>(a,b,c);
#else
        ::anpi::fallback::add(a,b,c);
#endif
      } else { // allocator seems to be unaligned
        ::anpi::fallback::add(a,b,c);
      }
    }

template<class M>
void printMat2(M c){
	std::cout << "MATRIX START HERE!" << std::endl;
	std::cout << "==================="<< std::endl;

	for (size_t i = 0; i < c.rows(); i++){
		for(size_t j = 0; j < c.cols();j++){
			std::cout << c[i][j] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << "==================="<< std::endl;
}


    // On-copy implementation c=a*b for SIMD-capable types
    template<typename T,
	     class Alloc,
	     typename std::enable_if<is_simd_type<T>::value,int>::type=0>
    inline void multiply(const Matrix<T,Alloc>& a,
                    const Matrix<T,Alloc>& b,
                    Matrix<T,Alloc>& c) {



      if (is_aligned_alloc<Alloc>::value) {        
#ifdef __SSE2__
        /*
        std::cout << "matrix A" << std::endl;
        printMat2<Matrix<T,Alloc>>(a);
        std::cout << "matrix B" << std::endl;
        printMat2<Matrix<T,Alloc>>(b);
        ::anpi::fallback::multiply(a,b,c);
        std::cout << "matrix C'" << std::endl;
        printMat2<Matrix<T,Alloc>>(c);
        */
        mulSIMD<T,Alloc,typename sse2_traits<T>::reg_type>(a,b,c);
        //std::cout << "matrix C" << std::endl;
        //printMat2<Matrix<T,Alloc>>(c);        
#else
        ::anpi::fallback::multiply(a,b,c);
#endif
      } else { // allocator seems to be unaligned
        ::anpi::fallback::multiply(a,b,c);
      }
    }






    // Non-SIMD types such as complex
    template<typename T,
             class Alloc,
             typename std::enable_if<!is_simd_type<T>::value,int>::type = 0>
    inline void add(const Matrix<T,Alloc>& a,
                    const Matrix<T,Alloc>& b,
                    Matrix<T,Alloc>& c) {
      
      ::anpi::fallback::add(a,b,c);
    }

    // In-place implementation a = a+b
    template<typename T,class Alloc>
    inline void add(Matrix<T,Alloc>& a,
                    const Matrix<T,Alloc>& b) {

      add(a,b,a);
    }


    /*
     * Subtraction
     */

    // Fall back implementations

    // In-copy implementation c=a-b
    template<typename T,class Alloc>
    inline void subtract(const Matrix<T,Alloc>& a,
                         const Matrix<T,Alloc>& b,
                         Matrix<T,Alloc>& c) {
      ::anpi::fallback::subtract(a,b,c);
    }

    // In-place implementation a = a-b
    template<typename T,class Alloc>
    inline void subtract(Matrix<T,Alloc>& a,
                         const Matrix<T,Alloc>& b) {

      ::anpi::fallback::subtract(a,b);
    }




    /*
     * Multiplication
     */

    // Fall back implementations

    // In-copy implementation c=a*b
    template<typename T,class Alloc,
             typename std::enable_if<!is_simd_type<T>::value,int>::type = 0>
    inline void multiply(const Matrix<T,Alloc>& a,
                         const Matrix<T,Alloc>& b,
                         Matrix<T,Alloc>& c) {
      ::anpi::fallback::multiply(a,b,c);
    }


    template<typename T,class Alloc>
    inline void multiply(const Matrix<T,Alloc>& a,
                         const std::vector<T>& b,
                         Matrix<T,Alloc>& c) {
      ::anpi::fallback::multiply(a,b,c);
    }


  } // namespace simd


  // The arithmetic implementation (aimpl) namespace
  // dispatches to the corresponding methods
#ifdef ANPI_ENABLE_SIMD
  namespace aimpl=simd;
#else
  namespace aimpl=fallback;
#endif
  
} // namespace anpi

#endif
