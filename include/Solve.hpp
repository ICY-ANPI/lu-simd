/*
 * Solve.hpp
 *
 *  Created on: Apr 4, 2018
 *      Author: cristian
 */


#include <cmath>
#include <limits>
#include <functional>
#include <vector>

#include "Exception.hpp"
#include "Matrix.hpp"


#include "LUDoolittle.hpp"
#include "QRdescomposition.hpp"

#ifndef INCLUDE_SOLVE_HPP_
#define INCLUDE_SOLVE_HPP_





namespace anpi {


#ifdef ANPI_ENABLE_SIMD
#include "bits/MatrixArithmetic.hpp"
#include "Intrinsics.hpp"

using namespace anpi::simd;

template<typename T,typename regType>
bool solveLUSIMD( const anpi::Matrix<T>& A, std::vector <T>& x ,const std::vector <T>& b){
		
		//Instance of  permut vector
		std::vector<size_t> p;
		//Instance of Matrices LU, L and U
		//Where LU is the packed matrix, L and U are the unpacked
		Matrix<T> LU,L,U;
		//Getting the LU matrix and p, the permutation vector
		luDoolittle(A,LU,p);
		unpackDoolittle(LU,L,U);
		//luCrout(A,LU,p);
		//unpackCrout(LU,L,U);

		//creating a vector y where we'll save the result of y=Lx
		std::vector<T,anpi::aligned_row_allocator<T>> y(A.rows());

		x = std::vector<T>(A.dcols());
		std::fill(x.begin(),x.end(),T(0));

		regType  val;

		size_t col_w =colw<T,regType>();
		regType minus_one = sse3_set1<T,regType>(T(-1));
		regType * ptr;
		regType * x_val;
		//L part
		//We are writting the vector y here
		for (size_t i = 0; i < L.rows(); i++) {
			//we are accesing the position in permutation vector using b[p[i]]
			ptr = reinterpret_cast<regType*>(L[i]);
			val = sse3_set_s<T,regType>(b[p[i]]);
			size_t j = 0;
			size_t ic = reg_mul_value<T,regType>(i);
			for(; j < ic; j+=col_w){
				regType mul = mm_mul<T,regType>(*ptr++,minus_one);
				x_val = reinterpret_cast<regType*>(&y.front() + j);
				mul = mm_mul<T,regType>(mul,*x_val++);
				val = mm_add<T,regType>(val,mul);
			}
			for(; j < i; j++){
				val = mm_add_s<T,regType>(val,sse3_set_s<T,regType>(T(-1)*y[j]*L[i][j]));
			}
			val = mm_hadd<T,regType>(val,val);
			y[i] = (mm_cvts<T,regType>(val)/L[i][i]);
		}


		//U part
		//now we'll do the same that previously done
		//but, from the botton of matrix. because the matrix U have a single value at row k
		//where k is the total of rows of the matrix.
		for (size_t i = U.rows()-1; i < U.rows(); i--) {
			val = sse3_set_s<T,regType>(y[i]);
			size_t j  = U.dcols() - col_w;
			ptr= reinterpret_cast<regType*>(U[i] + j);
			x_val = reinterpret_cast<regType*>( (x.data() + j));
			//Por el momento J va a irse hasta cero
			for(; j > i - i%col_w + col_w -1 ; j-=col_w){
				regType mul = mm_mul<T,regType>(*ptr--,minus_one);	
				mul = mm_mul<T,regType>(mul,*x_val--);
				val = mm_add<T,regType>(val,mul);
			}
			for(size_t k = i+1; k < j + col_w && k < U.cols(); k++){
				val = mm_add_s<T,regType>(val,sse3_set_s<T,regType>(T(-1)*x[k]*U[i][k]));
			}

			val = mm_hadd<T,regType>(val,val);
			x[i] = (mm_cvts<T,regType>(val))/(U[i][i]);
		}
		for(size_t k = U.dcols(); k > U.cols(); k--)x.pop_back();
		return true;







}

#endif


template<typename T>
bool solveLUAux( const anpi::Matrix<T>& A, std::vector <T>& x ,const std::vector <T>& b){
		std::vector<size_t> p;
		Matrix<T> LU,L,U;
		//Getting the LU matrix and p, the permutation vector
		//luDoolittle(A,LU,p);
		//unpackDoolittle(LU,L,U);
		luCrout(A,LU,p);
		unpackCrout(LU,L,U);

		//creating a vector y where we'll save the result of y=Lx
		std::vector<T> y;

		x = std::vector<T>(A.rows());


		T val;



		
		//L part
		//We are writting the vector y here
		for (size_t i = 0; i < A.rows(); i++) {
			//we are accesing the position in permutation vector using b[p[i]]
			val = b[p[i]];
			for(size_t j = 0; j < i; j++){
				//if j != of i add
				if (j != i)
					//y[j] is the previous found value
					val -= L[i][j]*y[j];
			}
			//then, divide by the current x_i
			val /= L[i][i];
			//and push the result of this to the vector y
			y.push_back(val);
		}


		//U part
		//now we'll do the same that previously done
		//but, from the botton of matrix. because the matrix U have a single value at row k
		//where k is the total of rows of the matrix.
		for (size_t i = 0; i < A.rows(); i++) {
			val = y[A.rows()- 1 - i];
			for(size_t j = 0; j < i; j++){
				if (j != i)
					val -= U[A.rows()- 1 - i][A.rows()- 1 - j]*x[A.rows()- 1 - j];
			}
			val /= U[A.rows()- 1 - i][A.rows()- 1 - i];
			x[A.rows()- 1 - i] = (val);
		}
		return true;
}

template<typename T>
bool solveLU( const anpi::Matrix<T>& A, std::vector <T>& x ,const std::vector <T>& b){

	try {
		#ifdef ANPI_ENABLE_SIMD
			#ifdef __SSE3__
				return solveLUSIMD<T,typename sse2_traits<T>::reg_type>(A,x ,b);
			#else
				return solveLUAux<T>(A,x ,b);
			#endif
		#else
			return solveLUAux<T>(A,x ,b);
		#endif
		
	} catch (anpi::Exception &e) {
		return false;
	}


}

template<typename T>
bool solveQR ( const anpi::Matrix<T>& A, std::vector <T>& x ,const std::vector <T>& b ){

	try {
		Matrix<T> Q,R;
		qr(A,Q,R);
		std::vector<T> b2;
		T val;
		//for some reason the multiplication of matrix and vector
		//didn't work here, for this reason we are making it
		for (size_t i = 0; i < b.size(); i++) {
			val = T(0);
			for (size_t j = 0; j < b.size(); ++j) {
				//virtually Q[j][i] is the Transpose position of Q[i][j]
				val += b[j]*Q[j][i];
			}
			//pushing the result in b2
			b2.push_back(val);
		}

		x = std::vector<T>(A.rows());


		//Because R is a superior matrix we are using
		//R[A.rows()- 1 - i][A.rows()- 1 - j] positions
		for (size_t i = 0; i < A.rows(); i++) {
			val = b2[A.rows()- 1 - i];
			for(size_t j = 0; j < i; j++){
				//if j and i are different add the oposite
				if (j != i)
					val -= R[A.rows()- 1 - i][A.rows()- 1 - j]*x[A.rows()- 1 - j];
			}
			//if j and i are equal divide.
			val /= R[A.rows()- 1 - i][A.rows()- 1 - i];
			x[A.rows()- 1 - i] = (val);
		}


		return true;

	} catch (anpi::Exception &e) {
		return false;
	}


}




}






#endif /* INCLUDE_SOLVE_HPP_ */
