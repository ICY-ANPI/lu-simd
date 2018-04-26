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
		
		std::cout << "LUSIMD called" << std::endl;
		std::vector<size_t> p;
		Matrix<T> LU,L,U;
		//Getting the LU matrix and p, the permutation vector
		luDoolittle(A,LU,p);
		unpackDoolittle(LU,L,U);
		//luCrout(A,LU,p);
		//unpackCrout(LU,L,U);

		//creating a vector y where we'll save the result of y=Lx
		std::vector<T,anpi::aligned_row_allocator<T>> y(A.rows());

		x = std::vector<T>(A.rows());

		T val;

		size_t col_w =colw<T,regType>();
		//L part
		//We are writting the vector y here
		for (size_t i = 0; i < L.rows(); i++) {
			std::cout << "L PART" << std::endl;
			//we are accesing the position in permutation vector using b[p[i]]
			regType * ptr = reinterpret_cast<regType*>(L[i]);
			regType  val = sse3_set_s<T,regType>(b[p[i]]);
			size_t j = 0;
			size_t ic = reg_mul_value<T,regType>(i);
			//size_t ic = column_correction<T,regType>(b.dcols());
			for(; j < ic; j+=col_w){
				//if j != of i add
				//if (j != i)
				//y[j] is the previous found value
				regType mul = mm_mul<T,regType>(*ptr,sse3_set1<T,regType>(T(-1)));
				std::cout << "ITERATION 1: " << j << " OF " << ic << std::endl;
				for (size_t k = 0;  k < y.size(); k++) std::cout << y[k] << " ";
				std::cout << std::endl;
				std::cout << "DIR IS: " << *(&y.front() + j) << " SIZE OF T IS: " << sizeof(T)<< std::endl;
				regType * tmpreg = reinterpret_cast<regType*>(&y.front() + j);
				std::cout << "TMP REG IS DEFINED\n";
				mul = mm_mul<T,regType>(mul,*tmpreg);
				std::cout << "ITERATION 2: " << j << " OF " << ic << std::endl;
				val = mm_add<T,regType>(val,mul);
				std::cout << "ITERATION 3: " << j << " OF " << ic << std::endl;
				ptr++;
			}
			std::cout << "L PART 2" << std::endl;
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
				std::cout << "LUSIMD called" << std::endl;
				return solveLUSIMD<T,typename sse2_traits<T>::reg_type>(A,x ,b);
			#else
				std::cout << "LUSIMD isn't called called" << std::endl;
				return solveLUAux<T>(A,x ,b);
			#endif
		#else
			std::cout << "LUAux is called" << std::endl;
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
