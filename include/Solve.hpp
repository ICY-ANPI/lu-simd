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
		//Where LU is the packed matrix of L and U matrices
		Matrix<T> LU;
		//Getting the LU matrix and p, the permutation vector
		luDoolittle(A,LU,p);
		//isDoolittle allow to do the doolittle method or crout method
		//true if is doolittle method
		//false if not.
		bool isDoolittle = true;
		//luCrout(A,LU,p);
		//bool isDoolittle = false;

		//creating a vector y where we'll save the result of y=Lx
		//it is important that this vector have an allocator
		std::vector<T,anpi::aligned_row_allocator<T>> y(A.rows());

		//Defining the size of x
		x = std::vector<T>(A.dcols());
		std::fill(x.begin(),x.end(),T(0));
		
		//val is the temporal value of x_i in SIMD
		regType  val;

		//col_w is the quantity of T values which can be added to regType
		//in sse3 sizeof(regType) took the value 16 bytes
		//and T could be 4 for floats or 8 for doubles
		//so if we are using floats col_w is 4, then for doubles col_w is going to be 2
		size_t col_w =colw<T,regType>();
		regType minus_one = sse3_set1<T,regType>(T(-1));
		//it is the pointer over the row in the LU matrix
		regType * ptr;
		//it is the pointer over the vector x
		regType * x_val;

		//is a temporary value where a multiplication of LU[k] * x[k] or LU[k] * y[k]
		//would be saved
		regType mul;

		//L part
		//We are writting the vector "y" here
		for (size_t i = 0; i < LU.rows(); i++) {
			//defining the pointer ptr at the begining of the current row
			ptr = reinterpret_cast<regType*>(LU[i]);
			//we are accesing the position in permutation vector using b[p[i]]
			//also we are defining "val" as SIMD vector of T values
			//in this case we are setting the first element as b[p[i]] value
			//Example for float: [0 0 0 x] 
			//Example for double [0 x]
			val = sse3_set_s<T,regType>(b[p[i]]);
			size_t j = 0;
			//ic the previous value of "i" which is a multiplier of col_w 
			//For example, if i is 5 and col_w is 4 ic is going to take the value of 4
			//or if "i" is 7 and col_w is 2 ic will take the value of 6
			size_t ic = reg_mul_value<T,regType>(i);

			//is the pointer which explore the y vector
			x_val = reinterpret_cast<regType*>(&y.front());

			//for each j, where j is col_w * k, where k is the number of repetitions
			for(; j < ic; j+=col_w){
				//change the sign of values in ptr
				mul = mm_mul<T,regType>(*ptr++,minus_one);
				//saving a vector of "- LU[k] * y[k]" in mul 
				mul = mm_mul<T,regType>(mul,*x_val++);
				//updating the value of  val
				val = mm_add<T,regType>(val,mul);
			}
			//for the values between ic and i do the same but i by 1
			for(; j < i; j++){
				val = mm_add_s<T,regType>(val,sse3_set_s<T,regType>(T(-1)*y[j]*LU[i][j]));
			}
			//values in val do an horizontal addition
			//val[0] + val[1] + val[2] ...
			val = mm_hadd<T,regType>(val,val);
			//cast the value of val to T value and save it in y[i]
			y[i] = (mm_cvts<T,regType>(val));
			//and if is a doolittle method divide by LU[i][i]
			if (!isDoolittle) y[i] = y[i]/LU[i][i];
		}


		//U part
		//now we'll do the same that previously done
		//but, from the botton of matrix. because the matrix U have a single value at row k
		//where k is the total of rows of the matrix.
		for (size_t i = LU.rows()-1; i < LU.rows(); i--) {
			//setting val as [0 0 ... 0 y[i]]
			val = sse3_set_s<T,regType>(y[i]);
			//
			size_t j  = LU.dcols() - col_w;
			ptr= reinterpret_cast<regType*>(LU[i] + j);
			x_val = reinterpret_cast<regType*>( (x.data() + j));
			size_t ic = reg_mul_value<T,regType>(i) + col_w -1;
			//it is:
			//x -= LU[k]*x[k] + LU[k-1]*x[k-1] + ... LU[ic]*x[ic] 
			for(; j > ic ; j-=col_w){
				mul = mm_mul<T,regType>(*ptr--,minus_one);	
				mul = mm_mul<T,regType>(mul,*x_val--);
				val = mm_add<T,regType>(val,mul);
			}
			//x -= LU[i+1] + ... + LU[ic-1]*x[ic-1]
			for(size_t k = i+1; k < j + col_w && k < LU.cols(); k++){
				val = mm_add_s<T,regType>(val,sse3_set_s<T,regType>(T(-1)*x[k]*LU[i][k]));
			}

			//horizontal addition of value
			val = mm_hadd<T,regType>(val,val);

			x[i] = (mm_cvts<T,regType>(val));
			//if is doolittle method, divide by LU[i][i]
			if (isDoolittle) x[i] = x[i]/(LU[i][i]);
		}
		//removing innecesary elements of x vector.
		for(size_t k = LU.cols(); k < LU.dcols(); k++)x.pop_back();
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
