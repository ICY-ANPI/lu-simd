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

template<typename T>
bool solveLU( const anpi::Matrix<T>& A, std::vector <T>& x ,const std::vector <T>& b){

	try {

		std::vector<size_t> p;
		Matrix<T> LU,L,U;
		//Getting the LU matrix and p, the permutation vector
		luDoolittle(A,LU,p);
		unpackDoolittle(LU,L,U);


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
