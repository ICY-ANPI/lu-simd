/*
 * MatrixInverse.hpp
 *
 *  Created on: Apr 5, 2018
 *      Author: cristian
 */



#include <cmath>
#include <limits>
#include <functional>
#include <vector>

#include "Exception.hpp"
#include "Matrix.hpp"


#include "LUDoolittle.hpp"
#include "LUCrout.hpp"




#ifndef INCLUDE_MATRIXINVERSE_HPP_
#define INCLUDE_MATRIXINVERSE_HPP_

namespace anpi{

template<typename T>
void invert ( const anpi::Matrix<T>& A, anpi::Matrix<T>& Ai ){
	std::vector<size_t> p;
	Matrix<T> LU,L,U;
	//Getting the LU matrix and p, the permutation vector
	luDoolittle(A,LU,p);
	//getting the L and U matrices
	unpackDoolittle(LU,L,U);


	Ai.allocate(A.rows(),A.cols());

	//creating a vector y where we'll save the result of y=Lx


	T val;


	for (size_t column = 0; column < A.cols(); column++) {
		//inicializing the vector z
		std::vector<T> z;
		//L part
		//in this part we are writting the vector z
		//Lz = Ck
		//where Ck is the column k of the identity matrix.
		//So, this is the forward sustitution
		for (size_t i = 0; i < A.rows(); i++) {
			//
			val = T(p[i] == column);
			for(size_t j = 0; j < i; j++){
				//if j != of i add
				if (j != i)
					//y[j] is the previous found value
					val -= L[i][j]*z[j];
			}
			//then, divide by the current z_i
			val /= L[i][i];
			//and push the result of this to the vector z
			z.push_back(val);
		}


		//U part
		//now we'll do the same that previously done
		//but, from the botton of matrix. because the matrix U have a single value at row k
		//where k is the total of rows of the matrix.
		for (size_t i = 0; i < A.rows(); i++) {
			val = z[A.rows()- 1 - i];
			for(size_t j = 0; j < i; j++){

				if (j != i){
					val -= U[A.rows()- 1 - i][A.rows()- 1 - j]*Ai[A.rows()- 1 - j][column];
				}
			}
			val /= U[A.rows()- 1 - i][A.rows()- 1 - i];
			Ai[A.rows()- 1 - i][column] = (val);
		}
	}

}


}



#endif /* INCLUDE_MATRIXINVERSE_HPP_ */
