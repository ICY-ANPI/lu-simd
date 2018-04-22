/*
 * QRdescomposition.hpp
 *
 *  Created on: Apr 2, 2018
 *      Author: cristian
 */


#include <cmath>
#include <limits>
#include <functional>

#include <vector>

#include "Exception.hpp"
#include "Matrix.hpp"


#include "LUDoolittle.hpp"




#ifndef INCLUDE_QRDESCOMPOSITION_HPP_
#define INCLUDE_QRDESCOMPOSITION_HPP_






namespace anpi {

using namespace std;


template<typename T>
void qr( const Matrix<T>& A, Matrix<T>& Q, Matrix<T>& R);



template<typename T>
void qr( const Matrix<T>& A, Matrix<T>& Q, Matrix<T>& R){

	//find the H matrix
	Q.allocate(A.rows(),A.cols());
	R.allocate(A.rows(),A.cols());

	//Q = A;


	//Setting identity matrix in Q
	//It doesn't affect the matrix product Q' = Q*H
	//if Q is an identity matrix we know that Q' = H
	for (size_t i = 0; i < A.rows(); i++) {
		for (size_t j = 0; j < A.rows(); j++) {
			if (i==j)
				Q[i][j] = T(1);
			else
				Q[i][j] = T(0);
		}
	}

	Matrix<T> H;
	H.allocate(A.rows(),A.cols());

	Matrix<T> D;
	D = A;

	for (size_t diagonal = 0; diagonal+1  < A.rows(); diagonal++) {

		//save the vector X
		vector<T> vectorX;
		T moduleX(T(0)),current;
		for (size_t i = diagonal; i < A.rows(); i++) {
			current = D[i][diagonal];
			vectorX.push_back(current);
			moduleX += current*current;
		}

		//vectorU = VectorX - || VectorX || * e_k
		//building the vector U for calculating the H matrix
		T sign = vectorX[0] <= T(0)? T(1): T(-1);
		T tmp = vectorX[0];
		vectorX[0] =  vectorX[0] + sign* std::sqrt(moduleX);
		//setting the vector U
		moduleX += vectorX[0]*vectorX[0] - tmp*tmp;



		//Building the external part of H matrix
		//      | I |  ZT  |
		// Q =  |---------|
		//      | Z | H_k |
		// Where the external part is composed by I and Z
		// Where I is the identity matrix
		// And Z is a matrix filled by zeros, and ZT the transposed of Z
		// for simplicity we are making the Q = H matrix as an identity matrix
		for (size_t i = 0; i < A.rows(); i++)
			for (size_t j = 0; j < A.rows(); j++)
					H[i][j] = T(i==j);

		//Building the H matrix
		//Now we are setting the values in H_k part.
		for (size_t i = diagonal; i < A.rows(); i++) {
			for (size_t j = diagonal; j < A.rows(); j++) {
				H[i][j] += T(-2) *(vectorX[i-diagonal]* (vectorX[j-diagonal])/moduleX);
			}
		}

		//QT is Qn-1 * Qn-2 * ... * Q1
		Q = H*Q;
		//D is the partial triangular matrix composed by the
		//matrix product of Q and A respectively
		D = Q*A;

	}

	R = Q*A;


	//This part is important, because we are setting the correct values of
	// the matrix Q
	// This matrix is tranposed. For this reason we are transposing it.
	T temp;
	for (size_t i = 0; i < A.rows(); i++) {
		for (size_t j = i; j < A.rows(); j++) {
			temp = Q[i][j];
			Q[i][j] = Q[j][i];
			Q[j][i] = temp;
		}
	}

}






}//end namespace anpi





#endif /* INCLUDE_QRDESCOMPOSITION_HPP_ */
