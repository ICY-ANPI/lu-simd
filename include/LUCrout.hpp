/**
 * Copyright (C) 2018
 * Área Académica de Ingeniería en Computadoras, ITCR, Costa Rica
 *
 * This file is part of the numerical analysis lecture CE3102 at TEC
 *
 * @Author: 
 * @Date  : 03.03.2018
 */

#include <cmath>
#include <limits>
#include <functional>

#include "Exception.hpp"
#include "Matrix.hpp"


#include "LUDoolittle.hpp"

#ifndef ANPI_LU_CROUT_HPP
#define ANPI_LU_CROUT_HPP

namespace anpi {

/**
 * Auxiliary method used to debug LU decomposition.
 *
 * It separates a packed LU matrix into the lower triangular matrix
 * L and the upper triangular matrix U, such that the diagonal of U
 * is composed by 1's.
 */
template<typename T>
void unpackCrout(const Matrix<T>& LU,
		Matrix<T>& L,
		Matrix<T>& U) {


	L.allocate(LU.rows(),LU.cols());
	U.allocate(LU.rows(),LU.cols());


	for (size_t i = 0; i < LU.rows(); ++i) {
		for (size_t j = 0; j < LU.cols(); ++j) {
			if(i > j){
				U[i][j] = T(0);
				L[i][j] = LU[i][j];
			}
			else if (i == j){
				U[i][j] = T(1);
				L[i][j] = LU[i][j];
			}
			else{
				L[i][j] = T(0);
				U[i][j] = LU[i][j];

			}
		}
	}

	/*
	std::cout << "LU, L y U son: " << std::endl;

	printMat<Matrix<T>>(LU);
	printMat<Matrix<T>>(L);
	printMat<Matrix<T>>(U);
	 */
	//throw anpi::Exception("To be implemented yet");

}

/**
 * Decompose the matrix A into a lower triangular matrix L and an
 * upper triangular matrix U.  The matrices L and U are packed into
 * a single matrix LU.
 *
 * Crout's way of packing assumes a diagonal of
 * 1's in the U matrix.
 *
 * @param[in] A a square matrix
 * @param[out] LU matrix encoding the L and U matrices
 * @param[out] permut permutation vector, holding the indices of the
 *             original matrix falling into the corresponding element.
 *             For example if permut[5]==3 holds, then the fifth row
 *             of the LU decomposition in fact is dealing with the third
 *             row of the original matrix.
 *
 * @throws anpi::Exception if matrix cannot be decomposed, or input
 *         matrix is not square.
 */
template<typename T>
void luCrout(const Matrix<T>& A,
		Matrix<T>& LU,
		std::vector<size_t>& permut) {


	//Verifying if the matrix is a 0x0 matrix
	if (A.cols() <= size_t(0) || A.rows() <= size_t(0))
		throw anpi::Exception("0x0 matrix found");

	//verifying if the matrix A is a squared matrix
	if (A.cols() != A.rows())
		throw anpi::Exception("The input matrix isn't a square matrix.");


	//setting the permut vector as 1,2,3,4.... n-1, with n the size of matrix nxn
	for (size_t x = 0; x < A.rows();x++)
		permut.push_back(x);


	//Allocating the matrix LU
	LU.allocate(A.rows(), A.rows());
	//Setting the initial values of LU, First row and first col
	//LU = A;

	for (size_t i = 0; i < A.cols(); i++) {
		for (size_t j = 0; j < A.cols(); j++) {
			LU[i][j] = 0;
			if (j == 0){
				LU[i][j] = A[i][j];
			}
			else if(i == 0 && j != 0)
				LU[i][j] = A[i][j]/A[0][0];
		}
	}
	//printMat<Matrix<T>>(LU);
	for (size_t col = 1; col < A.cols(); col++) {
		//setting the values for L
		for(size_t row = col; row < A.rows();row++){
			T L = A[row][col];
			for (size_t k = 0; k < col; k++) {
				L -= LU[row][k] * LU[k][col];
			}
			LU[row][col] = L;
		}

		//printMat<Matrix<T>>(LU);
		//setting the values for U
		for(size_t row = col+1; row < A.rows();row++){
			T U = A[col][row];
			T Lfactor = LU[col][col];


			//std::cout << "U(" << col << "," << row <<") = [" << U << " - (";

			if (Lfactor == T(0)) // use epsilon
				throw anpi::Exception("Cannont calculate U value");
			for (size_t k = 0; k < col; k++) {
				//std::cout << LU[col][k] << "*" << LU[k][row] << " + ";
				U -= LU[col][k] * LU[k][row];

			}
			//std::cout << "0)]/" << Lfactor << std::endl;
			LU[col][row] = U/Lfactor;
		}

		//printMat<Matrix<T>>(LU);

		//throw anpi::Exception("I want to stop here!");


	}


	//std::cout << "La matriz LU es: " << std::endl;

	//printMat<Matrix<T>>(LU);

	//throw anpi::Exception("To be implemented yet");
}

}

#endif

