/*
 * testQR.cpp
 *
 *  Created on: Apr 2, 2018
 *      Author: cristian
 */






#include <boost/test/unit_test.hpp>

#include "QRdescomposition.hpp"

#include <iostream>
#include <exception>
#include <cstdlib>
#include <complex>

#include <functional>

#include <cmath>

namespace anpi {
  namespace test {

    /// Test the given closed root finder
    template<typename T>
    void qrTest(const std::function<void(const Matrix<T>&,
                                         Matrix<T>&,
										 Matrix<T>&)>& iqr) {

    	anpi::Matrix<T> As[] = {
    			{ { 2, 0,1,2},{-1,-2,1,2},{ 1, 1,1,1},{-1,-1,0,1} },
				{{54,226,661},{544,112,174},{88,44,15}},
				{{22,221,33},{656,444,111},{111,555,55}},
				{{511,84,21},{54,659,14},{752,54,61}}
    	};
    	//Epsilon range

    	//Float Eps: 0.00119209
    	//Double Eps: 2.22045e-12

    	const T eps = std::numeric_limits<T>::epsilon()*T(10000);
    	std::cout << "Eps: " << eps << std::endl;

    	for (Matrix<T> A : As){
    	      // The result
    	    	Matrix<T> Q;
    	    	// The result
    	    	Matrix<T> R;
    	      // Test decomposition
    	      {
    	        // same matrix as before, but already permuted to force a clean decomposition
    	        iqr(A,Q,R);
    	        Matrix<T> Ar=Q*R;
    	        Matrix<T> Qt;
    	        Qt.allocate(Q.cols(),Q.rows());

    	        for (size_t i = 0; i < A.rows(); i++) {
    	        	for (size_t j = i; j < A.rows(); j++) {
    	        		Qt[j][i] = Q[i][j];
    	        		Qt[i][j] = Q[j][i];
    	        	}
    	        }

    	        //Making the resulting identity matrix
    	        //if Ir is an identity matrix it provee that
    	        //Q is an ortogonal matrix
    	        Matrix<T> Ir = Qt*Q;


    	        //Making an identity matrix
    	        Matrix<T> I;
    	        I.allocate(Q.rows(),Qt.cols());


    	        for (size_t i = 0; i < A.rows(); i++)
    	        	for (size_t j = 0; j < A.rows(); j++)
    	        		I[i][j] = T(i==j);

    	        //Checking
    	        BOOST_CHECK(Ar.rows()==A.rows());
    	        BOOST_CHECK(Ar.cols()==A.cols());
    	        bool iguales = true;
    	        for (size_t i=0;i<Ar.rows();++i) {
    	          for (size_t j=0;j<Ar.cols();++j) {
    	        	//testing if Ir is a identity matrix
    	        	BOOST_CHECK(std::abs(Ir(i,j)-I(i,j)) < eps);
    	        	//testing the A and Ar error
    	            BOOST_CHECK(std::abs(Ar(i,j)-A(i,j)) < eps);
    	            iguales = iguales && std::abs(Ir(i,j)-I(i,j)) < eps;
    	            //proving if R is a superior matrix
    	            if (i > j)BOOST_CHECK(std::abs(R(i,j)) < eps);
    	          }
    	        }

    	      }
    	}

    }

  } // test
}  // anpi

BOOST_AUTO_TEST_SUITE( QR )

BOOST_AUTO_TEST_CASE(qr)
{
  anpi::test::qrTest<float>(anpi::qr<float>);
  anpi::test::qrTest<double>(anpi::qr<double>);


}


BOOST_AUTO_TEST_SUITE_END()
