/*
 * testSolve.cpp
 *
 *  Created on: Apr 5, 2018
 *      Author: cristian
 */








#include <boost/test/unit_test.hpp>

#include "Solve.hpp"

#include "MatrixInverse.hpp"
#include "LUDoolittle.hpp"

#include <iostream>
#include <exception>
#include <cstdlib>
#include <complex>

#include <functional>

#include "Matrix.hpp"

#include <cmath>

namespace anpi {
  namespace test {

    /// Test the given closed root finder
    template<typename T>
    void solverTest(const std::function<void(const Matrix<T>&,
                                         std::vector<T>&,
										 const std::vector<T>&)>& solver) {

    	anpi::Matrix<T> A;
    	A.allocate(3,3);
    	A = {{ 1, 1,1},{-1,-2,1},{ 2,0,1}};
    	std::vector<T> b({9,12,8}),x,result({-T(6)/T(5),-T(1)/T(5),T(52)/T(5)});

    	solver(A,x,b);
    	const T eps = std::numeric_limits<T>::epsilon()*T(100);

    	for (size_t i = 0; i < x.size(); i++) {
    		BOOST_CHECK(std::abs(x[i] - result[i]) < eps);
			std::cout << "xi: " << x[i] << " res: " << result[i] << std::endl;
		}

    }


    template<typename T>
    void invertTest(){

    	anpi::Matrix<T> A,Ai,result;
    	A.allocate(3,3);
    	result.allocate(3,3);
    	//theorical value
    	result = {{-9,3,-4},{3,-1,1},{4,-1,2}};
    	A = {{1,2,1},{2,2,3},{-1,-3,0}};
    	//resulting value is Ai
    	anpi::invert<T>(A,Ai);
    	const T eps = std::numeric_limits<T>::epsilon();
    	for (size_t i = 0; i < 3; i++) {
    		for (size_t j = 0; j < 3; j++){
    			//Comparing the theorical value and Ai value
    			BOOST_CHECK(std::abs(Ai[i][j] - result[i][j]) < eps);
    		}
    	}

    }

  } // test
}  // anpi

BOOST_AUTO_TEST_SUITE( Solve )

BOOST_AUTO_TEST_CASE(qrSolver)
{
	anpi::test::solverTest<float>(anpi::solveQR<float>);
	anpi::test::solverTest<double>(anpi::solveQR<double>);


}


BOOST_AUTO_TEST_CASE(LUSolver)
{
	anpi::test::solverTest<float>(anpi::solveLU<float>);
	anpi::test::solverTest<double>(anpi::solveLU<double>);

}

BOOST_AUTO_TEST_CASE(Invert)
{
	anpi::test::invertTest<float>();
	anpi::test::invertTest<double>();

}



BOOST_AUTO_TEST_SUITE_END()

