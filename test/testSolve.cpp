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
		}

    }

	template<typename T>
    void solverTest2(const std::function<void(const Matrix<T>&,
                                         std::vector<T>&,
										 const std::vector<T>&)>& solver) {

    	anpi::Matrix<T> A;
    	A.allocate(6,6);
    	A = {{1,2,4,8,10,54},
			 {65,66,66,85,55,66},
			 {35,31,8,88,66,4},
			 {58,44,66,87,6,5},
			 {66,99,74,78,84,554},
			 {87,56,84,65,71,68}
			 
			 };
    	std::vector<T> b({37,13,66,64,84,6}),x,result({T(-1.38238373379),T(-3.26289178063),T(1.54533377877),T(2.09649074613),T(0.26131051497),T(0.35818150113)});
		//Matrix<T> LU;
		//std::vector<size_t> p;
		//luDoolittle<T>(A,LU,p);
    	solver(A,x,b);
		/*
		for(size_t i = 0 ; i < p.size();i++){
			std::cout << p[i]<<" ";
		}
		std::cout << std::endl;
		*/
    	const T eps = std::numeric_limits<T>::epsilon()*T(20000);

    	for (size_t i = 0; i < x.size(); i++) {
    		BOOST_CHECK(std::abs(x[i] - result[i]) < eps);
		}



    }


	template<typename T>
    void solverTest3(const std::function<void(const Matrix<T>&,
                                         std::vector<T>&,
										 const std::vector<T>&)>& solver) {
		anpi::Matrix<T> A;
    	A.allocate(7,7);
    	A = {
			 {5,8,4,5,1,7,96},
			 {5,44,88,56,47,44,5},
			 {87,42,11,27,84,54,21},
			 {58, 8,45,52,12,48,84},
			 {54,55,66,11,26,66,33},
			 {33,33,22,3,33,32,13},
			 {48,55,41,10,36,65,69}			 
			 };
    	std::vector<T> b2({65,8,66,54,32,73,74}),
			x2,result2({T(-13.924197024),
					  T(-64.693091981),
					  T(23.5281648033),
					  T(-40.3449819096),
					  T(35.2466805718),
					  T(32.5109196451),
					  T(5.17661305432)});
    	solver(A,x2,b2);
    	const T eps = std::numeric_limits<T>::epsilon()*T(500000);

    	for (size_t i = 0; i < x2.size(); i++) {
    		BOOST_CHECK(std::abs(x2[i] - result2[i]) < eps);
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
	anpi::test::solverTest2<float>(anpi::solveLU<float>);
	anpi::test::solverTest2<double>(anpi::solveLU<double>);
	anpi::test::solverTest3<float>(anpi::solveLU<float>);
	anpi::test::solverTest3<double>(anpi::solveLU<double>);

}

BOOST_AUTO_TEST_CASE(Invert)
{
	anpi::test::invertTest<float>();
	anpi::test::invertTest<double>();

}



BOOST_AUTO_TEST_SUITE_END()

