/**
 * Copyright (C) 2017 
 * Área Académica de Ingeniería en Computadoras, TEC, Costa Rica
 *
 * This file is part of the CE3102 Numerical Analysis lecture at TEC
 *
 * @Author: Pablo Alvarado
 * @Date  : 10.02.2018
 */

#include <boost/test/unit_test.hpp>

#include "LUCrout.hpp"
#include "LUDoolittle.hpp"

#include <iostream>
#include <exception>
#include <cstdlib>
#include <complex>

#include <functional>

#include <cmath>

#include "Solve.hpp"

namespace anpi {
  namespace test {
    
    /// Test the given closed root finder
    template<typename T>
    void luTest(const std::function<void(const Matrix<T>&,
                                         Matrix<T>&,
                                         std::vector<size_t>&)>& decomp,
                const std::function<void(const Matrix<T>&,
                                         Matrix<T>&,
                                         Matrix<T>&)>& unpack) {

      // The result
      Matrix<T> LU;

      // Test if a non-square matrix is successfully detected
      {
        Matrix<T> A = {{1,7,6,4},{2,17,27,17}};
        std::vector<size_t> p;
        try {
          decomp(A,LU,p);
          BOOST_CHECK_MESSAGE(false,"Rectangular matrix not properly catched");
        }
        catch(anpi::Exception& exc) {
          BOOST_CHECK_MESSAGE(true,"Rectangular matrix properly detected");
        }
      }

      // Test pivoting
      {
        anpi::Matrix<T> A = { {-1,-2,1,2},{ 2, 0,1,2},{-1,-1,0,1},{ 1, 1,1,1} };
        std::vector<size_t> p;
        decomp(A,LU,p);

        std::vector<size_t> gp= {1,0,3,2};
        BOOST_CHECK(gp==p);
      }
      
      // Test decomposition
      {
        // same matrix as before, but already permuted to force a clean decomposition
        anpi::Matrix<T> A = { { 2, 0,1,2},{-1,-2,1,2},{ 1, 1,1,1},{-1,-1,0,1} };
        std::vector<size_t> p;
        decomp(A,LU,p);
        Matrix<T> L,U;
        unpack(LU,L,U);
        Matrix<T> Ar=L*U;




        const T eps = std::numeric_limits<T>::epsilon();

        BOOST_CHECK(Ar.rows()==A.rows());
        BOOST_CHECK(Ar.cols()==A.cols());
        bool iguales = true;
        for (size_t i=0;i<Ar.rows();++i) {
          for (size_t j=0;j<Ar.cols();++j) {
            BOOST_CHECK(std::abs(Ar(i,j)-A(i,j)) < eps);
            iguales = iguales && std::abs(Ar(i,j)-A(i,j)) < eps;

          }
        }

      }
    }



  /// Test the given closed root finder
      template<typename T>
      void unpackTest(const std::function<void(const Matrix<T>&,
              Matrix<T>&,
              std::vector<size_t>&)>& decomp,

			  const std::function<void(const Matrix<T>&, Matrix<T>&,
              Matrix<T>&)>& unpack) {


    	  Matrix<T>  As[] = { {{54,226,661},{544,112,174},{88,44,15}},
    			  {{22,221,33},{656,4454,111},{111,555,55}},
				  {{5,84,21},{54,659,14},{752,54,61}},
          {
            {1,2,4,8,10,54},
			      {65,66,66,85,55,66},
			      {35,31,8,88,66,4},
			      {58,44,66,87,6,5},
			      {66,99,74,78,84,554},
			      {87,56,84,65,71,68}
			    },
          {
			      {5,8,4,5,1,7,96},
			      {5,44,88,56,47,44,5},
			      {87,452,211,227,884,54,212},
			      {58, 889,45,542,112,548,84},
			      {54,55,66,11,226,66,33},
			      {33,33,22,3,33,32,13},
			      {48,655,41,10,36,65,69}			 
			    }

    	  };

    	  for(Matrix<T> A : As){

        	  // The result
        	  Matrix<T> LU;

        	  // Test decomposition
        	  // same matrix as before, but already permuted to force a clean decomposition
        	  std::vector<size_t> p;
        	  Matrix<T> L,U;
        	  decomp(A,LU,p);

        	  unpack(LU,L,U);

        	  Matrix<T> P;
        	  P.allocate(A.rows(),A.cols());

        	  for (size_t i=0;i<A.rows();i++) {
        		  for (size_t j=0;j<A.cols();j++)
        			  P[i][j] = T(i == p[j]);

        	  }

        	  Matrix<T> Ar=P*(L*U);

        	  const T eps = std::numeric_limits<T>::epsilon()*T(1000);

        	  BOOST_CHECK(Ar.rows()==A.rows());
        	  BOOST_CHECK(Ar.cols()==A.cols());
        	  for (size_t i=0;i<Ar.rows();++i) {
        		  for (size_t j=0;j<Ar.cols();++j) {
        			  //std::cout << "A: " << A[i][j] << " Ar: " << Ar[i][j] << std::endl;
        			  BOOST_CHECK(std::abs(Ar(i,j)-A(i,j)) < eps);

        		  }
        	  }
    	  }
      }



  } // test
}  // anpi




BOOST_AUTO_TEST_SUITE( LU )

BOOST_AUTO_TEST_CASE(Doolittle) 
{
  anpi::test::luTest<float>(anpi::luDoolittle<float>,
                            anpi::unpackDoolittle<float>);
  anpi::test::luTest<double>(anpi::luDoolittle<double>,
                             anpi::unpackDoolittle<double>);
}

BOOST_AUTO_TEST_CASE(Crout) 
{
  anpi::test::luTest<float>(anpi::luCrout<float>,anpi::unpackCrout<float>);
  anpi::test::luTest<double>(anpi::luCrout<double>,anpi::unpackCrout<double>);
}


BOOST_AUTO_TEST_CASE(UnpackDoolittle)
{
	anpi::test::unpackTest<float>(anpi::luDoolittle<float>,
	                            anpi::unpackDoolittle<float>);
	anpi::test::unpackTest<double>(anpi::luDoolittle<double>,
	                             anpi::unpackDoolittle<double>);

}

BOOST_AUTO_TEST_CASE(UnpackCrout)
{

	anpi::test::unpackTest<float>(anpi::luCrout<float>,anpi::unpackCrout<float>);
	anpi::test::unpackTest<double>(anpi::luCrout<double>,anpi::unpackCrout<double>);


}


BOOST_AUTO_TEST_SUITE_END()
