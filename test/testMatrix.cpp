/**
 * Copyright (C) 2017 
 * Área Académica de Ingeniería en Computadoras, TEC, Costa Rica
 *
 * This file is part of the CE3102 Numerical Analysis lecture at TEC
 */


#include <boost/test/unit_test.hpp>

#include <iostream>
#include <exception>
#include <cstdlib>
#include <complex>

/**
 * Unit tests for the matrix class
 */

#include "Matrix.hpp"
#include "Allocator.hpp"

// Explicit instantiation of all methods of Matrix


// normal allocator


typedef std::complex<double> dcomplex;

typedef std::allocator<float> alloc;

template class anpi::Matrix<dcomplex,alloc>;
template class anpi::Matrix<double  ,alloc>;
template class anpi::Matrix<float   ,alloc>;
template class anpi::Matrix<int     ,alloc>;

typedef anpi::Matrix<dcomplex,alloc> cmatrix;
typedef anpi::Matrix<double  ,alloc> dmatrix;
typedef anpi::Matrix<float   ,alloc> fmatrix;
typedef anpi::Matrix<int     ,alloc> imatrix;



// aligned allocator
typedef anpi::aligned_allocator<float> aalloc;

template class anpi::Matrix<dcomplex,aalloc>;
template class anpi::Matrix<double  ,aalloc>;
template class anpi::Matrix<float   ,aalloc>;
template class anpi::Matrix<int     ,aalloc>;

typedef anpi::Matrix<dcomplex,aalloc> acmatrix;
typedef anpi::Matrix<double  ,aalloc> admatrix;
typedef anpi::Matrix<float   ,aalloc> afmatrix;
typedef anpi::Matrix<int     ,aalloc> aimatrix;

// row aligned allocator
typedef anpi::aligned_row_allocator<float> aralloc;

template class anpi::Matrix<dcomplex,aralloc>;
template class anpi::Matrix<double  ,aralloc>;
template class anpi::Matrix<float   ,aralloc>;
template class anpi::Matrix<int     ,aralloc>;

typedef anpi::Matrix<dcomplex,aralloc> arcmatrix;
typedef anpi::Matrix<double  ,aralloc> ardmatrix;
typedef anpi::Matrix<float   ,aralloc> arfmatrix;
typedef anpi::Matrix<int     ,aralloc> arimatrix;


#if 1
# define dispatchTest(func) \
  func<cmatrix>();          \
  func<dmatrix>();          \
  func<fmatrix>();          \
  func<imatrix>();          \
                            \
  func<acmatrix>();         \
  func<admatrix>();         \
  func<afmatrix>();         \
  func<aimatrix>();         \
                            \
  func<arcmatrix>();        \
  func<ardmatrix>();        \
  func<arfmatrix>();        \
  func<arimatrix>();

#else
# define dispatchTest(func) func<arfmatrix>(); 
#endif




#if 1
# define multiTest(func) \
  func<int>();			\
  func<float>();			\
  func<dcomplex>();			\
  func<double>();

#else
# define multiTest(func) func<float>();
#endif




BOOST_AUTO_TEST_SUITE( Matrix )

template<class M>
void testConstructors() {
  // Constructors
  { // default
    M a;
    BOOST_CHECK( a.rows() == 0);
    BOOST_CHECK( a.cols() == 0);
    BOOST_CHECK( a.dcols() == 0);
  }
  { // unitilialized
    M a(2,3,anpi::DoNotInitialize);
    BOOST_CHECK( a.rows() == 2);
    BOOST_CHECK( a.cols() == 3);
    BOOST_CHECK( a.dcols() >= 3);
  }
  { // default initialized
    M a(3,2);
    BOOST_CHECK( a.rows() == 3);
    BOOST_CHECK( a.cols() == 2);
    BOOST_CHECK( a(0,0) == typename M::value_type(0));
  }
  { // default initialized
    M a(3,2,typename M::value_type(4));
    BOOST_CHECK( a.rows() == 3);
    BOOST_CHECK( a.cols() == 2);
    BOOST_CHECK( a(0,0) == typename M::value_type(4));
  }
  { // initializer_list
    M a = { {1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15} };
    BOOST_CHECK( a.rows() == 3);
    BOOST_CHECK( a.cols() == 5);
    
    BOOST_CHECK( a(0,0) == typename M::value_type(1));
    BOOST_CHECK( a(1,2) == typename M::value_type(8));
    BOOST_CHECK( a(2,3) == typename M::value_type(14));
  }
  { // Copy constructor
    M a = { {1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15} };
    M b(a);

    BOOST_CHECK( a==b );
    BOOST_CHECK( b.rows() == 3 );
    BOOST_CHECK( b.cols() == 5 );
    BOOST_CHECK( b.data() != a.data());
  }

  { // Move constructor
    M a = { {1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15} };
    M b(std::move(a));

    BOOST_CHECK( b.rows() == 3 );
    BOOST_CHECK( b.cols() == 5 );

    BOOST_CHECK( a.empty() );
  }
  { // Mem constructor
    M a = { {1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15} };
    M b(a.rows(),a.cols(),a.data());

    BOOST_CHECK( a==b );
    BOOST_CHECK( b.rows() == 3 );
    BOOST_CHECK( b.cols() == 5 );
    BOOST_CHECK( b.data() != a.data() );
  }
}


/**
 * Instantiate and test the methods of the Matrix class
 */
BOOST_AUTO_TEST_CASE( Constructors ) {
  dispatchTest(testConstructors);
}

template<class M>
void testComparison() {
  // == and !=
  M a = { {1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15} };
  M b = { {1,2,3,4,5},{6,7,9,9,10},{11,12,13,14,15} };

  BOOST_CHECK( (a!=b) );
  
  b(1,2)=typename M::value_type(8);
  
  BOOST_CHECK( (a==b) );
}  

BOOST_AUTO_TEST_CASE(Comparison) 
{
  dispatchTest(testComparison);
}



//success case 1
// matrix * vector
template<class M,typename V>
void multiplicationTest1(){

	M a = { {1,2,-3},{4,0,-2}};
	V b = {1,2,3};
	M c = a*b;
	M result = {{-4},{-2}};
	BOOST_CHECK(c == result);

}

//success case 2
//matrix * matrix
template<class M,typename V>
void multiplicationTest2(){
	M a = {{5,84,21},{54,659,14},{752,54,61}};
	M b = {{54,226,661},{544,112,174},{88,44,15}};
	M result = {{47814,11462,18236},{362644,86628,150570},{75352,178684,507383}};
	M c = a*b;
	BOOST_CHECK(c == result);
}



//success case 3
//matrix * matrix
template<class M,typename V>
void multiplicationTest3(){
	M a = {{22,221,33},{656,4454,111},{111,555,55},{88,454,6565},{545,545,545}};
	M b = {{556,155,944,345},{126,545,545,544},{545,445,454,545}};
	M result = {{58063,138540,156195,145799},{986435,2578505,3097088,2709791},{161621,344155,432229,370190},{3684057,3182495,3311012,3855261},{668715,624025,1058935,781530}};

	M c = a*b;
	BOOST_CHECK(c == result);
}

//success case 4
//matrix * matrix
template<class M,typename V>
void multiplicationTest4(){
	M a = {{1,0,0},{0,1,0},{0,0,1}};
	M b = {{54,226,661},{544,112,174},{88,44,15}};
	M result = {{54,226,661},{544,112,174},{88,44,15}};


	M c = a*b;
	BOOST_CHECK(c == result);
}


//fail case 1
//inapropiate matrices size
template<class M,typename V>
void multiplicationTest5(){
	M a = { {1,2},{4,0}};
	M b = {{54,226,661},{544,112,174},{88,44,15}};
	try {
		M c = a*b;
		BOOST_CHECK(false && "Error at matrices bad size non-catched");

	} catch (anpi::Exception &e) {
		BOOST_CHECK(true && "Catched: matrix bad size");

	}
}

//fail case 2
//inapropiate matrix and vector size
template<class M,typename V>
void multiplicationTest6(){
	M a = { {1,2,-3},{4,0,-2},{4,0,-2},{4,0,-2}};
	V b = {1,2,3,4};
	try {
		M c = a*b;
		BOOST_CHECK(false && "Error at matrix or vector bad size non-catched");
	} catch (anpi::Exception &e) {
		BOOST_CHECK(true && "Catched: matrix or vector bad size");
	}
}


/**
 * This function test differents scenarios for matrix multiplication
 * verificate matrix and vector size before the multiplications
 */
template<typename T>
void multi3(){

	//success case 1
	multiplicationTest1<anpi::Matrix<T,std::allocator<T>>, std::vector<T>>();
	//success case 2
	multiplicationTest2<anpi::Matrix<T,std::allocator<T>>, std::vector<T>>();
	//success case 3
	multiplicationTest3<anpi::Matrix<T,std::allocator<T>>, std::vector<T>>();
	//success case 4
	multiplicationTest4<anpi::Matrix<T,std::allocator<T>>, std::vector<T>>();
	//fail case 1
	multiplicationTest5<anpi::Matrix<T,std::allocator<T>>, std::vector<T>>();
	//fail case 2
	multiplicationTest6<anpi::Matrix<T,std::allocator<T>>, std::vector<T>>();
}



BOOST_AUTO_TEST_CASE(Multi)
{
	multiTest(multi3);


}




template<class M>
void testAssignment() {
  { // Move assignment
    M a = { {1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15} };
    M c(a);
    M b;
    b=std::move(a);
    BOOST_CHECK(a.empty() );
    BOOST_CHECK(!b.empty() );
    BOOST_CHECK(b.rows()==3 );
    BOOST_CHECK(b.cols()==5 );
    BOOST_CHECK(b==c );
  }
  { // assignment
    M a = { {1,2,3,4,5},{5,6,7,8,9},{9,10,11,12,13} };
    M b;
    b=a;
    BOOST_CHECK(a==b );
  }  
  { // swap
    M a = { {1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15} };
    M b = { {13,14},{15,16} };

    M c(a);
    M d(b);

    BOOST_CHECK( a==c );
    BOOST_CHECK( d==b );
    
    c.swap(d);
    BOOST_CHECK( a==d );
    BOOST_CHECK( b==c );
  }
  { // column
    M a = { {1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15} };
    std::vector<typename M::value_type> col = a.column(1);
    std::vector<typename M::value_type> ref = {2,7,12};
    BOOST_CHECK( col == ref );
  }
}

BOOST_AUTO_TEST_CASE(Assignment)
{
  dispatchTest(testAssignment);
}

template<class M>
void testArithmetic() {
  
  {
    M a = { {1,2,3},{ 4, 5, 6} };
    M b = { {7,8,9},{10,11,12} };
    M r = { {8,10,12},{14,16,18} };
    
    M c(a);
    c+=b;
    BOOST_CHECK(c==r );
    c=a+b;
    BOOST_CHECK(c==r );


    c=M{ {1,2,3},{ 4, 5, 6} } + b;
    BOOST_CHECK(c==r );

    c=a+M{ {7,8,9},{10,11,12} };
    BOOST_CHECK(c==r );
  }

  {
    M a = { {1,2,3},{ 4, 5, 6} };
    M b = { {7,8,9},{10,11,12} };
    M r = { {-6,-6,-6},{-6,-6,-6} };
    
    M c(a);
    c-=b;
    BOOST_CHECK( c==r );
    c=a-b;
    BOOST_CHECK( c==r );


    c=M{ {1,2,3},{ 4, 5, 6} } - b;
    BOOST_CHECK( c==r );

    c=a-M{ {7,8,9},{10,11,12} };
    BOOST_CHECK( c==r );
  } 
}

BOOST_AUTO_TEST_CASE(Arithmetic) {
  dispatchTest(testArithmetic);  
}
  
BOOST_AUTO_TEST_SUITE_END()
