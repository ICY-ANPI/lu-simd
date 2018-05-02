#include "Exception.hpp"
#include "Matrix.hpp"
#include <vector>
#ifndef MATRIX_BUILDER_HPP_
#define MATRIX_BUILDER_HPP_

using namespace anpi;

template <typename T>
class MatrixBuilder{

    Matrix<T> currentMatrixSystem;
    std::vector<T> b,x;
    void nodesEquations(size_t sqr_size);
    void loopsEquations(const Matrix<T> & ResistiveValues);
public:

    MatrixBuilder();

    void buildMatrixSystem(const Matrix<T> & ResistiveValues);

    ~MatrixBuilder();

    
};


template <typename T>
void MatrixBuilder<T>::buildMatrixSystem(Matrix<T> & ResistiveValues){

    if (ResistiveValues.cols() != ResistiveValues.rows()){
        throw anpi::Exception("It can't be posible solve this equation system");
    }

    size_t sqr_size = 2*ResistiveValues.cols()*ResistiveValues.cols() - 2 * ResistiveValues.cols();
    currentMatrixSystem.allocate(sqr_size, sqr_size);

    b = std::vector<T>(sqr_size);
    x = std::vector<T>(sqr_size);

    
    nodesEquations();
    loopsEquations();

    
}


template <typename T>
void MatrixBuilder<T>::nodesEquations(size_t sqr_size){

    //currentMatrixSystem;
}


#endif //MATRIX_BUILDER_HPP_