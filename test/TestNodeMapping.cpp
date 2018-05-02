
#include <boost/test/unit_test.hpp>
#include "NodeMapping.hpp"
#include <iostream>

using namespace std;

namespace anpi {
  namespace test {


    void nodeToIndex(size_t sqrSize){
        NodeMapping nm(sqrSize,sqrSize);
        size_t counter = 0;
        for(size_t i = 0; i < sqrSize; i++){
            for(size_t j = 0; j < sqrSize-1; j++){
                BOOST_CHECK(nm.nodesToIndex(i,j,i,j+1) == counter++);
            }
            if (i < sqrSize)
                for(size_t j = 0; j < sqrSize; j++){
                    BOOST_CHECK(nm.nodesToIndex(i,j,i+1,j) == counter++);
                }
        }
      }

      void indexToNode(size_t sqrSize){

        NodeMapping nm(sqrSize,sqrSize);
        size_t counter = 0;
        size_t i1,j1,i2,j2;
        for(size_t i = 0; i < sqrSize; i++){
            for(size_t j = 0; j < sqrSize-1; j++){
                //there are the horizontal bands
                nm.indexToNodes(counter++,i1,j1,i2,j2);
                BOOST_CHECK(i1 == i && j1 == j);
                BOOST_CHECK(i1 == i2 && j1+1 == j2);
            }
            if (i < sqrSize)
                for(size_t j = 0; j < sqrSize; j++){
                    //there are the vertical bands
                    nm.indexToNodes(counter++,i1,j1,i2,j2);
                    BOOST_CHECK(i1 == i && j1 == j);
                    BOOST_CHECK(i1+1 == i2 && j1 == j2);
                }
        }

      }

  }
}





BOOST_AUTO_TEST_SUITE( NodeMappingTest )

BOOST_AUTO_TEST_CASE(NodeToIndex) {

    for(size_t i = 3; i < 50; i++)
        anpi::test::nodeToIndex(i);

}


BOOST_AUTO_TEST_CASE(IndexToNode) {
    for(size_t i = 3; i < 50; i++)
        anpi::test::indexToNode(i);
}


BOOST_AUTO_TEST_SUITE_END()

