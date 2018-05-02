

#include "Exception.hpp"
#ifndef NODE_MAPPING_HPP_
#define NODE_MAPPING_HPP_



class NodeMapping{
    size_t n,m,two_m_minus_one;

    bool areAdjacent(size_t i1, size_t j1, size_t i2, size_t j2);

public:
    /**
     * Where n is the size of rows and m is the size of columns.
     */
    NodeMapping(size_t n, size_t m);
    //throw an error if nodes aren't adjacent 
    size_t nodesToIndex(size_t i1, size_t j1, size_t i2, size_t j2) ;    
    void indexToNodes(size_t index, size_t &i1, size_t &j1, size_t &i2, size_t &j2);
};



NodeMapping::NodeMapping(size_t n, size_t m): n(n),m(m),two_m_minus_one(2*m - 1){
}

//verify if two nodes are adjacent
bool NodeMapping::areAdjacent(size_t i1, size_t j1, size_t i2, size_t j2){
    return (i1+1 == i2) ^ (j1+1 == j2);
}


//Transform two node of circuit to current index
size_t NodeMapping::nodesToIndex(size_t i1, size_t j1, size_t i2, size_t j2){
    if (!this->areAdjacent(i1,j1,i2,j2))
        throw anpi::Exception("Nodes aren't adjacent nodes");
    size_t formula = two_m_minus_one*i1 + (this->m-1)*(i2-i1) + j1;
    return formula;
}


//tranform a current index to nodes indexes.
void NodeMapping::indexToNodes(size_t index, size_t &i1, size_t &j1, size_t &i2, size_t &j2){

        //h represent if the current index is an horizontal value
        // -----v^v^-----v^v^----v^v^-----v^v^---- <- Horizontal resistances
        size_t h = index%two_m_minus_one;
        //it is a fix for the last element in vertical resistance values
        size_t s2 = (h == 2*m-2)? two_m_minus_one%(m) : 0;
        //is an horizontal resistance if h > of m-1
        //in other words, it is an overflow from 
        size_t horiz_flag = (h >= m-1)? 1:0;
        
        
        i1 = (index/two_m_minus_one);

        j1 = (index + horiz_flag)%two_m_minus_one%(m) + s2 * horiz_flag;
        //if is an horizontal index i2 = i1+1
        i2 = i1+horiz_flag;
        //if is an horizontal index j2=j1
        j2 = j1 + 1 - horiz_flag;
}



#endif