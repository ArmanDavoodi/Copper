#ifndef COPPER_BUFFER_H_
#define COPPER_BUFFER_H_

#include "common.h"
#include "interface/copper.h"

namespace Copper {

// class BufferDirectory {
// public:
//     virtual CopperVertex* Get_Vertex(VertexID) = 0;
//     virtual Vector* Get_Vector(VectorID) = 0;

//     virtual const CopperVertex* Get_Vertex(VertexID) const = 0;
//     virtual const Vector* Get_Vector(VectorID) const = 0;

//     virtual ReturnStatus Insert_Vector(VectorID, Vector*) = 0;
//     virtual ReturnStatus Insert_Vertex(VertexID, CopperVertex*) = 0;
//     virtual ReturnStatus Remove_Vector(VectorID) = 0;
//     virtual ReturnStatus Remove_Vertex(VertexID) = 0;
//     virtual ReturnStatus Update_Vector(VectorID, Vector*) = 0;
// };

class BufferManager {
public:
    virtual CopperVertex* Get_Vertex(VertexID) = 0;
    virtual CopperVertex* Get_Batch_Vertex(const std::vector<VertexID>&) = 0;

    virtual ReturnStatus Unpin_Vertex(VertexID) = 0;
    virtual ReturnStatus Unpin_Batch_Vertex(const std::vector<VertexID>&) = 0;

    /*
    TODO:
    SHould we put vectors in buckets for better locality, faster access, etc.
    If so, then maybe get vector by id is not a good thing as it has much overhead(
    each we read a vertex we have to insert all of its vectors into the directory
    and then we have to remove them + we then have to return vectors themselves not ids
     but it makes sense as we have to compute distance anyway so we have to load them)

     How about removing vectorID completly and let the application handle it if they so desire?
     Have to check benchmarks

     option 2 is to have pages for vectors but it will have bad locality and buf mangment and eviction
     would be a nightmere
    */
protected:
    const DataType _vtrtype;
    const Dimention _dim;
};

};

#endif