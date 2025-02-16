#ifndef COPPER_BUFFER_H_
#define COPPER_BUFFER_H_

#include "common.h"
#include "interface/buffer.h"

#include <unordered_map>

namespace Copper {

// struct SNIMVectorBufferSlot {
//     Vector _loc;

// };

class SNIMVertexBuffer { // I think this should be VertexBuffer
public:
protected:
    const DataType _vtrtype;
    const Dimention _dim;
    const uint64_t _max_size; // input -> leaf and internals have different sizes

    void* _vtx_buf;

    uint64_t _vtx_buf_next_free_slot;
    uint64_t _vtx_buf_last_free_slot;
    std::vector<uint64_t> free_slots;

    // todo convert index to vectorID and vice versa
    // todo for GPU we may need to replace buffers for better locality
};

class SNIMVectorBuffer {
public:
protected:
    const DataType _vtrtype;
    const Dimention _dim;

    void* _vec_buf;
    VectorID* _vec_ID_buf;

    uint64_t _vec_buf_next_free_slot;
    uint64_t _vec_buf_last_free_slot;
    std::vector<uint64_t> free_slots;

    // todo convert index to vectorID and vice versa
    // todo for GPU we may need to replace buffers for better locality
};

// class SNIMBufferDirectory : public BufferDirectory {
// public:
//     CopperVertex* Get_Vertex(VertexID);
//     Vector* Get_Vector(VectorID);

//     const CopperVertex* Get_Vertex(VertexID) const;
//     const Vector* Get_Vector(VectorID) const;

//     ReturnStatus Insert_Vector(VectorID, Vector*);
//     ReturnStatus Insert_Vertex(VertexID, CopperVertex*);
//     ReturnStatus Remove_Vector(VectorID);
//     ReturnStatus Remove_Vertex(VertexID);
//     ReturnStatus Update_Vector(VectorID, Vector*);

// protected:
//     // TODO better implementation?
//     std::unordered_map<VectorID, Vector*> _vdict;
//     std::unordered_map<VertexID, CopperVertex*> _ndict;
// };

class SNIMBufferManager : public BufferManager {
public:
    CopperVertex* Get_Vertex(VertexID);
    CopperVertex* Get_Batch_Vertex(const std::vector<VertexID>&);

    ReturnStatus Unpin_Vertex(VertexID);
    ReturnStatus Unpin_Batch_Vertex(const std::vector<VertexID>&);

    ReturnStatus Insert(Vector, VertexID, Vector&, VectorID&);
    ReturnStatus Delete(Vector, VectorID&, Vector&); // dir helps us find vectors by id

    Vector Get_Vector(VectorID);

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
    // SNIMMemoryManager *memmgr;
    std::vector<CopperVertex> _vertices;
    SNIMVertexBuffer _vtx_buf;
};

};

#endif