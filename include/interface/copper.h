#undef COPPER_INDEX_H_
#ifndef COPPER_INDEX_H_
#define COPPER_INDEX_H_

#include "common.h"
#include "interface/index.h"


namespace Copper {

class CopperVertex : public Index {
public:
    enum VertexType : uint8_t {
        Internal,
        Leaf
    };

    inline VertexType Vertex_Type() const {
        return _selfID.Is_Leaf() ? VertexType::Leaf : VertexType::Internal;
    }

    inline bool Is_Leaf() const {
        return _selfID.Is_Leaf();
    }

protected:
    virtual uint64_t Find_Nearest_Vector(const Vector&) = 0;

    const uint32_t _min_size, _max_size;
    const uint32_t _max_neighbors;
    const VertexID _selfID;

    // VertexID parentID;
    // Vector *_centroid;
    // Vector *_vectors;
    // VectorID *_vector_IDs;
    // VertexID *neighbours;
    // uint32_t _size;

friend class CopperIndex;
};

class CopperIndex : Index  {
    virtual uint64_t Get_Height() = 0;
public:
    virtual ReturnStatus Update(VectorID, Vector) = 0;

protected:
    virtual CopperVertex* Find_Leaf(const Vector&) = 0;

    const uint32_t _KI_min, _KI_max, _KL_min, _KL_max;
    const uint32_t _max_neighbors;

    CopperVertex *root;
};

};

#endif