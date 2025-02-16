#ifndef COPPER_INDEX_H_
#define COPPER_INDEX_H_

#include "common.h"
#include "interface/copper.h"
#include "single_node/in_memory/buffer_manager.h"

#include <unordered_map>

namespace Copper {

class SNIMCopperVertex : public CopperVertex {
public:
    ReturnStatus Build();
    ReturnStatus Load();

    ReturnStatus Insert(Vector, VectorID&);
    ReturnStatus Delete(VectorID);

    ReturnStatus KNearestNeighbours(const Vector&, uint16_t k, std::vector<VectorID>&) const;
    
    uint64_t Size() const;

protected:
    uint64_t Find_Nearest_Vector(const Vector&);

    SNIMBufferManager *_bufmgr;
    VertexID _parentID;
    std::vector<VertexID> _neighbours;
    Vector _centroid;
    void* _data;
    std::unordered_map<VectorID, Vector> _dir;
    uint32_t _size;

friend class SNIMCopperIndex;
};

class SNIMCopperIndex : public CopperIndex  {
    ReturnStatus Build();
    ReturnStatus Load();

    ReturnStatus Insert(Vector, VectorID&);
    ReturnStatus Delete(VectorID);
    ReturnStatus Update(VectorID, Vector);

    ReturnStatus KNearestNeighbours(const Vector&, uint16_t k, std::vector<VectorID>&) const;

    uint64_t Size() const;

protected:
    SNIMCopperVertex* Find_Leaf(const Vector&);

    SNIMCopperVertex _root;
    SNIMBufferManager *_bufmgr;
    uint64_t _height;
    uint64_t _num_vectors;

    const uint64_t _max_leaf;
    const uint64_t _max_internal_vertex;
};

};

#endif