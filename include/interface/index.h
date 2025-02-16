#ifndef COPPER_INTERFACE_H_
#define COPPER_INTERFACE_H_

#include "common.h"
#include "vector_utils.h"

namespace Copper {

class Index {
public:
    virtual ReturnStatus Build() = 0;
    virtual ReturnStatus Load() = 0;

    // TODO use a wrapper to make sure user cannot delete the vector?
    // virtual ReturnStatus Get_By_ID(VectorID, const Vector*) const; // should not be deleted by user
    // virtual ReturnStatus Get_By_ID(const std::vector<VectorID>&, std::vector<const Vector*>&) const; // should not be deleted by user

    virtual ReturnStatus Insert(Vector, VectorID&) = 0; // TODO && and const& implementations
    virtual ReturnStatus Delete(VectorID) = 0;

    virtual ReturnStatus KNearestNeighbours(const Vector&, uint16_t k, std::vector<VectorID>&) const = 0;

    virtual uint64_t Size() const = 0;

    // virtual ReturnStatus Find_ID(Vector, VectorID&) const;
protected:
    const DataType _vtrtype;
    const Dimention _dim;
    const Vector_Distance* _dist;
};

}

#endif