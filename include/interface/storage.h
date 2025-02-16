#ifndef COPPER_STORE_H_
#define COPPER_STORE_H_

#include "common.h"
#include "interface/copper.h"

namespace Copper {

class StorageDirectory {
public:
    virtual CopperVertex* Get_Vertex(VertexID) = 0;
    virtual Vector* Get_Vector(VectorID) = 0;

    virtual const CopperVertex* Get_Vertex(VertexID) const = 0;
    virtual const Vector* Get_Vector(VectorID) const = 0;

    virtual VectorID Insert_Vector(Vector*) = 0;
    virtual VertexID Insert_Vertex(CopperVertex*) = 0;
    virtual ReturnStatus Remove_Vector(VectorID) = 0;
    virtual ReturnStatus Remove_Vertex(VertexID) = 0;
    virtual ReturnStatus Update_Vector(VectorID, Vector*) = 0;
};

class StorageMangaer {
protected:
    
};

};

#endif