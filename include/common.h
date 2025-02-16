#ifndef COPPER_COMMON_H_
#define COPPER_COMMON_H_

#include <cstdint>
#include <vector>
#include <assert.h>

namespace Copper {

struct ReturnStatus {
    inline bool Is_OK() {

    }
};

constexpr uint64_t Invalid_ID = 0;

typedef uint8_t NodeID;

union VectorID { // can be a centroid
    uint64_t ID;
    struct {
        uint8_t level;
        uint8_t creator_node_id; // CNs are even and MNs are odd
        uint16_t i2;
        uint32_t i3;
    };

    inline bool Is_Centroid() const {
        assert((level > 0 && creator_node_id % 2 == 1) || (level == 0 && creator_node_id % 2 == 0));
        return level > 0;
    }

    inline uint8_t Get_Level() const {
        return level;
    }

    inline NodeID Get_Creator() const {
        return creator_node_id;
    }

    inline bool Is_Valid() const {
        return ID != Invalid_ID;
    }
};

union VertexID {
    uint64_t ID;
    struct {
        uint8_t level;
        uint8_t i1;
        uint16_t i2;
        uint32_t i3;
    };

    inline bool Is_Leaf() const {
        return level == 0;
    }

    inline uint8_t Get_Level() const {
        return level;
    }

    inline bool Is_Valid() const {
        return ID != Invalid_ID;
    }
};

enum Distance : uint8_t {
    L2
};

enum DataType : uint8_t {
    Uint8,
    Uint32,
    Float32
};

typedef uint16_t Dimention;

typedef void* Vector;

Vector Invalid_Vector = nullptr;
// struct Vector {
//     VectorData _data;
//     VectorID ID;
//     VertexID parent;
//     const Dimention dim;
//     const DataType type;

//     // TODO define other operators, copy, etc
// };



// could contain either children centroids or elements depending on vertex type.
// if not full, the rest are Invalid_ID
// template<typename T, uint16_t Dimention, uint64_t CAP> 
// struct Bucket {
//     Vector<T, Dimention> elements[CAP]; 
//     uint64_t IDs[CAP] = {Invalid_ID};
// };

};

#endif