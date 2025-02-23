#ifndef COPPER_COMMON_H
#define COPPER_COMMON_H

#include <vector>
#include <cstdint>
#include <cstring>

#include "debug.h"

namespace copper {

struct RetStatus {
    bool OK;
    const char* message;

    static inline RetStatus Success() {
        return RetStatus{true, "OK"};
    }

    static inline RetStatus Fail(const char* msg) {
        return RetStatus{false, msg};
    }

    inline bool Is_OK() const {
        return OK;
    }

    inline const char* Msg() const {
        return message;
    }
};

constexpr uint64_t INVALID_VECTOR_ID = 0;

union VectorID {
    uint64_t _id;
    struct {
        uint8_t _creator_node_id;
        uint8_t _level; // == 0 for leaves
        uint16_t a2;
        uint32_t a3;
    };

    VectorID(const uint64_t& ID) : _id(ID) {}

    inline VectorID Get_Next_ID() {
        VectorID new_id(*this);
        new_id.a3++;
        if (new_id.a3 == 0) {
            new_id.a2++;
        }
        if (new_id == INVALID_VECTOR_ID) {
            new_id.a3++;
        }
    }

    inline bool Is_Centroid() const {
        return _level > 0;
    }

    inline bool Is_Vector() const {
        return _level == 0;
    }

    inline bool Is_Leaf() const {
        return _level == 1;
    }

    inline void operator=(const VectorID& ID) {
        _id = ID._id;
    }

    inline bool operator==(const VectorID& ID) const {
        return _id == ID._id;
    }

    inline bool operator!=(const VectorID& ID) const {
        return _id != ID._id;
    }

    // inline bool operator<=(const VectorID& ID) const {
    //     return _id <= ID._id;
    // }

    // inline bool operator>=(const VectorID& ID) const {
    //     return _id >= ID._id;
    // }

    // inline bool operator<(const VectorID& ID) const {
    //     return _id < ID._id;
    // }

    // inline bool operator>(const VectorID& ID) const {
    //     return _id > ID._id;
    // }

    inline void operator=(const uint64_t& ID) {
        _id = ID;
    }

    inline bool operator==(const uint64_t& ID) const {
        return _id == ID;
    }

    inline bool operator!=(const uint64_t& ID) const {
        return _id != ID;
    }

    // inline bool operator<=(const uint64_t& ID) const {
    //     return _id <= ID;
    // }

    // inline bool operator>=(const uint64_t& ID) const {
    //     return _id >= ID;
    // }

    // inline bool operator<(const uint64_t& ID) const {
    //     return _id < ID;
    // }

    // inline bool operator>(const uint64_t& ID) const {
    //     return _id > ID;
    // }
};

};

#endif