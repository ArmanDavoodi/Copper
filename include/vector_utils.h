#ifndef DIVFTREE_VECTOR_UTILS_H_
#define DIVFTREE_VECTOR_UTILS_H_

#include "common.h"
#include "distributed_common.h"

#include <algorithm>

// #include <queue>

namespace divftree {

// struct VectorUpdate {
//     VectorID vector_id;
//     Address vector_data;

//     inline bool IsValid() const {
//         return vector_id.IsValid() && (vector_data != INVALID_ADDRESS);
//     }
// };

// class Vector {
// public:
//     Vector(Vector& other) = delete;
//     Vector(const Vector& other) = delete;
//     Vector& operator=(Vector& other) = delete;
//     Vector& operator=(const Vector& other) = delete;

//     Vector() : _data(nullptr), _delete_on_destroy(false) {}

//     /* Copy Constructors */
//     Vector(const Vector& other, uint16_t dim) : _data(other.IsValid() ? malloc(dim * sizeof(VTYPE)) : nullptr),
//                                                 _delete_on_destroy(other.IsValid())
// #ifdef MEMORY_DEBUG
//         , linkCnt(new std::atomic<uint64_t>(1))
// #endif
//     {
//         FatalAssert(dim > 0, LOG_TAG_VECTOR, "Cannot create vector with dimension 0.");
//         FatalAssert(!other.IsValid() || IsValid(), LOG_TAG_VECTOR, "Malloc failed");
//         if (IsValid()) {
//             memcpy(_data, other._data, dim * sizeof(VTYPE));
//         }
//         DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_MEMORY, "copy constructor: other=%p, this=%p, address=%p", &other, this, _data);
//     }

//     Vector(AddressToConst other, uint16_t dim) : _data(other != nullptr ? malloc(dim * sizeof(VTYPE)) : nullptr),
//                                                _delete_on_destroy(other != nullptr)
// #ifdef MEMORY_DEBUG
//         , linkCnt(new std::atomic<uint64_t>(1))
// #endif
//     {
//         FatalAssert(dim > 0, LOG_TAG_VECTOR, "Cannot create vector with dimension 0.");
//         FatalAssert((other == nullptr) || IsValid(), LOG_TAG_VECTOR, "Malloc failed");
//         if (IsValid()) {
//             memcpy(_data, other, dim * sizeof(VTYPE));
//         }
//         DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_MEMORY, "copy constructor: this=%p, address=%p", &other, this, _data);
//     }

//     /* Link Constructors */
//     Vector(Vector&& other) : _data(other._data), _delete_on_destroy(other._delete_on_destroy)
// #ifdef MEMORY_DEBUG
//         , linkCnt(other.linkCnt)
// #endif
//     {
//         other._data = nullptr;
//         other._delete_on_destroy = false;
// #ifdef MEMORY_DEBUG
//         other.linkCnt = nullptr;
// #endif
//         DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_MEMORY, "Move vector: other=%p to this=%p, address=%p, delete_on_destroy=%s",
//              &other, this, _data, (_delete_on_destroy ? "T" : "F"));
//     }

//     /*
//      * should not use AddressToConst as we would be able to change the data of that address.
//      * The user should ensure that if the address is valid, it is not freed or deleted while
//      * this vector(or other vectors) is linked to it.
//      */
//     Vector(Address data) : _data(data), _delete_on_destroy(false) {
//         DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_MEMORY, "link constructor: this=%p, address=%p",
//              this, _data);
//     }

//     ~Vector() {
//         if (!IsValid()) {
//             FatalAssert(_delete_on_destroy == false, LOG_TAG_VECTOR,
//                         "Vector is invalid but delete_on_destroy is true. This should not happen. this=%p", this);
//             return;
//         }

//         if (_delete_on_destroy) {
//             DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_MEMORY, "Destroy vector: this=%p, address=%p", this, _data);
// #ifdef MEMORY_DEBUG
//             FatalAssert((linkCnt == nullptr) || (linkCnt->load() == 1), LOG_TAG_VECTOR,
//                         "Vector is linked to other vectors. Cannot delete it. this=%p, address=%p, linkCnt=%lu",
//                         this, _data, linkCnt->load());
//             if (linkCnt != nullptr) {
//                 delete linkCnt;
//                 linkCnt = nullptr;
//             }
// #endif
//             free(_data);
//         } else {
//             DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_MEMORY, "Unlink vector: this=%p, address=%p", this, _data);
//         }
//         _data = nullptr;
//         _delete_on_destroy = false;
//     }

//     Vector& operator=(Vector&& other) {
//         FatalAssert(!(IsValid()), LOG_TAG_VECTOR, "Vector is not invalid.");
//         DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_MEMORY, "Move vector: other=%p to this=%p, address=%p, delete_on_destroy=%s",
//              &other, this, other._data, (other._delete_on_destroy ? "T" : "F"));
//         _data = other._data;
//         _delete_on_destroy = other._delete_on_destroy;
// #ifdef MEMORY_DEBUG
//         linkCnt = other.linkCnt;
//         other.linkCnt = nullptr;
// #endif
//         other._data = nullptr;
//         other._delete_on_destroy = false;
//         return *this;
//     }

//     inline constexpr bool IsValid() const {
//         return (_data != nullptr);
//     }

//     inline void Create(uint16_t dim) {
//         FatalAssert(!(IsValid()), LOG_TAG_VECTOR, "Vector is not invalid.");
//         FatalAssert(dim > 0, LOG_TAG_VECTOR, "Cannot create vector with dimension 0.");
//         _data = malloc(dim * sizeof(VTYPE));
//         _delete_on_destroy = true;
//         FatalAssert(IsValid(), LOG_TAG_VECTOR, "Malloc failed");
// #ifdef MEMORY_DEBUG
//         linkCnt = new std::atomic<uint64_t>(1);
// #endif
//         DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_MEMORY, "Created vector: this=%p, address=%p, dimension=%hu", this, _data, dim);
//     }

//     /*
//      * Will result in undefined behavior if another vector is linked to this vector.
//      */
//     inline void Destroy() {
//         FatalAssert(IsValid(), LOG_TAG_VECTOR, "Vector is invalid.");
//         FatalAssert(_delete_on_destroy, LOG_TAG_MEMORY,
//                     "Cannot destroy a linked vector. _delete_on_destroy is false. this=%p, address=%p", this, _data);
//         DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_MEMORY, "Destroy vector: this=%p, address=%p", this, _data);
// #ifdef MEMORY_DEBUG
//         FatalAssert((linkCnt == nullptr) || (linkCnt->load() == 1), LOG_TAG_VECTOR,
//                     "Vector is linked to other vectors. Cannot delete it. this=%p, address=%p, linkCnt=%lu",
//                     this, _data, linkCnt->load());
//         if (linkCnt != nullptr) {
//             delete linkCnt;
//             linkCnt = nullptr;
//         }
// #endif
//         free(_data);
//         _data = nullptr;
//         _delete_on_destroy = false;
//     }

//     /*
//      * If the src vector is a linked vector, the user should ensure that the associated data is not freed or deleted
//      * while this vector is linked to it.
//      */
//     inline void Link(Vector& src) {
//         FatalAssert(!(IsValid()), LOG_TAG_VECTOR, "Vector is valid");
//         _data = src._data;
//         _delete_on_destroy = false;
// #ifdef MEMORY_DEBUG
//         linkCnt = src.linkCnt;
//         if (linkCnt != nullptr) {
//             (void)linkCnt->fetch_add(1);
//         }
// #endif
//         DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_MEMORY, "Link vector: this=%p, address=%p", this, _data);
//     }

//     /*
//      * If the is not null, the user should ensure that it is not freed or deleted
//      * while this vector or other vectors are linked to it.
//      */
//     inline void Link(void* src) {
//         FatalAssert(!(IsValid()), LOG_TAG_VECTOR, "Vector is valid");
//         _data = src;
//         _delete_on_destroy = false;
//         DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_MEMORY, "Link vector: this=%p, address=%p", this, _data);
//     }

//     inline void Unlink() {
//         FatalAssert(IsValid(), LOG_TAG_VECTOR, "Vector is invalid.");
//         FatalAssert(!_delete_on_destroy, LOG_TAG_MEMORY,
//                     "Not a linked vector. _delete_on_destroy is true. this=%p, address=%p", this, _data);
//         DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_MEMORY, "Unlink vector: this=%p, address=%p", this, _data);
//         _data = nullptr;
//         _delete_on_destroy = false;
// #ifdef MEMORY_DEBUG
//         if (linkCnt != nullptr) {
//             FatalAssert((linkCnt->load() > 1), LOG_TAG_VECTOR,
//                         "This is the last linked vector and should not be unlinked. this=%p, address=%p, linkCnt=%lu",
//                         this, _data, linkCnt->load());
//             (void)linkCnt->fetch_sub(1);
//         }
// #endif
//     }

//     inline void CopyFrom(const Vector& src, uint16_t dim) {
//         FatalAssert(IsValid(), LOG_TAG_VECTOR, "Vector is invalid.");
//         FatalAssert(src.IsValid(), LOG_TAG_VECTOR, "Source Vector is not invalid.");

//         memcpy(_data, src._data, dim * sizeof(VTYPE));
//     }

//     inline void CopyFrom(AddressToConst src, uint16_t dim) {
//         FatalAssert(IsValid(), LOG_TAG_VECTOR, "Vector is invalid.");
//         FatalAssert(src != INVALID_ADDRESS, LOG_TAG_VECTOR, "Source should not be invalid.");

//         memcpy(_data, src, dim * sizeof(VTYPE));
//     }

//     inline Address GetData() {
//         return _data;
//     }

//     inline AddressToConst GetData() const {
//         return _data;
//     }

//     inline bool operator==(const Vector& other) const {
//         return (_data == other._data);
//     }

//     inline bool operator!=(const Vector& other) const {
//         return (_data != other._data);
//     }

//     inline bool Similar(const Vector& other, uint16_t dim) const {
//         if (*this == other) {
//             return true;
//         }

//         if (!(IsValid()) || !(other.IsValid())) {
//             return false;
//         }

//         return !(memcmp(_data, other._data, dim * sizeof(VTYPE)));
//     }

//     inline VTYPE& operator[](size_t idx) {
//         FatalAssert(IsValid(), LOG_TAG_VECTOR, "Vector is invalid.");
//         return (static_cast<VTYPE*>(_data))[idx];
//     }

//     inline const VTYPE& operator[](size_t idx) const {
//         FatalAssert(IsValid(), LOG_TAG_VECTOR, "Vector is invalid.");
//         return (static_cast<const VTYPE*>(_data))[idx];
//     }

//     inline String ToString(uint16_t dim) const {
//         if (!(IsValid())) {
//             return String("INV");
//         }
//         String str("(address=%p, linked=%s)<", _data, _delete_on_destroy ? "T" : "F");
//         for (uint16_t i = 0; i < dim; ++i) {
//             str += String(VTYPE_FMT, (*this)[i]) + String((i < (dim - 1)) ? ", " : ">");
//         }

//         return str;
//     }

// protected:
//     void* _data;
//     bool _delete_on_destroy; // If true, the vector will be deleted when destroyed.
// #ifdef MEMORY_DEBUG
//     std::atomic<uint64_t>* linkCnt = nullptr;
// #endif

// TESTABLE;
// };

String VectorToString(const VTYPE* vec, uint16_t dim) {
    String res = "{";
    for (uint16_t i = 0; i < dim; ++i) {
        res += String(VTYPE_FMT "%s", vec[i], (i == dim - 1 ? "}" : ", "));
    }
    return res;
}

enum VectorState : uint8_t {
    VECTOR_STATE_INVALID = 0,
    VECTOR_STATE_VALID = 1,
    VECTOR_STATE_MIGRATED = 2,
    VECTOR_STATE_OUTDATED = 3
};

inline String VectorStateToString(VectorState state) {
    switch (state)
    {
    case VECTOR_STATE_INVALID:
        return String("INVALID");
    case VECTOR_STATE_VALID:
        return String("VALID");
    case VECTOR_STATE_MIGRATED:
        return String("MIGRATED");
    case VECTOR_STATE_OUTDATED:
        return String("OUTDATED");
    default:
        return String("UNDEFINED");
    }
}

/* this struct is always moved and never copied! */
struct VectorBatch {
    VTYPE* data = nullptr;
    VectorID* id = nullptr;
    Version* version = nullptr;
    uint16_t size = 0;
    uint16_t capacity = 0;

    void Resize(uint16_t new_cap, bool is_leaf, uint16_t dim) {
        FatalAssert(new_cap > 0, LOG_TAG_BASIC, "New capacity must be greater than 0.");
        if (new_cap == capacity) {
            return;
        }

        VTYPE* new_data = new VTYPE[new_cap * dim];
        VectorID* new_id = new VectorID[new_cap];
        Version* new_version = nullptr;
        if (!is_leaf) {
            new_version = new Version[new_cap];
        }

        uint16_t copy_size = std::min(size, new_cap);
        if (data != nullptr) {
            memcpy((void*)new_data, (void*)data, copy_size * dim * sizeof(VTYPE));
            delete[] data;
        }
        if (id != nullptr) {
            memcpy((void*)new_id, (void*)id, copy_size * sizeof(VectorID));
            delete[] id;
        }
        if (version != nullptr) {
            FatalAssert(!is_leaf, LOG_TAG_BASIC, "Version array should not exist for leaf nodes.");
            memcpy((void*)new_version, (void*)version, copy_size * sizeof(Version));
            delete[] version;
        }

        data = new_data;
        id = new_id;
        version = new_version;
        size = copy_size;
        capacity = new_cap;
    }
};

template<typename T>
inline void ResizeArray(T*& array, uint16_t size, uint16_t new_cap) {
    FatalAssert(new_cap > 0, LOG_TAG_BASIC, "New capacity must be greater than 0.");
    if (new_cap == size) {
        return;
    }

    T* new_array = new T[new_cap];
    if (array != nullptr) {
        uint16_t copy_size = std::min(size, new_cap);
        memcpy(new_array, array, copy_size * sizeof(T));
        delete[] array;
    }

    array = new_array;
}

/* this struct is always moved and never copied! */
struct ConstVectorBatch {
    const VTYPE* data = nullptr;
    const VectorID* id = nullptr;
    const Version* version = nullptr;
    const uint16_t size = 0;

    ConstVectorBatch() = default;
    ConstVectorBatch(uint16_t s) : size(s) {};
    ConstVectorBatch(const ConstVectorBatch& other) = default;
    ConstVectorBatch(ConstVectorBatch&& other) = default;

    ConstVectorBatch(const VectorBatch& other) :
        data(other.data), id(other.id), version(other.version), size(other.size) {}

    ConstVectorBatch(const VTYPE* d, const VectorID* i, const Version* v, uint16_t s) :
        data(d), id(i), version(v), size(s) {}
};

/* todo: we may need to use packed attrbite for these in the multi node setup to save network bandwidth */
struct VectorMetaData {
    VectorID id;
    std::atomic<VectorState> state;
};

/* todo: we may need to use packed attrbite for these in the multi node setup to save network bandwidth */
struct CentroidMetaData {
    VectorID id;
    Version version;
    std::atomic<VectorState> state;
};

/*
 * todo: we may need to use packed attrbite for these in the multi node setup to save network bandwidth
 *
 * todo: put const variables in the vertex class to avoid reading them
 * each time and to avoid using too much memory in the local system
 *
 * todo: we may need to put this reserved_size, etc in a cache line of its own as unlike vertex header
 * it is read from remote + it is different from cluster data unless we do a single read/write for this and
 * combine it with the first block -> for read we have to read the header first in a seperate RDMA to avoid
 * out of order reads between header and data.
 */
struct ClusterHeader {
    std::atomic<uint16_t> reserved_size; // todo: do we need reserved_size in compute nodes?
    std::atomic<uint16_t> visible_size;
    std::atomic<uint16_t> num_deleted; // todo: do we need num deleted in compute nodes?
};
class Cluster {
public:
    inline static uint16_t BlockSize(size_t block_bytes, size_t meta_size, uint16_t cap, uint16_t dim) {
        return (uint16_t)(std::min(block_bytes / (meta_size + (sizeof(VTYPE) * (size_t)dim)), (size_t)cap));
    }

    inline static size_t BlockBytes(uint16_t block_size, size_t meta_size, uint16_t dim) {
        return ALIGNED_SIZE((size_t)block_size * ((size_t)meta_size + ((size_t)dim * sizeof(VTYPE))));
    }

    inline static size_t NumBlocks(uint16_t block_size, uint16_t cap) {
        return (cap + block_size - 1) / block_size;
    }

    inline static size_t DataBytes(bool is_leaf_vertex, uint16_t block_size, uint16_t cap, uint16_t dim) {
        const uint64_t header_bytes = (is_leaf_vertex ? sizeof(VectorMetaData) : sizeof(CentroidMetaData));
        const uint64_t data_bytes = sizeof(VTYPE) * (uint64_t)dim;
        uint64_t last_block_cap = cap % block_size;
        if (last_block_cap == 0 && cap > 0) {
            last_block_cap = block_size;
        }
        return ((NumBlocks(block_size, cap) - 1) * ALIGNED_SIZE(block_size * (header_bytes + data_bytes))) +
               ALIGNED_SIZE(last_block_cap * (header_bytes + data_bytes));
    }

    inline static size_t TotalBytes(bool is_leaf_vertex, uint16_t block_size, uint16_t cap, uint16_t dim) {
        return ALIGNED_SIZE(sizeof(ClusterHeader)) + DataBytes(is_leaf_vertex, block_size, cap, dim);
    }

    Cluster(bool is_leaf_vertex, uint16_t block_cap, uint16_t cap, uint16_t dim, bool set_zero = true) :
            header{0, 0, 0} {
        FatalAssert(block_cap > 0, LOG_TAG_CLUSTER, "Block size must be greater than 0. block_size=%hu", block_cap);
        FatalAssert(cap > 0, LOG_TAG_CLUSTER, "Capacity must be greater than 0. capacity=%hu", cap);
        FatalAssert(dim > 0, LOG_TAG_CLUSTER, "Dimension must be greater than 0. dimension=%hu", dim);
        if (set_zero) {
            memset(blocks, 0, DataBytes(is_leaf_vertex, block_cap, cap, dim));
        }
    }

    inline AddressToConst MetaData(uint16_t offset, bool is_leaf, uint16_t block_size,
                                   uint16_t capacity, uint16_t dimension) const {
        FatalAssert(offset < capacity, LOG_TAG_CLUSTER, "Offset is out of bounds. offset=%hu, capacity=%hu",
                    offset, capacity);
        const uint16_t block_number = offset / block_size;
        const uint16_t block_offset = offset % block_size;
        const uint64_t header_bytes = (is_leaf ? sizeof(VectorMetaData) : sizeof(CentroidMetaData));
        const uint64_t data_bytes = sizeof(VTYPE) * (uint64_t)dimension;
        const uint16_t block_bytes = ALIGNED_SIZE(block_size * (header_bytes + data_bytes));
        FatalAssert(block_number == 0, LOG_TAG_CLUSTER,
                    "Only single block clusters are supported currently. offset=%hu, "
                    "block_number=%hu, block_size=%hu", offset, block_number, block_size);
        return reinterpret_cast<AddressToConst>(ALIGNED_PTR(blocks) +
                                                block_number * block_bytes + block_offset * header_bytes);
    }

    inline Address MetaData(uint16_t offset, bool is_leaf, uint16_t block_size,
                            uint16_t capacity, uint16_t dimension) {
        FatalAssert(offset < capacity, LOG_TAG_CLUSTER, "Offset is out of bounds. offset=%hu, capacity=%hu",
                    offset, capacity);
        const uint16_t block_number = offset / block_size;
        const uint16_t block_offset = offset % block_size;
        const uint64_t header_bytes = (is_leaf ? sizeof(VectorMetaData) : sizeof(CentroidMetaData));
        const uint64_t data_bytes = sizeof(VTYPE) * (uint64_t)dimension;
        const uint16_t block_bytes = ALIGNED_SIZE(block_size * (header_bytes + data_bytes));
        FatalAssert(block_number == 0, LOG_TAG_CLUSTER,
                    "Only single block clusters are supported currently. offset=%hu, "
                    "block_number=%hu, block_size=%hu", offset, block_number, block_size);
        return reinterpret_cast<Address>(ALIGNED_PTR(blocks) +
                                         block_number * block_bytes + block_offset * header_bytes);
    }

    inline const VTYPE* Data(uint16_t offset, bool is_leaf, uint16_t block_size,
                             uint16_t capacity, uint16_t dimension) const {
        FatalAssert(offset < capacity, LOG_TAG_CLUSTER, "Offset is out of bounds. offset=%hu, capacity=%hu",
                    offset, capacity);
        const uint16_t block_number = offset / block_size;
        const uint16_t block_offset = offset % block_size;
        uint64_t last_block_size = capacity % block_size;
        if (last_block_size == 0 && capacity > 0) {
            last_block_size = block_size;
        }
        const uint64_t header_bytes = (is_leaf ? sizeof(VectorMetaData) : sizeof(CentroidMetaData));
        const uint64_t data_bytes = sizeof(VTYPE) * (uint64_t)dimension;
        const uint16_t block_bytes = ALIGNED_SIZE(block_size * (header_bytes + data_bytes));
        const uint16_t meta_data_bytes = (block_number == (block_size - 1) ?
                                          last_block_size : block_size) *
                                          header_bytes;
        FatalAssert(block_number == 0, LOG_TAG_CLUSTER,
                    "Only single block clusters are supported currently. offset=%hu, "
                    "block_number=%hu, block_size=%hu", offset, block_number, block_size);
        return reinterpret_cast<const VTYPE*>(ALIGNED_PTR(blocks) +
                                              block_number * block_bytes + meta_data_bytes + block_offset * data_bytes);
    }

    inline VTYPE* Data(uint16_t offset, bool is_leaf, uint16_t block_size,
                       uint16_t capacity, uint16_t dimension) {
        FatalAssert(offset < capacity, LOG_TAG_CLUSTER, "Offset is out of bounds. offset=%hu, capacity=%hu",
                    offset, capacity);
        const uint16_t block_number = offset / block_size;
        const uint16_t block_offset = offset % block_size;
        uint64_t last_block_size = capacity % block_size;
        if (last_block_size == 0 && capacity > 0) {
            last_block_size = block_size;
        }
        const uint64_t header_bytes = (is_leaf ? sizeof(VectorMetaData) : sizeof(CentroidMetaData));
        const uint64_t data_bytes = sizeof(VTYPE) * (uint64_t)dimension;
        const uint16_t block_bytes = ALIGNED_SIZE(block_size * (header_bytes + data_bytes));
        const uint16_t meta_data_bytes = (block_number == (block_size - 1) ?
                                          last_block_size : block_size) *
                                          header_bytes;
        FatalAssert(block_number == 0, LOG_TAG_CLUSTER,
                    "Only single block clusters are supported currently. offset=%hu, "
                    "block_number=%hu, block_size=%hu", offset, block_number, block_size);
        return reinterpret_cast<VTYPE*>(ALIGNED_PTR(blocks) +
                                        block_number * block_bytes + meta_data_bytes + block_offset * data_bytes);
    }
    /*
     * a cluster consists of a header and multiple blocks
     * ---------------------------------------------------
     * the blocks of an internal cluster have this structure:
     *
     * | VectorID | Version | State | ...
     * ---------------------------------------------------
     * | VectorData | ...
     *
     * the blocks of a leaf cluster have this structure:
     * | VectorID | State | ...
     * ---------------------------------------------------
     * | VectorData | ...
     *
     *
     *
     * NOTE: it is probably better to make each block cache alligned as we will be using different RDMA reads for them
     * but do not make the data inside a block cache alligned
     * to reduce the amount of data read from remote!
     */
    ClusterHeader header;
protected:
    alignas(CACHE_LINE_SIZE) char blocks[];

TESTABLE;
};
};
#endif