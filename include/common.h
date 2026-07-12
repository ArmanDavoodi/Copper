#ifndef DIVFTREE_COMMON_H_
#define DIVFTREE_COMMON_H_

#include "configs/support.h"

#include <vector>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <concepts>
#include <stdlib.h>
#include <functional>

#include "utils/string.h"
#include "utils/thread.h"
#include "utils/synchronization.h"
#include "utils/top_n.h"

#include "debug.h"

namespace divftree {

struct RetStatus {
    enum {
        SUCCESS,

        BATCH_CONFLICTING_OPERATIONS,
        BATCH_UNKNOWN_OPERATION,

        VECTOR_IS_INVALID,
        VECTOR_IS_MIGRATED,
        VECTOR_IS_INVISIBLE,
        VECTOR_NOT_FOUND,

        VERTEX_UPDATED,
        VERTEX_NEEDS_SPLIT, /* not enough space -> no vectors were placed in cluster -> split needed by caller */
        VERTEX_NEEDS_MERGE, /* may be solved with a single compaction? */
        VERTEX_NOT_ENOUGH_SPACE, /* returned when there is not enough space but someone else is splitting */

        FAILED_TO_CAS_ENTRY_STATE,
        FAILED_TO_CAS_VECTOR_STATE,

        TARGET_DELETED,
        TARGET_MIGRATED,
        TARGET_UPDATED,
        TARGET_IS_ROOT,
        NEW_CONTAINER_DELETED,
        TARGET_VERTEX_UPDATED,
        OLD_CONTAINER_UPDATED,
        NEW_CONTAINER_UPDATED,

        DEST_DELETED,
        DEST_UPDATED,
        SRC_DELETED,
        SRC_HAS_TOO_MANY_VECTORS,
        SRC_EMPTY,

        TREE_HIGHT_TOO_LOW,

        NO_HANDLE_FOUND_FOR_TARGET,
        ALREADY_HAS_A_HANDLE,

        VERTEX_DELETED,
        VERSION_NOT_APPLIED,
        OUTDATED_VERSION_DELETED,

        EMPTY_MIGRATION_BATCH,

        FAIL
    } stat;

    const char* message = nullptr;

    static inline RetStatus Success() {
        RetStatus status;
        status.stat = SUCCESS;
        status.message = nullptr;
        return status;
    }

    static inline RetStatus Fail(const char* msg) {
        RetStatus status;
        status.stat = FAIL;
        if (msg != nullptr) {
            status.message = new char[strlen(msg) + 1];
            strcpy(const_cast<char*>(status.message), msg);
        } else {
            status.message = nullptr;
        }
        return status;
    }

    inline bool IsOK() const {
        return stat == SUCCESS;
    }

    inline const char* Msg() const {
        return message;
    }

    ~RetStatus() {
        if (message != nullptr) {
            delete[] message;
            message = nullptr;
        }
    }

    inline void operator=(const RetStatus& other) {
        stat = other.stat;
        if (message != nullptr) {
            delete[] message;
            message = nullptr;
        }
        if (other.message != nullptr) {
            message = new char[strlen(other.message) + 1];
            strcpy(const_cast<char*>(message), other.message);
        }
    }

    inline void operator=(RetStatus&& other) {
        stat = other.stat;
        message = other.message;
        other.message = nullptr;
    }

    RetStatus(const RetStatus& other) {
        stat = other.stat;
        if (other.message != nullptr) {
            message = new char[strlen(other.message) + 1];
            strcpy(const_cast<char*>(message), other.message);
        } else {
            message = nullptr;
        }
    }

    RetStatus(RetStatus&& other) {
        stat = other.stat;
        message = other.message;
        other.message = nullptr;
    }

    RetStatus() = default;
};

typedef uint64_t RawVectorID;
constexpr RawVectorID INVALID_VECTOR_ID = UINT64_MAX;
constexpr uint16_t INVALID_OFFSET = UINT16_MAX;

/*
 * we should not need more than 10 levels even if we have 1 petabyte of raw vector data with each vector being 128 in
 * dimention and having 4 byte per element, then we would have 3.5*10^13 vectors and even if each cluster contains
 * 64 vectors at most, we would only need 8 levels.
 *
 * We can also look at it this way: if each container contains 64 vectors, then with 10 levels we can support up to
 * 10^18 vectors and even if each vector takes 1 byte of memory, then 10 levels can support up to 1 ZB of
 * raw vector data.
 */
constexpr uint8_t MAX_TREE_HIGHT = 10;
constexpr float COMPACTION_FACTOR = 1.15;

union VectorID {
    RawVectorID _id;
    struct {
        uint64_t _val : 48;
        uint64_t _level : 8; // == 0 for vectors, == 1 for leaves
        uint64_t _creator_node_id : 8;
    };

    static constexpr uint64_t MAX_ID_PER_LEVEL = 0x0000FFFFFFFFFFFF;
    static constexpr uint64_t VECTOR_LEVEL = 0;
    static constexpr uint64_t LEAF_LEVEL = 1;

    constexpr VectorID() : _id(INVALID_VECTOR_ID) {}
    constexpr VectorID(const RawVectorID& ID) : _id(ID) {}
    constexpr VectorID(const VectorID& ID) : _id(ID._id) {}
    constexpr VectorID(VectorID&& ID) : _id(ID._id) {}
    ~VectorID() = default;

    inline static VectorID AsID(RawVectorID id) {
        return VectorID(id);
    }

    inline bool IsValid() const {
        return (_id != INVALID_VECTOR_ID) && (_val < MAX_ID_PER_LEVEL);
    }

    inline bool IsCentroid() const {
        return _level > VECTOR_LEVEL;
    }

    inline bool IsVector() const {
        return _level == VECTOR_LEVEL;
    }

    inline bool IsLeaf() const {
        return _level == LEAF_LEVEL;
    }

    inline bool IsInternalVertex() const {
        return _level > LEAF_LEVEL;
    }

    inline constexpr void operator=(const VectorID& ID) {
        _id = ID._id;
    }

    inline constexpr void operator=(VectorID&& ID) {
        _id = ID._id;
    }

    inline bool operator==(const VectorID& ID) const {
        return _id == ID._id;
    }

    inline bool operator!=(const VectorID& ID) const {
        return _id != ID._id;
    }

    inline constexpr void operator=(const RawVectorID& ID) {
        _id = ID;
    }

    inline constexpr void operator=(RawVectorID&& ID) {
        _id = ID;
    }

    inline bool operator==(const RawVectorID& ID) const {
        return _id == ID;
    }

    inline bool operator!=(const RawVectorID& ID) const {
        return _id != ID;
    }

    inline bool operator<(const VectorID& ID) const {
        return _id < ID._id;
    }

    inline bool operator<=(const VectorID& ID) const {
        return _id < ID._id;
    }

    inline bool operator>(const VectorID& ID) const {
        return _id > ID._id;
    }

    inline bool operator>=(const VectorID& ID) const {
        return _id > ID._id;
    }
};

typedef uint32_t RawVersion;
union Version {
    RawVersion _raw;
    struct {
        uint32_t _compaction : 8;
        uint32_t _split : 24;
    };

    constexpr Version() : _raw(0) {}
    constexpr Version(const RawVersion& ver) : _raw(ver) {}
    constexpr Version(const Version& ver) : _raw(ver._raw) {}

    inline static Version AsVersion(RawVersion ver) {
        return Version(ver);
    }

    inline static RawVersion AsRawVersion(Version ver) {
        return ver._raw;
    }

    constexpr void operator=(const Version& ver) {
        _raw = ver._raw;
    }

    constexpr void operator=(const RawVersion& ver) {
        _raw = ver;
    }

    inline bool operator==(const Version& ver) const {
        return _raw == ver._raw;
    }

    inline bool operator==(const RawVersion& ver) const {
        return _raw == ver;
    }

    inline bool operator!=(const Version& ver) const {
        return _raw != ver._raw;
    }

    inline bool operator!=(const RawVersion& ver) const {
        return _raw != ver;
    }

    inline bool operator<(const Version& ver) const {
        FatalAssert((_raw < ver._raw) == (_split < ver._split ||
                    (_split == ver._split && _compaction < ver._compaction)),
                    LOG_TAG_BASIC, "Version comparison operator inconsistency");
        return _raw < ver._raw;
    }

    inline bool operator<(const RawVersion& ver) const {
        SANITY_CHECK(
            Version v = AsVersion(ver);
            FatalAssert((_raw < v._raw) == (_split < v._split ||
                        (_split == v._split && _compaction < v._compaction)),
                        LOG_TAG_BASIC, "Version comparison operator inconsistency");
        );
        return _raw < ver;
    }

    inline bool operator<=(const Version& ver) const {
        FatalAssert((_raw <= ver._raw) == (_split < ver._split ||
                    (_split == ver._split && _compaction <= ver._compaction)),
                    LOG_TAG_BASIC, "Version comparison operator inconsistency");
        return _raw <= ver._raw;
    }

    inline bool operator<=(const RawVersion& ver) const {
        SANITY_CHECK(
            Version v = AsVersion(ver);
            FatalAssert((_raw <= v._raw) == (_split < v._split ||
                        (_split == v._split && _compaction <= v._compaction)),
                        LOG_TAG_BASIC, "Version comparison operator inconsistency");
        );
        return _raw <= ver;
    }

    inline bool operator>(const Version& ver) const {
        FatalAssert((_raw > ver._raw) == (_split > ver._split ||
                    (_split == ver._split && _compaction > ver._compaction)),
                    LOG_TAG_BASIC, "Version comparison operator inconsistency");
        return _raw > ver._raw;
    }

    inline bool operator>(const RawVersion& ver) const {
        SANITY_CHECK(
            Version v = AsVersion(ver);
            FatalAssert((_raw > v._raw) == (_split > v._split ||
                        (_split == v._split && _compaction > v._compaction)),
                        LOG_TAG_BASIC, "Version comparison operator inconsistency");
        );
        return _raw > ver;
    }

    inline bool operator>=(const Version& ver) const {
        FatalAssert((_raw >= ver._raw) == (_split > ver._split ||
                    (_split == ver._split && _compaction >= ver._compaction)),
                    LOG_TAG_BASIC, "Version comparison operator inconsistency");
        return _raw >= ver._raw;
    }

    inline bool operator>=(const RawVersion& ver) const {
        SANITY_CHECK(
            Version v = AsVersion(ver);
            FatalAssert((_raw >= v._raw) == (_split > v._split ||
                        (_split == v._split && _compaction >= v._compaction)),
                        LOG_TAG_BASIC, "Version comparison operator inconsistency");
        );
        return _raw >= ver;
    }

    inline String ToString() const {
        return String("%u{split=%u, compaction=%u}", _raw, _split, _compaction);
    }

    inline Version NextCompaction() const {
        Version newVersion = *this;
        newVersion._raw += 1;
        /* If we have compaction overflow, it should be fine as we can consider a compaction to be split and it will
           only make things slower. */
        FatalAssert(((newVersion._compaction != 0) && (newVersion._compaction == _compaction + 1) &&
                     (newVersion._split == _split)) ||
                    ((newVersion._compaction == 0) && (newVersion._split == _split + 1)), LOG_TAG_BASIC,
                    "Version compaction overflow");
        /* todo: remove this and replace with stat collection code */
        DIVFLOG_IF_TRUE(newVersion._compaction == 0, LOG_LEVEL_DEBUG, LOG_TAG_BASIC,
                        "Version compaction overflow happened, performance may be degraded.");
        FatalAssert(newVersion != 0, LOG_TAG_BASIC, "Version overflow");
        return newVersion;
    }

    inline Version NextSplit() const {
        Version newVersion;
        newVersion._split = _split + 1;
        FatalAssert((newVersion._split == _split + 1) && (newVersion._compaction == 0), LOG_TAG_BASIC,
                    "Version split overflow");
        FatalAssert(newVersion != 0, LOG_TAG_BASIC, "Version overflow");
        return newVersion;
    }
};

struct IVFVectorID {
    uint64_t vector_hash;
    uint32_t value;

    inline bool operator==(const IVFVectorID& other) const {
        return (vector_hash == other.vector_hash) && (value == other.value);
    }

    inline bool operator!=(const IVFVectorID& other) const {
        return (vector_hash != other.vector_hash) || (value != other.value);
    }

    inline String ToString() const {
        return String("{hash=%lx, value=%u}", vector_hash, value);
    }
};

constexpr IVFVectorID INVALID_IVF_VECTOR_ID = {UINT64_MAX, UINT32_MAX};

template<uint8_t bits>
constexpr inline uint64_t GET_HALF_MASK() {
    if (bits == 1) {
        return (uint64_t)1;
    } else if (bits == 2) {
        return (uint64_t)3;
    } else if (bits == 3) {
        return (uint64_t)7;
    } else if (bits == 4) {
        return (uint64_t)15;
    } else if (bits == 5) {
        return (uint64_t)31;
    } else if (bits == 6) {
        return (uint64_t)63;
    } else if (bits == 7) {
        return (uint64_t)127;
    } else if (bits == 8) {
        return (uint64_t)255;
    }
    return (uint64_t)0;
}

template<uint8_t bytes>
constexpr inline uint64_t GET_MASK() {
    if (bytes == 1) {
        return (uint64_t)UINT8_MAX;
    } else if (bytes == 2) {
        return (uint64_t)UINT16_MAX;
    } else if (bytes == 3) {
        return ((((uint64_t)UINT16_MAX) << 8) | (uint64_t)UINT8_MAX);
    } else if (bytes == 4) {
        return (uint64_t)UINT32_MAX;
    } else if (bytes == 5) {
        return ((((uint64_t)UINT32_MAX) << 8) | (uint64_t)UINT8_MAX);
    } else if (bytes == 6) {
        return ((((uint64_t)UINT32_MAX) << 16) | (uint64_t)UINT16_MAX);
    } else if (bytes == 7) {
        return (((uint64_t)UINT32_MAX) << 24) | (((((uint64_t)UINT16_MAX) << 8) | (uint64_t)UINT8_MAX));
    } else if (bytes == 8) {
        return (uint64_t)UINT64_MAX;
    }
    return (uint64_t)0;
}

typedef uint64_t VectorDataHash;

inline uint32_t splitmix32(uint32_t x) {
    x += 0x9e3779b9;
    x = (x ^ (x >> 16)) * 0x85ebca6b;
    x = (x ^ (x >> 13)) * 0xc2b2ae35;
    return x ^ (x >> 16);
}

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline uint64_t wycombine(uint64_t a, uint64_t b) {
    a ^= b;
    a *= 0xd6e8feb86659fd93ULL;
    a ^= a >> 32;
    return a;
}

struct VersionHash {
    inline size_t operator()(const Version& p) const {
        return splitmix32(p._raw);
    }
};

struct VectorIDHash {
    inline size_t operator()(const VectorID& p) const {
        return splitmix64(p._id);
    }
};

struct VectorIDCMP {
    inline int operator()(const VectorID& a, const VectorID& b) const {
        if (a < b) {
            return -1;
        } else if (a > b) {
            return 1;
        } else {
            return 0;
        }
    }
};

struct VectorIDVersionPairCMP {
    inline int operator()(const std::pair<VectorID, Version>& a, const std::pair<VectorID, Version>& b) const {
        if (a.first < b.first) {
            return -1;
        } else if (a.first > b.first) {
            return 1;
        } else {
            if (a.second < b.second) {
                return -1;
            } else if (a.second > b.second) {
                return 1;
            } else {
                return 0;
            }
        }
    }
};

struct VectorIDVersionPairHash {
    inline size_t operator()(const std::pair<VectorID, Version>& p) const {
        if (p.first.IsCentroid()) {
            uint64_t h = wycombine(p.first._id, p.second._raw);

            h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ULL;
            h ^= h >> 27; h *= 0x94d049bb133111ebULL;
            h ^= h >> 31;
            return h;
        } else {
            return splitmix64(p.first._id);
        }
    }
};

template <typename T>
struct PtrHash {
    inline size_t operator()(T* const& p) const {
        return splitmix64(reinterpret_cast<uint64_t>(p));
    }
};

typedef void* Address;
typedef const void* AddressToConst;

constexpr Address INVALID_ADDRESS = nullptr;

// class DIVFTreeVertexInterface;
// class DIVFTreeInterface;
// class BufferManagerInterface;
// class DIVFTreeVertex;
// class DIVFTree;
// class BufferManager;
// class Cluster;

enum class ClusteringType : int8_t {
    Invalid,
    RoundRobin,
    KMeans,
    PCA1,
    GradientDescentLinearRegression, // GDLR
    /* first use a 2-mean(as in k mean) and if it produced higly imbalanced clusters use PCA1 or something becuase it usually means that the cluster */
    KMeansWithFallBackPCA1,
    KMeansWithFallBackGDLR,
    OutLierDetectionThenKMeansWithFallBackPCA1,
    OutLierDetectionThenKMeansWithFallBackGDLR,
    NumTypes
};
inline constexpr char* CLUSTERING_TYPE_NAME[(int8_t)(ClusteringType::NumTypes) + 1] =
    {"Invalid", "RoundRobin", "KMeans", "PCA1", "GradientDescentLinearRegression", "KMeansWithFallBackPCA1",
     "KMeansWithFallBackGDLR", "OutLierDetectionThenKMeansWithFallBackPCA1",
     "OutLierDetectionThenKMeansWithFallBackGDLR", "NumTypes"};
inline constexpr bool IsValid(ClusteringType type) {
    return ((type > ClusteringType::Invalid) && (type < ClusteringType::NumTypes));
}
inline constexpr ClusteringType CLUSTERING_NAME_TO_ENUM(const char * const name) {
    if (!strcmp(name, "RoundRobin")) {
        return ClusteringType::RoundRobin;
    } else if (!strcmp(name, "KMeans")) {
        return ClusteringType::KMeans;
    } else if (!strcmp(name, "PCA1")) {
        return ClusteringType::PCA1;
    } else if (!strcmp(name, "GradientDescentLinearRegression")) {
        return ClusteringType::GradientDescentLinearRegression;
    } else if (!strcmp(name, "KMeansWithFallBackPCA1")) {
        return ClusteringType::KMeansWithFallBackPCA1;
    } else if (!strcmp(name, "KMeansWithFallBackGDLR")) {
        return ClusteringType::KMeansWithFallBackGDLR;
    } else if (!strcmp(name, "OutLierDetectionThenKMeansWithFallBackPCA1")) {
        return ClusteringType::OutLierDetectionThenKMeansWithFallBackPCA1;
    } else if (!strcmp(name, "OutLierDetectionThenKMeansWithFallBackGDLR")) {
        return ClusteringType::OutLierDetectionThenKMeansWithFallBackGDLR;
    } else {
        return ClusteringType::Invalid;
    }
}

enum class DistanceType : int8_t {
    Invalid,
    L2,
    NumTypes
};
inline constexpr char* DISTANCE_TYPE_NAME[int8_t(DistanceType::NumTypes) + 1] = {"Invalid", "L2", "NumTypes"};
inline constexpr bool IsValid(DistanceType type) {
    return ((type > DistanceType::Invalid) && (type < DistanceType::NumTypes));
}

enum class SortType : int8_t {
    Unsorted,
    IncreasingSimilarity,
    DecreasingSimilarity
};


/*

    User local vector insert:
        1) Reserve a unique vector id at level 0
        2) Search for an appropriate leaf vertex to insert the vector and pin it
        3) TryLock the leaf in shared mode and try to reserve memory in cluster
            3.1) if lock failed it means there was an update and go to step 5
        4) If node is full split(*)
        5) If node is updating wait on cond var and retry based on the update type
            5.1)
        6) Create and insert entry in insertion state and increment _real_size
        7) Create buffer Entry for the entry
        8) change state from insertion to valid
        9) unlock leaf
        10) unpin leaf

    User local vector delete:
        1) Search for the vector in the buffer
        2) If not found return not found
        3) pin vertex and get lock in shared mode
            3.1) if lock failed it means there was an update and
                3.1.1)
        4) check the entry and CAS from Valid to Invalid
        5) if cas failed due to state:
            5.1)
        6) Create an empty buffer entry and cas it to the head of the version queue
            6.1) can cas fail here?
        7) Unpin vector and if pin is 0 delete it
            7.1) If deleted cas the new entry with nullptr and unpin it and if 0 delete
                7.2) can cas fail here?
        8) decrement _real_size and if it is less than min_size get exclusive, compact(*) and if needed merge(*)
        9) unlock vertex
        10) unpin vertex

    user local vector update:
        1) Search for the vector in the buffer
        2) If not found return not found
        3) pin vertex and get lock in shared mode
            3.1) if lock failed it means there was an update and
                3.1.1)
        4) check the entry and CAS from Valid to Valid|Move
        5) if cas failed due to state:
            5.1)
        6) Unlock vertex but do not unpin it!
        7) execute steps 2-7 of user local vector insert
        8) Cas the new buffer entry as the head of the version queue and make the current old version
            8.1) can cas fail? if not use exchange
            8.2) unpin the old version and if 0 delete it
        9) Unpin vector and if pin is 0 delete it
            9.1) If deleted cas the new entry with nullptr and unpin it and if 0 delete
        10) decrement _real_size and if it is less than min_size get exclusive, compact(*) and if needed merge(*)
        11) unlock vertex
        12) unpin vertex

*/
// enum VertexUpdateType : int8_t {
//     /* Main operations */
//     VERTEX_INSERT,
//     VERTEX_DELETE,
//     VERTEX_MOVE,
//     VERTEX_INPLACE_UPDATE,

//     /* Insertion subtypes */
//     VERTEX_MIGRATION_INSERT,
//     VERTEX_UPDATE_INSERT,

//     /* Deletion subtypes */
//     VERTEX_MIGRATION_DELETE,
//     VERTEX_UPDATE_DELETE,

//     NUM_VERTEX_UPDATE_TYPES
// };

// struct BatchVertexUpdateEntry {
//     VertexUpdateType type;
//     bool is_urgent;
//     VectorID vector_id;
//     RetStatus *result;

//     union {
//         struct {
//             AddressToConst vector_data; // for VERTEX_INSERT
//             Address *cluster_entry_addr; // if successful, this will set to the address of final cluster entry where vector is inserted
//         } insert_args;

//         struct {
//             AddressToConst vector_data; // for VERTEX_INPLACE_UPDATE
//         } inplace_update_args;
//     };
// };

/* todo need to think more about the cases for inserting a new operation as it
might cause deadlock or unnecessary errors*/
// struct BatchVertexUpdate {
//     std::map<VectorID, BatchVertexUpdateEntry> updates[NUM_VERTEX_UPDATE_TYPES];
//     VectorID target_cluster;
//     bool urgent = false;

//     inline RetStatus AddInsert(const VectorID& vector_id, AddressToConst vector_data,
//                                RetStatus *result = nullptr, bool is_urgent = false) {
//         FatalAssert(vector_id.IsValid(), LOG_TAG_DIVFTREE, "Invalid VectorID: " VECTORID_LOG_FMT, VECTORID_LOG(vector_id));
//         FatalAssert(vector_data != nullptr, LOG_TAG_DIVFTREE, "Vector data cannot be null.");
//         FatalAssert(target_cluster.IsValid(), LOG_TAG_DIVFTREE, "Target cluster is not set.");
//         FatalAssert(target_cluster.IsCentroid(), LOG_TAG_DIVFTREE, "Target cluster is not a centroid.");
//         FatalAssert(target_cluster._level == vector_id._level + 1,
//                     LOG_TAG_DIVFTREE, "Target cluster level (%hhu) does not match vector ID level (%hhu).",
//                     target_cluster._level, vector_id._level + 1);
//         RetStatus rs = RetStatus::Success();
//         if (updates[VERTEX_INSERT].find(vector_id) != updates[VERTEX_INSERT].end() ||
//             updates[VERTEX_MIGRATION_MOVE].find(vector_id) != updates[VERTEX_MIGRATION_MOVE].end() ||
//             updates[VERTEX_MIGRATION_DELETE].find(vector_id) != updates[VERTEX_MIGRATION_DELETE].end() ||
//             updates[VERTEX_INPLACE_UPDATE].find(vector_id) != updates[VERTEX_INPLACE_UPDATE].end()) {
//             DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_DIVFTREE,
//                  "VectorID " VECTORID_LOG_FMT " already exists in the batch with conflicting operations.",
//                  VECTORID_LOG(vector_id));
//             rs = RetStatus{
//                 .stat = RetStatus::BATCH_CONFLICTING_OPERATIONS
//             };
//             if (result != nullptr) {
//                 *result = rs;
//             }
//             return rs;
//         }
//         else if (updates[VERTEX_DELETE].find(vector_id) != updates[VERTEX_DELETE].end()) {
//             /* delete should override insertion as it was either caused by user  */
//             DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE,
//                  "VectorID " VECTORID_LOG_FMT
//                  " already exists in the batch with a delete operation. Overwriting it with INPLACE_UPDATE_OPT.",
//                  VECTORID_LOG(vector_id));
//             it->second.type = VERTEX_INPLACE_UPDATE;
//             it->second.is_urgent ||= is_urgent;
//             it->second.vector_id = vector_id;
//             urgent ||= is_urgent;
//             return rs;
//         }

//         updates[vector_id] = {.type = VERTEX_INSERT,
//                               .is_urgent = is_urgent,
//                               .vector_id = vector_id,
//                               .result = result,
//                               .insert_args.vector_data = vector_data};
//         urgent ||= is_urgent;
//         return RetStatus::Success();
//     }

//     inline RetStatus AddInplaceUpdate(const VectorID& vector_id, AddressToConst vector_data,
//                                       RetStatus *result = nullptr, bool is_urgent = false) {
//         FatalAssert(vector_id.IsValid(), LOG_TAG_DIVFTREE, "Invalid VectorID: " VECTORID_LOG_FMT, VECTORID_LOG(vector_id));
//         FatalAssert(vector_data != nullptr, LOG_TAG_DIVFTREE, "Vector data cannot be null.");
//         FatalAssert(target_cluster.IsValid(), LOG_TAG_DIVFTREE, "Target cluster is not set.");
//         FatalAssert(target_cluster.IsCentroid(), LOG_TAG_DIVFTREE, "Target cluster is not a centroid.");
//         FatalAssert(target_cluster._level == vector_id._level + 1,
//                     LOG_TAG_DIVFTREE, "Target cluster level (%hhu) does not match vector ID level (%hhu).",
//                     target_cluster._level, vector_id._level + 1);
//         RetStatus rs = RetStatus::Success();
//         if (updates[VERTEX_INSERT].find(vector_id) != updates[VERTEX_INSERT].end() ||
//             updates[VERTEX_MIGRATION_MOVE].find(vector_id) != updates[VERTEX_MIGRATION_MOVE].end() ||
//             updates[VERTEX_DELETE].find(vector_id) != updates[VERTEX_DELETE].end() ||
//             updates[VERTEX_INPLACE_UPDATE].find(vector_id) != updates[VERTEX_INPLACE_UPDATE].end()) {
//             DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_DIVFTREE,
//                  "VectorID " VECTORID_LOG_FMT " already exists in the batch with conflicting operations.",
//                  VECTORID_LOG(vector_id));
//             rs = RetStatus{
//                 .stat = RetStatus::BATCH_CONFLICTING_OPERATIONS
//             };
//             if (result != nullptr) {
//                 *result = rs;
//             }
//             return rs;
//         }

//         updates[vector_id] = {.type = VERTEX_INPLACE_UPDATE,
//                               .is_urgent = is_urgent,
//                               .vector_id = vector_id,
//                               .result = result,
//                               .inplace_update_args.vector_data = vector_data};
//         urgent ||= is_urgent;
//         return RetStatus::Success();
//     }

//     inline RetStatus AddMigrate(const VectorID& vector_id, RetStatus *result = nullptr,
//                                 bool is_urgent = false) {
//         FatalAssert(vector_id.IsValid(), LOG_TAG_DIVFTREE, "Invalid VectorID: " VECTORID_LOG_FMT, VECTORID_LOG(vector_id));
//         FatalAssert(target_cluster.IsValid(), LOG_TAG_DIVFTREE, "Target cluster is not set.");
//         FatalAssert(target_cluster.IsCentroid(), LOG_TAG_DIVFTREE, "Target cluster is not a centroid.");
//         FatalAssert(target_cluster._level == vector_id._level + 1,
//                     LOG_TAG_DIVFTREE, "Target cluster level (%hhu) does not match vector ID level (%hhu).",
//                     target_cluster._level, vector_id._level + 1);
//         RetStatus rs = RetStatus::Success();
//         if (updates[VERTEX_INSERT].find(vector_id) != updates[VERTEX_INSERT].end() ||
//             updates[VERTEX_MIGRATION_MOVE].find(vector_id) != updates[VERTEX_MIGRATION_MOVE].end() ||
//             updates[VERTEX_DELETE].find(vector_id) != updates[VERTEX_DELETE].end() ||
//             updates[VERTEX_INPLACE_UPDATE].find(vector_id) != updates[VERTEX_INPLACE_UPDATE].end()) {
//             DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_DIVFTREE,
//                  "VectorID " VECTORID_LOG_FMT " already exists in the batch with conflicting operations.",
//                  VECTORID_LOG(vector_id));
//             rs = RetStatus{
//                 .stat = RetStatus::BATCH_CONFLICTING_OPERATIONS
//             };
//             if (result != nullptr) {
//                 *result = rs;
//             }
//             return rs;
//         }

//         updates[vector_id] = {.type = VERTEX_MIGRATION_MOVE,
//                               .is_urgent = is_urgent,
//                               .vector_id = vector_id,
//                               .result = result};
//         urgent ||= is_urgent;
//         return RetStatus::Success();
//     }

//     inline RetStatus AddDelete(const VectorID& vector_id, RetStatus *result = nullptr,
//                                 bool is_urgent = false) {
//         FatalAssert(vector_id.IsValid(), LOG_TAG_DIVFTREE, "Invalid VectorID: " VECTORID_LOG_FMT, VECTORID_LOG(vector_id));
//         FatalAssert(target_cluster.IsValid(), LOG_TAG_DIVFTREE, "Target cluster is not set.");
//         FatalAssert(target_cluster.IsCentroid(), LOG_TAG_DIVFTREE, "Target cluster is not a centroid.");
//         FatalAssert(target_cluster._level == vector_id._level + 1,
//                     LOG_TAG_DIVFTREE, "Target cluster level (%hhu) does not match vector ID level (%hhu).",
//                     target_cluster._level, vector_id._level + 1);
//         RetStatus rs = RetStatus::Success();
//         if (updates[VERTEX_INSERT].find(vector_id) != updates[VERTEX_INSERT].end() ||
//             updates[VERTEX_MIGRATION_MOVE].find(vector_id) != updates[VERTEX_MIGRATION_MOVE].end() ||
//             updates[VERTEX_DELETE].find(vector_id) != updates[VERTEX_DELETE].end() ||
//             updates[VERTEX_INPLACE_UPDATE].find(vector_id) != updates[VERTEX_INPLACE_UPDATE].end()) {
//             DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_DIVFTREE,
//                  "VectorID " VECTORID_LOG_FMT " already exists in the batch with conflicting operations.",
//                  VECTORID_LOG(vector_id));
//             rs = RetStatus{
//                 .stat = RetStatus::BATCH_CONFLICTING_OPERATIONS
//             };
//             if (result != nullptr) {
//                 *result = rs;
//             }
//             return rs;
//         }

//         updates[vector_id] = {.type = VECTOR_DELETE,
//                               .is_urgent = is_urgent,
//                               .vector_id = vector_id,
//                               .result = result};
//         urgent ||= is_urgent;
//         return RetStatus::Success();
//     }
// };

#ifndef DIMENSION
#define DIMENSION 128
#error "DIMENSION is not defined"
#endif

#ifndef VECTOR_TYPE
#define VECTOR_TYPE UINT16
typedef uint16_t VTYPE;
#define VTYPE_FMT "%hu"
#endif

#ifndef CENTROID_TYPE
#define CENTROID_TYPE UINT16
typedef uint16_t CTYPE;
#define CTYPE_FMT "%hu"
#endif

#ifndef DISTANCE_TYPE
#define DISTANCE_TYPE FLOAT
typedef float DTYPE;
#define DTYPE_FMT "%0.2f"
#endif

#ifndef CENTROID_TYPE
#define CENTROID_TYPE UINT16
typedef uint16_t CENTROID;
#define CENTROID_FMT "%hu"
typedef double MVTYPE;
#define MVTYPE_FMT "%0.2f"
#endif



/* todo: use cpuid to get cahceline size */
// #if defined(__cpp_lib_hardware_interference_size) || defined(__GNUC__) || defined(_MSC_VER)
// #include <new>
// constexpr size_t CACHE_LINE_SIZE = std::hardware_destructive_interference_size;
// #else
constexpr size_t CACHE_LINE_SIZE = 64; // Fallback to common cache line size
// #endif

inline constexpr size_t ALIGNED_SIZE(size_t size, size_t align_bytes = CACHE_LINE_SIZE) {
    FatalAssert(align_bytes != 0, LOG_TAG_BASIC, "Alignment bytes cannot be zero.");
    return (size % align_bytes == 0) ? size : (size + (align_bytes - (size % align_bytes)));
}

inline constexpr size_t ALIGNEMENT(size_t size, size_t align_bytes = CACHE_LINE_SIZE) {
    FatalAssert(align_bytes != 0, LOG_TAG_BASIC, "Alignment bytes cannot be zero.");
    return (size % align_bytes == 0) ? 0 : (align_bytes - (size % align_bytes));
}

inline constexpr bool ALIGNED(void* ptr, size_t align_bytes = CACHE_LINE_SIZE) {
    FatalAssert(align_bytes != 0, LOG_TAG_BASIC, "Alignment bytes cannot be zero.");
    return (reinterpret_cast<uintptr_t>(ptr) % align_bytes == 0);
}

inline constexpr bool ALIGNED(size_t size, size_t align_bytes = CACHE_LINE_SIZE) {
    FatalAssert(align_bytes != 0, LOG_TAG_BASIC, "Alignment bytes cannot be zero.");
    return (reinterpret_cast<uintptr_t>(size) % align_bytes == 0);
}

inline constexpr void* ALIGNED_PTR(void* ptr, size_t align_bytes = CACHE_LINE_SIZE) {
    FatalAssert(align_bytes != 0, LOG_TAG_BASIC, "Alignment bytes cannot be zero.");
    return (reinterpret_cast<uintptr_t>(ptr) % align_bytes == 0) ? ptr :
            reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(ptr) +
                                    (align_bytes - (reinterpret_cast<uintptr_t>(ptr) % align_bytes))));
}

inline constexpr const void* ALIGNED_PTR(const void* ptr, size_t align_bytes = CACHE_LINE_SIZE) {
    FatalAssert(align_bytes != 0, LOG_TAG_BASIC, "Alignment bytes cannot be zero.");
    return (reinterpret_cast<uintptr_t>(const_cast<void*>(ptr)) % align_bytes == 0) ? ptr :
            reinterpret_cast<const void*>((reinterpret_cast<uintptr_t>(const_cast<void*>(ptr)) +
                                (align_bytes - (reinterpret_cast<uintptr_t>(const_cast<void*>(ptr)) %
                                                                           align_bytes))));
}


#define VECTORID_LOG_FMT "%s%lu(%lu, %lu, %lu)"
#define VECTORID_LOG(vid) (!((vid).IsValid()) ? "[INV]" : ""), (vid)._id, (vid)._creator_node_id, (vid)._level, (vid)._val
struct ANNVectorInfo {
    DTYPE distance_to_query;
    VectorID id;
    Version version;

    ANNVectorInfo() = default;
    ANNVectorInfo(DTYPE dist, VectorID vectorId, Version vectorVersion = 0) : distance_to_query(dist), id(vectorId),
                                                                              version(vectorVersion) {}

    inline bool operator==(const ANNVectorInfo& other) {
        return (id == other.id) && (version == other.version);
    }

    inline bool operator!=(const ANNVectorInfo& other) {
        return (id != other.id) || (version != other.version);
    }
};

String ANNVectorInfoToString(const ANNVectorInfo& target) {
    return String("{dist=" DTYPE_FMT ", id=" VECTORID_LOG_FMT ", Version=%s}",
                 target.distance_to_query, VECTORID_LOG(target.id), target.version.ToString().ToCStr());
}

struct L2DTYPEIDPairCMP;

struct IVFSearchTask {
    const VTYPE* query_vector;
    const uint32_t num_sibling_tasks;
    const VectorID cluster_id;
    uint32_t top_k;
    uint32_t cluster_partition_num_elements;
    void* cluster_partition_address;
    bool is_leaf;
    std::atomic<uint32_t>* num_tasks_completed;
    SXLock* neighbour_list_lock;
    union {
        std::vector<std::pair<DTYPE, IVFVectorID>>* top_vectors;
        std::vector<std::pair<DTYPE, VectorID>>* top_centroids;
    };
};

struct IVFSearchTaskFactory {
    const VTYPE* query_vector;
    uint32_t num_sibling_tasks;
    uint32_t top_k;
    bool is_leaf;
    std::atomic<uint32_t>* num_tasks_completed;
    SXLock* neighbour_list_lock;

    SANITY_CHECK(
    std::atomic<uint32_t>* num_tasks_created = new std::atomic<uint32_t>(0);
    DIVFThreadID creator_thread_id = DIVF_THREAD_ID;

    ~IVFSearchTaskFactory() {
        uint32_t expected_tasks = num_sibling_tasks;
        uint32_t created_tasks = num_tasks_created->load(std::memory_order_acquire);
        uint32_t completed_tasks = num_tasks_completed->load(std::memory_order_acquire);
        FatalAssert(creator_thread_id == DIVF_THREAD_ID, LOG_TAG_BASIC,
                    "IVFSearchTaskFactory destroyed by a different thread than it was created. Creator thread ID: %u, Destroyer thread ID: %u",
                    creator_thread_id, DIVF_THREAD_ID);
        FatalAssert(expected_tasks == completed_tasks,
                    LOG_TAG_BASIC,
                    "Not all IVFSearchTasks completed before IVFSearchTaskFactory destruction. "
                    "Expected tasks: %lu, Completed tasks: %lu, Creator thread ID: %u, Destroyer thread ID: %u",
                    expected_tasks, completed_tasks, creator_thread_id, DIVF_THREAD_ID);
        FatalAssert(expected_tasks == created_tasks,
                    LOG_TAG_BASIC,
                    "IVFSearchTaskFactory destroyed before all tasks are created. "
                    "Expected tasks: %lu, Created tasks: %lu, Creator thread ID: %u, Destroyer thread ID: %u",
                    expected_tasks, created_tasks, creator_thread_id, DIVF_THREAD_ID);
        delete num_tasks_created;
    }
    )
    union {
        std::vector<std::pair<DTYPE, IVFVectorID>>* top_vectors;
        std::vector<std::pair<DTYPE, VectorID>>* top_centroids;
    };

    inline IVFSearchTask* CreateTask(VectorID cluster_id, uint32_t cluster_partition_num_elements,
                                     void* cluster_partition_address) {
        SANITY_CHECK(
            uint32_t created_tasks = num_tasks_created->fetch_add(1) + 1;
            FatalAssert(created_tasks <= num_sibling_tasks, LOG_TAG_BASIC,
                        "Created more IVFSearchTasks than expected. Expected: %lu, Created: %lu",
                        num_sibling_tasks, created_tasks);
        );
        return new IVFSearchTask{
            .query_vector = query_vector,
            .num_sibling_tasks = num_sibling_tasks,
            .cluster_id = cluster_id,
            .top_k = top_k,
            .cluster_partition_num_elements = cluster_partition_num_elements,
            .cluster_partition_address = cluster_partition_address,
            .is_leaf = is_leaf,
            .num_tasks_completed = num_tasks_completed,
            .neighbour_list_lock = neighbour_list_lock,
            .top_vectors = top_vectors,
        };
    }
};

enum class IndexType : uint8_t {
    IVF_FLAT = 0,
    IVF_CAPPED = 1,
    IVF_TREE = 2
};

// enum DataType : int8_t {
//     Invalid = -1,
//     UInt16,
//     Float,
//     Double,
//     NumTypes
// };
// inline constexpr char* DATA_TYPE_NAME[DataType::NumTypes] = {"uint16", "float", "double"};
// inline constexpr size_t SIZE_OF_TYPE[DataType::NumTypes] = {sizeof(uint16_t), sizeof(float), sizeof(double)};
// inline constexpr bool IsValid(DataType type) {
//     return ((type != DataType::Invalid) && (type != DataType::NumTypes));
// }

// String DataToString(const void* data, DataType type) {
//     if (data == nullptr) {
//         return String("NULL");
//     }
//     switch (type) {
//         case DataType::UInt16: {
//             return String("%hu", *(reinterpret_cast<const uint16_t*>(data)));
//         }
//         case DataType::Float: {
//             return String("%f", *(reinterpret_cast<const float*>(data)));
//         }
//         case DataType::Double: {
//             return String("%lf", *(reinterpret_cast<const double*>(data)));
//         }
//         default:
//             return String("Invalid DataType");
//     }
// }
// Todo: Log DIVFTree: DIVFTree(RootID:%s(%lu, %lu, %lu), # levels:lu, # vertices:lu, # vectors:lu, size:lu)

// #ifndef PRINT_BUCKET
// #define PRINT_BUCKET false
// #endif

#define CHECK_MIN_MAX_SIZE(min_size, max_size, tag) \
    do { \
        FatalAssert((min_size) > 0, (tag), "Min size must be greater than 0."); \
        FatalAssert(((max_size) / 2) > (min_size), (tag), \
                    "Max size must be more than twice the min size. Min size: %hu, Max size: %hu", \
                    (min_size), (max_size)); \
    } while(0)

#define CHECK_VECTORID_IS_VALID(vid, tag) \
    FatalAssert((vid).IsValid(), (tag), "Invalid VectorID: " VECTORID_LOG_FMT, VECTORID_LOG((vid)))
#define CHECK_VECTORID_IS_CENTROID(vid, tag) \
    FatalAssert((vid).IsCentroid(), (tag), "VectorID " VECTORID_LOG_FMT " is not a centroid", VECTORID_LOG((vid)))
#define CHECK_VECTORID_IS_VECTOR(vid, tag) \
    FatalAssert((vid).IsVector(), (tag), "VectorID " VECTORID_LOG_FMT " is not a vector", VECTORID_LOG((vid)))
#define CHECK_VECTORID_IS_LEAF(vid, tag) \
    FatalAssert((vid).IsLeaf(), (tag), "VectorID " VECTORID_LOG_FMT " is not a leaf", VECTORID_LOG((vid)))
#define CHECK_VECTORID_IS_INTERNAL(vid, tag) \
    FatalAssert((!((vid).IsLeaf()) && ((vid).IsCentroid())), (tag), \
                "VectorID " VECTORID_LOG_FMT " is not a leaf", VECTORID_LOG((vid)))

};

#endif
