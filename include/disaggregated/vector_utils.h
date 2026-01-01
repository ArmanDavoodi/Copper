#ifndef DIVFTREE_VECTOR_UTILS_H_
#define DIVFTREE_VECTOR_UTILS_H_

#include "common.h"

#include <algorithm>


namespace divftree {


struct VectorBatchMeta {
    ClusterSizeType is_batch_size : 1;
    ClusterSizeType batch_size_or_last_offset : 15;
};

String VectorToString(const VTYPE* vec, uint16_t dim) {
    String res = "{";
    for (uint16_t i = 0; i < dim; ++i) {
        res += String(VTYPE_FMT "%s", vec[i], (i == dim - 1 ? "}" : ", "));
    }
    return res;
}

enum VectorStateDetail : uint8_t {
    VECTOR_STATE_NORMAL = 0,
    VECTOR_STATE_MIGRATED = 1,
    VECTOR_STATE_OUTDATED = 2,
    VECTOR_STATE_DELETED = 3
};

enum VectorLockState : uint8_t {
    VECTOR_LOCK_UNLOCKED = 0,
    VECTOR_LOCK_IN_PROGRESS = 1,
    VECTOR_LOCK_LOCKED = 2,
};

struct VectorState {
    uint8_t is_state_valid : 1;
    VectorStateDetail detail : 2;
    VectorLockState lock_state : 2; /* Unused for now */
    uint8_t unused : 3;

    inline bool operator==(const VectorState& other) const {
        return (is_state_valid == other.is_state_valid) &&
               (detail == other.detail);
    }
};

inline void ChangeVectorState(std::atomic<VectorState>& state, VectorState& expected, VectorStateDetail& target) {
    if (expected.is_state_valid) {
        FatalAssert(expected.detail == VECTOR_STATE_NORMAL, LOG_TAG_CLUSTER,
                    "Only NORMAL state can be changed with this function!");
        FatalAssert(target != VECTOR_STATE_NORMAL, LOG_TAG_CLUSTER,
                    "Target state cannot be NORMAL in this function!");
        FatalAssert(state.load(std::memory_order_acquire) == expected,
                    LOG_TAG_CLUSTER,
                    "Expected state does not match the actual state!");
        state.store(VectorState{true, target, VECTOR_LOCK_UNLOCKED, 0},
                    std::memory_order_release);
        return;
    }

    VectorState desired{true, target, VECTOR_LOCK_UNLOCKED, 0};
    if (expected.detail == VECTOR_STATE_NORMAL) {
        /* Expecting Empty Vector Slot(Invalid) */
        if (target != VECTOR_STATE_NORMAL) {
            /* Empty -> Migrated/Outdated/Deleted (Invalid state) -> Out-of-order update */
            desired.is_state_valid = false;
            if (!state.compare_exchange_strong(expected, desired)) {
                /* In this case, expected should have become valid-normal */
                FatalAssert(expected.is_state_valid &&
                            expected.detail == VECTOR_STATE_NORMAL,
                            LOG_TAG_CLUSTER,
                            "Expected state does not match the actual state after failed CAS!");
                desired.is_state_valid = true;
                state.store(desired, std::memory_order_release);
            }
            return;
        }

        /* Empty -> Valid (Insertion) */
        if (state.compare_exchange_strong(expected, desired)) {
            return;
        }
        /* In this case, state should have become Invalid-(migrated/deleted/outdated)
           and we should now change state from Invalid-non normal to valid-non Normal */
        target = expected.detail;
        desired.detail = target;
    }

    /* Expecting an Invalid non-normal state */
    FatalAssert(!expected.is_state_valid,
                LOG_TAG_CLUSTER,
                "Expected state cannot be valid");
    FatalAssert(expected.detail != VECTOR_STATE_NORMAL,
                LOG_TAG_CLUSTER,
                "Expected state cannot be NORMAL");
    FatalAssert(target == expected.detail,
                LOG_TAG_CLUSTER,
                "Target state must be equal to expected state in this case");
    FatalAssert(desired.detail == target,
                LOG_TAG_CLUSTER,
                "Desired state must be equal to target state in this case");
    FatalAssert(desired.is_state_valid,
                LOG_TAG_CLUSTER,
                "Desired state must be valid in this case");
    state.store(desired, std::memory_order_release);
    return;
}

/* tries to change state to target. will returns the old state and sets target to current state */
inline VectorState ChangeVectorState(std::atomic<VectorState>& state, VectorStateDetail& target) {
    VectorState expected = state.load(std::memory_order_acquire);
    ChangeVectorState(state, expected, target);
    return expected;
}

inline String VectorStateDetailToString(const VectorStateDetail& state) {
    switch (state)
    {
    case VECTOR_STATE_NORMAL:
        return String("NORMAL");
    case VECTOR_STATE_MIGRATED:
        return String("MIGRATED");
    case VECTOR_STATE_OUTDATED:
        return String("OUTDATED");
    case VECTOR_STATE_DELETED:
        return String("DELETED");
    default:
        return String("UNDEFINED");
    }
}

inline String VectorLockStateToString(const VectorLockState& state) {
    switch (state)
    {
    case VECTOR_LOCK_UNLOCKED:
        return String("UNLOCKED");
    case VECTOR_LOCK_IN_PROGRESS:
        return String("IN_PROGRESS");
    case VECTOR_LOCK_LOCKED:
        return String("LOCKED");
    default:
        return String("UNDEFINED");
    }
}

inline String VectorStateToString(const VectorState& state) {
    return String(state.is_state_valid ? "VALID-" : "INVALID-") +
           VectorStateDetailToString(state.detail) + "-" +
           VectorLockStateToString(state.lock_state);
}

/* this struct is always moved and never copied! */
struct VectorBatch {
    VTYPE* data = nullptr;
    VectorID* id = nullptr;
    Version* version = nullptr;
    ClusterSizeType size = 0;
};

// enum VectorBatchState : uint8_t {
//     BATCH_STATE_VALID = 0,
//     BATCH_STATE_INVALID = 1,
//     BATCH_STATE_CHAINED_INVALID = 2
//     /* chained invalid means that there is another batch somewhere that is invalid because of this batch */
// };

/* this struct is always moved and never copied! */
struct ConstVectorBatch {
    const VTYPE* data = nullptr;
    const VectorID* id = nullptr;
    const Version* version = nullptr;
    const ClusterSizeType size = 0;

    ConstVectorBatch() = default;
    ConstVectorBatch(ClusterSizeType s) : size(s) {};
    ConstVectorBatch(const ConstVectorBatch& other) = default;
    ConstVectorBatch(ConstVectorBatch&& other) = default;

    ConstVectorBatch(const VectorBatch& other) :
        data(other.data), id(other.id), version(other.version), size(other.size) {}

    ConstVectorBatch(const VTYPE* d, const VectorID* i, const Version* v, ClusterSizeType s) :
        data(d), id(i), version(v), size(s) {}
};

/* todo: we may need to use packed attrbite for these in the multi node setup to save network bandwidth */
struct VectorMetaData {
    VectorID id;
    VectorBatchMeta batch_meta;
    std::atomic<bool> batch_valid;
    std::atomic<VectorState> state;
};

/* todo: we may need to use packed attrbite for these in the multi node setup to save network bandwidth */
struct CentroidMetaData {
    VectorID id;
    Version version;
    VectorBatchMeta batch_meta;
    std::atomic<bool> batch_valid;
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
class Cluster {
public:
    inline static ClusterSizeType BlockSize(size_t block_bytes, size_t meta_size, ClusterSizeType cap, uint16_t dim) {
        return (ClusterSizeType)(std::min(block_bytes / (meta_size + (sizeof(VTYPE) * (size_t)dim)), (size_t)cap));
    }

    inline static size_t BlockBytes(ClusterSizeType block_size, size_t meta_size, uint16_t dim) {
        return ALIGNED_SIZE((size_t)block_size * ((size_t)meta_size + ((size_t)dim * sizeof(VTYPE))));
    }

    inline static size_t NumBlocks(ClusterSizeType block_size, ClusterSizeType cap) {
        return (cap + block_size - 1) / block_size;
    }

    inline static size_t TotalBytes(bool is_leaf_vertex, ClusterSizeType block_size, ClusterSizeType cap, uint16_t dim) {
        const uint64_t header_bytes = (is_leaf_vertex ? sizeof(VectorMetaData) : sizeof(CentroidMetaData));
        const uint64_t data_bytes = sizeof(VTYPE) * (uint64_t)dim;
        uint64_t last_block_cap = cap % block_size;
        if (last_block_cap == 0 && cap > 0) {
            last_block_cap = block_size;
        }
        return ((NumBlocks(block_size, cap) - 1) * ALIGNED_SIZE(block_size * (header_bytes + data_bytes))) +
               ALIGNED_SIZE(last_block_cap * (header_bytes + data_bytes)) + 1;
    }

    Cluster(bool is_leaf_vertex, ClusterSizeType block_cap, ClusterSizeType cap, uint16_t dim, bool set_valid = true,
            bool set_zero = true) {
        FatalAssert(block_cap > 0, LOG_TAG_CLUSTER, "Block size must be greater than 0. block_size=%hu", block_cap);
        FatalAssert(cap > 0, LOG_TAG_CLUSTER, "Capacity must be greater than 0. capacity=%hu", cap);
        FatalAssert(dim > 0, LOG_TAG_CLUSTER, "Dimension must be greater than 0. dimension=%hu", dim);
        const size_t bytes = TotalBytes(is_leaf_vertex, block_cap, cap, dim);
        if (set_zero) {
            memset(blocks, 0, bytes);
        }
        if (set_valid) {
            blocks[bytes - 1] |= 0b1; // set the last bit to indicate valid cluster for the purpose of RDMA Writes
        }
    }

    inline AddressToConst MetaData(ClusterSizeType offset, bool is_leaf, ClusterSizeType block_size,
                                   ClusterSizeType capacity, uint16_t dimension) const {
        FatalAssert(offset < capacity, LOG_TAG_CLUSTER, "Offset is out of bounds. offset=%hu, capacity=%hu",
                    offset, capacity);
        const ClusterSizeType block_number = offset / block_size;
        const ClusterSizeType block_offset = offset % block_size;
        const uint64_t header_bytes = (is_leaf ? sizeof(VectorMetaData) : sizeof(CentroidMetaData));
        const uint64_t data_bytes = sizeof(VTYPE) * (uint64_t)dimension;
        const uint64_t block_bytes = ALIGNED_SIZE(block_size * (header_bytes + data_bytes));
        FatalAssert(block_number == 0, LOG_TAG_CLUSTER,
                    "Only single block clusters are supported currently. offset=%hu, "
                    "block_number=%hu, block_size=%hu", offset, block_number, block_size);
        return reinterpret_cast<AddressToConst>(ALIGNED_PTR(blocks) +
                                                block_number * block_bytes + block_offset * header_bytes);
    }

    inline Address MetaData(ClusterSizeType offset, bool is_leaf, ClusterSizeType block_size,
                            ClusterSizeType capacity, uint16_t dimension) {
        FatalAssert(offset < capacity, LOG_TAG_CLUSTER, "Offset is out of bounds. offset=%hu, capacity=%hu",
                    offset, capacity);
        const ClusterSizeType block_number = offset / block_size;
        const ClusterSizeType block_offset = offset % block_size;
        const uint64_t header_bytes = (is_leaf ? sizeof(VectorMetaData) : sizeof(CentroidMetaData));
        const uint64_t data_bytes = sizeof(VTYPE) * (uint64_t)dimension;
        const uint64_t block_bytes = ALIGNED_SIZE(block_size * (header_bytes + data_bytes));
        FatalAssert(block_number == 0, LOG_TAG_CLUSTER,
                    "Only single block clusters are supported currently. offset=%hu, "
                    "block_number=%hu, block_size=%hu", offset, block_number, block_size);
        return reinterpret_cast<Address>(ALIGNED_PTR(blocks) +
                                         block_number * block_bytes + block_offset * header_bytes);
    }

    inline const VTYPE* Data(ClusterSizeType offset, bool is_leaf, ClusterSizeType block_size,
                             ClusterSizeType capacity, uint16_t dimension) const {
        FatalAssert(offset < capacity, LOG_TAG_CLUSTER, "Offset is out of bounds. offset=%hu, capacity=%hu",
                    offset, capacity);
        const ClusterSizeType block_number = offset / block_size;
        const ClusterSizeType block_offset = offset % block_size;
        uint64_t last_block_size = capacity % block_size;
        if (last_block_size == 0 && capacity > 0) {
            last_block_size = block_size;
        }
        const uint64_t header_bytes = (is_leaf ? sizeof(VectorMetaData) : sizeof(CentroidMetaData));
        const uint64_t data_bytes = sizeof(VTYPE) * (uint64_t)dimension;
        const uint64_t block_bytes = ALIGNED_SIZE(block_size * (header_bytes + data_bytes));
        const uint64_t meta_data_bytes = (block_number == (block_size - 1) ?
                                          last_block_size : block_size) *
                                          header_bytes;
        FatalAssert(block_number == 0, LOG_TAG_CLUSTER,
                    "Only single block clusters are supported currently. offset=%hu, "
                    "block_number=%hu, block_size=%hu", offset, block_number, block_size);
        return reinterpret_cast<const VTYPE*>(ALIGNED_PTR(blocks) +
                                              block_number * block_bytes + meta_data_bytes + block_offset * data_bytes);
    }

    inline VTYPE* Data(ClusterSizeType offset, bool is_leaf, ClusterSizeType block_size,
                       ClusterSizeType capacity, uint16_t dimension) {
        FatalAssert(offset < capacity, LOG_TAG_CLUSTER, "Offset is out of bounds. offset=%hu, capacity=%hu",
                    offset, capacity);
        const ClusterSizeType block_number = offset / block_size;
        const ClusterSizeType block_offset = offset % block_size;
        uint64_t last_block_size = capacity % block_size;
        if (last_block_size == 0 && capacity > 0) {
            last_block_size = block_size;
        }
        const uint64_t header_bytes = (is_leaf ? sizeof(VectorMetaData) : sizeof(CentroidMetaData));
        const uint64_t data_bytes = sizeof(VTYPE) * (uint64_t)dimension;
        const uint64_t block_bytes = ALIGNED_SIZE(block_size * (header_bytes + data_bytes));
        const uint64_t meta_data_bytes = (block_number == (block_size - 1) ?
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
     * | VectorID | Version | BatchMeta | State | ...
     * ---------------------------------------------------
     * | VectorData | ...
     *
     * the blocks of a leaf cluster have this structure:
     * | VectorID | BatchMeta | State | ...
     * ---------------------------------------------------
     * | VectorData | ...
     *
     *
     * Finally there is a single bit at the end of cluster indicating whether the RDMAWrite is finished
     *
     * NOTE: it is probably better to make each block cache alligned as we will be using different RDMA reads for them
     * but do not make the data inside a block cache alligned
     * to reduce the amount of data read from remote!
     */
protected:
    char* blocks;
    // alignas(CACHE_LINE_SIZE) char blocks[];

TESTABLE;
};
};
#endif