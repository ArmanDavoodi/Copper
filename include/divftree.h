#ifndef DIVFTREE_H_
#define DIVFTREE_H_

#include "common.h"
#include "vector_utils.h"
#include "buffer.h"
#include "distance.h"
#include "distributed_common.h"

#include "utils/synchronization.h"
#include "utils/concurrent_datastructures.h"

#include "interface/divftree.h"

#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <vector>

// Todo: better logs and asserts -> style: <Function name>(self data(a=?), input data): msg, additonal variables if needed

/*
 * Todo: There are two approaches we can take:
 * 1) Bottom-up nearest approach:
 *      This is our current approach in which we try to gurantee(lazily) that each vector is always in its closest vertex.
 *      However, this approach requires us to either have migration per insert or
 *      to check every possible leaf to see which one is closest.
 *      Imagin this example with 1 dim vectors:
 *      Layer 2:        1          2
 *      Layer 1:    0.9   1.1  1.6   2.6
 *      Layer 0:   ........................
 *      Now if we want to insert 1.4, it is closer to 1 so it will be inserted in leaf with centroid 1.1 while the closest
 *      leaf centroid to it is 1.6.
 *      probably better to have larger buckets.
 * 2) top-down nearest approach:
 *      In this approach, I gurantee that we can always take the greedy path from the root of the tree to
 *      find a vector.
 *      To do so, at the time of split of vertex k, we have to check migrataion for every vector in any layer in that subtree
 *      to other sibling vertices of vertex k.(sibling vertices are vertices with the same parent)
 *      As a result, a split in higher levels is more costly. and the cost only grows. lazy approach will cause us to have
 *      bad accuracy for a long time.
 *      probably having small buckets is better here.
 *
 */

namespace divftree {

struct MigrationCheckTask {
    VectorID first;
    VectorID second;
};

struct MergeTask {
    VectorID target;
};

struct CompactionTask {
    VectorID target;
};

struct SearchTask {
    DIVFThreadID master;
    uint64_t taskId;
    uint64_t num_tasks;
    VectorID target;
    Version version;
    const VTYPE* query;
    size_t k;

    std::atomic<bool>* taken;
    std::atomic<bool>* done;

    bool done_waiting;
    SortedList<ANNVectorInfo, SimilarityComparator>* neighbours;

    std::atomic<size_t>* viewed;
    std::atomic<bool>* shared_data;
    SearchTask** task_set;

    SearchTask() = default;
    SearchTask(uint64_t task_id, uint64_t nt, VectorID id, Version ver, const VTYPE* q,
               size_t span, std::atomic<bool>* tk, std::atomic<bool>* dn, std::atomic<size_t>* vd,
               std::atomic<bool>* sd, SearchTask** ts) :
        master(threadSelf->ID()), taskId(task_id), num_tasks(nt),
        target(id), version(ver), query(q), k(span),
        taken(tk), done(dn), done_waiting(false), neighbours(nullptr), viewed(vd), shared_data(sd), task_set(ts) {
        CHECK_VECTORID_IS_VALID(id, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(q, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(tk, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(dn, LOG_TAG_DIVFTREE);
        FatalAssert(span > 0, LOG_TAG_DIVFTREE, "k cannot be 0!");
    }

    inline void CopyFrom(const SearchTask& other) {
        master = other.master;
        taskId = other.taskId;
        target = other.target;
        version = other.version;
        query = other.query;
        k = other.k;
        neighbours = other.neighbours;
    }
};

struct MigrationInfo {
    VectorID id;
    uint16_t offset;
    Version version;

    MigrationInfo() = default;
    MigrationInfo(VectorID vec_id, uint16_t vec_off, Version ver = 0) : id(vec_id), offset(vec_off), version(ver) {}

    inline bool operator<(const MigrationInfo& other) const {
        return id < other.id;
    }
};

class DIVFTreeVertex : public DIVFTreeVertexInterface {
public:
    DIVFTreeVertex() = delete;
    DIVFTreeVertex(DIVFTreeVertex&) = delete;
    DIVFTreeVertex(const DIVFTreeVertex&) = delete;
    DIVFTreeVertex(DIVFTreeVertex&&) = delete;

    DIVFTreeVertex(const DIVFTreeVertexAttributes& attributes) : attr(attributes), unpinCount{0},
                                                                 cluster(attributes.centroid_id.IsLeaf(),
                                                                         attributes.block_size,
                                                                         attributes.cap,
                                                                         attributes.index->GetAttributes().dimension,
                                                                         true) {
        CHECK_MIN_MAX_SIZE(attr.min_size, attr.cap, LOG_TAG_DIVFTREE_VERTEX);
        CHECK_VECTORID_IS_VALID(attr.centroid_id, LOG_TAG_DIVFTREE_VERTEX);
        CHECK_VECTORID_IS_CENTROID(attr.centroid_id, LOG_TAG_DIVFTREE_VERTEX);
        CHECK_NOT_NULLPTR(attr.index, LOG_TAG_DIVFTREE_VERTEX);
        FatalAssert(attr.block_size > 0, LOG_TAG_DIVFTREE_VERTEX, "Block size cannot be 0!");
#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE_VERTEX, "%p constructed -> ID:" VECTORID_LOG_FMT ", Version:%u",
                this, VECTORID_LOG(attributes.centroid_id), attributes.version);
#endif
    }

    DIVFTreeVertex(const DIVFTreeAttributes& attributes, VectorID id, DIVFTreeInterface* index) :
        attr(id, 0,
             id.IsLeaf() ? attributes.leaf_min_size : attributes.internal_min_size,
             id.IsLeaf() ? attributes.leaf_max_size: attributes.internal_max_size,
             id.IsLeaf() ? attributes.leaf_blck_size: attributes.internal_blck_size, index),
             unpinCount{0}, cluster(id.IsLeaf(),
                                    id.IsLeaf() ? attributes.leaf_blck_size: attributes.internal_blck_size,
                                    id.IsLeaf() ? attributes.leaf_max_size: attributes.internal_max_size,
                                    attributes.dimension, true) {
        CHECK_MIN_MAX_SIZE(attr.min_size, attr.cap, LOG_TAG_DIVFTREE_VERTEX);
        CHECK_VECTORID_IS_VALID(attr.centroid_id, LOG_TAG_DIVFTREE_VERTEX);
        CHECK_VECTORID_IS_CENTROID(attr.centroid_id, LOG_TAG_DIVFTREE_VERTEX);
        CHECK_NOT_NULLPTR(attr.index, LOG_TAG_DIVFTREE_VERTEX);
        FatalAssert(attr.block_size > 0, LOG_TAG_DIVFTREE_VERTEX, "Block size cannot be 0!");
        FatalAssert(attr.version == 0, LOG_TAG_DIVFTREE_VERTEX, "The first version should be 0!");
#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE_VERTEX, "%p constructed -> ID:" VECTORID_LOG_FMT ", Version:%u",
                this, VECTORID_LOG(id), 0);
#endif
    }

    ~DIVFTreeVertex() override {
        FatalAssert(unpinCount.load(std::memory_order_relaxed) == 0, LOG_TAG_DIVFTREE_VERTEX,
                    "the vertex is still pinned!");
#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE_VERTEX, "%p destroyed -> ID:" VECTORID_LOG_FMT ", Version:%u",
                this, VECTORID_LOG(attr.centroid_id), attr.version);
#endif
        if (attr.centroid_id.IsLeaf()) {
            return;
        }

        uint16_t size = cluster.header.reserved_size.load(std::memory_order_relaxed);
        FatalAssert(size == cluster.header.visible_size.load(std::memory_order_relaxed), LOG_TAG_DIVFTREE_VERTEX,
                    "mismatch between visited and reserved!");
        const uint16_t dim = attr.index->GetAttributes().dimension;
        CentroidMetaData* vmd =
            reinterpret_cast<CentroidMetaData*>(cluster.MetaData(0, attr.centroid_id.IsLeaf(),
                                                                 attr.block_size, attr.cap, dim));
        FatalAssert(cluster.NumBlocks(attr.block_size, attr.cap) == 1, LOG_TAG_NOT_IMPLEMENTED,
                    "cannot support more than one block!");
        for (uint16_t i = 0; i < size; ++i) {
            VectorState state = vmd[i].state.load(std::memory_order_relaxed);
            /*
             * if state is valid, it means that there was a compaction and we did not change the state nor did we
             * unpined it as it was simply moved to the new memory location
             */
            if (state == VECTOR_STATE_MIGRATED || state == VECTOR_STATE_OUTDATED) {
                FatalAssert(vmd[i].id != INVALID_VECTOR_ID, LOG_TAG_DIVFTREE_VERTEX, "cannot unpin invalid vector");
                BufferManager::GetInstance()->UnpinVertexVersion(vmd[i].id, vmd[i].version);
            }
        }
    }

    inline void Unpin() override {
        uint64_t curr_cnt = unpinCount.fetch_add(1) + 1;
#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE_VERTEX, "%p cluster unpinned -> ID:" VECTORID_LOG_FMT
                ", Version:%u, unpinCnt=%lu", this, VECTORID_LOG(attr.centroid_id), attr.version, curr_cnt);
#endif
        if (curr_cnt == 0) {
            // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE_VERTEX, "Unpin and delete vertex " VECTORID_LOG_FMT
            //         ", ver:%u, addr:%p", VECTORID_LOG(attr.centroid_id), attr.version, this);
            /* todo: we need to handle all the children(unpining their versions and stuff) in the destructor */
            this->~DIVFTreeVertex();
            BufferManager::GetInstance()->Recycle(this);
            return;
        }
        else {
            // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE_VERTEX, "Unpin vertex " VECTORID_LOG_FMT,
            //      VECTORID_LOG(attr.centroid_id));
        }
    }

    inline void MarkForRecycle(uint64_t pinCount) override {
        FatalAssert(pinCount >= unpinCount.load(), LOG_TAG_DIVFTREE_VERTEX,
                    "The cluster was unpinned more times than it was pinned. pinCount=%lu, unpinCount=%lu",
                    pinCount, unpinCount.load());
        uint64_t newPin = unpinCount.fetch_sub(pinCount) - pinCount;
        // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE_VERTEX, "Mark vertex " VECTORID_LOG_FMT
        //     " ver:%u, addr=%p, for recycle with pin count %lu", VECTORID_LOG(attr.centroid_id),
        //     attr.version, this, pinCount);
#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE_VERTEX, "%p cluster marked for recycke -> ID:" VECTORID_LOG_FMT
                ", Version:%u, pinCount=%lu, unpinCnt=%lu", this, VECTORID_LOG(attr.centroid_id), attr.version,
                pinCount, newPin);
#endif
        if (newPin == 0) {
            // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE_VERTEX, "Delete vertex " VECTORID_LOG_FMT ", ver:%u, addr:%p",
            //      VECTORID_LOG(attr.centroid_id), attr.version, this);
            /* todo: we need to handle all the children(unpining their versions and stuff) in the destructor */
            this->~DIVFTreeVertex();
            BufferManager::GetInstance()->Recycle(this);
            return;
        }
    }

    /* todo: make sure the vertex is locked in shared/exclusive mode when calling this function! */
    RetStatus BatchInsert(const ConstVectorBatch& batch, DuplicateState* dup_state, uint16_t num_duplicates_found,
                          uint16_t checked_offset, uint16_t marked_for_update = INVALID_OFFSET) override {
        // CHECK_VERTEX_SELF_IS_VALID(LOG_TAG_DIVFTREE_VERTEX, false);
        FatalAssert(batch.size > 0, LOG_TAG_DIVFTREE_VERTEX,
                    "Batch size must be greater than zero.");
        FatalAssert(batch.data != nullptr, LOG_TAG_DIVFTREE_VERTEX,
                    "Batch data must not be null.");
        FatalAssert(batch.id != nullptr, LOG_TAG_DIVFTREE_VERTEX,
                    "Batch id must not be null.");

        CHECK_NOT_NULLPTR(dup_state, LOG_TAG_DIVFTREE_VERTEX);
        FatalAssert(attr.centroid_id.IsLeaf() || num_duplicates_found == 0, LOG_TAG_DIVFTREE_VERTEX,
                    "For internal vertices, all vectors in the batch must be valid.");

        if (num_duplicates_found == batch.size) {
            BufferManager* bufferMgr = BufferManager::GetInstance();
            FatalAssert(bufferMgr != nullptr, LOG_TAG_DIVFTREE_VERTEX,
                        "BufferManager is not initialized.");
            for (uint16_t i = 0; i < batch.size; ++i) {
                FatalAssert(dup_state[i] == DuplicateState::DUPLICATE_NOT_INSERTED, LOG_TAG_DIVFTREE_VERTEX,
                            "All vectors in the batch must be marked as duplicate not inserted.");
                bufferMgr->UpdateVectorLocation(batch.id[i], INVALID_VECTOR_LOCATION, true);
            }
            return RetStatus::Success();
        }

        uint16_t inserted_size = batch.size - num_duplicates_found;
        uint16_t offset = cluster.header.reserved_size.fetch_add(inserted_size);
        if (offset + inserted_size >= attr.cap) {
            cluster.header.reserved_size.fetch_sub(inserted_size);
            return RetStatus{.stat = RetStatus::VERTEX_NOT_ENOUGH_SPACE, .message=nullptr};
        }

        FatalAssert(offset < cluster.header.reserved_size.load(), LOG_TAG_DIVFTREE_VERTEX, "Overflow detected!");
        BufferManager* bufferMgr = BufferManager::GetInstance();
        FatalAssert(bufferMgr != nullptr, LOG_TAG_DIVFTREE_VERTEX,
                    "BufferManager is not initialized.");
        const uint16_t dim = attr.index->GetAttributes().dimension;
        VectorID marked_id = INVALID_VECTOR_ID;
        CentroidMetaData* markedMeta = nullptr;
        if (marked_for_update != INVALID_OFFSET) {
            FatalAssert(num_duplicates_found == 0, LOG_TAG_DIVFTREE_VERTEX,
                        "Cannot have duplicates when marking a vector for update.");
            FatalAssert(marked_for_update < offset, LOG_TAG_DIVFTREE_VERTEX,
                        "Marked for update offset %hu is not less than the start of newly inserted batch %hu.",
                        marked_for_update, offset);
            FatalAssert(attr.centroid_id.IsInternalVertex(), LOG_TAG_DIVFTREE_VERTEX,
                        "Marked for update offset can only be set for internal vertices.");
            markedMeta = reinterpret_cast<CentroidMetaData*>(
                cluster.MetaData(marked_for_update, false, attr.block_size, attr.cap, dim));
            marked_id = markedMeta->id;
        }
        VTYPE* dest = cluster.Data(offset, attr.centroid_id.IsLeaf(), attr.block_size, attr.cap, dim);
        Address meta = cluster.MetaData(offset, attr.centroid_id.IsLeaf(), attr.block_size, attr.cap, dim);
        if (num_duplicates_found == 0) {
            memcpy(dest, batch.data, batch.size * dim * sizeof(VTYPE));
        }

        uint16_t cluster_idx = 0;
        for (uint16_t i = 0; i < batch.size; ++i) {
            if (attr.centroid_id.IsLeaf()) {
                FatalAssert(dup_state[i] != DuplicateState::DUPLICATE_INSERTED, LOG_TAG_DIVFTREE_VERTEX,
                            "Vector cannot be marked as DUPLICATE_INSERTED in leaf insertion.");
                if (dup_state[i] == DuplicateState::DUPLICATE_NOT_INSERTED) {
                    continue;
                }

                VectorMetaData* vmd = reinterpret_cast<VectorMetaData*>(meta);
                FatalAssert(&vmd[cluster_idx] ==
                            &((VectorMetaData*)(cluster.MetaData(0, true, attr.block_size, attr.cap, dim)))[cluster_idx + offset],
                            LOG_TAG_DIVFTREE_VERTEX, "address mismatch!");
                if (num_duplicates_found != 0) {
                    memcpy(&dest[cluster_idx * dim], &batch.data[i * dim], dim * sizeof(VTYPE));
                }
                vmd[cluster_idx].id = batch.id[i];
                FatalAssert(vmd[cluster_idx].state.load(std::memory_order_relaxed) == VECTOR_STATE_INVALID,
                            LOG_TAG_DIVFTREE_VERTEX, "Reserved space at offset %hu is not in INVALID state.",
                            offset + cluster_idx);
                if (offset == 0) {
                    vmd[cluster_idx].state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
                }
#ifdef EXCESS_LOGING
                DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE_VERTEX,
                    "Inserting vector={id= " VECTORID_LOG_FMT ", data=%s} into %s: stored vector data = %s",
                    VECTORID_LOG(batch.id[i]), VectorToString(&batch.data[i * dim], dim).ToCStr(),
                    VectorLocation(attr.centroid_id, attr.version, offset + cluster_idx).ToString().ToCStr(),
                    VectorToString(&dest[cluster_idx * dim], dim).ToCStr());
#endif
                ++cluster_idx;
            } else {
                FatalAssert(batch.version != nullptr, LOG_TAG_DIVFTREE_VERTEX,
                            "Batch version must not be null for centroid insertion.");
                FatalAssert(dup_state[i] == DuplicateState::UNIQUE, LOG_TAG_DIVFTREE_VERTEX,
                            "All vectors in the batch must be unique for centroid insertion.");
                FatalAssert(cluster_idx == i, LOG_TAG_DIVFTREE_VERTEX,
                            "Cluster index mismatch for centroid insertion.");
                CentroidMetaData* vmd = reinterpret_cast<CentroidMetaData*>(meta);
                FatalAssert(&vmd[i] ==
                            &((CentroidMetaData*)(cluster.MetaData(0, false, attr.block_size, attr.cap, dim)))[i + offset],
                            LOG_TAG_DIVFTREE_VERTEX, "address mismatch!");
                vmd[i].id = batch.id[i];
                /* todo: should we pin version for the marked vertex?! yeah because its version is differnet */
                bufferMgr->PinVertexVersion(batch.id[i], batch.version[i]);
                vmd[i].version = batch.version[i];
                FatalAssert(vmd[i].state.load(std::memory_order_relaxed) == VECTOR_STATE_INVALID,
                            LOG_TAG_DIVFTREE_VERTEX, "Reserved space at offset %hu is not in INVALID state.",
                            offset + i);
                vmd[i].state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
#ifdef EXCESS_LOGING
                DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE_VERTEX,
                    "Inserting vector={id= " VECTORID_LOG_FMT ", version=%u, data=%s} into %s: stored vector data = %s",
                    VECTORID_LOG(batch.id[i]), batch.version[i], VectorToString(&batch.data[i * dim], dim).ToCStr(),
                    VectorLocation(attr.centroid_id, attr.version, offset + i).ToString().ToCStr(),
                    VectorToString(&dest[i * dim], dim).ToCStr());
#endif
                ++cluster_idx;
            }
        }

        FatalAssert(cluster_idx == inserted_size, LOG_TAG_DIVFTREE_VERTEX,
                    "Cluster index does not match the number of inserted vectors.");

        uint16_t num_dup_inserted = 0;
        if (attr.centroid_id.IsLeaf() && offset != 0) {
            cluster_idx = 0;
            Exists(batch, offset, dup_state, num_duplicates_found, DuplicateState::DUPLICATE_INSERTED, checked_offset);
            for (uint16_t i = 0; i < batch.size; ++i) {
                if (dup_state[i] == DuplicateState::UNIQUE) {
                    reinterpret_cast<VectorMetaData*>(meta)[cluster_idx].state.store(VECTOR_STATE_VALID,
                                                                                     std::memory_order_relaxed);
                } else if (dup_state[i] == DuplicateState::DUPLICATE_INSERTED) {
                    FatalAssert(batch.id[i] ==
                                reinterpret_cast<VectorMetaData*>(meta)[cluster_idx].id,
                                LOG_TAG_DIVFTREE_VERTEX,
                                "Vector already exists in the cluster but IDs do not match.");
                    ++num_dup_inserted;
                    bufferMgr->UpdateVectorLocation(batch.id[i], INVALID_VECTOR_LOCATION, true);
                    FatalAssert(reinterpret_cast<VectorMetaData*>(meta)[cluster_idx].
                                    state.load(std::memory_order_relaxed) ==
                                    VECTOR_STATE_INVALID,
                                LOG_TAG_DIVFTREE_VERTEX,
                                "Vector already exists in the cluster but its state is VALID.");
                } else {
                    bufferMgr->UpdateVectorLocation(batch.id[i], INVALID_VECTOR_LOCATION, true);
                    continue;
                }
                ++cluster_idx;
            }
            FatalAssert(cluster_idx == inserted_size, LOG_TAG_DIVFTREE_VERTEX,
                        "Cluster index does not match the number of inserted vectors.");
        }

        /* we still have to do a while loop here even for the leaf nodes because we may see all vectors before
           everything before us is visible */
        /* todo: do it in a lock-free way! */
        while (cluster.header.visible_size.load(std::memory_order_acquire) < offset) {
            DIVFTREE_YIELD();
        }
        cluster.header.visible_size.store(offset + inserted_size, std::memory_order_release);
        cluster.header.num_deleted.fetch_add(num_dup_inserted);
        cluster_idx = 0;
        for (uint16_t i = 0; i < batch.size; ++i) {
            // VectorLocation oldLoc = bufferMgr->LoadCurrentVectorLocation(batch.id[i], false);
            if (dup_state[i] == DuplicateState::UNIQUE) {
                // DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_DIVFTREE_VERTEX, "UpdateVectorLocation in batch insert: %hu: "
                //         VECTORID_LOG_FMT ", %u, %hu -> " VECTORID_LOG_FMT ", %u, %hu", i,
                //         VECTORID_LOG(oldLoc.detail.containerId), Version::AsRawVersion(oldLoc.detail.containerVersion), oldLoc.detail.entryOffset,
                //         VECTORID_LOG(attr.centroid_id), Version::AsRawVersion(attr.version), offset + cluster_idx);
                bufferMgr->
                    UpdateVectorLocation(batch.id[i],
                                         VectorLocation(attr.centroid_id, attr.version, offset + cluster_idx),
                                         batch.id[i] != marked_id);
            } else {
                // DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_DIVFTREE_VERTEX, "UpdateVectorLocation in batch insert: %hu: "
                //         VECTORID_LOG_FMT ", %u, %hu -> INV", i,
                //         VECTORID_LOG(oldLoc.detail.containerId), Version::AsRawVersion(oldLoc.detail.containerVersion), oldLoc.detail.entryOffset);
                FatalAssert(bufferMgr->LoadCurrentVectorLocation(batch.id[i], false) == INVALID_VECTOR_LOCATION,
                            LOG_TAG_DIVFTREE_VERTEX,
                            "Vector location must be invalid for duplicate vectors when checked_offset is set.");
            }
            // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE_VERTEX, "UpdateVectorLocation in batch insert:");
            if (dup_state[i] != DuplicateState::DUPLICATE_NOT_INSERTED) {
                ++cluster_idx;
            }
        }
        FatalAssert(cluster_idx == inserted_size, LOG_TAG_DIVFTREE_VERTEX,
                    "Cluster index does not match the number of inserted vectors.");

        if (markedMeta != nullptr) {
            FatalAssert(inserted_size == batch.size, LOG_TAG_DIVFTREE_VERTEX,
                        "When marking a vector for update, there cannot be any duplicates in the batch.");
            markedMeta->state.store(VECTOR_STATE_OUTDATED);
            cluster.header.num_deleted.fetch_add(1);
            /* todo: do I need to check cluster size and prune the cluster here if needed? */
            /*
             * we cannot unpin the outdated version here because if an ANN search has read this version but
             * has not pinned it, and the time it was not outdated,
             * the version will be lost.
             */
            // bufferMgr->UnpinVertexVersion(markedMeta->id, markedMeta->version);
        }

        return RetStatus::Success();
    }

    /*
     * if id.level is vertex/cluster and targetState is Invalid then the cluster should be locked in exclusive mode
     * otherwise if id.level is vertex/cluster then target should be locked in shared mode
     */
    inline RetStatus ChangeVectorState(VectorID target, uint16_t targetOffset,
                                       VectorState& expectedState, VectorState finalState,
                                       Version* version = nullptr) override {
        CHECK_VECTORID_IS_VALID(target, LOG_TAG_DIVFTREE_VERTEX);
        FatalAssert(target._level + 1 == attr.centroid_id._level, LOG_TAG_DIVFTREE_VERTEX, "level mismatch");
        FatalAssert(expectedState == VECTOR_STATE_VALID || expectedState == VECTOR_STATE_MIGRATED,
                    LOG_TAG_DIVFTREE_VERTEX, "expected state can be either VALID or MIGRATED");
        FatalAssert(finalState == VECTOR_STATE_VALID || finalState == VECTOR_STATE_MIGRATED ||
                    finalState == VECTOR_STATE_INVALID, LOG_TAG_DIVFTREE_VERTEX,
                    "final state can only be valid, invalid or migrated!");
        FatalAssert((expectedState == VECTOR_STATE_MIGRATED) == (finalState == VECTOR_STATE_VALID),
                    LOG_TAG_DIVFTREE_VERTEX, "excpected state is migrated is migrated if and only "
                    "if final state is valid. this can only happen when a migration fails and "
                    "we want to revert our state back to valid");
        SANITY_CHECK(
            uint16_t size = cluster.header.visible_size.load(std::memory_order_acquire);
            FatalAssert(targetOffset < size, LOG_TAG_DIVFTREE_VERTEX, "offset out of range!");
        );

        Address meta = cluster.MetaData(targetOffset, attr.centroid_id.IsLeaf(),
                                        attr.block_size, attr.cap, attr.index->GetAttributes().dimension);
        VectorID* id = nullptr;
        std::atomic<VectorState>* state = nullptr;
        if (attr.centroid_id.IsLeaf()) {
            VectorMetaData* vmd = static_cast<VectorMetaData*>(meta);
            id = &(vmd->id);
            state = &(vmd->state);
        } else {
            CentroidMetaData* vmd = static_cast<CentroidMetaData*>(meta);
            id = &(vmd->id);
            if (version != nullptr) {
                *version = vmd->version;
            }
            state = &(vmd->state);
        }

        FatalAssert(*id == target, LOG_TAG_DIVFTREE_VERTEX, "id mismatch!");
        if (expectedState == VECTOR_STATE_MIGRATED) {
            FatalAssert((*state).load(std::memory_order_acquire) == VECTOR_STATE_MIGRATED, LOG_TAG_DIVFTREE_VERTEX,
                        "mismatch state!");
            state->store(finalState, std::memory_order_release);
            return RetStatus::Success();
        }

        if (target.IsCentroid()) {
            if (finalState == VECTOR_STATE_INVALID) {
                /* deletion */
                /* todo: check target should be locked in exclusive mode! */
                FatalAssert((*state).load(std::memory_order_acquire) == VECTOR_STATE_VALID, LOG_TAG_DIVFTREE_VERTEX,
                            "mismatch state!");
                state->store(finalState, std::memory_order_release);
                return RetStatus::Success();
            } else {
                /* todo: check target should be locked in shared mode! */
                FatalAssert((*state).load(std::memory_order_acquire) != VECTOR_STATE_INVALID, LOG_TAG_DIVFTREE_VERTEX,
                            "mismatch state!");
            }
        }

        return (state->compare_exchange_strong(expectedState, finalState) ?
                RetStatus::Success() :
                RetStatus{.stat=RetStatus::FAILED_TO_CAS_VECTOR_STATE, .message=nullptr});
    }

    /*
     * if id.level is vertex/cluster and targetState is Invalid then the cluster should be locked in exclusive mode
     * otherwise if id.level is vertex/cluster then target should be locked in shared mode
     */
    inline RetStatus ChangeVectorState(Address meta, VectorState& expectedState, VectorState finalState,
                                       Version* version = nullptr) override {
        FatalAssert(expectedState == VECTOR_STATE_VALID || expectedState == VECTOR_STATE_MIGRATED,
                    LOG_TAG_DIVFTREE_VERTEX, "expected state can be either VALID or MIGRATED");
        FatalAssert(finalState == VECTOR_STATE_VALID || finalState == VECTOR_STATE_MIGRATED ||
                    finalState == VECTOR_STATE_INVALID, LOG_TAG_DIVFTREE_VERTEX,
                    "final state can only be valid, invalid or migrated!");
        FatalAssert((expectedState == VECTOR_STATE_MIGRATED) == (finalState == VECTOR_STATE_VALID),
                    LOG_TAG_DIVFTREE_VERTEX, "excpected state is migrated is migrated if and only "
                    "if final state is valid. this can only happen when a migration fails and "
                    "we want to revert our state back to valid");

        // VectorID* id = nullptr;
        std::atomic<VectorState>* state = nullptr;
        if (attr.centroid_id.IsLeaf()) {
            VectorMetaData* vmd = static_cast<VectorMetaData*>(meta);
            // id = &(vmd->id);
            state = &(vmd->state);
        } else {
            CentroidMetaData* vmd = static_cast<CentroidMetaData*>(meta);
            // id = &(vmd->id);
            if (version != nullptr) {
                *version = vmd->version;
            }
            state = &(vmd->state);
        }

        if (expectedState == VECTOR_STATE_MIGRATED) {
            FatalAssert((*state).load(std::memory_order_acquire) == VECTOR_STATE_MIGRATED, LOG_TAG_DIVFTREE_VERTEX,
                        "mismatch state!");
            state->store(finalState, std::memory_order_release);
            return RetStatus::Success();
        }

        if (attr.centroid_id.IsInternalVertex()) {
            if (finalState == VECTOR_STATE_INVALID) {
                /* deletion */
                /* todo: check target should be locked in exclusive mode! */
                FatalAssert((*state).load(std::memory_order_acquire) == VECTOR_STATE_VALID, LOG_TAG_DIVFTREE_VERTEX,
                            "mismatch state!");
                state->store(finalState, std::memory_order_release);
                return RetStatus::Success();
            } else {
                /* todo: check target should be locked in shared mode! */
                FatalAssert((*state).load(std::memory_order_acquire) != VECTOR_STATE_INVALID, LOG_TAG_DIVFTREE_VERTEX,
                            "mismatch state!");
            }
        }

        return (state->compare_exchange_strong(expectedState, finalState) ?
                RetStatus::Success() :
                RetStatus{.stat=RetStatus::FAILED_TO_CAS_VECTOR_STATE, .message=nullptr});
    }

    void Search(const VTYPE* query, size_t k, SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
                std::unordered_set<std::pair<VectorID, Version>, VectorIDVersionPairHash>& seen) override {
        /* todo: should I also ignore same vectors with different versions? e.g. if it is newer replace o.w ignore */
        CHECK_NOT_NULLPTR(query, LOG_TAG_DIVFTREE_VERTEX);
        CHECK_NOT_NULLPTR(neighbours, LOG_TAG_DIVFTREE_VERTEX);
        FatalAssert(k > 0, LOG_TAG_DIVFTREE_VERTEX, "k must be greater than 0!");
        FatalAssert(neighbours->Size() <= k, LOG_TAG_DIVFTREE_VERTEX, "neighbour list size out of bounds!");
        const uint16_t dim = attr.index->GetAttributes().dimension;
        const DistanceType dtype = attr.index->GetAttributes().distanceAlg;
        VTYPE* data = cluster.Data(0, attr.centroid_id.IsLeaf(), attr.block_size, attr.cap, dim);
        Address meta = cluster.MetaData(0, attr.centroid_id.IsLeaf(), attr.block_size, attr.cap, dim);
        uint16_t old_size = 0;
        uint16_t curr_size = cluster.header.visible_size.load(std::memory_order_acquire);
        // std::unordered_map<VectorID, std::pair<uint16_t, DTYPE>, VectorIDHash> in_list;
        std::unordered_set<VectorID, VectorIDHash> in_cluster;
        in_cluster.reserve(curr_size);
        // in_list.reserve(curr_size);
        // while(curr_size != old_size) {
            /* this will overflow and i will get greater than size when it gets to 0 */
            uint16_t last_seen = old_size - 1;
            for (uint16_t i = curr_size - 1; i != last_seen; --i) {
                ANNVectorInfo new_vector;
                if (attr.centroid_id.IsLeaf()) {
                    VectorMetaData* vmd = reinterpret_cast<VectorMetaData*>(meta);
                    if (seen.find(std::make_pair(vmd[i].id, 0)) != seen.end()) {
                        continue;
                    }

                    switch (vmd[i].state.load(std::memory_order_acquire)) {
                    case VECTOR_STATE_VALID:
                    case VECTOR_STATE_MIGRATED:
                        new_vector = ANNVectorInfo(Distance(query, &data[i * dim], dim, dtype), vmd[i].id);
                        neighbours->Insert(new_vector);
                        seen.emplace(new_vector.id, new_vector.version);
                        if (neighbours->Size() > k) {
                            neighbours->PopBack();
                        }
                        break;
                    case VECTOR_STATE_OUTDATED:
                        FatalAssert(false, LOG_TAG_DIVFTREE_VERTEX, "a pure vector cannot become outdated!");
                    default:
                        continue;
                    }
                } else {
                    CentroidMetaData* vmd = reinterpret_cast<CentroidMetaData*>(meta);
                    VectorState state = vmd[i].state.load(std::memory_order_acquire);
                    if ((state == VECTOR_STATE_VALID) || (state == VECTOR_STATE_MIGRATED) ||
                        (state == VECTOR_STATE_INVALID)) {
                        in_cluster.emplace(vmd[i].id);
                    }

                    if (seen.find(std::make_pair(vmd[i].id, vmd[i].version)) != seen.end()) {
                        if ((state == VECTOR_STATE_OUTDATED) &&
                            (in_cluster.find(vmd[i].id) == in_cluster.end())) {
                            in_cluster.emplace(vmd[i].id);
                        }
                        continue;
                    }

                    std::unordered_map<VectorID, std::pair<uint16_t, DTYPE>, VectorIDHash>::iterator emplace_res;
                    std::unordered_map<VectorID, std::pair<uint16_t, DTYPE>, VectorIDHash>::iterator check;
                    switch (state) {
                    case VECTOR_STATE_OUTDATED:
                        if (in_cluster.find(vmd[i].id) != in_cluster.end()) {
                            break;
                        }
                        in_cluster.emplace(vmd[i].id);
                    case VECTOR_STATE_VALID:
                    case VECTOR_STATE_MIGRATED:
                        // check = in_list.find(vmd[i].id);
                        // if (check != in_list.end()) { /* todo: we can handle this case much more efficiently! */
                        //     /* a vector was outdated in a previous pass */
                        //     SANITY_CHECK(
                        //         VectorState oldState = vmd[check->second.first].state.load(std::memory_order_acquire);
                        //         FatalAssert((vmd[check->second.first].id == vmd[i].id) &&
                        //                     ((oldState == VECTOR_STATE_OUTDATED) ||
                        //                      (oldState == VECTOR_STATE_MIGRATED)) &&
                        //                     (vmd[check->second.first].version < vmd[i].version),
                        //                     LOG_TAG_DIVFTREE_VERTEX,
                        //                     "the one that we have seen before should be outdated/migrated!");
                        //     );
                        //     ANNVectorInfo outdated(check->second.second, vmd[check->second.first].id,
                        //                            vmd[check->second.first].version);
                        //     auto it = neighbours->Find(outdated);
                        //     FatalAssert((it != neighbours->end()),
                        //                 LOG_TAG_DIVFTREE_VERTEX, "Could not find the outdated version!");
                        //     neighbours->Erase(it);
                        //     check->second.first = i;
                        //     check->second.second = Distance(query, &data[i * dim], dim, dtype);
                        //     emplace_res = check;
                        // } else {
                        //     emplace_res = in_list.insert({new_vector.id,
                        //                                  {i, Distance(query, &data[i * dim], dim, dtype)}}).first;
                        // }
                        new_vector = ANNVectorInfo(Distance(query, &data[i * dim], dim, dtype),
                                                   vmd[i].id, vmd[i].version);
                        neighbours->Insert(new_vector);
                        seen.emplace(new_vector.id, new_vector.version);
                        if (neighbours->Size() > k) {
                            // auto to_be_deleted = std::prev(neighbours->end());
                            // if (new_vector == *to_be_deleted) {
                            //     in_list.erase(emplace_res);
                            // } else {
                            //     auto it = in_list.find(new_vector.id);
                            //     if (it != in_list.end()) {
                            //         in_list.erase(it);
                            //     }
                            // }
                            // neighbours->Erase(to_be_deleted);
                            neighbours->PopBack();
                        }
                        break;
                    // case VECTOR_STATE_OUTDATED:
                        // if (in_cluster.find(vmd[i].id) == in_cluster.end()) {
                        //     old_size = curr_size;
                        // }
                        // break;
                    case VECTOR_STATE_INVALID:
                        break;
                    default:
                        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_DIVFTREE_VERTEX, "invalid state");
                    }
                }
            }

            /* todo: find out the reason why this assertion fails! although I am going to change the algorithm,
            this can cause a lot of issues later if it is something serious! */
            // if (old_size == curr_size) {
            //     curr_size = cluster.header.visible_size.load(std::memory_order_acquire);
            //     // while (curr_size == old_size) {
            //     //     DIVFTREE_YIELD();
            //     //     curr_size = cluster.header.visible_size.load(std::memory_order_acquire);
            //     // }
            //     FatalAssert(curr_size > old_size, LOG_TAG_DIVFTREE_VERTEX,
            //                 "if we have seen an outdated vector our size should have increased!");
            //     in_cluster.clear();
            // } else {
            //     old_size = curr_size;
            // }
        // }
    }

    void Exists(const ConstVectorBatch batch, uint16_t expected_size, DuplicateState* dup_state, uint16_t& num_found,
                DuplicateState set_if_found, uint16_t checked_for_duplicates = UINT16_MAX) override {
        /* todo: use a bloom filter or something */
        CHECK_NOT_NULLPTR(dup_state, LOG_TAG_DIVFTREE_VERTEX);
        CHECK_NOT_NULLPTR(batch.data, LOG_TAG_DIVFTREE_VERTEX);
        FatalAssert(batch.size > 0, LOG_TAG_DIVFTREE_VERTEX, "batch size must be greater than 0");
        const uint16_t dim = attr.index->GetAttributes().dimension;
        FatalAssert(attr.centroid_id.IsLeaf(), LOG_TAG_DIVFTREE_VERTEX,
                    "Exists can only be called on leaf vertices!");
        FatalAssert(set_if_found != DuplicateState::UNIQUE, LOG_TAG_DIVFTREE_VERTEX,
                    "set_if_found cannot be UNIQUE!");
        VTYPE* data = cluster.Data(0, attr.centroid_id.IsLeaf(), attr.block_size, attr.cap, dim);
        VectorMetaData* vmd =
            reinterpret_cast<VectorMetaData*>(
                cluster.MetaData(0, attr.centroid_id.IsLeaf(), attr.block_size, attr.cap, dim));
        uint16_t curr_size = cluster.header.visible_size.load(std::memory_order_acquire);
        FatalAssert(curr_size <= expected_size, LOG_TAG_DIVFTREE_VERTEX,
                    "curr_size cannot be greater than expected_size!");
        FatalAssert(expected_size <= cluster.header.reserved_size.load(std::memory_order_acquire),
                    LOG_TAG_DIVFTREE_VERTEX, "expected_size cannot be greater than the reserved size!");
        uint16_t to_see = checked_for_duplicates + 1;
        while (to_see < expected_size) {
            FatalAssert(curr_size <= expected_size, LOG_TAG_DIVFTREE_VERTEX,
                        "curr_size cannot be greater than expected_size!");
            /* this will overflow and i will get greater than size when it gets to 0 */
            for (; to_see < expected_size && to_see < curr_size; ++to_see) {
                VectorState state = vmd[to_see].state.load(std::memory_order_acquire);
                switch (state) {
                case VECTOR_STATE_VALID:
                case VECTOR_STATE_MIGRATED:
                    for (uint16_t b = 0; b < batch.size; ++b) {
                        if (batch.id[b] == vmd[to_see].id) {
                            FatalAssert(state == VECTOR_STATE_MIGRATED, LOG_TAG_DIVFTREE_VERTEX,
                                        "If IDs match, state must be MIGRATED!");
                            continue;
                        }

                        if (dup_state[b] != DuplicateState::UNIQUE) {
                            continue;
                        }
                        /* todo: maybe use distance and say if distance is less than some number */
                        if (memcmp(&batch.data[b * dim], &data[to_see * dim], dim * sizeof(VTYPE)) == 0) {
                            dup_state[b] = set_if_found;
                            num_found++;
                            if (num_found == batch.size) {
                                return;
                            }
                        }
                    }
                    break;
                case VECTOR_STATE_OUTDATED:
                    FatalAssert(false, LOG_TAG_DIVFTREE_VERTEX, "a pure vector cannot become outdated!");
                default:
                    continue;
                }
            }

            if (to_see < expected_size) {
                DIVFTREE_YIELD();
                curr_size = cluster.header.visible_size.load(std::memory_order_acquire);
            }
        }
        FatalAssert(curr_size == expected_size, LOG_TAG_DIVFTREE_VERTEX, "curr_size mismatch!");
        FatalAssert(to_see == expected_size, LOG_TAG_DIVFTREE_VERTEX, "to_see mismatch!");
    }

    inline Cluster& GetCluster() override {
        return cluster;
    }

    inline const DIVFTreeVertexAttributes& GetAttributes() const override {
        return attr;
    }

    inline VectorID CentroidID() const override {
        return attr.centroid_id;
    }

    inline Version VertexVersion() const override {
        return attr.version;
    }

    String ToString(bool detailed = false) const override {
        uint16_t reserved = cluster.header.reserved_size.load(std::memory_order_acquire);
        uint16_t visible = cluster.header.visible_size.load(std::memory_order_acquire);
        uint16_t deleted = cluster.header.num_deleted.load(std::memory_order_acquire);
        String rs = String("{self=%p, unpinCount:%lu, reserved_size:%hu, visible_size:%hu, num_deleted:%hu, attr:%s",
                           this, unpinCount.load(std::memory_order_acquire),
                           reserved, visible, deleted, attr.ToString().ToCStr());
        if (!detailed) {
            return rs+String("}");
        }

        FatalAssert(Cluster::NumBlocks(attr.block_size, attr.cap) == 1,
                    LOG_TAG_NOT_IMPLEMENTED, "Currently cannot handle more than one block!");
        rs += String(", Cluster:{num_blocks=1, blocks=[{");

        uint16_t dim = attr.index->GetAttributes().dimension;
        const VTYPE* vec = cluster.Data(0, attr.centroid_id.IsLeaf(), attr.block_size, attr.cap, dim);
        const void* meta = cluster.MetaData(0, attr.centroid_id.IsLeaf(), attr.block_size, attr.cap, dim);
        for (uint16_t i = 0; i < visible; ++i) {
            if (attr.centroid_id.IsLeaf()) {
                const VectorMetaData* vmd = static_cast<const VectorMetaData*>(meta);
                rs += String("{meta:{ID: " VECTORID_LOG_FMT ", State: %s}, data:{", VECTORID_LOG(vmd[i].id),
                             VectorStateToString(vmd[i].state.load(std::memory_order_acquire)).ToCStr());
                for (uint16_t j = 0; j < dim; ++j) {
                    rs += String(VTYPE_FMT, vec[i * dim + j]);
                    if (j < dim - 1) {
                        rs += String(", ");
                    }
                }
            } else {
                const CentroidMetaData* vmd = static_cast<const CentroidMetaData*>(meta);
                rs += String("{meta:{ID: " VECTORID_LOG_FMT " Version: %lu, State: %s}, data:{",
                             VECTORID_LOG(vmd[i].id), vmd[i].version,
                             VectorStateToString(vmd[i].state.load(std::memory_order_acquire)).ToCStr());
                for (uint16_t j = 0; j < dim; ++j) {
                    rs += String(VTYPE_FMT, vec[i * dim + j]);
                    if (j < dim - 1) {
                        rs += String(", ");
                    }
                }
            }
            rs += String((i == visible - 1 ? "}}" : "}}, "));
        }
        rs += String("}]}}");

        return rs;
    }


protected:
    const DIVFTreeVertexAttributes attr;
    std::atomic<uint64_t> unpinCount;
    alignas(CACHE_LINE_SIZE) Cluster cluster;

TESTABLE;
friend class DIVFTree;
};

class DIVFTree : public DIVFTreeInterface {
public:
    DIVFTree(DIVFTreeAttributes attributes) : attr(attributes), real_size(0), end_signal(false) {
        Thread* _self = new Thread(attributes.random_base_perc);
        _self->InitDIVFThread(attr.dimension);
        attr.similarityComparator = GetDistancePairSimilarityComparator(attr.distanceAlg, false);
        attr.reverseSimilarityComparator = GetDistancePairSimilarityComparator(attr.distanceAlg, true);
        FatalAssert(attr.internal_max_size > 1 && attr.internal_max_size < MAX_CLUSTER_SIZE,
                    LOG_TAG_DIVFTREE, "Internal max size out of bounds!");
        FatalAssert(attr.leaf_max_size > 1 && attr.leaf_max_size < MAX_CLUSTER_SIZE,
                    LOG_TAG_DIVFTREE, "Leaf max size out of bounds!");
        FatalAssert(attr.dimension > 0 && attr.dimension <= MAX_VECTOR_DIMENSION,
                    LOG_TAG_DIVFTREE, "Dimension out of bounds!");
        FatalAssert(attr.internal_min_size < attr.internal_max_size,
                    LOG_TAG_DIVFTREE, "Internal min size must be less than max size!");
        FatalAssert(attr.leaf_min_size < attr.leaf_max_size,
                    LOG_TAG_DIVFTREE, "Leaf min size must be less than max size!");
        if (attr.use_block_bytes) {
            attr.leaf_blck_size = Cluster::BlockSize(attr.leaf_blck_bytes, sizeof(VectorMetaData),
                                                     attr.leaf_max_size, attr.dimension);
            attr.internal_blck_size = Cluster::BlockSize(attr.internal_blck_bytes, sizeof(CentroidMetaData),
                                                         attr.internal_max_size, attr.dimension);
        } else {
            attr.leaf_blck_bytes = Cluster::BlockBytes(attr.leaf_blck_size, sizeof(VectorMetaData),
                                                       attr.dimension);
            attr.internal_blck_bytes = Cluster::BlockBytes(attr.internal_blck_size, sizeof(CentroidMetaData),
                                                           attr.dimension);
        }
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_DIVFTREE, "Create DIVFTree Index Start");
#ifdef ENABLE_STAT_COLLECTION
        if (attr.collect_stats) {
            StartStatsCollection();
        } else {
            StopStatsCollection();
        }

        bg_migration_iterations = new std::atomic<uint64_t>[attr.num_migrators];
        bg_migration_tasks_completed = new std::atomic<uint64_t>[attr.num_migrators];
        bg_migration_num_migrated_vectors = new std::atomic<uint64_t>[attr.num_migrators];

        bg_merge_iterations = new std::atomic<uint64_t>[attr.num_mergers];
        bg_merge_tasks_completed = new std::atomic<uint64_t>[attr.num_mergers];
        bg_merge_num_merged_clusters = new std::atomic<uint64_t>[attr.num_mergers];

        bg_compaction_iterations = new std::atomic<uint64_t>[attr.num_compactors];
        bg_compaction_tasks_completed = new std::atomic<uint64_t>[attr.num_compactors];
        bg_compaction_num_compacted_clusters = new std::atomic<uint64_t>[attr.num_compactors];

        total_bg_search_tasks = 0;
        bg_search_iterations = new std::atomic<uint64_t>[attr.num_searchers];
        bg_search_tasks_completed = new std::atomic<uint64_t>[attr.num_searchers];

        ClearStats(true);
#endif
        VectorID root_id;
        BufferVertexEntry* root_entry =
            BufferManager::Init(sizeof(DIVFTreeVertex) - ALIGNED_SIZE(sizeof(ClusterHeader)),
                                attr.leaf_blck_size, attr.internal_blck_size,
                                attr.leaf_max_size, attr.internal_max_size, attr.dimension, attr.memory_pool_size_gb);
        CHECK_NOT_NULLPTR(root_entry, LOG_TAG_DIVFTREE);
        (void)(new (root_entry->ReadLatestVersion(false)) DIVFTreeVertex(attr, root_entry->centroidMeta.selfId, this));
        root_entry->headerLock.Unlock();
        BufferManager::GetInstance()->
            ReleaseBufferEntryIfNotNull(root_entry, ReleaseBufferEntryFlags(true, true));

        StartBGThreads();
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_DIVFTREE, "Create DIVFTree Index End");
    }

    ~DIVFTree() override {
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_DIVFTREE, "Shutdown DIVFTree Index Start");

        StopStatsCollection();

        end_signal.store(true, std::memory_order_release);

        DestroyBGThreads();

#ifdef ENABLE_STAT_COLLECTION
        delete[] bg_migration_iterations;
        delete[] bg_migration_tasks_completed;
        delete[] bg_migration_num_migrated_vectors;
        delete[] bg_merge_iterations;
        delete[] bg_merge_tasks_completed;
        delete[] bg_merge_num_merged_clusters;
        delete[] bg_compaction_iterations;
        delete[] bg_compaction_tasks_completed;
        delete[] bg_compaction_num_compacted_clusters;
        delete[] bg_search_iterations;
        delete[] bg_search_tasks_completed;
#endif

        BufferManager::GetInstance()->Shutdown();
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_DIVFTREE, "Shutdown DIVFTree Index End");

        FatalAssert(threadSelf != nullptr, LOG_TAG_DIVFTREE, "threadself not inited!");
        Thread* _self = threadSelf;
        threadSelf->DestroyDIVFThread();
        delete _self;
    }

    /* todo: for now since it is single node, the create_completion_notification here does not do anything and returning
       from this function indicates that the vector is inserted. But we need to change
       this in the multi node environment */
    RetStatus Insert(const VTYPE* vec, VectorID& vec_id, uint8_t search_span,
                     bool create_completion_notification = false) override {
        CHECK_NOT_NULLPTR(vec, LOG_TAG_DIVFTREE);
        UNUSED_VARIABLE(create_completion_notification);
        FatalAssert(search_span > 0, LOG_TAG_DIVFTREE, "Number of neighbours cannot be 0.");
        RetStatus rs = RetStatus::Success();
        FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                    "should not hold any lock here!");

#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
             "Insert BEGIN: Vector=%s",
             VectorToString(vec, attr.dimension).ToCStr());
#endif

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

        BufferVectorEntry* vector_entry = nullptr;
        bufferMgr->BatchCreateVectorEntry(1, &vector_entry, false);
        CHECK_NOT_NULLPTR(vector_entry, LOG_TAG_DIVFTREE);
        vec_id = vector_entry->selfId;
        CHECK_VECTORID_IS_VALID(vec_id, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_VECTOR(vec_id, LOG_TAG_DIVFTREE);
        const ConstVectorBatch batch(vec, &vec_id, nullptr, 1);

        std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*> layers;
        VectorID target_leaf;
        Version target_version;
        BufferVertexEntry* leaf_entry = nullptr;

        do {
            while (true) {
                DIVFTreeVertex* root =
                    static_cast<DIVFTreeVertex*>(bufferMgr->ReadAndPinRoot());
                CHECK_NOT_NULLPTR(root, LOG_TAG_DIVFTREE);

                if (root->attr.centroid_id.IsLeaf()) {
                    target_leaf = root->attr.centroid_id;
                    target_version = root->attr.version;
                    root->Unpin();
                    break;
                }

                layers.reserve(root->attr.centroid_id._level + 1);
                /* level 0 represents vectors and we do not need them */
                layers.emplace_back(nullptr);
                for (uint64_t i = 1; i <= root->attr.centroid_id._level; ++i) {
                    layers.emplace_back(
                        new SortedList<ANNVectorInfo, SimilarityComparator>(attr.similarityComparator));
                }

                layers[root->attr.centroid_id._level]->Insert(ANNVectorInfo(0, root->attr.centroid_id,
                                                                            root->attr.version));
                rs = ANNSearch(vec, 1, search_span, 1,
                               (uint8_t)(root->attr.centroid_id._level), (uint8_t)VectorID::LEAF_LEVEL,
                               layers, root);
                root->Unpin();

                if (rs.IsOK()) {
                    FatalAssert(layers[VectorID::LEAF_LEVEL]->Size() == 1, LOG_TAG_DIVFTREE,
                                "if rs is OK we should have found the leaf!");
                    target_leaf = (*layers[VectorID::LEAF_LEVEL])[0].id;
                    target_version = (*layers[VectorID::LEAF_LEVEL])[0].version;
                }

                for (uint64_t i = 1; i < layers.size(); ++i) {
                    delete layers[i];
                }
                layers.clear();

                if (rs.IsOK()) {
                    break;
                }
                /* todo: check how many retries! */
            }

            leaf_entry = bufferMgr->ReadBufferEntry(target_leaf, SX_SHARED);
            if (leaf_entry == nullptr || leaf_entry->currentVersion._split != target_version._split) {
                /* todo: if there is a difference in version we may be able to solve it using update logs! */
                /* todo: check how many retries! */
                bufferMgr->ReleaseBufferEntryIfNotNull(leaf_entry, ReleaseBufferEntryFlags(false, false));
                rs = RetStatus::Fail(nullptr);
                continue;
            }
            /* todo: maybe we can read the parent and only go one layer up instead of rereading from root */
            rs = BatchInsertInto(leaf_entry, batch, true);
        } while(!rs.IsOK());

        real_size.fetch_add(1);

#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
             "Insert END: Vector=%s, vector_id=" VECTORID_LOG_FMT,
             VectorToString(vec, attr.dimension).ToCStr(), VECTORID_LOG(vec_id));
#endif
        FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                    "should not hold any lock here!");
        return rs;
    }

    /* todo: for now since it is single node, the create_completion_notification here does not do anything and returning
       from this function indicates that the vector is deleted. But we need to change
       this in the multi node environment */
    RetStatus Delete(VectorID vec_id, bool create_completion_notification = false) override {
        UNUSED_VARIABLE(create_completion_notification);
        CHECK_VECTORID_IS_VALID(vec_id, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_VECTOR(vec_id, LOG_TAG_DIVFTREE);
        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);
#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
             "DELETE START: vector_id=" VECTORID_LOG_FMT,
             VECTORID_LOG(vec_id));
#endif
        FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                    "should not hold any lock here!");
        BufferVectorEntry* entry = bufferMgr->GetVectorEntry(vec_id);
        if (entry == nullptr) {
#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_DIVFTREE,
             "DELETE ERR: vector_id=" VECTORID_LOG_FMT " does not exists!",
             VECTORID_LOG(vec_id));
#endif
            FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                        "should not hold any lock here!");
            return RetStatus{.stat=RetStatus::VECTOR_NOT_FOUND, .message="Vector does not exist in this index"};
        }
        VectorLocation loc;
        BufferVertexEntry* parent = nullptr;
        do {
            parent = entry->ReadParentEntry(loc);
            if (parent == nullptr) {
#ifdef EXCESS_LOGING
            DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_DIVFTREE,
                "DELETE ERR: vector_id=" VECTORID_LOG_FMT " is either deleted or is not yet inserted!",
                VECTORID_LOG(vec_id));
#endif
                FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                            "should not hold any lock here!");
                return RetStatus{.stat=RetStatus::VECTOR_NOT_FOUND,
                                .message="Vector is either deleted or is not yet inserted."};
            }
        } while(!DeleteVectorAndReleaseContainer(vec_id, parent, loc).IsOK());

        real_size.fetch_sub(1);

#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
             "DELETE END: vector_id=" VECTORID_LOG_FMT,
             VECTORID_LOG(vec_id));
#endif
        FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                    "should not hold any lock here!");
        return RetStatus::Success();
    }

    RetStatus ApproximateKNearestNeighbours(const VTYPE* query,
                                            size_t k, uint8_t internal_node_search_span, uint8_t leaf_node_search_span,
                                            SortType sort_type, std::vector<ANNVectorInfo>& neighbours) override {

        CHECK_NOT_NULLPTR(query, LOG_TAG_DIVFTREE);

        FatalAssert(k > 0, LOG_TAG_DIVFTREE, "Number of neighbours cannot be 0.");
        FatalAssert(internal_node_search_span > 0, LOG_TAG_DIVFTREE,
                    "Number of internal vertex neighbours cannot be 0.");
        FatalAssert(leaf_node_search_span > 0, LOG_TAG_DIVFTREE, "Number of leaf neighbours cannot be 0.");

#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
             "ApproximateKNearestNeighbours BEGIN: query=%s",
             VectorToString(query, attr.dimension).ToCStr());
#endif

        FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                    "should not hold any lock here!");
        RetStatus rs = RetStatus::Success();

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);
        std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*> layers;

        while (real_size.load(std::memory_order_acquire) > 0) {
            DIVFTreeVertex* root =
                static_cast<DIVFTreeVertex*>(bufferMgr->ReadAndPinRoot());
            CHECK_NOT_NULLPTR(root, LOG_TAG_DIVFTREE);

            layers.reserve(root->attr.centroid_id._level + 1);
            for (uint64_t i = 0; i <= root->attr.centroid_id._level; ++i) {
                layers.emplace_back(new SortedList<ANNVectorInfo, SimilarityComparator>(attr.similarityComparator));
            }

            layers[root->attr.centroid_id._level]->Insert(ANNVectorInfo(0, root->attr.centroid_id,
                                                                        root->attr.version));
            rs = ANNSearch(query, k, internal_node_search_span, leaf_node_search_span,
                           (uint8_t)(root->attr.centroid_id._level), (uint8_t)VectorID::VECTOR_LEVEL, layers, root);
            root->Unpin();

#ifdef EXCESS_LOGING
        if (rs.IsOK()) {
            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
                "ApproximateKNearestNeighbours END: query=%s, AKNN=%s",
                VectorToString(query, attr.dimension).ToCStr(), layers[0]->ToString<ANNVectorInfoToString>().ToCStr());
        }
#endif
            layers[0]->Extract(neighbours, (sort_type == SortType::IncreasingSimilarity));
            for (uint64_t i = 0; i < layers.size(); ++i) {
                delete layers[i];
            }
            layers.clear();

            if (rs.IsOK()) {
                break;
            }

            neighbours.clear();
        }

        FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                    "should not hold any lock here!");
        return rs;
    }

    size_t Size() const override {
        return real_size.load(std::memory_order_acquire);
    }

    const DIVFTreeAttributes& GetAttributes() const override {
        return attr;
    }

    inline void EndBGThreads() override {
        end_signal.store(true, std::memory_order_release);
        for (size_t i = 0; i < attr.num_compactors; ++i) {
            CHECK_NOT_NULLPTR(bg_compactors[i], LOG_TAG_DIVFTREE);
            bg_compactors[i]->WaitForThreadToFinish();
        }

        for (size_t i = 0; i < attr.num_mergers; ++i) {
            CHECK_NOT_NULLPTR(bg_mergers[i], LOG_TAG_DIVFTREE);
            bg_mergers[i]->WaitForThreadToFinish();
        }

        for (size_t i = 0; i < attr.num_migrators; ++i) {
            CHECK_NOT_NULLPTR(bg_migrators[i], LOG_TAG_DIVFTREE);
            bg_migrators[i]->WaitForThreadToFinish();
        }

        for (size_t i = 0; i < attr.num_searchers; ++i) {
            CHECK_NOT_NULLPTR(searchers[i], LOG_TAG_DIVFTREE);
            searchers[i]->WaitForThreadToFinish();
        }

#ifdef HANG_DETECTION
        end_bghang_detector.store(true, std::memory_order_release);
        CHECK_NOT_NULLPTR(bg_hang_detector, LOG_TAG_DIVFTREE);
        bg_hang_detector->WaitForThreadToFinish();
#endif
    }

    inline String GetStatistics(std::string title_extention = "", bool clear_stats = false) override {
#ifdef ENABLE_STAT_COLLECTION
        stats_lock.Lock(SX_EXCLUSIVE);
        String out("\n************************ %s%sStatistics ************************\n\n",
                   title_extention.empty() ? "" : " ", title_extention.c_str());
        uint64_t total_bg_migration_tasks_completed = 0;
        uint64_t total_bg_migration_num_migrated_vectors = 0;
        uint64_t total_bg_merge_tasks_completed = 0;
        uint64_t total_bg_merge_num_merged_clusters = 0;
        uint64_t total_bg_compaction_tasks_completed = 0;
        uint64_t total_bg_compaction_num_compacted_clusters = 0;
        uint64_t total_bg_search_tasks_completed = 0;

        uint64_t* bg_migration_iterations_cpy = nullptr;
        uint64_t* bg_migration_tasks_completed_cpy = nullptr;
        uint64_t* bg_migration_num_migrated_vectors_cpy = nullptr;
        uint64_t* bg_merge_iterations_cpy = nullptr;
        uint64_t* bg_merge_tasks_completed_cpy = nullptr;
        uint64_t* bg_merge_num_merged_clusters_cpy = nullptr;
        uint64_t* bg_compaction_iterations_cpy = nullptr;
        uint64_t* bg_compaction_tasks_completed_cpy = nullptr;
        uint64_t* bg_compaction_num_compacted_clusters_cpy = nullptr;
        uint64_t* bg_search_iterations_cpy = nullptr;
        uint64_t* bg_search_tasks_completed_cpy = nullptr;

        if (attr.num_migrators > 0) {
            CHECK_NOT_NULLPTR(bg_migration_tasks_completed, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_migration_num_migrated_vectors, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_migration_iterations, LOG_TAG_DIVFTREE);
            FatalAssert(bg_migrators.size() == attr.num_migrators, LOG_TAG_DIVFTREE,
                        "bg_migrators size mismatch with attr.num_migrators");

            bg_migration_tasks_completed_cpy = new uint64_t[attr.num_migrators];
            bg_migration_num_migrated_vectors_cpy = new uint64_t[attr.num_migrators];
            bg_migration_iterations_cpy = new uint64_t[attr.num_migrators];

            for (size_t i = 0; i < attr.num_migrators; ++i) {
                uint64_t tasks_completed = bg_migration_tasks_completed[i].load(std::memory_order_acquire);
                uint64_t num_migrated_vectors = bg_migration_num_migrated_vectors[i].load(std::memory_order_acquire);
                uint64_t iterations = bg_migration_iterations[i].load(std::memory_order_acquire);

                bg_migration_tasks_completed_cpy[i] = tasks_completed;
                bg_migration_num_migrated_vectors_cpy[i] = num_migrated_vectors;
                bg_migration_iterations_cpy[i] = iterations;
                total_bg_migration_tasks_completed += tasks_completed;
                total_bg_migration_num_migrated_vectors += num_migrated_vectors;
            }

            out += String("BGMigration: num_bg_threads=%lu, tasks_completed=%lu, num_migrated_vectors=%lu:\n",
                          attr.num_migrators, total_bg_migration_tasks_completed,
                          total_bg_migration_num_migrated_vectors);

            for (size_t i = 0; i < attr.num_migrators; ++i) {
                double task_completed_to_total =
                    (total_bg_migration_tasks_completed == 0) ? 0 :
                                                                (bg_migration_tasks_completed_cpy[i] * 100) /
                                                                (double)total_bg_migration_tasks_completed;
                double num_migrated_vectors_to_total =
                    (total_bg_migration_num_migrated_vectors == 0) ? 0 :
                                                                    (bg_migration_num_migrated_vectors_cpy[i] * 100) /
                                                                    (double)total_bg_migration_num_migrated_vectors;
                uint64_t num_migrated_vectors_per_total_checks = 0;
                double active_tasks_to_total = 0;
                if (bg_migration_iterations_cpy[i] > 0) {
                    num_migrated_vectors_per_total_checks =
                        bg_migration_num_migrated_vectors_cpy[i] / bg_migration_iterations_cpy[i];
                    active_tasks_to_total =
                        (bg_migration_tasks_completed_cpy[i] * 100) /
                        (double)bg_migration_iterations_cpy[i];
                }
                out += String("\t* Migrator Thread %lu: tasks_completed=%lu, num_migrated_vectors=%lu, iterations=%lu |"
                              " task_completed_to_total=%.2f%%, num_migrated_vectors_to_total=%.2f%% | "
                              " num_migrated_vectors_per_total_checks=%lu, "
                              "active_tasks_to_total=%.2f%%\n",
                              bg_migrators[i]->ID(), bg_migration_tasks_completed_cpy[i],
                              bg_migration_num_migrated_vectors_cpy[i], bg_migration_iterations_cpy[i],
                              task_completed_to_total, num_migrated_vectors_to_total,
                              num_migrated_vectors_per_total_checks, active_tasks_to_total);
            }

            delete[] bg_migration_iterations_cpy;
            delete[] bg_migration_tasks_completed_cpy;
            delete[] bg_migration_num_migrated_vectors_cpy;
        } else {
            out += String("No Background Migration Tasks!\n");
        }

        out += String("\n");

        if (attr.num_mergers > 0) {
            CHECK_NOT_NULLPTR(bg_merge_tasks_completed, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_merge_num_merged_clusters, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_merge_iterations, LOG_TAG_DIVFTREE);
            FatalAssert(bg_mergers.size() == attr.num_mergers, LOG_TAG_DIVFTREE,
                        "bg_mergers size mismatch with attr.num_mergers");

            bg_merge_tasks_completed_cpy = new uint64_t[attr.num_mergers];
            bg_merge_num_merged_clusters_cpy = new uint64_t[attr.num_mergers];
            bg_merge_iterations_cpy = new uint64_t[attr.num_mergers];

            for (size_t i = 0; i < attr.num_mergers; ++i) {
                uint64_t tasks_completed = bg_merge_tasks_completed[i].load(std::memory_order_acquire);
                uint64_t num_merged_clusters = bg_merge_num_merged_clusters[i].load(std::memory_order_acquire);
                uint64_t iterations = bg_merge_iterations[i].load(std::memory_order_acquire);

                bg_merge_tasks_completed_cpy[i] = tasks_completed;
                bg_merge_num_merged_clusters_cpy[i] = num_merged_clusters;
                bg_merge_iterations_cpy[i] = iterations;
                total_bg_merge_tasks_completed += tasks_completed;
                total_bg_merge_num_merged_clusters += num_merged_clusters;
            }

            out += String("BGMerge: num_bg_threads=%lu, tasks_completed=%lu, num_merged_clusters=%lu, "
                          "merge_trigger_rate=%.2f%%:\n",
                          attr.num_mergers, total_bg_merge_tasks_completed,
                          total_bg_merge_num_merged_clusters,
                          total_bg_merge_tasks_completed == 0 ? 0 : (total_bg_merge_num_merged_clusters * 100) /
                                                                    (double)total_bg_merge_tasks_completed);

            for (size_t i = 0; i < attr.num_mergers; ++i) {
                double task_completed_to_total = (total_bg_merge_tasks_completed == 0) ? 0 :
                                                            (bg_merge_tasks_completed_cpy[i] * 100) /
                                                            (double)total_bg_merge_tasks_completed;
                double merge_trigger_rate = bg_merge_tasks_completed_cpy[i] == 0 ?
                    0 : (bg_merge_num_merged_clusters_cpy[i] * 100) / (double)bg_merge_tasks_completed_cpy[i];
                double valid_iterations = bg_merge_iterations_cpy[i] == 0 ?
                    0 : (bg_merge_tasks_completed_cpy[i] * 100) / (double)bg_merge_iterations_cpy[i];

                out += String("\t* Merger Thread %lu: tasks_completed=%lu, num_merged_clusters=%lu, iterations=%lu |"
                              " task_completed_to_total=%.2f%% | merge_trigger_rate=%.2f%%, "
                              "valid_iterations=%.2f%%\n",
                              bg_mergers[i]->ID(), bg_merge_tasks_completed_cpy[i],
                              bg_merge_num_merged_clusters_cpy[i], bg_merge_iterations_cpy[i],
                              task_completed_to_total, merge_trigger_rate, valid_iterations);
            }

            delete[] bg_merge_iterations_cpy;
            delete[] bg_merge_tasks_completed_cpy;
            delete[] bg_merge_num_merged_clusters_cpy;
        } else {
            out += String("No Background Merge Tasks!\n");
        }

        out += String("\n");

        if (attr.num_compactors > 0) {
            CHECK_NOT_NULLPTR(bg_compaction_tasks_completed, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_compaction_num_compacted_clusters, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_compaction_iterations, LOG_TAG_DIVFTREE);
            FatalAssert(bg_compactors.size() == attr.num_compactors, LOG_TAG_DIVFTREE,
                        "bg_compactors size mismatch with attr.num_compactors");

            bg_compaction_tasks_completed_cpy = new uint64_t[attr.num_compactors];
            bg_compaction_num_compacted_clusters_cpy = new uint64_t[attr.num_compactors];
            bg_compaction_iterations_cpy = new uint64_t[attr.num_compactors];

            for (size_t i = 0; i < attr.num_compactors; ++i) {
                uint64_t tasks_completed = bg_compaction_tasks_completed[i].load(std::memory_order_acquire);
                uint64_t num_compacted_clusters = bg_compaction_num_compacted_clusters[i].load(std::memory_order_acquire);
                uint64_t iterations = bg_compaction_iterations[i].load(std::memory_order_acquire);

                bg_compaction_tasks_completed_cpy[i] = tasks_completed;
                bg_compaction_num_compacted_clusters_cpy[i] = num_compacted_clusters;
                bg_compaction_iterations_cpy[i] = iterations;
                total_bg_compaction_tasks_completed += tasks_completed;
                total_bg_compaction_num_compacted_clusters += num_compacted_clusters;
            }

            out += String("BGCompaction: num_bg_threads=%lu, tasks_completed=%lu, num_compacted_clusters=%lu, "
                          "compaction_trigger_rate=%.2f%%:\n",
                          attr.num_compactors, total_bg_compaction_tasks_completed,
                          total_bg_compaction_num_compacted_clusters,
                          total_bg_compaction_tasks_completed == 0 ?
                            0 : (total_bg_compaction_num_compacted_clusters * 100) /
                                (double)total_bg_compaction_tasks_completed);

            for (size_t i = 0; i < attr.num_compactors; ++i) {
                double task_completed_to_total =
                    total_bg_compaction_tasks_completed == 0 ? 0 :
                                                                (bg_compaction_tasks_completed_cpy[i] * 100) /
                                                                (double)total_bg_compaction_tasks_completed;
                double compaction_trigger_rate = bg_compaction_tasks_completed_cpy[i] == 0 ?
                    0 : (bg_compaction_num_compacted_clusters_cpy[i] * 100) /
                        (double)bg_compaction_tasks_completed_cpy[i];
                double valid_iterations = bg_compaction_iterations_cpy[i] == 0 ?
                    0 : (bg_compaction_tasks_completed_cpy[i] * 100) /
                        (double)bg_compaction_iterations_cpy[i];
                out += String("\t* Compactor Thread %lu: tasks_completed=%lu, num_compacted_clusters=%lu, iterations=%lu |"
                              " task_completed_to_total=%.2f%% | compaction_trigger_rate=%.2f%%, "
                              "valid_iterations=%.2f%%\n",
                              bg_compactors[i]->ID(), bg_compaction_tasks_completed_cpy[i],
                              bg_compaction_num_compacted_clusters_cpy[i], bg_compaction_iterations_cpy[i],
                              task_completed_to_total, compaction_trigger_rate, valid_iterations);
            }

            delete[] bg_compaction_tasks_completed_cpy;
            delete[] bg_compaction_num_compacted_clusters_cpy;
            delete[] bg_compaction_iterations_cpy;
        } else {
            out += String("No Background Compaction Tasks!\n");
        }

        if (attr.num_searchers > 0) {
            CHECK_NOT_NULLPTR(bg_search_iterations, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_search_tasks_completed, LOG_TAG_DIVFTREE);
            FatalAssert(searchers.size() == attr.num_searchers, LOG_TAG_DIVFTREE,
                        "searchers size mismatch with attr.num_searchers");

            bg_search_tasks_completed_cpy = new uint64_t[attr.num_searchers];
            bg_search_iterations_cpy = new uint64_t[attr.num_searchers];

            for (size_t i = 0; i < attr.num_searchers; ++i) {
                uint64_t tasks_completed = bg_search_tasks_completed[i].load(std::memory_order_acquire);
                uint64_t iterations = bg_search_iterations[i].load(std::memory_order_acquire);

                bg_search_tasks_completed_cpy[i] = tasks_completed;
                bg_search_iterations_cpy[i] = iterations;
                total_bg_search_tasks_completed += tasks_completed;
            }

            uint64_t tasks_created = total_bg_search_tasks.load(std::memory_order_acquire);

            out += String("AsyncSearch: num_bg_threads=%lu, tasks_completed=%lu, tasks_created=%lu, task_completed_to_total=%.2f%%:\n",
                          attr.num_searchers, total_bg_search_tasks_completed, tasks_created,
                          tasks_created == 0 ? 0 : (total_bg_search_tasks_completed * 100) / (double)tasks_created);

            for (size_t i = 0; i < attr.num_searchers; ++i) {
                double task_completed_to_total = tasks_created == 0 ? 0 :
                                                            (bg_search_tasks_completed_cpy[i] * 100) /
                                                            (double)tasks_created;
                double valid_iterations = bg_search_iterations_cpy[i] == 0 ? 0 :
                    (bg_search_tasks_completed_cpy[i] * 100) / (double)bg_search_iterations_cpy[i];
                out += String("\t* Searcher Thread %lu: tasks_completed=%lu, iterations=%lu |"
                              " task_completed_to_total=%.2f%% | valid_iterations=%.2f%%\n",
                              searchers[i]->ID(), bg_search_tasks_completed_cpy[i],
                              bg_search_iterations_cpy[i], task_completed_to_total, valid_iterations);
            }

            delete[] bg_search_tasks_completed_cpy;
            delete[] bg_search_iterations_cpy;
        } else {
            out += String("No Background Search Tasks!\n");
        }

        out += String("\n");


        out += String("************************************************************\n");

        if (clear_stats) {
            ClearStats(false);
        }
        stats_lock.Unlock();
        return out;
#else
        return String("\n************************ Statistics collection is disabled! ************************\n");
#endif
    }

    inline void StartStatsCollection() override {
#ifdef ENABLE_STAT_COLLECTION
        collect_stats.store(true, std::memory_order_release);
#endif
    }

    inline void StopStatsCollection() override {
#ifdef ENABLE_STAT_COLLECTION
        collect_stats.store(false, std::memory_order_release);
#endif
    }

    inline void ClearStats() override {
        ClearStats(true);
    }


    // inline String ToString(bool detailed = false) const override {
    //     String out("{Attr=<Dim=%hu, ClusteringAlg=%s, DistanceAlg=%s, "
    //                "LeafMinSize=%hu, LeafMaxSize=%hu, InternalMinSize=%hu, InternalMaxSize=%hu, "
    //                "SplitInternal=%hu, SplitLeaf=%hu, Size=%lu, Levels=%lu>, RootID=" VECTORID_LOG_FMT
    //                ", Vertexs=[", core_attr.dimension, CLUSTERING_TYPE_NAME[core_attr.clusteringAlg],
    //                DISTANCE_TYPE_NAME[core_attr.distanceAlg], leaf_min_size, leaf_max_size,
    //                internal_min_size, internal_max_size, split_internal, split_leaf,
    //                _size, _levels, VECTORID_LOG(_root));
    //     std::vector<VectorID> curr_level_stack, next_level_stack;
    //     size_t seen = 0;
    //     if (_root.IsValid()) {
    //         curr_level_stack.push_back(_root);
    //     }
    //     bool go_to_next_level = true;
    //     uint64_t curr_level = 0;
    //     while (!curr_level_stack.empty()) {
    //         VectorID curr_vertexId = curr_level_stack.back();
    //         curr_level_stack.pop_back();
    //         CHECK_VECTORID_IS_VALID(curr_vertexId, LOG_TAG_DIVFTREE);
    //         if (go_to_next_level) {
    //             curr_level = curr_vertexId._level;
    //             out += String("Level %lu:{Size=%lu, Vertexs=[", curr_level, curr_level_stack.size() + 1);
    //             go_to_next_level = false;
    //         }
    //         else {
    //             FatalAssert(curr_vertexId._level == curr_level, LOG_TAG_DIVFTREE,
    //                         "Current vertex level (%hhu) does not match expected level (%lu).",
    //                         curr_vertexId._level, curr_level);
    //         }
    //         DIVFTreeVertex* vertex = static_cast<DIVFTreeVertex*>(_bufmgr.GetVertex(curr_vertexId));
    //         CHECK_VERTEX_IS_VALID(vertex, LOG_TAG_DIVFTREE, false);
    //         out += String("<VertexID=" VECTORID_LOG_FMT ", Bucket=%s>",
    //                     VECTORID_LOG(curr_vertexId), vertex->BucketToString().ToCStr());
    //         if (!vertex->IsLeaf()) {
    //             for (uint16_t i = 0; i < vertex->Size(); ++i) {
    //                 const VectorPair& vectorPair = vertex->cluster[i];
    //                 CHECK_VECTORID_IS_VALID(vectorPair.id, LOG_TAG_DIVFTREE);
    //                 FatalAssert(vectorPair.id._level == curr_vertexId._level - 1, LOG_TAG_DIVFTREE,
    //                             "VectorID level (%hhu) does not match current vertex level (%hhu).",
    //                             vectorPair.id._level, curr_vertexId._level - 1);
    //                 FatalAssert(vectorPair.vec.IsValid(), LOG_TAG_DIVFTREE,
    //                             "Invalid vector in vertex %s at index %hu.", VECTORID_LOG(curr_vertexId), i);
    //                 next_level_stack.emplace_back(vectorPair.id);
    //             }
    //         }
    //         else {
    //             seen += vertex->Size();
    //         }
    //         if (curr_level_stack.empty()) {
    //             out += String("]}");
    //             go_to_next_level = true;
    //             next_level_stack.swap(curr_level_stack);
    //             if (!curr_level_stack.empty()) {
    //                 out += String(", ");
    //             }
    //             else {
    //                 out += String("]}");
    //             }
    //         }
    //         else {
    //             out += String(", ");
    //         }
    //     }
    //     FatalAssert(seen == _size, LOG_TAG_DIVFTREE,
    //                 "Seen vertices (%lu) does not match expected size (%lu).", seen, _size);
    //     return out;
    // }

protected:
    DIVFTreeAttributes attr;
    std::atomic<uint64_t> real_size;
    std::atomic<bool> end_signal;
#ifdef HANG_DETECTION
    std::atomic<bool> end_bghang_detector = false;
    static inline constexpr uint64_t BG_HANG_DETECTOR_SLEEP_MS = 60000; /* 60 seconds */
    Thread* bg_hang_detector = nullptr;
#endif

    std::vector<Thread*> bg_migrators;
    BlockingQueue<MigrationCheckTask> migration_tasks;
    std::vector<Thread*> bg_mergers;
    BlockingQueue<MergeTask> merge_tasks;
    std::vector<Thread*> bg_compactors;
    BlockingQueue<CompactionTask> compaction_tasks;
    std::vector<Thread*> searchers;
    BlockingQueue<SearchTask*> search_tasks;

#ifdef ENABLE_STAT_COLLECTION
    std::atomic<bool> collect_stats = true;
    SXSpinLock stats_lock;

    std::atomic<uint64_t>* bg_migration_iterations = nullptr;
    std::atomic<uint64_t>* bg_migration_tasks_completed = nullptr;
    std::atomic<uint64_t>* bg_migration_num_migrated_vectors = nullptr;

    std::atomic<uint64_t>* bg_merge_iterations = nullptr;
    std::atomic<uint64_t>* bg_merge_tasks_completed = nullptr;
    std::atomic<uint64_t>* bg_merge_num_merged_clusters = nullptr;

    std::atomic<uint64_t>* bg_compaction_iterations = nullptr;
    std::atomic<uint64_t>* bg_compaction_tasks_completed = nullptr;
    std::atomic<uint64_t>* bg_compaction_num_compacted_clusters = nullptr;

    std::atomic<uint64_t> total_bg_search_tasks = 0;
    std::atomic<uint64_t>* bg_search_iterations = nullptr;
    std::atomic<uint64_t>* bg_search_tasks_completed = nullptr;

#ifdef COLLECT_LATENCY_STATS
#endif
#endif

    inline void ClearStats(bool need_lock) {
#ifdef ENABLE_STAT_COLLECTION
        if (need_lock) {
            stats_lock.Lock(SX_EXCLUSIVE);
        }
        threadSelf->SanityCheckLockHeldInModeByMe(&stats_lock, SX_EXCLUSIVE);

        if (attr.num_migrators > 0) {
            CHECK_NOT_NULLPTR(bg_migration_iterations, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_migration_tasks_completed, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_migration_num_migrated_vectors, LOG_TAG_DIVFTREE);
            for (size_t i = 0; i < attr.num_migrators; ++i) {
                bg_migration_iterations[i].store(0, std::memory_order_release);
                bg_migration_tasks_completed[i].store(0, std::memory_order_release);
                bg_migration_num_migrated_vectors[i].store(0, std::memory_order_release);
            }
        }

        if (attr.num_mergers > 0) {
            CHECK_NOT_NULLPTR(bg_merge_iterations, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_merge_tasks_completed, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_merge_num_merged_clusters, LOG_TAG_DIVFTREE);
            for (size_t i = 0; i < attr.num_mergers; ++i) {
                bg_merge_iterations[i].store(0, std::memory_order_release);
                bg_merge_tasks_completed[i].store(0, std::memory_order_release);
                bg_merge_num_merged_clusters[i].store(0, std::memory_order_release);
            }
        }

        if (attr.num_compactors > 0) {
            CHECK_NOT_NULLPTR(bg_compaction_iterations, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_compaction_tasks_completed, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_compaction_num_compacted_clusters, LOG_TAG_DIVFTREE);
            for (size_t i = 0; i < attr.num_compactors; ++i) {
                bg_compaction_iterations[i].store(0, std::memory_order_release);
                bg_compaction_tasks_completed[i].store(0, std::memory_order_release);
                bg_compaction_num_compacted_clusters[i].store(0, std::memory_order_release);
            }
        }

        if (attr.num_searchers > 0) {
            CHECK_NOT_NULLPTR(bg_search_iterations, LOG_TAG_DIVFTREE);
            CHECK_NOT_NULLPTR(bg_search_tasks_completed, LOG_TAG_DIVFTREE);
            for (size_t i = 0; i < attr.num_searchers; ++i) {
                bg_search_iterations[i].store(0, std::memory_order_release);
                bg_search_tasks_completed[i].store(0, std::memory_order_release);
            }
        }
        total_bg_search_tasks.store(0, std::memory_order_release);
        if (need_lock) {
            stats_lock.Unlock();
        }
#endif
    }

    void BGMigrationStatsUpdate(uint64_t thread_index, bool completed_task, uint64_t num_migrated_vectors) {
#ifdef ENABLE_STAT_COLLECTION
        CHECK_NOT_NULLPTR(bg_migration_iterations, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(bg_migration_tasks_completed, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(bg_migration_num_migrated_vectors, LOG_TAG_DIVFTREE);
        if (collect_stats.load(std::memory_order_acquire)) {
            stats_lock.Lock(SX_SHARED);
            bg_migration_iterations[thread_index].fetch_add(1, std::memory_order_release);
            if (completed_task) {
                bg_migration_tasks_completed[thread_index].fetch_add(1, std::memory_order_release);
            }
             bg_migration_num_migrated_vectors[thread_index].fetch_add(num_migrated_vectors,
                                                                       std::memory_order_release);
            stats_lock.Unlock();
        }
#endif
    }

    void BGMergeStatsUpdate(uint64_t thread_index, bool completed_task, bool cluster_merged) {
#ifdef ENABLE_STAT_COLLECTION
        CHECK_NOT_NULLPTR(bg_merge_iterations, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(bg_merge_tasks_completed, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(bg_merge_num_merged_clusters, LOG_TAG_DIVFTREE);
        if (collect_stats.load(std::memory_order_acquire)) {
            stats_lock.Lock(SX_SHARED);
            bg_merge_iterations[thread_index].fetch_add(1, std::memory_order_release);
            if (completed_task) {
                bg_merge_tasks_completed[thread_index].fetch_add(1, std::memory_order_release);
                if (cluster_merged) {
                    bg_merge_num_merged_clusters[thread_index].fetch_add(1, std::memory_order_release);
                }
            }
            stats_lock.Unlock();
        }
#endif
    }

    void BGCompactionStatsUpdate(uint64_t thread_index, bool completed_task, bool cluster_compacted) {
#ifdef ENABLE_STAT_COLLECTION
        CHECK_NOT_NULLPTR(bg_compaction_iterations, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(bg_compaction_tasks_completed, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(bg_compaction_num_compacted_clusters, LOG_TAG_DIVFTREE);
        if (collect_stats.load(std::memory_order_acquire)) {
            stats_lock.Lock(SX_SHARED);
            bg_compaction_iterations[thread_index].fetch_add(1, std::memory_order_release);
            if (completed_task) {
                bg_compaction_tasks_completed[thread_index].fetch_add(1, std::memory_order_release);
                if (cluster_compacted) {
                    bg_compaction_num_compacted_clusters[thread_index].fetch_add(1, std::memory_order_release);
                }
            }
            stats_lock.Unlock();
        }
#endif
    }

    void BGSearchStatsUpdate(uint64_t thread_index, bool completed_task) {
#ifdef ENABLE_STAT_COLLECTION
        CHECK_NOT_NULLPTR(bg_search_iterations, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(bg_search_tasks_completed, LOG_TAG_DIVFTREE);
        if (collect_stats.load(std::memory_order_acquire)) {
            stats_lock.Lock(SX_SHARED);
            bg_search_iterations[thread_index].fetch_add(1, std::memory_order_release);
            if (completed_task) {
                bg_search_tasks_completed[thread_index].fetch_add(1, std::memory_order_release);
            }
            stats_lock.Unlock();
        }
#endif
    }

    void BGSearchStatsUpdateCreatedTask(uint64_t num_tasks) {
#ifdef ENABLE_STAT_COLLECTION
        if (collect_stats.load(std::memory_order_acquire)) {
            stats_lock.Lock(SX_SHARED);
            total_bg_search_tasks.fetch_add(num_tasks,  std::memory_order_release);
            stats_lock.Unlock();
        }
#endif
    }

    RetStatus CompactAndInsert(BufferVertexEntry* container_entry, const ConstVectorBatch& batch,
                               DuplicateState* dup_state = nullptr, uint16_t num_duplicates = 0,
                               uint16_t marked_for_update = INVALID_OFFSET) {
        CHECK_NOT_NULLPTR(container_entry, LOG_TAG_DIVFTREE);
        FatalAssert(((batch.size == 0) || ((batch.data != nullptr) && (batch.id != nullptr))), LOG_TAG_DIVFTREE,
                    "Invalid batch");
        FatalAssert((batch.size != 0) || (marked_for_update == INVALID_OFFSET), LOG_TAG_DIVFTREE,
                    "If there is no batch, marked_for_update should be invalid!");
        FatalAssert(num_duplicates <= batch.size, LOG_TAG_DIVFTREE,
                    "num_duplicates cannot be larger than batch.size!");
        threadSelf->SanityCheckLockHeldInModeByMe(&container_entry->clusterLock, SX_EXCLUSIVE);

        DIVFTreeVertex* current =
            static_cast<DIVFTreeVertex*>(container_entry->ReadLatestVersion(marked_for_update != INVALID_OFFSET));
        CHECK_NOT_NULLPTR(current, LOG_TAG_DIVFTREE);
        FatalAssert(current->attr.index == this, LOG_TAG_DIVFTREE,
                    "input entry does not belong to this index!");
        FatalAssert(current->cluster.header.num_deleted.load(std::memory_order_relaxed) > 0, LOG_TAG_DIVFTREE,
                    "Cannot compact a cluster with no deleted elemetns!");

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

        Version nextVersion = container_entry->currentVersion.NextCompaction();
        DIVFTreeVertexAttributes newAttr = current->attr;
        newAttr.version = nextVersion;
        DIVFTreeVertex* compacted =
            new(bufferMgr->AllocateMemoryForVertex(container_entry->centroidMeta.selfId._level))
            DIVFTreeVertex(newAttr);
#ifndef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE, "compacting vertex id: " VECTORID_LOG_FMT
                "old ver: %s, new ver: %s, old addr: %p, new addr: %p",
                VECTORID_LOG(container_entry->centroidMeta.selfId), container_entry->currentVersion.ToString().ToCStr(),
                nextVersion.ToString().ToCStr(), current, compacted);
#endif

        uint16_t size = current->cluster.header.reserved_size.load(std::memory_order_relaxed);
        const bool is_leaf = container_entry->centroidMeta.selfId.IsLeaf();
        const uint16_t dim = attr.dimension;
        const uint16_t cap = current->attr.cap;
        const uint16_t blckSize = current->attr.cap;
        /* Todo: this will not work if we have more than one block! */
        FatalAssert(current->cluster.NumBlocks(blckSize, cap) == 1,
                    LOG_TAG_NOT_IMPLEMENTED, "Currently cannot handle more than one block!");

        FatalAssert(is_leaf || (batch.size == 0) || (batch.version != nullptr), LOG_TAG_DIVFTREE,
                    "Batch of versions to insert is null!");

        Address srcMetaData = current->cluster.MetaData(0, is_leaf, blckSize, cap, dim);
        VTYPE* srcData = current->cluster.Data(0, is_leaf, blckSize, cap, dim);
        Address destMetaData = compacted->cluster.MetaData(0, is_leaf, blckSize, cap, dim);
        VTYPE* destData = compacted->cluster.Data(0, is_leaf, blckSize, cap, dim);
        uint16_t new_size = 0;
        std::vector<uint16_t> offsets;
        offsets.reserve(size -
                        current->cluster.header.num_deleted.load(std::memory_order_relaxed));
        Address marked_for_update_meta = INVALID_ADDRESS;
        VectorID marked_for_update_id = INVALID_VECTOR_ID;
        SANITY_CHECK(
            Version marked_version = 0;
            bool found_marked = false;
        );
        /*
         * Note: During compaction, we will update vector locations but
         * we haven't updated the container cluster ptr yet. As a result, a reader may see that offset mismatch
         * caused by this. As long as we can get the container in S mode and we have ourself in X mode, our location
         * should not change. so in cases where only offset has changed, we can simply reread the location to
         * see the new one!
         * This may be a bit funky for vectors though as they do not have any locks associated with them.
         */
        for (uint16_t i = 0; i < size; ++i) {
            if (is_leaf) {
                VectorMetaData* src_vmd = reinterpret_cast<VectorMetaData*>(srcMetaData);
                VectorMetaData* dest_vmd = reinterpret_cast<VectorMetaData*>(destMetaData);
                VectorState vstate = src_vmd[i].state.load(std::memory_order_relaxed);
                FatalAssert(marked_for_update == INVALID_OFFSET, LOG_TAG_DIVFTREE,
                            "Since vectors cannot become outdated, marked_for_update should be invalid in leaf compaction!");
                if (vstate == VECTOR_STATE_VALID) {
                    memcpy(&destData[new_size * dim], &srcData[i * dim], dim * sizeof(VTYPE));
                    dest_vmd[new_size].id = src_vmd[i].id;
                    dest_vmd[new_size].state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
                    offsets.push_back(i);
                    ++new_size;
                } SANITY_CHECK( else if (vstate == VECTOR_STATE_MIGRATED) {
                    VectorLocation outdatedLoc(container_entry->centroidMeta.selfId,
                                               container_entry->currentVersion, i);
                    VectorLocation currentLoc = bufferMgr->LoadCurrentVectorLocation(src_vmd[i].id, false);
                    FatalAssert(outdatedLoc != currentLoc, LOG_TAG_DIVFTREE,
                                "current location is migrated!");
                })
            } else {
                FatalAssert(num_duplicates == 0, LOG_TAG_NOT_IMPLEMENTED,
                            "Compaction with duplicates not implemented for internal nodes yet!");
                CentroidMetaData* src_vmd = reinterpret_cast<CentroidMetaData*>(srcMetaData);
                CentroidMetaData* dest_vmd = reinterpret_cast<CentroidMetaData*>(destMetaData);
                VectorState vstate = src_vmd[i].state.load(std::memory_order_relaxed);
                if (i == marked_for_update) {
                    FatalAssert(vstate == VECTOR_STATE_VALID, LOG_TAG_DIVFTREE,
                                "marked for update vector is not valid!");
                    marked_for_update_meta = &src_vmd[i];
                    marked_for_update_id = src_vmd[i].id;
                    SANITY_CHECK(
                        VectorLocation markedLoc(container_entry->centroidMeta.selfId,
                                                 container_entry->currentVersion, i);
                        FatalAssert(markedLoc == bufferMgr->LoadCurrentVectorLocation(src_vmd[i].id, false), LOG_TAG_DIVFTREE,
                                    "marked entry's location should be here!");
                        marked_version = src_vmd[i].version;
                    );
                    continue;
                }
                if (vstate == VECTOR_STATE_VALID) {
                    memcpy(&destData[new_size * dim], &srcData[i * dim], dim * sizeof(VTYPE));
                    dest_vmd[new_size].id = src_vmd[i].id;
                    dest_vmd[new_size].version = src_vmd[i].version;
                    dest_vmd[new_size].state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
                    offsets.push_back(i);
                    ++new_size;
                } SANITY_CHECK( else if (vstate == VECTOR_STATE_MIGRATED) {
                    VectorLocation outdatedLoc(container_entry->centroidMeta.selfId,
                                               container_entry->currentVersion, i);
                    FatalAssert(outdatedLoc != bufferMgr->LoadCurrentVectorLocation(src_vmd[i].id, false), LOG_TAG_DIVFTREE,
                                "current location is migrated!");
                })
            }
        }
        FatalAssert(current->cluster.header.visible_size.load(std::memory_order_relaxed) ==
                    current->cluster.header.reserved_size.load(std::memory_order_relaxed), LOG_TAG_DIVFTREE,
                    "not all updates went through!");
        FatalAssert(new_size == (current->cluster.header.reserved_size.load(std::memory_order_relaxed) -
                                 current->cluster.header.num_deleted.load(std::memory_order_relaxed) -
                                 (marked_for_update != INVALID_OFFSET ? 1 : 0)),
                    LOG_TAG_DIVFTREE, "not all deletes went through!");
        uint16_t num_inserted = batch.size - num_duplicates;
        FatalAssert(new_size + num_inserted >= new_size, LOG_TAG_DIVFTREE, "overflow detected!");
        FatalAssert(new_size + num_inserted <= cap, LOG_TAG_DIVFTREE,
                    "Cluster cannot contain all of these vectors!");
        FatalAssert(offsets.size() == new_size, LOG_TAG_DIVFTREE, "offsets size mismatch!");

        uint16_t old_reserved = new_size;

        for (uint16_t i = 0; i < batch.size; ++i) {
            if (dup_state != nullptr && dup_state[i] != DuplicateState::UNIQUE) {
                FatalAssert(dup_state[i] == DuplicateState::DUPLICATE_NOT_INSERTED, LOG_TAG_DIVFTREE,
                            "Since exists was called before compaction, we do not need to insert the duplicates!");
                bufferMgr->UpdateVectorLocation(batch.id[i], INVALID_VECTOR_LOCATION, true);
                continue;
            }

            memcpy(&destData[new_size * dim], &batch.data[i * dim], dim * sizeof(VTYPE));
            SANITY_CHECK(
                if (batch.id[i] == marked_for_update_id) {
                    FatalAssert(!found_marked, LOG_TAG_DIVFTREE,
                                "marked for update vector found multiple times in batch!");
                    found_marked = true;

                    if (!is_leaf) {
                        FatalAssert(batch.version[i] == marked_version.NextCompaction() ||
                                    batch.version[i] == marked_version.NextSplit(), LOG_TAG_DIVFTREE,
                                    "marked for update version mismatch!");
                    }
                }
            );

            if (is_leaf) {
                VectorMetaData* dest_vmd = reinterpret_cast<VectorMetaData*>(destMetaData);
                dest_vmd[new_size].id = batch.id[i];
                dest_vmd[new_size].state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
            } else {
                CentroidMetaData* dest_vmd = reinterpret_cast<CentroidMetaData*>(destMetaData);
                dest_vmd[new_size].id = batch.id[i];
                dest_vmd[new_size].version = batch.version[i];
                dest_vmd[new_size].state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
                bufferMgr->PinVertexVersion(batch.id[i], batch.version[i]);
            }

            ++new_size;
        }

        FatalAssert(new_size == old_reserved + (batch.size - num_duplicates), LOG_TAG_DIVFTREE,
                    "new size mismatch after insertion!");

        compacted->cluster.header.num_deleted.store(0, std::memory_order_relaxed);
        compacted->cluster.header.reserved_size.store(new_size, std::memory_order_relaxed);
        compacted->cluster.header.visible_size.store(new_size, std::memory_order_relaxed);

        BufferVertexEntry* parent = nullptr;
        while (true) {
            VectorLocation currentLocation = INVALID_VECTOR_LOCATION;
            parent = container_entry->ReadParentEntry(currentLocation);
            if (parent == nullptr) {
                FatalAssert(currentLocation == INVALID_VECTOR_LOCATION, LOG_TAG_DIVFTREE,
                            "null parent with valid location!");
                /* In this case, we are compacting the root and we do not need to expand the tree */
                break;
            }

            CHECK_NOT_NULLPTR(parent, LOG_TAG_DIVFTREE);
            FatalAssert(currentLocation != INVALID_VECTOR_LOCATION, LOG_TAG_DIVFTREE,
                        "null parent with valid location!");
            DIVFTreeVertex* parent_vertex =
                static_cast<DIVFTreeVertex*>(parent->ReadLatestVersion(false));
            CHECK_NOT_NULLPTR(parent_vertex, LOG_TAG_DIVFTREE);
            FatalAssert(parent_vertex->cluster.header.visible_size.load(std::memory_order_acquire) >
                        currentLocation.detail.entryOffset, LOG_TAG_DIVFTREE, "current location offset out of bound!");
            FatalAssert(currentLocation.detail.containerId == parent->centroidMeta.selfId, LOG_TAG_DIVFTREE,
                        "current location containerId mismatch with parent id!");
            FatalAssert(currentLocation.detail.containerVersion == parent_vertex->attr.version, LOG_TAG_DIVFTREE,
                        "current location version mismatch with parent version!");
            SANITY_CHECK(
                CentroidMetaData* selfMeta = reinterpret_cast<CentroidMetaData*>(parent_vertex->cluster.MetaData(
                    currentLocation.detail.entryOffset, false,
                    parent_vertex->attr.block_size, parent_vertex->attr.cap, attr.dimension));
                FatalAssert(selfMeta->id == container_entry->centroidMeta.selfId, LOG_TAG_DIVFTREE,
                            "current location id mismatch with container id!");
                FatalAssert(selfMeta->state.load(std::memory_order_acquire) == VECTOR_STATE_VALID,
                            LOG_TAG_DIVFTREE, "current location is not valid!");
                FatalAssert(selfMeta->version == container_entry->currentVersion, LOG_TAG_DIVFTREE,
                            "current location version mismatch with container version!")
            );

            ConstVectorBatch centroid(1);
            centroid.version = &nextVersion;
            centroid.id = &container_entry->centroidMeta.selfId;
            centroid.data = parent_vertex->cluster.Data(
                currentLocation.detail.entryOffset, false,
                parent_vertex->attr.block_size, parent_vertex->attr.cap, attr.dimension);
            RetStatus rs = BatchInsertInto(parent, centroid, false, currentLocation.detail.entryOffset);
            if (!rs.IsOK()) {
                bufferMgr->ReleaseBufferEntry(parent, ReleaseBufferEntryFlags(false, false));
                parent = nullptr;
                continue;
            }

            break;
        }

#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE, "compacted vertex id: " VECTORID_LOG_FMT
                "ver: %u, old vertex: (%p)%s, new vertex: (%p)%s", VECTORID_LOG(container_entry->centroidMeta.selfId),
                container_entry->currentVersion, current, current->ToString(true).ToCStr(),
                compacted, compacted->ToString(true).ToCStr());
#endif
        if (parent != nullptr) {
            threadSelf->SanityCheckLockHeldInModeByMe(&parent->clusterLock, SX_SHARED);
            container_entry->UpdateClusterPtr(compacted, nextVersion, false);
            bufferMgr->ReleaseBufferEntry(parent, ReleaseBufferEntryFlags(false, false));
            parent = nullptr;
        } else {
            container_entry->UpdateClusterPtr(compacted, nextVersion, true);
        }

        /* todo: can we merge these two loops together? */
        // for (uint16_t i = 0; i < old_reserved; ++i) {
        for (uint16_t i = 0; i < new_size; ++i) {
            SANITY_CHECK(
                if (i < old_reserved) {
                    if (is_leaf) {
                        VectorMetaData* vmd = reinterpret_cast<VectorMetaData*>(destMetaData);
                        FatalAssert(vmd[i].id != marked_for_update_id, LOG_TAG_DIVFTREE,
                                    "marked for update vector found in old reserved during compaction!");
                    } else {
                        CentroidMetaData* vmd = reinterpret_cast<CentroidMetaData*>(destMetaData);
                        FatalAssert(vmd[i].id != marked_for_update_id, LOG_TAG_DIVFTREE,
                                    "marked for update vector found in old reserved during compaction!");
                    }
                }
            );
            // if (offsets[i] == i) {
            //     continue;
            // }
            // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE_VERTEX, "UpdateVectorLocation in compaction:");
            if (is_leaf) {
                VectorMetaData* vmd = reinterpret_cast<VectorMetaData*>(destMetaData);
                bufferMgr->
                    UpdateVectorLocation(vmd[i].id, VectorLocation(container_entry->centroidMeta.selfId,
                                                                   nextVersion, i));
                // bufferMgr->UpdateVectorLocationOffset(vmd[i].id, i);
            } else {
                CentroidMetaData* vmd = reinterpret_cast<CentroidMetaData*>(destMetaData);
                bufferMgr->
                    UpdateVectorLocation(vmd[i].id, VectorLocation(container_entry->centroidMeta.selfId,
                                                                   nextVersion, i));
                // bufferMgr->UpdateVectorLocationOffset(vmd[i].id, i);
            }
        }

        // for (uint16_t i = old_reserved; i < new_size; ++i) {
        //     if (is_leaf) {
        //         VectorMetaData* vmd = reinterpret_cast<VectorMetaData*>(destMetaData);
        //         // if (vmd[i].id == marked_for_update_id && i == marked_for_update) {
        //         //     continue;
        //         // }
        //         // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE_VERTEX, "UpdateVectorLocation in compaction insert:");
        //         bufferMgr->UpdateVectorLocation(vmd[i].id,
        //             VectorLocation(container_entry->centroidMeta.selfId, nextVersion, i));
        //     } else {
        //         CentroidMetaData* vmd = reinterpret_cast<CentroidMetaData*>(destMetaData);
        //         // if (vmd[i].id == marked_for_update_id && i == marked_for_update) {
        //         //     continue;
        //         // }
        //         // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE_VERTEX, "UpdateVectorLocation in compaction insert:");
        //         bufferMgr->UpdateVectorLocation(vmd[i].id,
        //             VectorLocation(container_entry->centroidMeta.selfId, nextVersion, i));
        //     }
        // }

        if (marked_for_update_meta != INVALID_ADDRESS) {
            FatalAssert(!is_leaf, LOG_TAG_DIVFTREE,
                        "marked for update vector metadata should be in internal node!");
            CentroidMetaData* vmd = reinterpret_cast<CentroidMetaData*>(marked_for_update_meta);
            vmd->state.store(VECTOR_STATE_OUTDATED, std::memory_order_release);
            current->Unpin();
        }

        current = nullptr;

        SANITY_CHECK({
            for (uint16_t i = 0; i < batch.size; ++i) {
                VectorLocation loc = bufferMgr->LoadCurrentVectorLocation(batch.id[i], false);
                if (dup_state[i] == DuplicateState::UNIQUE) {
                    FatalAssert(loc.detail.containerId == container_entry->centroidMeta.selfId,
                                LOG_TAG_DIVFTREE, "Vector location container ID mismatch after insertion!");
                    FatalAssert(loc.detail.containerVersion == nextVersion,
                                LOG_TAG_DIVFTREE, "Vector location container version mismatch after insertion!");
                    FatalAssert((loc.detail.entryOffset < new_size) && (loc.detail.entryOffset >= old_reserved),
                                LOG_TAG_DIVFTREE, "Vector location offset out of bounds after insertion!");
                } else {
                    FatalAssert(loc == INVALID_VECTOR_LOCATION,
                                LOG_TAG_DIVFTREE, "Duplicate vector has valid location after insertion!");
                }
            }
        });

        return RetStatus::Success();
    }

    inline bool CompactionCheck(VectorID target) {
        CHECK_VECTORID_IS_VALID(target, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_CENTROID(target, LOG_TAG_DIVFTREE);

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

        BufferVertexEntry* entry = bufferMgr->ReadBufferEntry(target, SX_EXCLUSIVE);
        if (entry == nullptr) {
            return false;
        }

        DIVFTreeVertex* cluster = static_cast<DIVFTreeVertex*>(entry->ReadLatestVersion(false));
        CHECK_NOT_NULLPTR(cluster, LOG_TAG_DIVFTREE);

        if (cluster->cluster.header.num_deleted.load(std::memory_order_relaxed) == 0) {
            bufferMgr->RemoveCompactionTask(target, entry);
            bufferMgr->ReleaseBufferEntry(entry, ReleaseBufferEntryFlags(false, false));
            return false;
        }

        entry->state.store(CLUSTER_FULL, std::memory_order_release);
        VectorBatch dummy;
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE, "BGCompaction triggered for " VECTORID_LOG_FMT,
                VECTORID_LOG(target));
        RetStatus rs = CompactAndInsert(entry, dummy);
        bufferMgr->RemoveCompactionTask(target, entry);
        bufferMgr->ReleaseBufferEntry(entry, ReleaseBufferEntryFlags(true, true));

        return rs.IsOK();
    }

    inline void RoundRobinClustering(const DIVFTreeVertex* base, const ConstVectorBatch& batch,
                                     DuplicateState* dup_state, uint16_t num_duplicates,
                                     BufferVertexEntry**& entries, DIVFTreeVertex**& clusters, VectorBatch& centroids,
                                     uint16_t marked_for_update = INVALID_OFFSET) {
        UNUSED_VARIABLE(dup_state);
        UNUSED_VARIABLE(num_duplicates);
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_NOT_IMPLEMENTED,
                "RoundRobinClustering is not implemented anymore, please use KMeansClustering!");
        FatalAssert(base->attr.index == this, LOG_TAG_DIVFTREE, "index mismatch!");
        uint16_t num_vectors = base->cluster.header.reserved_size.load(std::memory_order_relaxed) -
                               base->cluster.header.num_deleted.load(std::memory_order_relaxed) + batch.size -
                               (marked_for_update != INVALID_OFFSET ? 1 : 0);
        const bool is_leaf = base->attr.centroid_id.IsLeaf();
        centroids.size = (is_leaf ? attr.split_leaf : attr.split_internal);
        if (num_vectors / centroids.size < base->attr.min_size) {
            centroids.size = num_vectors / base->attr.min_size + ((num_vectors % base->attr.min_size) > 0);
        }

        if (centroids.size < 2) {
            centroids.size = 2;
        }

        centroids.data = new VTYPE[attr.dimension * centroids.size];
        centroids.version = new Version[centroids.size];
        centroids.id = new VectorID[centroids.size];
        entries = new BufferVertexEntry*[centroids.size];
        clusters = new DIVFTreeVertex*[centroids.size];

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_CLUSTERING);
        centroids.id[0] = base->attr.centroid_id;
        centroids.version[0] = base->attr.version.NextSplit();
        clusters[0] = new (bufferMgr->AllocateMemoryForVertex(centroids.id[0]._level))
            DIVFTreeVertex(DIVFTreeVertexAttributes(centroids.id[0], centroids.version[0], base->attr.min_size,
                                                    base->attr.cap, base->attr.block_size, base->attr.index));
        CHECK_NOT_NULLPTR(clusters[0], LOG_TAG_CLUSTERING);
        entries[0] = nullptr;
        DIVFTreeVertexInterface** clusterMemories = reinterpret_cast<DIVFTreeVertexInterface**>(&clusters[1]);
        bufferMgr->BatchCreateBufferEntry(centroids.size - 1, base->attr.centroid_id._level, &entries[1],
                                          clusterMemories, &centroids.id[1], &centroids.version[1]);

        for (uint16_t i = 1; i < centroids.size; ++i) {
            CHECK_NOT_NULLPTR(clusters[i], LOG_TAG_CLUSTERING);
            CHECK_NOT_NULLPTR(entries[i], LOG_TAG_CLUSTERING);
            new (clusters[i]) DIVFTreeVertex(
                DIVFTreeVertexAttributes(centroids.id[i], centroids.version[i], base->attr.min_size,
                                         base->attr.cap, base->attr.block_size, base->attr.index));
        }

        FatalAssert(base->cluster.NumBlocks(base->attr.block_size, base->attr.cap) == 1,
                    LOG_TAG_NOT_IMPLEMENTED, "Currently cannot handle more than one block!");
        const VTYPE* vectors = base->cluster.Data(0, is_leaf, base->attr.block_size, base->attr.cap, attr.dimension);
        const void* meta = base->cluster.MetaData(0, is_leaf, base->attr.block_size, base->attr.cap, attr.dimension);
        uint16_t num_seen = 0;
        for (uint16_t i = 0; i < base->attr.cap; ++i) {
            if (i == marked_for_update) {
                continue;
            }

            const uint16_t destSize =
                clusters[num_seen % centroids.size]->cluster.header.reserved_size.load(std::memory_order_relaxed);
            void* destMeta =
                clusters[num_seen % centroids.size]->cluster.MetaData(destSize, is_leaf, base->attr.block_size,
                                                                    base->attr.cap, attr.dimension);
            VTYPE* destData =
                clusters[num_seen % centroids.size]->cluster.Data(destSize, is_leaf, base->attr.block_size,
                                                                base->attr.cap, attr.dimension);
            if (is_leaf) {
                const VectorMetaData* vmt = reinterpret_cast<const VectorMetaData*>(meta);
                VectorMetaData* destVmt = reinterpret_cast<VectorMetaData*>(destMeta);
                if (vmt[i].state.load(std::memory_order_relaxed) != VECTOR_STATE_VALID) {
                    continue;
                }
                memcpy(destData, &vectors[i * attr.dimension], sizeof(VTYPE) * attr.dimension);
                destVmt->id = vmt[i].id;
                destVmt->state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
            } else {
                const CentroidMetaData* vmt = reinterpret_cast<const CentroidMetaData*>(meta);
                CentroidMetaData* destVmt = reinterpret_cast<CentroidMetaData*>(destMeta);
                if (vmt[i].state.load(std::memory_order_relaxed) != VECTOR_STATE_VALID) {
                    continue;
                }
                memcpy(destData, &vectors[i * attr.dimension], sizeof(VTYPE) * attr.dimension);
                destVmt->id = vmt[i].id;
                destVmt->version = vmt[i].version;
                destVmt->state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
            }
            clusters[num_seen % centroids.size]->cluster.header.reserved_size.store(destSize + 1,
                                                                                    std::memory_order_relaxed);
            ++num_seen;
        }

        for (uint16_t i = 0; i < batch.size; ++i) {
            const uint16_t destSize =
                clusters[num_seen % centroids.size]->cluster.header.reserved_size.load(std::memory_order_relaxed);
            void* destMeta =
                clusters[num_seen % centroids.size]->cluster.MetaData(destSize, is_leaf, base->attr.block_size,
                                                                    base->attr.cap, attr.dimension);
            VTYPE* destData =
                clusters[num_seen % centroids.size]->cluster.Data(destSize, is_leaf, base->attr.block_size,
                                                                base->attr.cap, attr.dimension);
            if (is_leaf) {
                VectorMetaData* destVmt = reinterpret_cast<VectorMetaData*>(destMeta);
                memcpy(destData, &(batch.data[i * attr.dimension]), sizeof(VTYPE) * attr.dimension);
                destVmt->id = batch.id[i];
                destVmt->state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
            } else {
                CentroidMetaData* destVmt = reinterpret_cast<CentroidMetaData*>(destMeta);
                memcpy(destData, &(batch.data[i * attr.dimension]), sizeof(VTYPE) * attr.dimension);
                destVmt->id = batch.id[i];
                destVmt->version = batch.version[i];
                destVmt->state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
                bufferMgr->PinVertexVersion(batch.id[i], batch.version[i]);
            }
            clusters[num_seen % centroids.size]->cluster.header.reserved_size.store(destSize + 1,
                                                                                    std::memory_order_relaxed);
            ++num_seen;
        }

        /* Now we have to compute centroids and update num visible for each cluster and do a sanityt check for their size */
        for (uint16_t i = 0; i < centroids.size; ++i) {
            uint16_t size = clusters[i]->cluster.header.reserved_size.load(std::memory_order_relaxed);
            FatalAssert((base->attr.min_size <= size) && (size <= base->attr.cap), LOG_TAG_CLUSTERING, "invalid size!");
            clusters[i]->cluster.header.num_deleted.store(0, std::memory_order_relaxed);
            clusters[i]->cluster.header.visible_size.store(size, std::memory_order_relaxed);

            bool res = ComputeCentroid(clusters[i]->cluster.Data(0, is_leaf, base->attr.block_size,
                                                                 base->attr.cap, attr.dimension),
                                       size, nullptr, 0, attr.dimension, attr.distanceAlg,
                                       &centroids.data[i * attr.dimension], threadSelf->centroid_compute_buffer);
            UNUSED_VARIABLE(res);
            FatalAssert(res, LOG_TAG_CLUSTERING, "cluster is empty!");
        }

    }

    inline bool ChooseDataAsCentroid(const VTYPE* cluster_vectors, const VTYPE* batch_vectors,
                                     uint16_t* cluster_offsets, uint16_t* batch_offsets,
                                     uint16_t cluster_size, uint16_t batch_size,
                                     uint16_t cluster_total_size, uint16_t* cluster_valid_offsets, VTYPE* centroids,
                                     uint16_t target_centroid_idx, uint16_t new_sibling_idx) {
        FatalAssert(cluster_size + batch_size >= 2, LOG_TAG_DIVFTREE, "There should be at least two vectors!");
        uint16_t batch_offset = 0;
        bool chosen = false;
        UNUSED_VARIABLE(cluster_total_size);
        if (cluster_size == 0) {
            memcpy(&centroids[target_centroid_idx * attr.dimension],
                   &batch_vectors[batch_offsets[0] * attr.dimension], sizeof(VTYPE) * attr.dimension);
            // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_CLUSTERING, "Chosen centroid %hu from batch -> offset:%u",
            //         target_centroid_idx, batch_offsets[0] + cluster_total_size);
            batch_offset = 1;
        } else {
            memcpy(&centroids[target_centroid_idx * attr.dimension],
                   &cluster_vectors[cluster_valid_offsets[cluster_offsets[0]] * attr.dimension],
                   sizeof(VTYPE) * attr.dimension);
            // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_CLUSTERING, "Chosen centroid %hu from cluster -> offset %u",
                    // target_centroid_idx, cluster_valid_offsets[cluster_offsets[0]]);
            for (uint16_t i = 1; i < cluster_size; ++i) {
                if (memcmp(&cluster_vectors[cluster_valid_offsets[cluster_offsets[i]] * attr.dimension],
                           &centroids[target_centroid_idx * attr.dimension], sizeof(VTYPE) * attr.dimension) != 0) {
                    memcpy(&centroids[new_sibling_idx * attr.dimension],
                           &cluster_vectors[cluster_valid_offsets[cluster_offsets[i]] * attr.dimension],
                           sizeof(VTYPE) * attr.dimension);
                    // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_CLUSTERING, "Chosen centroid %hu from cluster -> offset %hu",
                    //         new_sibling_idx, cluster_valid_offsets[cluster_offsets[i]]);
                    chosen = true;
                    break;
                }
            }
        }

        if (chosen) {
            return true;
        }

        for (uint16_t i = batch_offset; i < batch_size; ++i) {
            if (memcmp(&batch_vectors[batch_offsets[i] * attr.dimension],
                       &centroids[target_centroid_idx * attr.dimension], sizeof(VTYPE) * attr.dimension) != 0) {
                memcpy(&centroids[new_sibling_idx * attr.dimension],
                       &batch_vectors[batch_offsets[i] * attr.dimension], sizeof(VTYPE) * attr.dimension);
                chosen = true;
                // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_CLUSTERING, "Chosen centroid %hu from batch -> offset %hu",
                //         new_sibling_idx, batch_offsets[i] + cluster_total_size);
                break;
            }
        }

        return chosen;
    }

    inline void TwoMeansDetail(const VTYPE* cluster_vectors, const VTYPE* batch_vectors,
                               uint16_t* cluster_offsets, uint16_t* batch_offsets,
                               uint16_t cluster_size, uint16_t batch_size,
                               uint16_t cluster_total_size,
                               uint16_t* cluster_valid_offsets, VTYPE* centroids,
                               uint16_t target_centroid_idx, uint16_t new_sibling_idx,
                               size_t max_iterations,
                               uint16_t* new_cluster_size, uint16_t* assignments) {
        for (size_t iteration = 0; iteration < max_iterations && max_iterations != 0; ++iteration) {
            /* Assignment Step */
            if (iteration == 0) {
                bool chosen = ChooseDataAsCentroid(cluster_vectors, batch_vectors,
                                                   cluster_offsets, batch_offsets,
                                                   cluster_size, batch_size,
                                                   cluster_total_size,
                                                   cluster_valid_offsets, centroids,
                                                   target_centroid_idx, new_sibling_idx);
                FatalAssert(chosen, LOG_TAG_CLUSTERING, "Cannot choose distinct initial centroids!");
            } else {
                ComputeCentroids(cluster_vectors, batch_vectors,
                                 cluster_offsets, batch_offsets,
                                 cluster_size, batch_size,
                                 cluster_total_size,
                                 cluster_valid_offsets, centroids,
                                 target_centroid_idx, new_sibling_idx,
                                 assignments, new_cluster_size,
                                 attr.dimension, threadSelf->centroid_compute_buffer, attr.distanceAlg);
                // String first_cent_log = "";
                // String second_cent_log = "";
                // for (uint16_t d = 0; d < attr.dimension; ++d) {
                //     first_cent_log += String(DTYPE_FMT ", ", centroids[target_centroid_idx * attr.dimension + d]);
                //     second_cent_log += String(DTYPE_FMT ", ", centroids[new_sibling_idx * attr.dimension + d]);
                // }
                // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_CLUSTERING,
                //         "centroid %hu updated to [%s] with %hu vectors assigned",
                //         target_centroid_idx, first_cent_log.ToCStr(), new_cluster_size[target_centroid_idx]);
                // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_CLUSTERING,
                //         "centroid %hu updated to [%s] with %hu vectors assigned",
                //         new_sibling_idx, second_cent_log.ToCStr(), new_cluster_size[new_sibling_idx]);
            }

            new_cluster_size[target_centroid_idx] = 0;
            new_cluster_size[new_sibling_idx] = 0;
            bool converged = (iteration != 0);
            for (uint16_t i = 0; i < cluster_size; ++i) {
                uint16_t old_assignment = assignments[cluster_offsets[i]];
                FatalAssert(iteration == 0 ||
                            old_assignment == target_centroid_idx || old_assignment == new_sibling_idx,
                            LOG_TAG_DIVFTREE, "invalid old assignment!");
                DTYPE target_distance =
                    Distance(&cluster_vectors[cluster_valid_offsets[cluster_offsets[i]] * attr.dimension],
                             &centroids[target_centroid_idx * attr.dimension], attr.dimension, attr.distanceAlg);
                DTYPE new_sibling_distance =
                    Distance(&cluster_vectors[cluster_valid_offsets[cluster_offsets[i]] * attr.dimension],
                             &centroids[new_sibling_idx * attr.dimension], attr.dimension, attr.distanceAlg);
                if (MoreSimilar(new_sibling_distance, target_distance, attr.distanceAlg) == 1) {
                    // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_CLUSTERING,
                    //         "Vector %hu is assigned to centroid %u(dist:" DTYPE_FMT ") over centroid %u(dist:" DTYPE_FMT ")",
                    //         cluster_offsets[i], new_sibling_idx, new_sibling_distance,
                    //         target_centroid_idx, target_distance);
                    assignments[cluster_offsets[i]] = new_sibling_idx;
                    ++new_cluster_size[new_sibling_idx];
                } else {
                    // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_CLUSTERING,
                    //         "Vector %hu is assigned to centroid %u(dist:" DTYPE_FMT ") over centroid %u(dist:" DTYPE_FMT ")",
                    //         cluster_offsets[i], target_centroid_idx, target_distance,
                    //         new_sibling_idx, new_sibling_distance);
                    assignments[cluster_offsets[i]] = target_centroid_idx;
                    ++new_cluster_size[target_centroid_idx];
                }

                if (assignments[cluster_offsets[i]] != old_assignment) {
                    // DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_CLUSTERING,
                    //         "Vector %hu changed assignment from %hu to %hu",
                    //         cluster_offsets[i], old_assignment,
                    //         assignments[cluster_offsets[i]]);
                    converged = false;
                }
            }

            for (uint16_t i = 0; i < batch_size; ++i) {
                uint16_t old_assignment = assignments[batch_offsets[i] + cluster_total_size];
                FatalAssert(iteration == 0 ||
                            old_assignment == target_centroid_idx || old_assignment == new_sibling_idx,
                            LOG_TAG_DIVFTREE, "invalid old assignment!");
                DTYPE target_distance =
                    Distance(&batch_vectors[batch_offsets[i] * attr.dimension],
                             &centroids[target_centroid_idx * attr.dimension], attr.dimension, attr.distanceAlg);
                DTYPE new_sibling_distance =
                    Distance(&batch_vectors[batch_offsets[i] * attr.dimension],
                             &centroids[new_sibling_idx * attr.dimension], attr.dimension, attr.distanceAlg);
                if (MoreSimilar(new_sibling_distance, target_distance, attr.distanceAlg) == 1) {
                    // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_CLUSTERING,
                    //         "Vector %hu is assigned to centroid %u(dist:" DTYPE_FMT ") over centroid %u(dist:" DTYPE_FMT ")",
                    //         batch_offsets[i] + cluster_total_size,
                    //         new_sibling_idx, new_sibling_distance,
                    //         target_centroid_idx, target_distance);
                    assignments[batch_offsets[i] + cluster_total_size] = new_sibling_idx;
                    ++new_cluster_size[new_sibling_idx];
                } else {
                    // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_CLUSTERING,
                    //         "Vector %hu is assigned to centroid %u(dist:" DTYPE_FMT ") over centroid %u(dist:" DTYPE_FMT ")",
                    //         batch_offsets[i] + cluster_total_size,
                    //         target_centroid_idx, target_distance,
                    //         new_sibling_idx, new_sibling_distance);
                    assignments[batch_offsets[i] + cluster_total_size] = target_centroid_idx;
                    ++new_cluster_size[target_centroid_idx];
                }

                if (assignments[batch_offsets[i] + cluster_total_size] != old_assignment) {
                    // DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_CLUSTERING,
                    //         "Vector %hu changed assignment from %hu to %hu",
                    //         batch_offsets[i] + cluster_total_size, old_assignment,
                    //         assignments[batch_offsets[i] + cluster_total_size]);
                    converged = false;
                }
            }

            if (converged) {
                break;
            }
        }
    }

    inline void KMeans(const DIVFTreeVertex* base, const ConstVectorBatch& batch,
                       DuplicateState* dup_state, uint16_t num_duplicates, BufferVertexEntry**& entries,
                       DIVFTreeVertex**& clusters, VectorBatch& centroids,
                       size_t max_iterations = 20, /* 0 means until convergence */
                       uint16_t marked_for_update = INVALID_OFFSET) {
        FatalAssert(base->attr.index == this, LOG_TAG_DIVFTREE, "index mismatch!");
        FatalAssert(attr.split_leaf == 2 && attr.split_internal == 2, LOG_TAG_NOT_IMPLEMENTED,
                    "KMeans currently only supports binary split!");
        uint16_t num_centroids = attr.split_leaf;
        uint16_t cluster_size = base->cluster.header.reserved_size.load(std::memory_order_relaxed);
        uint16_t num_total_vectors = cluster_size + batch.size;
        uint16_t num_vectors = num_total_vectors - base->cluster.header.num_deleted.load(std::memory_order_relaxed) -
                               num_duplicates - (marked_for_update != INVALID_OFFSET ? 1 : 0);

        const bool is_leaf = base->attr.centroid_id.IsLeaf();
        centroids.size = (is_leaf ? attr.split_leaf : attr.split_internal);
        uint16_t expected_max_centroid = std::max((uint16_t)(num_vectors / 10), num_centroids) + 1;

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_CLUSTERING);

        centroids.data = nullptr;
        centroids.version = nullptr;
        centroids.id = nullptr;
        entries = nullptr;
        clusters = nullptr;
        centroids.Resize(expected_max_centroid, false, attr.dimension);
        centroids.size = num_centroids;
        uint16_t batch_size = batch.size - num_duplicates;
        uint16_t* batch_offsets = new uint16_t[batch_size];
        uint16_t valid_off_idx = 0;
        for (uint16_t i = 0; i < batch.size; ++i) {
            if (dup_state[i] == DuplicateState::UNIQUE) {
                FatalAssert(valid_off_idx < batch_size, LOG_TAG_DIVFTREE, "batch size mismatch!");
                batch_offsets[valid_off_idx] = i;
                ++valid_off_idx;
            } else {
                FatalAssert(is_leaf, LOG_TAG_DIVFTREE, "Invalid duplicate state in KMeans clustering!");
                FatalAssert(dup_state[i] == DuplicateState::DUPLICATE_NOT_INSERTED, LOG_TAG_DIVFTREE,
                            "Invalid duplicate state in KMeans clustering!");
                bufferMgr->UpdateVectorLocation(batch.id[i], INVALID_VECTOR_LOCATION, true);
                FatalAssert(bufferMgr->LoadCurrentVectorLocation(batch.id[i], false) == INVALID_VECTOR_LOCATION,
                            LOG_TAG_DIVFTREE, "Failed to invalidate duplicate vector location!");
            }
        }
        FatalAssert(valid_off_idx == batch_size, LOG_TAG_DIVFTREE, "batch size mismatch!");

        ResizeArray(entries, 0, expected_max_centroid);
        ResizeArray(clusters, 0, expected_max_centroid);
        uint16_t* assignments = new uint16_t[num_total_vectors];
        uint16_t* cluster_valid_offsets = new uint16_t[cluster_size];
        uint16_t* cluster_offsets = new uint16_t[cluster_size];
        uint16_t* new_cluster_size = nullptr;
        ResizeArray(new_cluster_size, 0, expected_max_centroid);

        centroids.id[0] = base->attr.centroid_id;
        centroids.version[0] = base->attr.version.NextSplit();
        clusters[0] = new (bufferMgr->AllocateMemoryForVertex(centroids.id[0]._level))
            DIVFTreeVertex(DIVFTreeVertexAttributes(centroids.id[0], centroids.version[0], base->attr.min_size,
                                                    base->attr.cap, base->attr.block_size, base->attr.index));
        CHECK_NOT_NULLPTR(clusters[0], LOG_TAG_CLUSTERING);
        entries[0] = nullptr;
        DIVFTreeVertexInterface** clusterMemories = reinterpret_cast<DIVFTreeVertexInterface**>(&clusters[1]);
        bufferMgr->BatchCreateBufferEntry(centroids.size - 1, base->attr.centroid_id._level, &entries[1],
                                          clusterMemories, &centroids.id[1], &centroids.version[1]);

        for (uint16_t i = 1; i < centroids.size; ++i) {
            CHECK_NOT_NULLPTR(clusters[i], LOG_TAG_CLUSTERING);
            CHECK_NOT_NULLPTR(entries[i], LOG_TAG_CLUSTERING);
            new (clusters[i]) DIVFTreeVertex(
                DIVFTreeVertexAttributes(centroids.id[i], centroids.version[i], base->attr.min_size,
                                         base->attr.cap, base->attr.block_size, base->attr.index));
        }

        FatalAssert(base->cluster.NumBlocks(base->attr.block_size, base->attr.cap) == 1,
                    LOG_TAG_NOT_IMPLEMENTED, "Currently cannot handle more than one block!");
        const VTYPE* vectors = base->cluster.Data(0, is_leaf, base->attr.block_size, base->attr.cap, attr.dimension);
        const void* meta = base->cluster.MetaData(0, is_leaf, base->attr.block_size, base->attr.cap, attr.dimension);
        uint16_t num_seen = 0;
        for (uint16_t i = 0; i < base->attr.cap; ++i) {
            if (i == marked_for_update) {
                continue;
            }
            if (is_leaf) {
                const VectorMetaData* vmt = reinterpret_cast<const VectorMetaData*>(meta);
                if (vmt[i].state.load(std::memory_order_relaxed) != VECTOR_STATE_VALID) {
                    continue;
                }
            } else {
                const CentroidMetaData* vmt = reinterpret_cast<const CentroidMetaData*>(meta);
                if (vmt[i].state.load(std::memory_order_relaxed) != VECTOR_STATE_VALID) {
                    continue;
                }
            }

            cluster_valid_offsets[num_seen] = i;
            cluster_offsets[num_seen] = num_seen;
            // String vecLog = "";
            // for (uint16_t d = 0; d < attr.dimension; ++d) {
            //     vecLog += String(DTYPE_FMT ", ", vectors[i * attr.dimension + d]);
            // }
            // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_CLUSTERING,
            //             "vector[%hu] from cluster -> offset = %hu = [%s]",
            //             num_seen, i, vecLog.ToCStr());

            ++num_seen;
        }

        TwoMeansDetail(vectors, batch.data,
                       cluster_offsets, batch_offsets,
                       num_seen, batch_size,
                       num_seen,
                       cluster_valid_offsets, centroids.data,
                       0, 1,
                       max_iterations,
                       new_cluster_size, assignments);

        std::vector<uint16_t> need_split_centroids;
        for (uint16_t i = 0; i < centroids.size; ++i) {
            if ((float)new_cluster_size[i] * COMPACTION_FACTOR > (float)base->attr.cap) {
                need_split_centroids.push_back(i);
            }
        }

        /* todo: stat collection */
        if (need_split_centroids.size() > 0) {
            uint16_t* curr_batch_offsets = new uint16_t[batch_size];
            while (need_split_centroids.size() > 0) {
                uint16_t centroid_to_split = need_split_centroids.back();
                if (centroids.size >= expected_max_centroid) {
                    uint16_t old_size = expected_max_centroid;
                    expected_max_centroid *= 2;
                    centroids.Resize(expected_max_centroid, false, attr.dimension);
                    ResizeArray(entries, old_size, expected_max_centroid);
                    ResizeArray(clusters, old_size, expected_max_centroid);
                    ResizeArray(new_cluster_size, old_size, expected_max_centroid);
                }
                clusterMemories = reinterpret_cast<DIVFTreeVertexInterface**>(&clusters[centroids.size]);
                bufferMgr->BatchCreateBufferEntry(1, base->attr.centroid_id._level, &entries[centroids.size],
                                                    clusterMemories, &centroids.id[centroids.size],
                                                    &centroids.version[centroids.size]);
                new (clusters[centroids.size]) DIVFTreeVertex(
                    DIVFTreeVertexAttributes(centroids.id[centroids.size], centroids.version[centroids.size],
                                            base->attr.min_size, base->attr.cap, base->attr.block_size,
                                            base->attr.index));
                need_split_centroids.pop_back();
                // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_CLUSTERING,
                //         "Centroid %hu needs to be split with %hu vectors assigned",
                //         centroid_to_split, new_cluster_size[centroid_to_split]);
                uint16_t num_vectors_in_cluster = 0;
                uint16_t num_vectors_in_batch = 0;
                for (uint16_t i = 0; i < num_seen; ++i) {
                    if (assignments[i] == centroid_to_split) {
                        cluster_offsets[num_vectors_in_cluster] = i;
                        ++num_vectors_in_cluster;
                    }
                }
                for (uint16_t i = 0; i < batch_size; ++i) {
                    if (assignments[batch_offsets[i] + num_seen] == centroid_to_split) {
                        curr_batch_offsets[num_vectors_in_batch] = batch_offsets[i];
                        ++num_vectors_in_batch;
                    }
                }
                FatalAssert(num_vectors_in_cluster + num_vectors_in_batch == new_cluster_size[centroid_to_split],
                            LOG_TAG_DIVFTREE, "Inconsistent cluster size!");
                TwoMeansDetail(vectors, batch.data,
                               cluster_offsets, curr_batch_offsets,
                               num_vectors_in_cluster, num_vectors_in_batch,
                               num_seen,
                               cluster_valid_offsets, centroids.data,
                               centroid_to_split, centroids.size,
                               max_iterations,
                               new_cluster_size, assignments);
                if ((float)new_cluster_size[centroids.size] * COMPACTION_FACTOR > (float)base->attr.cap) {
                    need_split_centroids.push_back(centroids.size);
                }
                if ((float)new_cluster_size[centroid_to_split] * COMPACTION_FACTOR > (float)base->attr.cap) {
                    need_split_centroids.push_back(centroid_to_split);
                }
                ++centroids.size;
            }
            delete[] curr_batch_offsets;
        }

        SANITY_CHECK(std::unordered_set<VectorID, VectorIDHash> vectors_in_cluster;);

        for (uint16_t i = 0; i < num_seen; ++i) {
            uint16_t centroid = assignments[i];
            FatalAssert(centroid < centroids.size, LOG_TAG_DIVFTREE, "invalid centroid assignment!");
            uint16_t destSize =
                clusters[centroid]->cluster.header.reserved_size.load(std::memory_order_relaxed);

            void* destMeta =
                clusters[centroid]->cluster.MetaData(destSize, is_leaf, base->attr.block_size,
                                                                    base->attr.cap, attr.dimension);
            VTYPE* destData =
                clusters[centroid]->cluster.Data(destSize, is_leaf, base->attr.block_size,
                                                                base->attr.cap, attr.dimension);
            if (is_leaf) {
                const VectorMetaData* src_vmd = reinterpret_cast<const VectorMetaData*>(meta);
                VectorMetaData* dest_vmd = reinterpret_cast<VectorMetaData*>(destMeta);
                memcpy(destData, &(vectors[cluster_valid_offsets[i] * attr.dimension]), sizeof(VTYPE) * attr.dimension);
                dest_vmd->id = src_vmd[cluster_valid_offsets[i]].id;
                dest_vmd->state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
                SANITY_CHECK({
                    FatalAssert(vectors_in_cluster.find(dest_vmd->id) == vectors_in_cluster.end(),
                                LOG_TAG_DIVFTREE, "Duplicate vector in clustering!");
                    vectors_in_cluster.insert(dest_vmd->id);
                });
            } else {
                const CentroidMetaData* src_vmd = reinterpret_cast<const CentroidMetaData*>(meta);
                CentroidMetaData* dest_vmd = reinterpret_cast<CentroidMetaData*>(destMeta);
                memcpy(destData, &(vectors[cluster_valid_offsets[i] * attr.dimension]), sizeof(VTYPE) * attr.dimension);
                dest_vmd->id = src_vmd[cluster_valid_offsets[i]].id;
                dest_vmd->version = src_vmd[cluster_valid_offsets[i]].version;
                dest_vmd->state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
                SANITY_CHECK({
                    FatalAssert(vectors_in_cluster.find(dest_vmd->id) == vectors_in_cluster.end(),
                                LOG_TAG_DIVFTREE, "Duplicate vector in clustering!");
                    vectors_in_cluster.insert(dest_vmd->id);
                });
            }
            clusters[centroid]->cluster.header.reserved_size.store(destSize + 1, std::memory_order_relaxed);
        }

        for (uint16_t i = 0; i < batch_size; ++i) {
            FatalAssert(batch_offsets[i] < batch.size, LOG_TAG_DIVFTREE, "invalid batch offset!");
            FatalAssert(dup_state[batch_offsets[i]] == DuplicateState::UNIQUE,
                        LOG_TAG_DIVFTREE, "invalid duplicate state!");
            uint16_t centroid = assignments[batch_offsets[i] + num_seen];
            FatalAssert(centroid < centroids.size, LOG_TAG_DIVFTREE, "invalid centroid assignment!");
            uint16_t destSize =
                clusters[centroid]->cluster.header.reserved_size.load(std::memory_order_relaxed);
            void* destMeta =
                clusters[centroid]->cluster.MetaData(destSize, is_leaf, base->attr.block_size,
                                                     base->attr.cap, attr.dimension);
            VTYPE* destData =
                clusters[centroid]->cluster.Data(destSize, is_leaf, base->attr.block_size,
                                                 base->attr.cap, attr.dimension);
            if (is_leaf) {
                VectorMetaData* destVmt = reinterpret_cast<VectorMetaData*>(destMeta);
                memcpy(destData, &(batch.data[batch_offsets[i] * attr.dimension]), sizeof(VTYPE) * attr.dimension);
                destVmt->id = batch.id[batch_offsets[i]];
                destVmt->state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
                SANITY_CHECK({
                    FatalAssert(vectors_in_cluster.find(destVmt->id) == vectors_in_cluster.end(),
                                LOG_TAG_DIVFTREE, "Duplicate vector in clustering!");
                    vectors_in_cluster.insert(destVmt->id);
                });
            } else {
                FatalAssert(batch_offsets[i] == i, LOG_TAG_DIVFTREE,
                            "batch offsets should be identity mapping in non-leaf nodes!");
                CentroidMetaData* destVmt = reinterpret_cast<CentroidMetaData*>(destMeta);
                memcpy(destData, &(batch.data[batch_offsets[i] * attr.dimension]), sizeof(VTYPE) * attr.dimension);
                destVmt->id = batch.id[batch_offsets[i]];
                destVmt->version = batch.version[batch_offsets[i]];
                destVmt->state.store(VECTOR_STATE_VALID, std::memory_order_relaxed);
                bufferMgr->PinVertexVersion(batch.id[batch_offsets[i]], batch.version[batch_offsets[i]]);
                SANITY_CHECK({
                    FatalAssert(vectors_in_cluster.find(destVmt->id) == vectors_in_cluster.end(),
                                LOG_TAG_DIVFTREE, "Duplicate vector in clustering!");
                    vectors_in_cluster.insert(destVmt->id);
                });
            }
            clusters[centroid]->cluster.header.reserved_size.store(destSize + 1, std::memory_order_relaxed);
        }

        /* Now we have to compute centroids and update num visible for each cluster and do a sanityt check for their size */
        for (uint16_t i = 0; i < centroids.size; ++i) {
            uint16_t size = clusters[i]->cluster.header.reserved_size.load(std::memory_order_relaxed);
            FatalAssert(size > 0, LOG_TAG_CLUSTERING, "cluster is empty!");
            FatalAssert(size <= base->attr.cap, LOG_TAG_CLUSTERING, "invalid size!");
            clusters[i]->cluster.header.num_deleted.store(0, std::memory_order_relaxed);
            clusters[i]->cluster.header.visible_size.store(size, std::memory_order_relaxed);
        }

        delete[] assignments;
        delete[] cluster_offsets;
        delete[] cluster_valid_offsets;
        delete[] new_cluster_size;
        delete[] batch_offsets;
    }

    /*
     * will only fill in the raw centroid vectors to
     * the centroids batch and allocates memory for version and ids but does not fill them
     */
    inline void Clustering(const DIVFTreeVertex* base, const ConstVectorBatch& batch, DuplicateState* dup_state,
                           uint16_t num_duplicates, BufferVertexEntry**& entries, DIVFTreeVertex**& clusters,
                           VectorBatch& centroids, uint16_t marked_for_update = INVALID_OFFSET) {
        switch (attr.clusteringAlg)
        {
        case ClusteringType::RoundRobin:
            RoundRobinClustering(base, batch, dup_state, num_duplicates, entries,
                                 clusters, centroids, marked_for_update);
            break;
        case ClusteringType::KMeans:
            KMeans(base, batch, dup_state, num_duplicates, entries, clusters, centroids, 100, marked_for_update);
            break;
        default:
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_DIVFTREE,
                 "Cluster: Invalid clustering type: %hhu", (uint8_t)(attr.clusteringAlg));
        }
    }

    inline BufferVertexEntry* ExpandTree(VectorID expRootId) {
        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);
        /* todo: remove */
        VectorID oldRootId = bufferMgr->GetCurrentRootId();
        UNUSED_VARIABLE(oldRootId);
        BufferVertexEntry* new_root = bufferMgr->CreateNewRootEntry(expRootId);
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE, "Expanding old root id: " VECTORID_LOG_FMT ", new root id: "
                VECTORID_LOG_FMT, VECTORID_LOG(oldRootId), VECTORID_LOG(new_root->centroidMeta.selfId));
        (void)(new (new_root->ReadLatestVersion(false)) DIVFTreeVertex(attr, new_root->centroidMeta.selfId, this));
        new_root->headerLock.Unlock();
        new_root->DowngradeAccessToShared();
        return new_root;
    }

    RetStatus SplitAndInsert(BufferVertexEntry* container_entry, const ConstVectorBatch& batch,
                             DuplicateState* dup_state, uint16_t num_duplicates,
                             uint16_t marked_for_update = INVALID_OFFSET) {
        CHECK_NOT_NULLPTR(container_entry, LOG_TAG_DIVFTREE);
        FatalAssert(batch.data != nullptr, LOG_TAG_DIVFTREE, "Batch of vectors to insert is null.");
        FatalAssert(batch.id != nullptr, LOG_TAG_DIVFTREE, "Batch of vector IDs to insert is null.");
        CHECK_NOT_NULLPTR(dup_state, LOG_TAG_DIVFTREE);
        FatalAssert(num_duplicates <= batch.size, LOG_TAG_DIVFTREE,
                    "Batch size is larger than batch total size.");
        threadSelf->SanityCheckLockHeldInModeByMe(&container_entry->clusterLock, SX_EXCLUSIVE);

        RetStatus rs = RetStatus::Success();
        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

        DIVFTreeVertex* current =
            static_cast<DIVFTreeVertex*>(container_entry->ReadLatestVersion(marked_for_update != INVALID_OFFSET));
        CHECK_NOT_NULLPTR(current, LOG_TAG_DIVFTREE);

#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE, "Splitting vertex (%p)%s",
                current, current->ToString(true).ToCStr());
#else
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE, "Splitting vertex with id: " VECTORID_LOG_FMT " ver:%u",
                VECTORID_LOG(container_entry->centroidMeta.selfId), container_entry->currentVersion);
#endif

        // uint16_t size = current->cluster.header.reserved_size.load(std::memory_order_relaxed);
        const bool is_leaf = container_entry->centroidMeta.selfId.IsLeaf();
        FatalAssert(!is_leaf || (marked_for_update == INVALID_OFFSET), LOG_TAG_DIVFTREE,
                    "Only internal nodes can have marked for update vectors!");
        const uint16_t dim = attr.dimension;
        const uint16_t cap = current->attr.cap;
        const uint16_t blckSize = current->attr.block_size;
        // const uint8_t level = (uint8_t)(container_entry->centroidMeta.selfId._level);
        /* Todo: this will not work if we have more than one block! */
        FatalAssert(current->cluster.NumBlocks(blckSize, cap) == 1,
                    LOG_TAG_NOT_IMPLEMENTED, "Currently cannot handle more than one block!");

        FatalAssert(is_leaf == (batch.version == nullptr), LOG_TAG_DIVFTREE, "Batch of versions to insert is null!");
        // uint16_t total_size = current->cluster.header.reserved_size.load(std::memory_order_relaxed) -
        //                       current->cluster.header.num_deleted.load(std::memory_order_relaxed) +
        //                       batch.size;

        Address marked_for_update_meta = INVALID_ADDRESS;
        if (marked_for_update != INVALID_OFFSET) {
            FatalAssert(marked_for_update < current->cluster.header.reserved_size.load(std::memory_order_relaxed),
                        LOG_TAG_DIVFTREE, "marked for update offset is out of bounds!");
            marked_for_update_meta = current->cluster.MetaData(marked_for_update, is_leaf, blckSize, cap, dim);
        }

        VectorBatch centroids;
        DIVFTreeVertex** clusters = nullptr;
        BufferVertexEntry** entries = nullptr;
        Clustering(current, batch, dup_state, num_duplicates, entries, clusters, centroids, marked_for_update);
        CHECK_NOT_NULLPTR(entries, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(clusters, LOG_TAG_DIVFTREE);
        FatalAssert(centroids.size >= 2, LOG_TAG_DIVFTREE, "numclusters should at least be 2!");
        CHECK_NOT_NULLPTR(centroids.data, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(centroids.id, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(centroids.version, LOG_TAG_DIVFTREE);
#ifdef EXCESS_LOGING
        for (uint16_t i = 0; i < centroids.size; ++i) {
            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE, "new cluster: centroid=%s, data=(%p)%s",
                    VectorToString(&centroids.data[i * attr.dimension], attr.dimension).ToCStr(),
                    clusters[i], clusters[i]->ToString(true).ToCStr());
        }
#endif
        entries[0] = container_entry;

        BufferVertexEntry* parent = nullptr;
        while (true) {
            VectorLocation currentLocation = INVALID_VECTOR_LOCATION;
            parent = container_entry->ReadParentEntry(currentLocation);
            if (parent == nullptr) {
                FatalAssert(currentLocation == INVALID_VECTOR_LOCATION, LOG_TAG_DIVFTREE,
                            "null parent with valid location!");
                /* changing the root and location are not done atomically together hence this assert may fail */
                // FatalAssert(container_entry->centroidMeta.selfId == bufferMgr->GetCurrentRootId(), LOG_TAG_DIVFTREE,
                //             "null parent but we are not root!");
                parent = ExpandTree(container_entry->centroidMeta.selfId);
            } else {
                FatalAssert(currentLocation != INVALID_VECTOR_LOCATION, LOG_TAG_DIVFTREE,
                        "null parent with valid location!");
            }

            CHECK_NOT_NULLPTR(parent, LOG_TAG_DIVFTREE);
            rs = BatchInsertInto(parent, centroids, false, currentLocation.detail.entryOffset);
            if (!rs.IsOK()) {
                bufferMgr->ReleaseBufferEntry(parent, ReleaseBufferEntryFlags(false, false));
                parent = nullptr;
                continue;
            }

            break;
        }

        threadSelf->SanityCheckLockHeldInModeByMe(&parent->clusterLock, SX_SHARED);

        container_entry->UpdateClusterPtr(clusters[0], centroids.version[0]);
        bufferMgr->ReleaseBufferEntry(parent, ReleaseBufferEntryFlags(false, false));
        parent = nullptr;

        for (uint16_t i = 0; i < centroids.size; ++i) {
            uint16_t size = clusters[i]->cluster.header.reserved_size.load(std::memory_order_relaxed);
            FatalAssert(size == clusters[i]->cluster.header.visible_size.load(std::memory_order_relaxed),
                        LOG_TAG_DIVFTREE, "mismatch between sizes when cluster is locked in X mode");
            FatalAssert(clusters[i]->cluster.header.num_deleted.load(std::memory_order_relaxed) == 0,
                        LOG_TAG_DIVFTREE, "num deleted should be 0 at thsi stage!");
            void* meta = clusters[i]->cluster.MetaData(0, is_leaf, blckSize, cap, dim);
            CHECK_NOT_NULLPTR(meta, LOG_TAG_DIVFTREE);
            for (uint16_t offset = 0; offset < size; ++offset) {
                // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE_VERTEX, "UpdateVectorLocation in split:");
                if (is_leaf) {
                    VectorMetaData* vmt = reinterpret_cast<VectorMetaData*>(meta);
                    bufferMgr->UpdateVectorLocation(vmt[offset].id,
                                                    VectorLocation(centroids.id[i], centroids.version[i], offset));
                } else {
                    CentroidMetaData* vmt = reinterpret_cast<CentroidMetaData*>(meta);
                    bufferMgr->UpdateVectorLocation(vmt[offset].id,
                                                    VectorLocation(centroids.id[i], centroids.version[i], offset));
                }
            }
        }

        SANITY_CHECK({
            std::unordered_set<std::pair<VectorID, Version>, VectorIDVersionPairHash> batch_elems;
            std::unordered_set<VectorID, VectorIDHash> batch_elem_ids;
            for (uint16_t i = 0; i < batch.size; ++i) {
                if (is_leaf) {
                    FatalAssert(batch_elems.find(std::make_pair(batch.id[i], Version{0})) ==
                                batch_elems.end(), LOG_TAG_DIVFTREE, "Duplicate vector IDs in batch!");
                    batch_elems.insert(std::make_pair(batch.id[i], Version{0}));
                } else {
                    FatalAssert(batch_elems.find(std::make_pair(batch.id[i], batch.version[i])) ==
                                batch_elems.end(), LOG_TAG_DIVFTREE, "Duplicate vector IDs in batch!");
                    batch_elems.insert(std::make_pair(batch.id[i], batch.version[i]));
                }
                FatalAssert(batch_elem_ids.find(batch.id[i]) == batch_elem_ids.end(),
                            LOG_TAG_DIVFTREE, "Duplicate vector IDs in batch!");
                batch_elem_ids.insert(batch.id[i]);
                VectorLocation loc = bufferMgr->LoadCurrentVectorLocation(batch.id[i], false);
                if (dup_state[i] == DuplicateState::UNIQUE) {
                    bool found = false;
                    for (uint16_t centroid = 0; centroid < centroids.size; ++centroid) {
                        if (loc.detail.containerId == centroids.id[centroid] &&
                            loc.detail.containerVersion == centroids.version[centroid]) {
                            found = true;
                            break;
                        }
                    }
                    FatalAssert(found, LOG_TAG_DIVFTREE, "Inserted vector not found in any new centroid!");
                } else {
                    FatalAssert(loc == INVALID_VECTOR_LOCATION,
                                LOG_TAG_DIVFTREE, "Duplicate vector has valid location after insertion!");
                }
            }
        });

        for (uint16_t i = centroids.size; i > 1; --i) {
            bufferMgr->ReleaseBufferEntry(entries[i-1], ReleaseBufferEntryFlags(true, true));
        }
        // this is released in the caller
        // bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(true, true));

        if (marked_for_update_meta != INVALID_ADDRESS) {
            if (is_leaf) {
                VectorMetaData* vmd = reinterpret_cast<VectorMetaData*>(marked_for_update_meta);
                vmd->state.store(VECTOR_STATE_OUTDATED, std::memory_order_release);
            } else {
                CentroidMetaData* vmd = reinterpret_cast<CentroidMetaData*>(marked_for_update_meta);
                vmd->state.store(VECTOR_STATE_OUTDATED, std::memory_order_release);
            }
            current->Unpin();
            current = nullptr;
        }

        delete[] clusters;
        delete[] entries;
        delete[] centroids.data;
        delete[] centroids.id;
        delete[] centroids.version;

        return rs;
    }

    RetStatus BatchInsertInto(BufferVertexEntry* container_entry, const ConstVectorBatch& batch, bool releaseEntry,
                              uint16_t marked_for_update = INVALID_OFFSET, bool no_major_updates = false) {
        CHECK_NOT_NULLPTR(container_entry, LOG_TAG_DIVFTREE);
        FatalAssert(batch.size != 0, LOG_TAG_DIVFTREE, "Batch of vectors to insert is empty.");
        FatalAssert(batch.data != nullptr, LOG_TAG_DIVFTREE, "Batch of vectors to insert is null.");
        FatalAssert(batch.id != nullptr, LOG_TAG_DIVFTREE, "Batch of vector IDs to insert is null.");
        RetStatus rs = RetStatus::Success();
        uint16_t checked_offset_for_duplicates = UINT16_MAX;
        uint16_t num_duplicates_found = 0;
        DuplicateState* dup_state = new DuplicateState[batch.size];
        memset(dup_state, 0, sizeof(DuplicateState) * batch.size);

        threadSelf->SanityCheckLockHeldInModeByMe(&container_entry->clusterLock, SX_SHARED);
        FatalAssert(container_entry->state.load() != CLUSTER_DELETED, LOG_TAG_DIVFTREE, "Container vertex is deleted.");
        while (true) {
            /*
             * we do not need to pin the cluster here because the version pin should remain at least one as long as we
             * have a lock on it.
             */
            DIVFTreeVertex* container_vertex =
                static_cast<DIVFTreeVertex*>(container_entry->ReadLatestVersion(false));
            CHECK_NOT_NULLPTR(container_vertex, LOG_TAG_DIVFTREE);
            // CHECK_VERTEX_IS_VALID(container_vertex, LOG_TAG_DIVFTREE, RetStatus::FAIL);
            rs = container_vertex->BatchInsert(batch, dup_state, num_duplicates_found, checked_offset_for_duplicates,
                                               marked_for_update);
            if (rs.IsOK() || no_major_updates) {
                break;
            }

            FatalAssert(rs.stat == RetStatus::VERTEX_NOT_ENOUGH_SPACE,
                        LOG_TAG_DIVFTREE, "Batch insert failed with error: %s", rs.Msg());
            BufferVertexEntryState state = CLUSTER_FULL;
            rs = container_entry->UpgradeAccessToExclusive(state);
            if (!rs.IsOK()) {
                FatalAssert(rs.stat == RetStatus::FAILED_TO_CAS_ENTRY_STATE, LOG_TAG_DIVFTREE,
                        "UpgradeAccessToExclusive failed with error: %s", rs.Msg());
                switch(state) {
                case BufferVertexEntryState::CLUSTER_DELETE_IN_PROGRESS: /* Can this happen? */
                case BufferVertexEntryState::CLUSTER_FULL:
                case BufferVertexEntryState::CLUSTER_DELETED: /* Can this happen? */
                    /* it is possible that the update was not applied though! */
                    rs = RetStatus{.stat=RetStatus::VERTEX_UPDATED, .message=nullptr};
                    break;
                default:
                    DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_DIVFTREE, "Invalid entry state!");
                    rs = RetStatus::Fail("Invalid State!");
                    break;
                }
                break;
            }

            FatalAssert((void*)(container_vertex) ==
                        (void*)(container_entry->ReadLatestVersion(false)), LOG_TAG_DIVFTREE,
                        "ClusterPtr has changed during the atomic upgrade from shared to exclusive!");
            FatalAssert(container_vertex->cluster.header.visible_size.load(std::memory_order_relaxed) ==
                        container_vertex->cluster.header.reserved_size.load(std::memory_order_relaxed),
                        LOG_TAG_DIVFTREE, "Some updates did not go through!");
            FatalAssert(container_vertex->cluster.header.reserved_size.load(std::memory_order_relaxed) >=
                        container_vertex->cluster.header.num_deleted.load(std::memory_order_relaxed),
                        LOG_TAG_DIVFTREE, "More vectors are deleted than inserted!");
            uint16_t reserved_size =
                container_vertex->cluster.header.reserved_size.load(std::memory_order_relaxed);

            if (container_entry->centroidMeta.selfId.IsLeaf() && reserved_size != 0) {
                container_vertex->Exists(batch, reserved_size, dup_state, num_duplicates_found,
                                         DuplicateState::DUPLICATE_NOT_INSERTED, checked_offset_for_duplicates);
                checked_offset_for_duplicates = reserved_size - 1;
            }

            uint16_t num_inserted = batch.size - num_duplicates_found;
            reserved_size += num_inserted;
            uint16_t real_vertex_size =
                reserved_size - (container_vertex->cluster.header.num_deleted.load(std::memory_order_relaxed) +
                                 (marked_for_update != INVALID_OFFSET ? 1 : 0));

            if ((container_vertex->cluster.header.num_deleted.load(std::memory_order_relaxed) > 0) &&
                ((float)real_vertex_size * COMPACTION_FACTOR <= (float)container_vertex->attr.cap)) {
                rs = CompactAndInsert(container_entry, batch, dup_state, num_duplicates_found, marked_for_update);
            } else if (reserved_size < container_vertex->attr.cap) {
                container_entry->DowngradeAccessToShared();
                continue;
            } else {
                rs = SplitAndInsert(container_entry, batch, dup_state, num_duplicates_found, marked_for_update);
            }

            FatalAssert(rs.IsOK(), LOG_TAG_DIVFTREE, "Major update failed!");
            if (releaseEntry) {
                BufferManager::GetInstance()->
                    ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(true, true));
            } else {
                container_entry->DowngradeAccessToShared();
            }

            delete[] dup_state;
            return rs;
        }

        SANITY_CHECK({
            if (rs.IsOK()) {
                for (uint16_t i = 0; i < batch.size; ++i) {
                    VectorLocation loc =
                            BufferManager::GetInstance()->LoadCurrentVectorLocation(batch.id[i], false);
                    if (dup_state[i] == DuplicateState::UNIQUE) {
                        // FatalAssert(loc.detail.containerId == container_entry->centroidMeta.selfId,
                        //             LOG_TAG_DIVFTREE, "Vector location container ID mismatch after insertion!");
                        // FatalAssert(loc.detail.containerVersion == container_entry->currentVersion,
                        //             LOG_TAG_DIVFTREE, "Vector location container version mismatch after insertion!");
                    } else {
                        FatalAssert(loc == INVALID_VECTOR_LOCATION,
                                    LOG_TAG_DIVFTREE, "Duplicate vector has valid location after insertion!");
                    }
                }
            }
        });

        if (releaseEntry) {
            BufferManager::GetInstance()->
                ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(false, false));
        }
        delete[] dup_state;
        return rs;
    }

    inline void PruneAtRootAndRelease(BufferVertexEntry* container_entry) {
        threadSelf->SanityCheckLockHeldInModeByMe(&container_entry->clusterLock, SX_EXCLUSIVE);
        CHECK_NOT_NULLPTR(container_entry, LOG_TAG_DIVFTREE);

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

        FatalAssert(container_entry->centroidMeta.selfId == bufferMgr->GetCurrentRootId(),
                    LOG_TAG_DIVFTREE, "we should be the root!");

        VectorID rootId = bufferMgr->GetCurrentRootId();
        CHECK_VECTORID_IS_VALID(container_entry->centroidMeta.selfId , LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_INTERNAL(container_entry->centroidMeta.selfId , LOG_TAG_DIVFTREE);

        DIVFTreeVertex* container = static_cast<DIVFTreeVertex*>(container_entry->ReadLatestVersion(false));
        CHECK_NOT_NULLPTR(container, LOG_TAG_DIVFTREE);
        uint16_t size = container->cluster.header.reserved_size.load(std::memory_order_relaxed);
        uint16_t num_deleted = container->cluster.header.num_deleted.load(std::memory_order_relaxed);

        FatalAssert(size == container->cluster.header.visible_size.load(std::memory_order_relaxed), LOG_TAG_DIVFTREE,
                    "size should be stable!");
        FatalAssert(size >= num_deleted, LOG_TAG_DIVFTREE,
                    "we cannot delete more vectors than there are in the cluster!");
        if (size - num_deleted == 0) {
            /* all vectors in the index are deleted */
            /* todo: delete this node and then create a new root? what if an insertion comes in at this point? */
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_NOT_IMPLEMENTED, "we do not handle this case for now");
        }

        if (((size - num_deleted) > 1)) {
            /* no need to prune */
            bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(true, true));
            return;
        }

        /* we should see exactly one child in this cluster and that child will become the new root */
        FatalAssert(container->cluster.NumBlocks(container->attr.block_size, container->attr.cap) == 1,
                    LOG_TAG_NOT_IMPLEMENTED, "cannot handle multiple blocks");
        CentroidMetaData* vmd =
            static_cast<CentroidMetaData*>(container->
                cluster.MetaData(0, false, container->attr.block_size, container->attr.cap, attr.dimension));
        VectorID newRootId = INVALID_VECTOR_ID;
        Version newRootVersion = 0;
        for (uint16_t i = 0; i < size; ++i) {
            if (vmd[i].state.load(std::memory_order_relaxed) == VECTOR_STATE_VALID) {
                newRootId = vmd[i].id;
                newRootVersion = vmd[i].version;
                break;
            }
        }
        CHECK_VECTORID_IS_VALID(newRootId, LOG_TAG_DIVFTREE);
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE, "Pruned old root " VECTORID_LOG_FMT ". new root is "
                VECTORID_LOG_FMT, VECTORID_LOG(container_entry->centroidMeta.selfId), VECTORID_LOG(newRootId));
        // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE_VERTEX, "UpdateVectorLocation prune invalidation");
        bufferMgr->UpdateVectorLocation(newRootId, INVALID_VECTOR_LOCATION);
        bufferMgr->UpdateRoot(newRootId, newRootVersion, container_entry);
        container_entry->InitiateDelete();
        bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(true, false));
    }

    inline void PruneEmptyVertexAndRelease(BufferVertexEntry* container_entry) {
        threadSelf->SanityCheckLockHeldInModeByMe(&container_entry->clusterLock, SX_EXCLUSIVE);
        CHECK_NOT_NULLPTR(container_entry, LOG_TAG_DIVFTREE);

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

        DIVFTreeVertex* container = static_cast<DIVFTreeVertex*>(container_entry->ReadLatestVersion(false));
        CHECK_NOT_NULLPTR(container, LOG_TAG_DIVFTREE);
        uint16_t size = container->cluster.header.reserved_size.load(std::memory_order_relaxed);
        uint16_t num_deleted = container->cluster.header.num_deleted.load(std::memory_order_relaxed);

        FatalAssert(size == container->cluster.header.visible_size.load(std::memory_order_relaxed), LOG_TAG_DIVFTREE,
                    "size should be stable!");
        FatalAssert(size >= num_deleted, LOG_TAG_DIVFTREE,
                    "we cannot delete more vectors than there are in the cluster!");
        if (((size - num_deleted) >= 1)) {
            /* no need to prune */
            bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(true, true));
            return;
        }

        VectorLocation loc = bufferMgr->LoadCurrentVectorLocation(container_entry->centroidMeta.selfId);
        BufferVertexEntry* parent = nullptr;
        if ((loc == INVALID_VECTOR_LOCATION) || ((parent = container_entry->ReadParentEntry(loc)) == nullptr)) {
            FatalAssert((parent == nullptr) && (loc == INVALID_VECTOR_LOCATION), LOG_TAG_DIVFTREE,
                        "mismatch between parent ptr and location");
            PruneAtRootAndRelease(container_entry);
            return;
        }
        FatalAssert((parent != nullptr) && (loc != INVALID_VECTOR_LOCATION), LOG_TAG_DIVFTREE,
                    "mismatch between parent ptr and location");
        FatalAssert(container_entry->centroidMeta.selfId != bufferMgr->GetCurrentRootId(),
                    LOG_TAG_DIVFTREE, "we should not be the root!");
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE, "Pruned vertex " VECTORID_LOG_FMT,
                VECTORID_LOG(container_entry->centroidMeta.selfId));
        RetStatus rs = DeleteVectorAndReleaseContainer(container_entry->centroidMeta.selfId, parent, loc);
        UNUSED_VARIABLE(rs);
        FatalAssert(rs.IsOK(), LOG_TAG_DIVFTREE, "this deletion should not fail!");
        container_entry->InitiateDelete();
        bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(true, false));
    }

    inline void PruneIfNeededAndRelease(BufferVertexEntry* container_entry) {
        while (true) {
            threadSelf->SanityCheckLockHeldInModeByMe(&container_entry->clusterLock, SX_SHARED);
            CHECK_NOT_NULLPTR(container_entry, LOG_TAG_DIVFTREE);

            BufferManager* bufferMgr = BufferManager::GetInstance();
            CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

            VectorID rootId = bufferMgr->GetCurrentRootId();
            CHECK_VECTORID_IS_VALID(rootId, LOG_TAG_DIVFTREE);
            CHECK_VECTORID_IS_CENTROID(rootId, LOG_TAG_DIVFTREE);
            if (rootId.IsLeaf()) {
                FatalAssert(container_entry->centroidMeta.selfId == rootId, LOG_TAG_DIVFTREE, "we should be the root!");
                bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(false, false));
                return;
            }

            DIVFTreeVertex* container = static_cast<DIVFTreeVertex*>(container_entry->ReadLatestVersion(false));
            FatalAssert((container_entry->state.load() == CLUSTER_DELETED) == (container == nullptr), LOG_TAG_DIVFTREE,
                            "container is null but not deleted!");
            if (container == nullptr) {
                bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(false, false));
                return;
            }
            CHECK_NOT_NULLPTR(container, LOG_TAG_DIVFTREE);
            if (container_entry->centroidMeta.selfId == rootId) {
                /* we can be sure that we remain root as changing the current root requires exclusive lock on it */
                /* todo: can it cause problems since we are loading reserve first then deleted? -> we may see num_deleted > reserved */
                if ((container->cluster.header.reserved_size.load(std::memory_order_acquire) -
                    container->cluster.header.num_deleted.load(std::memory_order_acquire)) > 1) {
                    bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(false, false));
                    return;
                }
                BufferVertexEntryState state = BufferVertexEntryState::CLUSTER_DELETE_IN_PROGRESS;
                if (container_entry->UpgradeAccessToExclusive(state).IsOK()) {
                    PruneAtRootAndRelease(container_entry);
                    return;
                } else if (state == BufferVertexEntryState::CLUSTER_DELETE_IN_PROGRESS ||
                           state == BufferVertexEntryState::CLUSTER_DELETED) {
                    /* someone else has handled it */
                    bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(false, false));
                    return;
                } else if (state == BufferVertexEntryState::CLUSTER_FULL) {
                    /* recheck if pruning is needed */
                    continue;
                } else {
                    DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_DIVFTREE, "we shouldn't get here! Invalid state");
                }
            }

            uint16_t deleted = container->cluster.header.num_deleted.load(std::memory_order_acquire);
            uint16_t reserved = container->cluster.header.reserved_size.load(std::memory_order_acquire);
            if ((reserved - deleted) >= 1) {
                if ((reserved - deleted) < container->attr.min_size) {
                    if (bufferMgr->AddMergeTaskIfNotExists(container_entry->centroidMeta.selfId,
                                                           container_entry)) {
                        bool res = merge_tasks.Push(MergeTask{container_entry->centroidMeta.selfId});
                        UNUSED_VARIABLE(res);
                        FatalAssert(res, LOG_TAG_DIVFTREE, "this should not fail!");
                    }
                } else if ((deleted > 0) && (threadSelf->UniformBinary((uint32_t)((double)attr.random_base_perc *
                                             (double)deleted / (double)reserved)))) {
                    if (bufferMgr->AddCompactionTaskIfNotExists(container_entry->centroidMeta.selfId,
                                                                container_entry)) {
                        bool res = compaction_tasks.Push(CompactionTask{container_entry->centroidMeta.selfId});
                        UNUSED_VARIABLE(res);
                        FatalAssert(res, LOG_TAG_DIVFTREE, "this should not fail!");
                    }
                }
                bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(false, false));
                return;
            }
            BufferVertexEntryState state = BufferVertexEntryState::CLUSTER_DELETE_IN_PROGRESS;
            if (container_entry->UpgradeAccessToExclusive(state).IsOK()) {
                if ((container->cluster.header.reserved_size.load(std::memory_order_acquire) -
                    container->cluster.header.num_deleted.load(std::memory_order_acquire)) >= 1) {
                    bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(true, true));
                    return;
                }

                rootId = bufferMgr->GetCurrentRootId();
                if (rootId.IsLeaf()) {
                    FatalAssert(container_entry->centroidMeta.selfId == rootId, LOG_TAG_DIVFTREE,
                                "we should be the root!");
                    bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(true, true));
                    return;
                }
                if (container_entry->centroidMeta.selfId == rootId) {
                    PruneAtRootAndRelease(container_entry);
                } else {
                    PruneEmptyVertexAndRelease(container_entry);
                }
                return;
            } else if (state == BufferVertexEntryState::CLUSTER_DELETE_IN_PROGRESS ||
                       state == BufferVertexEntryState::CLUSTER_DELETED) {
                /* someone else has handled it */
                bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(false, false));
                return;
            } else if (state == BufferVertexEntryState::CLUSTER_FULL) {
                /* recheck if pruning is needed */
                continue;
            } else {
                DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_DIVFTREE, "we shouldn't get here! Invalid state");
            }
        }
    }

    RetStatus DeleteVectorAndReleaseContainer(VectorID target, BufferVertexEntry* container_entry, VectorLocation loc) {
        CHECK_NOT_NULLPTR(container_entry, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_VALID(target, LOG_TAG_DIVFTREE);
        FatalAssert(target._level + 1 == container_entry->centroidMeta.selfId._level, LOG_TAG_DIVFTREE,
                    "target cannot be child of container");
        FatalAssert(loc.detail.containerId == container_entry->centroidMeta.selfId, LOG_TAG_DIVFTREE, "Location Id mismatch");
        FatalAssert(container_entry->currentVersion == loc.detail.containerVersion, LOG_TAG_DIVFTREE,
                    "Location version mismatch");
        threadSelf->SanityCheckLockHeldInModeByMe(&container_entry->clusterLock, SX_SHARED);
        DIVFTreeVertex* container = static_cast<DIVFTreeVertex*>(container_entry->ReadLatestVersion(false));
        CHECK_NOT_NULLPTR(container, LOG_TAG_DIVFTREE);
        Version version = 0;
        VectorState expState = VECTOR_STATE_VALID;
        RetStatus rs = container->ChangeVectorState(target, loc.detail.entryOffset, expState, VECTOR_STATE_INVALID,
                                                    &version);
        FatalAssert(target.IsVector() || rs.IsOK(), LOG_TAG_DIVFTREE,
                    "state change will never fail if target is a vertex as it should be locked in X mode.");
        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);
        if (rs.IsOK()) {
#ifdef EXCESS_LOGING
            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
                    "DELETEING vector_id=" VECTORID_LOG_FMT " in Location=%s: Vector=%s",
                    VECTORID_LOG(target), loc.ToString().ToCStr(),
                    VectorToString(container->cluster.Data(loc.detail.entryOffset, container->attr.centroid_id.IsLeaf(),
                                                           container->attr.block_size, container->attr.cap,
                                                           attr.dimension), attr.dimension).ToCStr());
#endif
            // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE_VERTEX, "UpdateVectorLocation in delete invalidation:");
            bufferMgr->UpdateVectorLocation(target, INVALID_VECTOR_LOCATION);
            if (target.IsCentroid()) {
                bufferMgr->UnpinVertexVersion(target, version);
            }
            container->cluster.header.num_deleted.fetch_add(1);
            PruneIfNeededAndRelease(container_entry);
        } else {
            bufferMgr->ReleaseBufferEntry(container_entry, ReleaseBufferEntryFlags(false, false));
        }

        return rs;
    }

    inline RetStatus ReadAndCheckVersion(VectorID containerId, Version containerVersion,
                                         BufferVertexEntry** entries, uint16_t max_entries, uint16_t& num_entries,
                                         LockMode mode) {
        BufferVertexEntry*& containerEntry = entries[max_entries - num_entries - 1];
        BufferManager* bufferMgr = BufferManager::GetInstance();
        RetStatus rs = RetStatus::Success();

        /*
         * we do not care if we are blocked because the block does not necessarily lead to Major Updates +
         * Compaction does not change the version
         */
        containerEntry = bufferMgr->ReadBufferEntry(containerId, mode);
        if (containerEntry == nullptr) {
            rs = RetStatus{.stat=RetStatus::TARGET_DELETED, .message=nullptr};
        }

        if (rs.IsOK()) {
            ++num_entries;
            if (containerEntry->currentVersion._split != containerVersion._split) {
                rs = RetStatus{.stat=RetStatus::TARGET_UPDATED, .message=nullptr};
            }
        }

        if (!rs.IsOK()) {
            bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                               ReleaseBufferEntryFlags(false, false));
        }

        return rs;
    }

    /* Todo: recheck to see if everything works fine */
    /* todo: need to refactor */
    RetStatus Migrate(std::vector<MigrationInfo> targetBatch,
                      VectorID src_id, VectorID dest_id,
                      Version src_ver, Version dest_ver, uint64_t& num_migrated) {
        CHECK_VECTORID_IS_VALID(src_id, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_CENTROID(src_id, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_VALID(dest_id, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_CENTROID(dest_id, LOG_TAG_DIVFTREE);
        FatalAssert(src_id != dest_id, LOG_TAG_DIVFTREE,
                    "containers should not be the same!");
        FatalAssert(src_id._level == dest_id._level, LOG_TAG_DIVFTREE,
                    "containers should be on the same level!");

        RetStatus rs = RetStatus::Success();

        if (targetBatch.empty()) {
            return rs;
        }
        FatalAssert(!targetBatch.empty(), LOG_TAG_DIVFTREE, "Empty batch!");

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);
        uint16_t max_entries = 2;
        if (src_id.IsInternalVertex()) {
            max_entries += targetBatch.size();
        }
        std::vector<BufferVertexEntry*> entries;
        BufferVertexEntry** src_entry;
        BufferVertexEntry** dest_entry;
        entries.resize(max_entries, nullptr);
        uint16_t num_entries = 0;

        if (src_id.IsInternalVertex()) {
            for (size_t i = 0; i < targetBatch.size(); ++i) {
                FatalAssert((i == 0) || (targetBatch[i - 1].id <= targetBatch[i].id), LOG_TAG_DIVFTREE,
                            "target batch should be sorder in the increasing order of IDs");
                FatalAssert(targetBatch[i].id._level + 1 == src_id._level, LOG_TAG_DIVFTREE, "level mismatch!");
                rs = ReadAndCheckVersion(targetBatch[i].id, targetBatch[i].version, &entries[0], max_entries,
                                         num_entries, SX_SHARED);
                if (!rs.IsOK()) {
                    return rs;
                }
            }
        }

        /* Todo: if we have version logs, maybe we can use that to handle migration when there is an update */
        /* Todo: maybe we can do a simple distance computation in case there was any updates to avoid too many fails */
        /* Todo: collect stats on the failure percentages, the reason and the target level */

        /* We should first lock the child and then the parents in order of their values to avoid deadlocks! */
        if (src_id <= dest_id) {
            rs = ReadAndCheckVersion(src_id, src_ver, &entries[0], max_entries, num_entries, SX_SHARED);
            if (!rs.IsOK()) {
                return rs;
            }

            src_entry = &entries[max_entries - num_entries];

            rs = ReadAndCheckVersion(dest_id, dest_ver, &entries[0], max_entries, num_entries, SX_SHARED);
            if (!rs.IsOK()) {
                return rs;
            }
            dest_entry = &entries[max_entries - num_entries];
        } else {
            rs = ReadAndCheckVersion(dest_id, dest_ver, &entries[0], max_entries, num_entries, SX_SHARED);
            if (!rs.IsOK()) {
                return rs;
            }
            dest_entry = &entries[max_entries - num_entries];

            rs = ReadAndCheckVersion(src_id, src_ver, &entries[0], max_entries, num_entries, SX_SHARED);
            if (!rs.IsOK()) {
                return rs;
            }

            src_entry = &entries[max_entries - num_entries];
        }

        DIVFTreeVertex* src_cluster = static_cast<DIVFTreeVertex*>((*src_entry)->ReadLatestVersion(false));
        CHECK_NOT_NULLPTR(src_cluster, LOG_TAG_DIVFTREE);
        VTYPE* data = src_cluster->cluster.Data(0, src_id.IsLeaf(), src_cluster->attr.block_size,
                                                    src_cluster->attr.cap, attr.dimension);
        void* meta = src_cluster->cluster.MetaData(0, src_id.IsLeaf(), src_cluster->attr.block_size,
                                                   src_cluster->attr.cap, attr.dimension);
        FatalAssert(src_cluster->cluster.NumBlocks(src_cluster->attr.block_size, src_cluster->attr.cap) == 1,
                    LOG_TAG_NOT_IMPLEMENTED, "cannot handle more than 1 block!");

        VectorBatch batch;
        batch.size = 0;
        batch.data = new VTYPE[targetBatch.size() * attr.dimension];
        batch.id = new VectorID[targetBatch.size()];
        batch.version = nullptr;
        uint16_t* migrated_offsets = new uint16_t[targetBatch.size()];

        if (src_id.IsInternalVertex()) {
            batch.version = new Version[targetBatch.size()];
        }

        Version curr_src_ver = (*src_entry)->currentVersion;
        FatalAssert(curr_src_ver._split == src_ver._split, LOG_TAG_DIVFTREE,
                    "the source entry version should not have been splited!");
        bool compacted = (curr_src_ver._compaction != src_ver._compaction);
        for (size_t i = 0; i < targetBatch.size(); ++i) {
            if (compacted) {
                VectorLocation cur_loc = bufferMgr->LoadCurrentVectorLocation(targetBatch[i].id);
                if (cur_loc.detail.containerId != src_id) {
                    continue;
                }

                FatalAssert(cur_loc.detail.containerVersion._split == src_ver._split, LOG_TAG_DIVFTREE,
                            "Eventhough the src version is the same as what was used container version has changed!");
                targetBatch[i].offset = cur_loc.detail.entryOffset;
            }

            if (src_id.IsLeaf()) {
                VectorMetaData* vmd = static_cast<VectorMetaData*>(meta);
                FatalAssert(vmd[targetBatch[i].offset].id == targetBatch[i].id, LOG_TAG_DIVFTREE,
                            "there was a compaction but we didn't detect it!");
                // if (vmd[targetBatch[i].offset].id != targetBatch[i].id) {
                //     /* There was a compaction */
                //     FatalAssert(!compacted, LOG_TAG_DIVFTREE, "Already checked for compaction!");
                //     compacted = true;
                //     --i;
                //     continue;
                // }

                VectorState excp_state = VECTOR_STATE_VALID;
                rs = src_cluster->ChangeVectorState(&vmd[targetBatch[i].offset],
                                                    excp_state, VECTOR_STATE_MIGRATED);
                if (!rs.IsOK()) {
                    continue;
                }

                memcpy(&batch.data[(size_t)batch.size * (size_t)attr.dimension],
                       &data[(size_t)attr.dimension * (size_t)targetBatch[i].offset],
                       sizeof(VTYPE) * (size_t)attr.dimension);
                batch.id[batch.size] = vmd[targetBatch[i].offset].id;
                migrated_offsets[batch.size] = targetBatch[i].offset;
                ++batch.size;
            } else {
                CentroidMetaData* vmd = static_cast<CentroidMetaData*>(meta);
                FatalAssert(vmd[targetBatch[i].offset].id == targetBatch[i].id, LOG_TAG_DIVFTREE,
                            "there was a compaction but we didn't detect it!");
                // if (vmd[targetBatch[i].offset].id != targetBatch[i].id) {
                //     /* There was a compaction */
                //     compacted = true;
                //     --i;
                //     continue;
                // }

                VectorState excp_state = VECTOR_STATE_VALID;
                rs = src_cluster->ChangeVectorState(&vmd[targetBatch[i].offset],
                                                    excp_state, VECTOR_STATE_MIGRATED);
                if (!rs.IsOK()) {
                    continue;
                }

                memcpy(&batch.data[(size_t)batch.size * (size_t)attr.dimension],
                       &data[(size_t)attr.dimension * (size_t)targetBatch[i].offset],
                       sizeof(VTYPE) * (size_t)attr.dimension);
                batch.id[batch.size] = vmd[targetBatch[i].offset].id;
                batch.version[batch.size] = vmd[targetBatch[i].offset].version;
                migrated_offsets[batch.size] = targetBatch[i].offset;
                ++batch.size;
            }
        }

        /* todo: loop on all instead of yield on one */
        for (size_t i = 0; i < batch.size; ++i) {
            VectorLocation cur_loc;
            while ((cur_loc = bufferMgr->LoadCurrentVectorLocation(batch.id[i])) !=
                   VectorLocation(src_id, curr_src_ver, migrated_offsets[i])) {
                /* wait until the location is updated to the expected one */
                DIVFTREE_YIELD();
            }
        }

        /* todo: we may be able to avoid this extra copy */
        if (batch.size > 0) {
            rs = BatchInsertInto((*dest_entry), batch, true, INVALID_OFFSET, dest_id < src_id);
            *dest_entry = nullptr;

            threadSelf->SanityCheckLockHeldInModeByMe(&(*src_entry)->clusterLock, SX_SHARED);
            SANITY_CHECK({
                if (rs.IsOK()) {
                    for (size_t i = 0; i < batch.size; ++i) {
                        FatalAssert(bufferMgr->LoadCurrentVectorLocation(batch.id[i]) !=
                                    VectorLocation(src_id, curr_src_ver, migrated_offsets[i]),
                                    LOG_TAG_DIVFTREE, "the location should have changed!");
                    }
                } else {
                    for (size_t i = 0; i < batch.size; ++i) {
                        FatalAssert(bufferMgr->LoadCurrentVectorLocation(batch.id[i]) ==
                                    VectorLocation(src_id, curr_src_ver, migrated_offsets[i]),
                                    LOG_TAG_DIVFTREE, "the location should not have changed!");
                    }
                }
            });
        }
        delete[] batch.data;
        batch.data = nullptr;

        if (batch.size > 0) {
            if (!rs.IsOK()) {
                for (uint16_t i = 0; i < batch.size; ++i) {
                    VectorState expState = VECTOR_STATE_MIGRATED;
                    rs = src_cluster->ChangeVectorState(batch.id[i], migrated_offsets[i],
                                                        expState, VECTOR_STATE_VALID);
                    FatalAssert(rs.IsOK(), LOG_TAG_DIVFTREE, "this should not fail!");
                }
                rs = RetStatus{.stat=RetStatus::NEW_CONTAINER_UPDATED, .message=nullptr};
                bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                            ReleaseBufferEntryFlags(false, false));
            } else {
                uint16_t num_deleted = src_cluster->cluster.header.num_deleted.fetch_add(batch.size) + batch.size;
                uint16_t reserved = src_cluster->cluster.header.reserved_size.load(std::memory_order_acquire);
                FatalAssert(reserved >= num_deleted, LOG_TAG_DIVFTREE,
                            "we cannot delete more vectors than there are in the cluster!");
                num_migrated += batch.size;
                if ((reserved - num_deleted) < src_cluster->attr.min_size) {
                    if (bufferMgr->AddMergeTaskIfNotExists(src_id, *src_entry)) {
                        bool res = merge_tasks.Push(MergeTask{src_id});
                        UNUSED_VARIABLE(res);
                        FatalAssert(res, LOG_TAG_DIVFTREE, "this should not fail!");
                    }
                } else if (threadSelf->UniformBinary((uint32_t)((double)attr.random_base_perc *
                                                    ((double)num_deleted / (double)reserved)))) {
                    if (bufferMgr->AddCompactionTaskIfNotExists(src_id, *src_entry)) {
                        bool res = compaction_tasks.Push(CompactionTask{src_id});
                        UNUSED_VARIABLE(res);
                        FatalAssert(res, LOG_TAG_DIVFTREE, "this should not fail!");
                    }
                }
                BufferVertexEntry* src_entry_cpy = *src_entry;
                *src_entry = nullptr;
                // SANITY_CHECK(
                //     if (batch.version != nullptr) {
                //         for (uint16_t i = 0; i < batch.size; ++i) {
                //             DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE, "Migrated (" VECTORID_LOG_FMT ", ver:%u) from "
                //                     VECTORID_LOG_FMT " to " VECTORID_LOG_FMT,
                //                     VECTORID_LOG(batch.id[i]), batch.version[i], VECTORID_LOG(src_id), VECTORID_LOG(dest_id));
                //         }
                //     } else {
                //         for (uint16_t i = 0; i < batch.size; ++i) {
                //             DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE, "Migrated (" VECTORID_LOG_FMT ") from "
                //                     VECTORID_LOG_FMT " to " VECTORID_LOG_FMT,
                //                     VECTORID_LOG(batch.id[i]), VECTORID_LOG(src_id), VECTORID_LOG(dest_id));
                //         }
                //     }
                // )
                bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                                ReleaseBufferEntryFlags(false, false));
                PruneIfNeededAndRelease(src_entry_cpy);
            }
        } else {
            rs = RetStatus{.stat=RetStatus::EMPTY_MIGRATION_BATCH, .message=nullptr};
            bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                               ReleaseBufferEntryFlags(false, false));
        }

        delete[] migrated_offsets;
        migrated_offsets = nullptr;
        delete[] batch.id;
        batch.id = nullptr;
        if (src_id.IsInternalVertex()) {
            delete[] batch.version;
            batch.version = nullptr;
        }

        return rs;
    }

    uint64_t MigrationCheck(VectorID first_cluster, VectorID second_cluster) {
        CHECK_VECTORID_IS_VALID(first_cluster, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_CENTROID(first_cluster, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_VALID(second_cluster, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_CENTROID(second_cluster, LOG_TAG_DIVFTREE);
        FatalAssert(first_cluster != second_cluster, LOG_TAG_DIVFTREE,
                    "clusters should be different!");
        FatalAssert(first_cluster._level == second_cluster._level, LOG_TAG_DIVFTREE,
                    "clusters should be on the same level!");

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);
        constexpr uint8_t max_entries = 2;
        BufferVertexEntry* entries[max_entries];
        DIVFTreeVertex* clusters[max_entries];
        DIVFTreeVertex* parents[max_entries];
        VTYPE* centroidData[max_entries];
        uint16_t visible_size[max_entries];
        Version versions[max_entries];
        VectorID ids[max_entries];
        uint8_t num_entries = 0;

        if (first_cluster > second_cluster) {
            std::swap(first_cluster, second_cluster);
        }
        ids[1] = first_cluster;
        ids[0] = second_cluster;

        entries[1] = bufferMgr->ReadBufferEntry(ids[1], SX_SHARED);
        if (entries[1] == nullptr) {
            return 0;
        }
        ++num_entries;
        entries[0] = bufferMgr->ReadBufferEntry(ids[0], SX_SHARED);
        if (entries[0] == nullptr) {
            bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                               ReleaseBufferEntryFlags(false, false));
            return 0;
        }
        ++num_entries;

        versions[1] = entries[1]->currentVersion;
        versions[0] = entries[0]->currentVersion;

        clusters[1] = static_cast<DIVFTreeVertex*>(entries[1]->ReadLatestVersion(false));
        CHECK_NOT_NULLPTR(clusters[1], LOG_TAG_DIVFTREE);
        visible_size[1] = clusters[1]->cluster.header.visible_size.load(std::memory_order_acquire);
        if (visible_size[1] == 0) {
            bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                               ReleaseBufferEntryFlags(false, false));
            return 0;
        }

        clusters[0] = static_cast<DIVFTreeVertex*>(entries[0]->ReadLatestVersion(false));
        CHECK_NOT_NULLPTR(clusters[0], LOG_TAG_DIVFTREE);
        visible_size[0] = clusters[0]->cluster.header.visible_size.load(std::memory_order_acquire);
        if (visible_size[0] == 0) {
            bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                               ReleaseBufferEntryFlags(false, false));
            return 0;
        }


        VectorLocation loc;
        parents[1] = static_cast<DIVFTreeVertex*>(entries[1]->centroidMeta.ReadAndPinParent(loc));
        if (parents[1] == nullptr) {
            bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                               ReleaseBufferEntryFlags(false, false));
            return 0;
        }
#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE, "MigrationCheck: cluster pinned: %p:" VECTORID_LOG_FMT, parents[1],
                VECTORID_LOG(parents[1]->attr.centroid_id));
#endif
        FatalAssert(
            (static_cast<CentroidMetaData*>(parents[1]->
                cluster.MetaData(loc.detail.entryOffset, false, parents[1]->attr.block_size,
                                 parents[1]->attr.cap, attr.dimension))->id == ids[1]) &&
            (static_cast<CentroidMetaData*>(parents[1]->
                cluster.MetaData(loc.detail.entryOffset, false, parents[1]->attr.block_size,
                                 parents[1]->attr.cap, attr.dimension))->version == versions[1]),
            LOG_TAG_DIVFTREE,
            "Invalid centroid location returned!"
        );

        // if ((static_cast<CentroidMetaData*>(parents[1]->
        //         cluster.MetaData(loc.entryOffset, false, parents[1]->attr.block_size,
        //                          parents[1]->attr.cap, attr.dimension))->id != ids[1])) {
        //     parents[1]->Unpin();
        //     bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
        //                                        ReleaseBufferEntryFlags(false, false));
        //     return 0;

        // }

        parents[0] = static_cast<DIVFTreeVertex*>(entries[0]->centroidMeta.ReadAndPinParent(loc));
        if (parents[0] == nullptr) {
#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE, "MigrationCheck: cluster unpinned: %p:" VECTORID_LOG_FMT,
                parents[1], VECTORID_LOG(parents[1]->attr.centroid_id));
#endif
            parents[1]->Unpin();
            bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                               ReleaseBufferEntryFlags(false, false));
            return 0;
        }

#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE, "MigrationCheck: cluster pinned: %p:" VECTORID_LOG_FMT, parents[0],
                VECTORID_LOG(parents[0]->attr.centroid_id));
#endif

        FatalAssert(
            static_cast<CentroidMetaData*>(parents[0]->
                cluster.MetaData(loc.detail.entryOffset, false, parents[0]->attr.block_size,
                                 parents[0]->attr.cap, attr.dimension))->id == ids[0] &&
            (static_cast<CentroidMetaData*>(parents[0]->
                cluster.MetaData(loc.detail.entryOffset, false, parents[0]->attr.block_size,
                                 parents[0]->attr.cap, attr.dimension))->version == versions[0]),
            LOG_TAG_DIVFTREE,
            "Invalid centroid location returned!"
        );

        // if ((static_cast<CentroidMetaData*>(parents[0]->
        //         cluster.MetaData(loc.entryOffset, false, parents[0]->attr.block_size,
        //                          parents[0]->attr.cap, attr.dimension))->id != ids[0])) {
        //     parents[1]->Unpin();
        //     parents[0]->Unpin();
        //     bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
        //                                        ReleaseBufferEntryFlags(false, false));
        //     return 0;

        // }

        std::vector<MigrationInfo> migration_batch[max_entries];
        for (uint8_t cn = 0; cn < max_entries; ++cn) {
            centroidData[cn] = parents[cn]->cluster.Data(loc.detail.entryOffset, false, parents[cn]->attr.block_size,
                                                         parents[cn]->attr.cap, attr.dimension);
            FatalAssert(visible_size[cn] > 0, LOG_TAG_DIVFTREE, "size should be greater than 0!");
            migration_batch[cn].reserve(visible_size[cn]);
        }

        FatalAssert(clusters[0]->cluster.NumBlocks(clusters[0]->attr.block_size, clusters[0]->attr.cap) == 1,
                    LOG_TAG_NOT_IMPLEMENTED, "cannot handle more than 1 block!");

        for (int cn = 0; cn < max_entries; ++cn) {
            VTYPE* data = clusters[cn]->cluster.Data(0, ids[1].IsLeaf(), clusters[cn]->attr.block_size,
                                                     clusters[cn]->attr.cap, attr.dimension);
            void* meta = clusters[cn]->cluster.MetaData(0, ids[1].IsLeaf(), clusters[cn]->attr.block_size,
                                                        clusters[cn]->attr.cap, attr.dimension);
            std::unordered_set<VectorID, VectorIDHash> seen;
            seen.reserve(visible_size[cn]);
            for (uint16_t i = visible_size[cn] - 1; i != UINT16_MAX; --i) {
                if (ids[1].IsLeaf()) {
                    VectorMetaData* vmd = static_cast<VectorMetaData*>(meta);
                    if ((seen.find(vmd[i].id) != seen.end()) ||
                        (vmd[i].state.load(std::memory_order_acquire) != VECTOR_STATE_VALID)) {
                        continue;
                    }
                    seen.emplace(vmd[i].id);
                    /* todo: is it worth the additional memory usage to
                    cache the distance of each vector to it's centroid? */
                    if (MoreSimilar(Distance(&data[(size_t)i * (size_t)attr.dimension], centroidData[1 - cn],
                                             attr.dimension, attr.distanceAlg),
                                    Distance(&data[(size_t)i * (size_t)attr.dimension], centroidData[cn],
                                             attr.dimension, attr.distanceAlg),
                                    attr.distanceAlg) == 1) {
                        migration_batch[cn].emplace_back(vmd[i].id, i);
                    }
                } else {
                    CentroidMetaData* vmd = static_cast<CentroidMetaData*>(meta);
                    if ((seen.find(vmd[i].id) != seen.end()) ||
                        (vmd[i].state.load(std::memory_order_acquire) != VECTOR_STATE_VALID)) {
                        continue;
                    }
                    seen.emplace(vmd[i].id);
                    /* todo: is it worth the additional memory usage to
                    cache the distance of each vector to it's centroid? */
                    if (MoreSimilar(Distance(&data[(size_t)i * (size_t)attr.dimension], centroidData[1 - cn],
                                             attr.dimension, attr.distanceAlg),
                                    Distance(&data[(size_t)i * (size_t)attr.dimension], centroidData[cn],
                                             attr.dimension, attr.distanceAlg),
                                    attr.distanceAlg) == 1) {
                        migration_batch[cn].emplace_back(vmd[i].id, i, vmd[i].version);
                    }
                }
            }

            if (ids[1].IsInternalVertex()) {
                std::sort(migration_batch[cn].begin(), migration_batch[cn].end());
            }
            seen.clear();
#ifdef MEMORY_DEBUG
            DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE, "MigrationCheck: cluster unpinned: %p:" VECTORID_LOG_FMT,
                    parents[cn], VECTORID_LOG(parents[cn]->attr.centroid_id));
#endif
        }

        for (uint8_t cn = 0; cn < max_entries; ++cn) {
            parents[cn]->Unpin();
        }

        bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                           ReleaseBufferEntryFlags(false, false));

        /* todo: have to change for multi ndoe setup */
        uint64_t num_migrated = 0;
        if (threadSelf->UniformBinary(attr.random_base_perc / 2)) {
            Migrate(migration_batch[0], ids[0], ids[1], versions[0], versions[1], num_migrated);
            Migrate(migration_batch[1], ids[1], ids[0], versions[1], versions[0], num_migrated);
        } else {
            Migrate(migration_batch[1], ids[1], ids[0], versions[1], versions[0], num_migrated);
            Migrate(migration_batch[0], ids[0], ids[1], versions[0], versions[1], num_migrated);
        }
        return num_migrated;
    }

    /* todo: need to refactor */
    RetStatus Merge(VectorID srcId, Version srcVersion,
                    VectorID destId, Version destVersion) {
        CHECK_VECTORID_IS_VALID(srcId, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_CENTROID(srcId, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_VALID(destId, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_CENTROID(destId, LOG_TAG_DIVFTREE);
        FatalAssert(srcId != destId, LOG_TAG_DIVFTREE,
                    "clusters should be on the same level!");
        FatalAssert(srcId._level == destId._level, LOG_TAG_DIVFTREE,
                    "clusters should be on the same level!");

        RetStatus rs = RetStatus::Success();
        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);
        constexpr uint16_t max_entries = 2;
        BufferVertexEntry* entries[max_entries];
        uint16_t num_entries = 0;
        BufferVertexEntry** srcEntry;
        BufferVertexEntry** destEntry;
        uint16_t reservedSize = 0;
        uint16_t totalSize = 0;
        DIVFTreeVertex* srcCluster = nullptr;

        /* Todo: if we have version logs, maybe we can use that to handle migration when there is an update */
        /* Todo: maybe we can do a simple distance computation in case there was any updates to avoid too many fails */
        /* Todo: collect stats on the failure percentages, the reason and the target level */

        /* We should first lock the child and then the parents in order of their values to avoid deadlocks! */
        if (srcId < destId) {
            rs = ReadAndCheckVersion(srcId, srcVersion, entries, max_entries, num_entries, SX_EXCLUSIVE);
            if (!rs.IsOK()) {
                bufferMgr->RemoveMergeTask(srcId);
                return rs;
            }
            srcEntry = &entries[max_entries - num_entries];
            srcCluster = static_cast<DIVFTreeVertex*>((*srcEntry)->ReadLatestVersion(false));
            CHECK_NOT_NULLPTR(srcCluster, LOG_TAG_DIVFTREE);
            reservedSize = srcCluster->cluster.header.reserved_size.load(std::memory_order_relaxed);
            FatalAssert(reservedSize == srcCluster->cluster.header.visible_size.load(std::memory_order_relaxed),
                        LOG_TAG_DIVFTREE, "reserved and visible size should be the same!");
            FatalAssert(reservedSize > srcCluster->cluster.header.num_deleted.load(std::memory_order_relaxed),
                        LOG_TAG_DIVFTREE, "size should be greater than to number of deleted entries!");
            /* size should not be 0 because the one who 0ed the size will also delete the cluster! */
            totalSize = reservedSize - srcCluster->cluster.header.num_deleted.load(std::memory_order_relaxed);
            if (totalSize >= srcCluster->attr.min_size) {
                bufferMgr->RemoveMergeTask(srcId, *srcEntry);
                bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                                   ReleaseBufferEntryFlags(false, false));
                return RetStatus{.stat=RetStatus::SRC_HAS_TOO_MANY_VECTORS, .message=nullptr};
            }

            if (totalSize == 0) {
                bufferMgr->RemoveMergeTask(srcId, *srcEntry);
                /* todo: check if state is not delete in progress, we should delete the cluster ourselves here? */
                bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                                   ReleaseBufferEntryFlags(false, false));
                return RetStatus{.stat=RetStatus::SRC_EMPTY, .message=nullptr};
            }

            rs = ReadAndCheckVersion(destId, destVersion, entries, max_entries, num_entries, SX_SHARED);
            if (!rs.IsOK()) {
                bufferMgr->RemoveMergeTask(srcId, *srcEntry);
                return rs;
            }
            destEntry = &entries[max_entries - num_entries];
        } else {
            rs = ReadAndCheckVersion(destId, destVersion, entries, max_entries, num_entries, SX_SHARED);
            if (!rs.IsOK()) {
                bufferMgr->RemoveMergeTask(srcId);
                return rs;
            }
            destEntry = &entries[max_entries - num_entries];

            rs = ReadAndCheckVersion(srcId, srcVersion, entries, max_entries, num_entries, SX_EXCLUSIVE);
            if (!rs.IsOK()) {
                bufferMgr->RemoveMergeTask(srcId);
                return rs;
            }

            srcEntry = &entries[max_entries - num_entries];
            srcCluster = static_cast<DIVFTreeVertex*>((*srcEntry)->ReadLatestVersion(false));
            CHECK_NOT_NULLPTR(srcCluster, LOG_TAG_DIVFTREE);
            reservedSize = srcCluster->cluster.header.reserved_size.load(std::memory_order_relaxed);
            FatalAssert(reservedSize == srcCluster->cluster.header.visible_size.load(std::memory_order_relaxed),
                        LOG_TAG_DIVFTREE, "reserved and visible size should be the same!");
            FatalAssert(reservedSize > srcCluster->cluster.header.num_deleted.load(std::memory_order_relaxed),
                        LOG_TAG_DIVFTREE, "size should be greater than to number of deleted entries!");
            /* size should not be 0 because the one who 0ed the size will also delete the cluster! */
            totalSize = reservedSize - srcCluster->cluster.header.num_deleted.load(std::memory_order_relaxed);
            if (totalSize >= srcCluster->attr.min_size) {
                bufferMgr->RemoveMergeTask(srcId, *srcEntry);
                bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                                   ReleaseBufferEntryFlags(false, false));
                return RetStatus{.stat=RetStatus::SRC_HAS_TOO_MANY_VECTORS, .message=nullptr};
            }

            if (totalSize == 0) {
                bufferMgr->RemoveMergeTask(srcId, *srcEntry);
                /* todo: check if state is not delete in progress, we should delete the cluster ourselves here? */
                bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                                   ReleaseBufferEntryFlags(false, false));
                return RetStatus{.stat=RetStatus::SRC_EMPTY, .message=nullptr};
            }
        }

        /* todo: use the original pointers to write these  */
        VectorBatch batch;
        batch.size = totalSize;
        batch.id = new VectorID[totalSize];
        batch.data = new VTYPE[totalSize * attr.dimension];
        batch.version = nullptr;
        uint16_t* offsets = new uint16_t[totalSize];
        bool is_leaf = srcId.IsLeaf();
        if (!is_leaf) {
            batch.version = new Version[totalSize];
        }

        VTYPE* data = srcCluster->cluster.Data(0, is_leaf, srcCluster->attr.block_size, srcCluster->attr.cap,
                                               attr.dimension);
        void* meta = srcCluster->cluster.MetaData(0, is_leaf, srcCluster->attr.block_size, srcCluster->attr.cap,
                                               attr.dimension);
        FatalAssert(srcCluster->cluster.NumBlocks(srcCluster->attr.block_size, srcCluster->attr.cap) == 1,
                    LOG_TAG_NOT_IMPLEMENTED, "cannot handle more than 1 block!");
        uint16_t i = 0;
        for (uint16_t offset = 0; (offset < reservedSize) && (i < totalSize); ++offset) {
            VectorState expected = VECTOR_STATE_VALID;
            if (is_leaf) {
                VectorMetaData* vmd = static_cast<VectorMetaData*>(meta);
                if (vmd[offset].state.load(std::memory_order_acquire) != VECTOR_STATE_VALID) {
                    continue;
                }

                if (srcCluster->ChangeVectorState(&vmd[offset], expected, VECTOR_STATE_MIGRATED).IsOK()) {
                    memcpy(&data[offset * attr.dimension], &batch.data[i * attr.dimension],
                           sizeof(VTYPE) * attr.dimension);
                    batch.id[i] = vmd[offset].id;
                    offsets[i] = offset;
                    ++i;
                    SANITY_CHECK(
                        VectorLocation loc = bufferMgr->LoadCurrentVectorLocation(vmd[offset].id);
                        FatalAssert(loc == VectorLocation(srcId, srcCluster->attr.version, offset),
                                    LOG_TAG_DIVFTREE, "location mismatch!");
                    )
                }
            } else {
                CentroidMetaData* vmd = static_cast<CentroidMetaData*>(meta);
                if (vmd[offset].state.load(std::memory_order_acquire) != VECTOR_STATE_VALID) {
                    continue;
                }

                if (srcCluster->ChangeVectorState(&vmd[offset], expected, VECTOR_STATE_MIGRATED).IsOK()) {
                    memcpy(&data[offset * attr.dimension], &batch.data[i * attr.dimension],
                           sizeof(VTYPE) * attr.dimension);
                    batch.id[i] = vmd[offset].id;
                    batch.version[i] = vmd[offset].version;
                    offsets[i] = offset;
                    ++i;
                    SANITY_CHECK(
                        VectorLocation loc = bufferMgr->LoadCurrentVectorLocation(vmd[offset].id);
                        /* if the vertex is just created, it is possible that loadCurrent returns invalid! */
                        FatalAssert(loc == INVALID_VECTOR_LOCATION ||
                                    loc == VectorLocation(srcId, srcCluster->attr.version, offset),
                                    LOG_TAG_DIVFTREE, "location mismatch!");
                    )
                }
            }
        }
        FatalAssert(i == totalSize, LOG_TAG_DIVFTREE, "fewer elements than excpected!");

        if (!is_leaf) {
            /* todo: loop on all instead of yield on one */
            for (size_t i = 0; i < batch.size; ++i) {
                VectorLocation cur_loc;
                while ((cur_loc = bufferMgr->LoadCurrentVectorLocation(batch.id[i])) !=
                        VectorLocation(srcId, srcCluster->attr.version, offsets[i])) {
                    /* wait until the location is updated to the expected one */
                    FatalAssert(cur_loc == INVALID_VECTOR_LOCATION, LOG_TAG_DIVFTREE,
                                "the vector location should be invalid here!");
                    DIVFTREE_YIELD();
                }
            }
        }

        rs = BatchInsertInto((*destEntry), batch, true, INVALID_OFFSET, destId < srcId);
        (*destEntry) = nullptr;
        if (rs.IsOK()) {
            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE, "Merged " VECTORID_LOG_FMT " into " VECTORID_LOG_FMT,
                    VECTORID_LOG(srcId), VECTORID_LOG(destId));
            srcCluster->cluster.header.num_deleted.store(reservedSize, std::memory_order_release);
            (*srcEntry)->state.store(CLUSTER_DELETE_IN_PROGRESS, std::memory_order_release);
            /* no need to unset the merge flag here as the entry is deleted */
            PruneEmptyVertexAndRelease(*srcEntry);
            delete[] batch.data;
            if (!is_leaf) {
                delete[] batch.version;
            }
            delete[] batch.id;
            delete[] offsets;
            return rs;
        } else {
            /* todo: add the task of checking the src to merge check */
        }

        for (uint16_t i = 0; i < totalSize; ++i) {
            if (is_leaf) {
                VectorMetaData* vmd = static_cast<VectorMetaData*>(meta);
                FatalAssert((vmd[offsets[i]].id == batch.id[i]) &&
                            (vmd[offsets[i]].state.load(std::memory_order_relaxed) == VECTOR_STATE_MIGRATED),
                            LOG_TAG_DIVFTREE, "mismatch!");
                vmd[offsets[i]].state.store(VECTOR_STATE_VALID, std::memory_order_release);
            } else {
                CentroidMetaData* vmd = static_cast<CentroidMetaData*>(meta);
                FatalAssert((vmd[offsets[i]].id == batch.id[i]) &&
                            (vmd[offsets[i]].state.load(std::memory_order_relaxed) == VECTOR_STATE_MIGRATED),
                            LOG_TAG_DIVFTREE, "mismatch!");
                vmd[offsets[i]].state.store(VECTOR_STATE_VALID, std::memory_order_release);
            }
        }

        bufferMgr->ReleaseEntriesIfNotNull(&entries[max_entries - num_entries], num_entries,
                                           ReleaseBufferEntryFlags(false, false));
        delete[] batch.data;
        if (!is_leaf) {
            delete[] batch.version;
        }
        delete[] batch.id;
        delete[] offsets;

        return RetStatus{.stat=RetStatus::DEST_UPDATED, .message=nullptr};
    }

    bool MergeCheck(VectorID target) {
        CHECK_VECTORID_IS_VALID(target, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_CENTROID(target, LOG_TAG_DIVFTREE);

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);
        BufferVertexEntry* target_entry = bufferMgr->ReadBufferEntry(target, SX_SHARED);
        if (target_entry == nullptr) {
            return false;
        }
        Version target_ver = target_entry->currentVersion;

        VectorLocation target_loc;
        BufferVertexEntry* parent_entry = target_entry->ReadParentEntry(target_loc);
        if (parent_entry == nullptr) {
            bufferMgr->RemoveMergeTask(target, target_entry);
            bufferMgr->ReleaseBufferEntry(target_entry, ReleaseBufferEntryFlags(false, false));
            return false;
        }

        DIVFTreeVertex* target_cluster = static_cast<DIVFTreeVertex*>(target_entry->ReadLatestVersion(false));
        CHECK_NOT_NULLPTR(target_cluster, LOG_TAG_DIVFTREE);
        DIVFTreeVertex* parent_cluster = static_cast<DIVFTreeVertex*>(parent_entry->ReadLatestVersion(false));
        CHECK_NOT_NULLPTR(parent_cluster, LOG_TAG_DIVFTREE);

        uint16_t target_num_deleted = target_cluster->cluster.header.num_deleted.load(std::memory_order_acquire);
        uint16_t target_size = target_cluster->cluster.header.visible_size.load(std::memory_order_acquire);
        if (target_size - target_num_deleted == 0) {
            bufferMgr->RemoveMergeTask(target, target_entry);
            bufferMgr->ReleaseBufferEntry(parent_entry, ReleaseBufferEntryFlags(false, false));
            bufferMgr->ReleaseBufferEntry(target_entry, ReleaseBufferEntryFlags(false, false));
            return false;
        }

        VTYPE target_centroid[attr.dimension];
        bool res = ComputeCentroid(target_cluster->cluster, target_cluster->attr.block_size, target_cluster->attr.cap,
                                   target.IsLeaf(), nullptr, 0, attr.dimension, attr.distanceAlg, target_centroid,
                                   threadSelf->centroid_compute_buffer);
        if (!res) {
            bufferMgr->RemoveMergeTask(target, target_entry);
            bufferMgr->ReleaseBufferEntry(parent_entry, ReleaseBufferEntryFlags(false, false));
            bufferMgr->ReleaseBufferEntry(target_entry, ReleaseBufferEntryFlags(false, false));
            return false;
        }

        FatalAssert(parent_cluster->cluster.NumBlocks(parent_cluster->attr.block_size, parent_cluster->attr.cap) == 1,
                    LOG_TAG_NOT_IMPLEMENTED, "cannot handle more than 1 block!");
        VTYPE* data = parent_cluster->cluster.Data(0, false, parent_cluster->attr.block_size, parent_cluster->attr.cap,
                                                   attr.dimension);
        CentroidMetaData* vmd =
            static_cast<CentroidMetaData*>(parent_cluster->cluster.MetaData(0, false, parent_cluster->attr.block_size,
                                           parent_cluster->attr.cap, attr.dimension));

        uint16_t parent_size = parent_cluster->cluster.header.visible_size.load(std::memory_order_acquire);
        std::unordered_set<VectorID, VectorIDHash> seen;
        seen.reserve(parent_size);
        DTYPE best_dist;
        VectorID best_id = INVALID_VECTOR_ID;
        Version best_version;
        for (uint16_t i = parent_size - 1; i != UINT16_MAX; --i) {
            if ((vmd[i].id == target) || (seen.find(vmd[i].id) != seen.end()) ||
                vmd[i].state.load(std::memory_order_acquire) != VECTOR_STATE_VALID) {
                continue;
            }
            seen.emplace(vmd[i].id);

            if (best_id == INVALID_VECTOR_ID) {
                best_id = vmd[i].id;
                best_version = vmd[i].version;
                best_dist = Distance(target_centroid, &data[i * attr.dimension], attr.dimension, attr.distanceAlg);
            } else {
                DTYPE dist = Distance(target_centroid, &data[i * attr.dimension], attr.dimension, attr.distanceAlg);
                if (MoreSimilar(dist, best_dist, attr.distanceAlg) == 1) {
                    best_id = vmd[i].id;
                    best_version = vmd[i].version;
                    best_dist = dist;
                }
            }
        }

        bufferMgr->ReleaseBufferEntry(parent_entry, ReleaseBufferEntryFlags(false, false));
        bufferMgr->ReleaseBufferEntry(target_entry, ReleaseBufferEntryFlags(false, false));

        /* todo: change in multi node */
        if (best_id != INVALID_VECTOR_ID) {
            RetStatus rs = Merge(target, target_ver, best_id, best_version);
            if (!rs.IsOK() && (rs.stat == RetStatus::DEST_UPDATED)) {
                bool res = merge_tasks.Push(MergeTask{target});
                UNUSED_VARIABLE(res);
                FatalAssert(res, LOG_TAG_DIVFTREE, "this should not fail!");
            }

            return rs.IsOK();
        } else {
            bufferMgr->RemoveMergeTask(target, target_entry);
            return false;
        }
    }

    void SearchRoot(const VTYPE* query, size_t span,
                    std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers,
                    DIVFTreeVertex* pinned_root_version) {
        BufferManager* bufferMgr = BufferManager::GetInstance();
        FatalAssert(bufferMgr != nullptr, LOG_TAG_DIVFTREE, "BufferManager is not initialized.");
        CHECK_NOT_NULLPTR(pinned_root_version, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_VALID(pinned_root_version->attr.centroid_id, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_CENTROID(pinned_root_version->attr.centroid_id, LOG_TAG_DIVFTREE);
        uint8_t level = pinned_root_version->attr.centroid_id._level;
        FatalAssert(layers[level]->Size() == 1 &&
                    (*layers[level])[0].id == pinned_root_version->attr.centroid_id &&
                    (*layers[level])[0].version == pinned_root_version->attr.version, LOG_TAG_DIVFTREE,
                    "the highest level should only contain the pinned version of the root");
        FatalAssert(span > 0, LOG_TAG_DIVFTREE, "span cannot be 0");
        std::unordered_set<std::pair<VectorID, Version>, VectorIDVersionPairHash> seen;
        seen.reserve(pinned_root_version->attr.cap);
#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
            "Searching for query=%s in root=%s, neighbours=(%p)%s", VectorToString(query, attr.dimension).ToCStr(),
            pinned_root_version->ToString(true).ToCStr(), layers[level - 1],
            layers[level - 1]->ToString<ANNVectorInfoToString>().ToCStr());
#endif
        pinned_root_version->Search(query, span, layers[level - 1], seen);
#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
            "Searching for query=%s in root with id " VECTORID_LOG_FMT ", neighbours=(%p)%s",
            VectorToString(query, attr.dimension).ToCStr(), VECTORID_LOG(pinned_root_version->attr.centroid_id),
            layers[level - 1], layers[level - 1]->ToString<ANNVectorInfoToString>().ToCStr());
#endif
    }

    void SearchVertex(VectorID id, Version version, const VTYPE* query, size_t span,
                      SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
                      std::unordered_set<std::pair<VectorID, Version>, VectorIDVersionPairHash>& seen) {
        CHECK_VECTORID_IS_VALID(id, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_CENTROID(id, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(query, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(neighbours, LOG_TAG_DIVFTREE);
        FatalAssert(span > 0, LOG_TAG_DIVFTREE, "span should be larget than 0!");

        BufferManager* bufferMgr = BufferManager::GetInstance();
        FatalAssert(bufferMgr != nullptr, LOG_TAG_DIVFTREE, "BufferManager is not initialized.");
        bool outdated;
        DIVFTreeVertexInterface* vertex_tmp = nullptr;
        while (vertex_tmp == nullptr) {
            RetStatus rs = bufferMgr->ReadAndPinVertex(id, version, vertex_tmp, &outdated);
            /*
             * todo use stats to make sure this does not happen often! this can only happen when there was a split and
             * the new version is inserted into the parent but the clusterPtr is not yet updated.
             */
            if (vertex_tmp == nullptr) {
                if (rs.stat == RetStatus::VERSION_NOT_APPLIED) {
                    DIVFTREE_YIELD();
                    continue;
                }
                return;
            }
        }

        CHECK_NOT_NULLPTR(vertex_tmp, LOG_TAG_DIVFTREE);
        DIVFTreeVertex* vertex = static_cast<DIVFTreeVertex*>(vertex_tmp);
        if (!outdated) {
            uint16_t deleted = vertex->cluster.header.num_deleted.load(std::memory_order_acquire);
            uint16_t reserved = vertex->cluster.header.reserved_size.load(std::memory_order_acquire);
            if ((deleted > 0) && (threadSelf->UniformBinary((uint32_t)((double)attr.random_base_perc *
                                    (double)deleted / (double)reserved)))) {
                if (bufferMgr->AddCompactionTaskIfNotExists(vertex->attr.centroid_id)) {
                    bool res = compaction_tasks.Push(CompactionTask{vertex->attr.centroid_id});
                    UNUSED_VARIABLE(res);
                    FatalAssert(res, LOG_TAG_DIVFTREE, "this should not fail!");
                }
            }
        }
#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
            "Searching for query=%s in vertex=%s, neighbours=(%p)%s", VectorToString(query, attr.dimension).ToCStr(),
            vertex->ToString(true).ToCStr(), neighbours, neighbours->ToString<ANNVectorInfoToString>().ToCStr());
#endif
        vertex->Search(query, span, neighbours, seen);
#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
            "Searching for query=%s in vertex with id " VECTORID_LOG_FMT ", neighbours=(%p)%s",
            VectorToString(query, attr.dimension).ToCStr(), VECTORID_LOG(vertex->attr.centroid_id), neighbours,
            neighbours->ToString<ANNVectorInfoToString>().ToCStr());
#endif
        vertex->Unpin();
    }

    /* todo: use multiple threads for searching each layer -> what if we use a single pool for all searches?
       if there are few threads, they will do the search layer themselves but if there are free threads they
       can help each other */
    void SearchLayer(const VTYPE* query, size_t span,
                     std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers, uint8_t level) {
        FatalAssert(level < layers.size() - 1, LOG_TAG_DIVFTREE, "level out of bounds!");
        FatalAssert((uint64_t)level > VectorID::VECTOR_LEVEL, LOG_TAG_DIVFTREE, "level out of bounds!");
        FatalAssert(layers[level - 1]->Empty(), LOG_TAG_DIVFTREE, "next level should be empty!");
        std::unordered_set<std::pair<VectorID, Version>, VectorIDVersionPairHash> seen;
        seen.reserve(layers[level]->Size() * (((uint64_t)level == VectorID::LEAF_LEVEL) ? attr.leaf_max_size :
                                                                                         attr.internal_max_size));

#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
            "SearchLayer BEGIN: query=%s, layer=%hhu, upper_layer=(%p)%s ",
            VectorToString(query, attr.dimension).ToCStr(), level-1, layers[level],
            layers[level]->ToString<ANNVectorInfoToString>().ToCStr());
#endif
        if (layers[level]->Size() == 1) {
            FatalAssert((*layers[level])[0].id._level == (uint64_t)level, LOG_TAG_DIVFTREE, "mismatch level!");
            SearchVertex((*layers[level])[0].id, (*layers[level])[0].version, query, span, layers[level - 1], seen);
#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
            "SearchLayer END single: query=%s, layer=%hhu, upper_layer=(%p)%s, lower_layer=(%p)%s",
            VectorToString(query, attr.dimension).ToCStr(),
            level-1, layers[level]->ToString<ANNVectorInfoToString>().ToCStr(),
            layers[level-1]->ToString<ANNVectorInfoToString>().ToCStr());
#endif
            return;
        }

        uint64_t taskId = threadSelf->GetNextTaskID();
        SearchTask** task_ptrs = new SearchTask*[layers[level]->Size()];
        std::atomic<bool>* sync_bools = new std::atomic<bool>[layers[level]->Size() * 2];
        std::atomic<size_t>* viewed = new std::atomic<size_t>(0);
        for (size_t i = 0; i < layers[level]->Size(); ++i) {
            FatalAssert((*layers[level])[i].id._level == (uint64_t)level, LOG_TAG_DIVFTREE, "mismatch level!");
            task_ptrs[i] =
                new SearchTask(taskId, layers[level]->Size(), (*layers[level])[i].id, (*layers[level])[i].version,
                               query, span, &sync_bools[i], &sync_bools[i + layers[level]->Size()], viewed, sync_bools,
                               task_ptrs);
        }
#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
            "SearchLayer: query=%s, layer=%hhu, taskId=%lu ",
            VectorToString(query, attr.dimension).ToCStr(), level-1, taskId);
#endif

        BGSearchStatsUpdateCreatedTask(layers[level]->Size());
        bool rs = search_tasks.BatchPush(task_ptrs, layers[level]->Size());
        UNUSED_VARIABLE(rs);
        FatalAssert(rs, LOG_TAG_DIVFTREE, "should not fail!");

        for (size_t i = layers[level]->Size() - 1; i != UINT64_MAX; --i) {
            bool exp = false;
            if (!task_ptrs[i]->taken->compare_exchange_strong(exp, true)) {
                continue;
            }

            SearchVertex(task_ptrs[i]->target, task_ptrs[i]->version, query, span, layers[level - 1], seen);
            task_ptrs[i]->done->store(true, std::memory_order_relaxed);
        }

        size_t num_done = 0;
        bool first_time = true;
        while(num_done != layers[level]->Size()) {
            if (first_time) {
                first_time = false;
            } else {
                /* todo: is this too much and if so maybe we should use wait and notify instead or use YEILD */
                usleep(1);
            }
            for (size_t i = 0; i < layers[level]->Size(); ++i) {
                if (task_ptrs[i]->done_waiting) {
                    continue;
                }
                if (task_ptrs[i]->done->load(std::memory_order_acquire)) {
                    ++num_done;
                    task_ptrs[i]->done_waiting = true;
                }
            }
        }

        /* todo: check if I need a better algorithm for this */
        std::unordered_set<uintptr_t> lists;
        for (size_t i = 0; i < layers[level]->Size(); ++i) {
            if (task_ptrs[i]->neighbours == nullptr) {
                continue;
            }
            if (lists.find(reinterpret_cast<uintptr_t>(task_ptrs[i]->neighbours)) != lists.end()) {
                task_ptrs[i]->neighbours = nullptr;
                continue;
            }

            lists.insert(reinterpret_cast<uintptr_t>(task_ptrs[i]->neighbours));
            layers[level - 1]->MergeWith(*task_ptrs[i]->neighbours, span, false);
            delete task_ptrs[i]->neighbours;
            task_ptrs[i]->neighbours = nullptr;
        }

        uint8_t num_viewed = viewed->fetch_add(1) + 1;
        FatalAssert(num_viewed <= (layers[level]->Size() + 1), LOG_TAG_DIVFTREE, "too many views!");
        if (num_viewed == (layers[level]->Size() + 1)) {
            delete viewed;
            delete[] sync_bools;
            for (size_t i = 0; i < layers[level]->Size(); ++i) {
                delete task_ptrs[i];
            }
            delete[] task_ptrs;
        }

#ifdef EXCESS_LOGING
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
            "SearchLayer END: query=%s, layer=%hhu, taskId=%lu, upper_layer=(%p)%s, lower_layer=(%p)%s",
            VectorToString(query, attr.dimension).ToCStr(), level-1, taskId, layers[level],
            layers[level]->ToString<ANNVectorInfoToString>().ToCStr(), layers[level-1],
            layers[level-1]->ToString<ANNVectorInfoToString>().ToCStr());
#endif
    }

    RetStatus ANNSearch(const VTYPE* query, size_t k, uint8_t internal_node_search_span, uint8_t leaf_node_search_span,
                        uint8_t start_level, uint8_t end_level,
                        std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers,
                        DIVFTreeVertex* pinned_root_version) {
        BufferManager* bufferMgr = BufferManager::GetInstance();
        FatalAssert(bufferMgr != nullptr, LOG_TAG_DIVFTREE, "BufferManager is not initialized.");
        CHECK_NOT_NULLPTR(pinned_root_version, LOG_TAG_DIVFTREE);
        /*
         * Since a version of the root is pinned, we basicaly have an MVCC snapshop of the
         * database based on that root version and it's children are not deleted unless they are unpinned or empty
         */
        FatalAssert(pinned_root_version->attr.centroid_id._level >= start_level, LOG_TAG_DIVFTREE,
                    "start_level should be lower than or equal to root level!");
        FatalAssert(end_level < start_level, LOG_TAG_DIVFTREE, "last level cannot be higher than start level!");

        uint8_t current_level = start_level;
        uint8_t span;
        while (current_level > end_level) {
            /* migration trigger check */
            if ((layers[current_level]->Size() > 1) &&
                (threadSelf->UniformBinary(attr.migration_check_triger_rate))) {
                VectorID firstId = INVALID_VECTOR_ID;
                VectorID secondId = INVALID_VECTOR_ID;
                if (threadSelf->UniformBinary(attr.migration_check_triger_single_rate)) {
                    uint64_t index = threadSelf->UniformRange64(0, layers[current_level]->Size() - 1);
                    firstId = (*layers[current_level])[index].id;
                    /* todo: this will not work when we have multi node system as it relies on creator node id to be 0*/
                    secondId = bufferMgr->GetRandomCentroidIdAtLayer(current_level, firstId);
                    if (secondId == INVALID_VECTOR_ID) {
                        firstId = INVALID_VECTOR_ID;
                    }
                } else {
                    auto indices = threadSelf->UniformRangeTwo64(0, layers[current_level]->Size() - 1);
                    firstId = (*layers[current_level])[indices.first].id;
                    secondId = (*layers[current_level])[indices.second].id;
                    uint64_t num_retry = 0;
                    while(secondId == firstId) {
                        uint64_t index = threadSelf->UniformRange64(0, layers[current_level]->Size() - 1);
                        secondId = (*layers[current_level])[index].id;
                        ++num_retry;
                        if (num_retry >= Thread::MAX_RETRY) {
                            firstId = INVALID_VECTOR_ID;
                            secondId = INVALID_VECTOR_ID;
                            break;
                        }
                    }
                }

                if ((secondId != INVALID_VECTOR_ID) && (firstId != INVALID_VECTOR_ID) &&
                    bufferMgr->AddMigrationTaskIfNotExists(firstId, secondId)) {
                    bool res = migration_tasks.Push(MigrationCheckTask{.first=firstId, .second=secondId});
                    UNUSED_VARIABLE(res);
                    FatalAssert(res, LOG_TAG_DIVFTREE, "this should not fail!");
                }
            }


            uint8_t next_level = current_level - 1;
            if (next_level == end_level) {
                span = k;
            } else if ((uint64_t)next_level > VectorID::LEAF_LEVEL) {
                span = internal_node_search_span;
            } else {
                FatalAssert((uint64_t)next_level == VectorID::LEAF_LEVEL, LOG_TAG_DIVFTREE,
                            "since current cannot be lower than end level, "
                            "if it is at vector level, the first condition will handle it.");
                span = leaf_node_search_span;
            }
            FatalAssert(!layers[current_level]->Empty(), LOG_TAG_DIVFTREE,
                        "current level cannot be empty!");
            FatalAssert(layers[next_level]->Empty(), LOG_TAG_DIVFTREE,
                        "next level should be empty!");
            FatalAssert(span > 0, LOG_TAG_DIVFTREE,
                        "span should be at least 1");
            if (pinned_root_version->attr.centroid_id._level == current_level) {
                FatalAssert(layers[current_level]->Size() == 1 &&
                            (*layers[current_level])[0].id == pinned_root_version->attr.centroid_id &&
                            (*layers[current_level])[0].version == pinned_root_version->attr.version, LOG_TAG_DIVFTREE,
                            "the highest level should only contain the pinned version of the root");
                SearchRoot(query, span, layers, pinned_root_version);
            } else {
                SearchLayer(query, span, layers, current_level);
            }

            if (layers[next_level]->Empty()) {
                layers[current_level]->Clear();
                if (current_level == pinned_root_version->attr.centroid_id._level) {
                    return RetStatus{.stat=RetStatus::FAIL, .message=nullptr};
                }
                ++current_level;
                continue;
            }
            --current_level;
        }

        return RetStatus::Success();
    }

    inline void AsyncSearch(Thread* self, uint64_t idx) {
        CHECK_NOT_NULLPTR(self, LOG_TAG_DIVFTREE);
        self->InitDIVFThread(attr.dimension);

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

        std::unordered_set<std::pair<VectorID, Version>, VectorIDVersionPairHash> seen;
        seen.reserve(std::max(attr.internal_max_size, attr.leaf_max_size));
        SearchTask oldTask;
        oldTask.target = INVALID_VECTOR_ID;

        while (!end_signal.load(std::memory_order_acquire) || !search_tasks.Empty()) {
            SearchTask* nextTask = nullptr;
            self->LoopIncrement();
            FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                        "should not hold any lock here!");
            if (search_tasks.PopHead(nextTask)) {
                bool exp = false;
                if (!(nextTask->taken->compare_exchange_strong(exp, true))) {
                    size_t num_tasks = nextTask->num_tasks;
                    uint8_t num_viewed = nextTask->viewed->fetch_add(1) + 1;
                    FatalAssert(num_viewed <= (num_tasks + 1), LOG_TAG_DIVFTREE, "too many views!");
                    if (num_viewed == (num_tasks + 1)) {
                        delete nextTask->viewed;
                        delete[] nextTask->shared_data;
                        SearchTask** task_set = nextTask->task_set;
                        for (size_t i = 0; i < num_tasks; ++i) {
                            delete task_set[i];
                        }
                        delete[] task_set;
                    }
                    continue;
                }
                CHECK_VECTORID_IS_VALID(nextTask->target, LOG_TAG_DIVFTREE);

                if ((nextTask->master != oldTask.master) || (nextTask->taskId != oldTask.taskId) ||
                    (oldTask.target == INVALID_VECTOR_ID)) {
                    /* todo: we can add another bool pointer to this and when we change the neighbours, we set that
                       to true so that the user knows it can proceed with that list */
                    nextTask->neighbours =
                        new SortedList<ANNVectorInfo, SimilarityComparator>(attr.similarityComparator, nextTask->k);
                    seen.clear();
                } else {
                    CHECK_NOT_NULLPTR(oldTask.neighbours, LOG_TAG_DIVFTREE);
                    nextTask->neighbours = oldTask.neighbours;
                }
#ifdef EXCESS_LOGING
            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
                    "BGSearch BEGIN: taskId=%lu", nextTask->taskId);
#endif
                oldTask.CopyFrom(*nextTask);

                SearchVertex(nextTask->target, nextTask->version, nextTask->query, nextTask->k,
                             nextTask->neighbours, seen);
#ifdef EXCESS_LOGING
            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
                    "BGSearch END: taskId=%lu", nextTask->taskId);
#endif
                nextTask->done->store(true, std::memory_order_release);
                size_t num_tasks = nextTask->num_tasks;
                uint8_t num_viewed = nextTask->viewed->fetch_add(1) + 1;
                FatalAssert(num_viewed <= (num_tasks + 1), LOG_TAG_DIVFTREE, "too many views!");
                if (num_viewed == (num_tasks + 1)) {
                    delete nextTask->viewed;
                    delete[] nextTask->shared_data;
                    SearchTask** task_set = nextTask->task_set;
                    for (size_t i = 0; i < num_tasks; ++i) {
                        delete task_set[i];
                    }
                    delete[] task_set;
                }

                BGSearchStatsUpdate(idx, true);
            } else {
                BGSearchStatsUpdate(idx, false);
            }
            FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                        "should not hold any lock here!");
        }
        self->DestroyDIVFThread();
    }

    inline void BGMigration(Thread* self, uint64_t idx) {
        CHECK_NOT_NULLPTR(self, LOG_TAG_DIVFTREE);
        self->InitDIVFThread(attr.dimension);

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

        while (!end_signal.load(std::memory_order_acquire)) {
            self->LoopIncrement();
            MigrationCheckTask nextTask;
            uint64_t num_migrated = 0;
            FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                        "should not hold any lock here!");
            if (migration_tasks.TryPopHead(nextTask)) {
#ifdef EXCESS_LOGING
            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
                    "BGMigration existing task: 1) " VECTORID_LOG_FMT " 2) " VECTORID_LOG_FMT,
                    VECTORID_LOG(nextTask.first), VECTORID_LOG(nextTask.second));
#endif
                num_migrated = MigrationCheck(nextTask.first, nextTask.second);
                bufferMgr->RemoveMigrationTask(nextTask.first, nextTask.second);
                BGMigrationStatsUpdate(idx, true, num_migrated);
            } else {
                auto ids = bufferMgr->GetTwoRandomCentroidIdAtNonRootLayer();
                if ((ids.first != INVALID_VECTOR_ID) && (ids.second != INVALID_VECTOR_ID)) {
#ifdef EXCESS_LOGING
                    DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
                            "BGMigration new task: 1) " VECTORID_LOG_FMT " 2) " VECTORID_LOG_FMT,
                            VECTORID_LOG(ids.first), VECTORID_LOG(ids.second));
#endif
                    num_migrated = MigrationCheck(ids.first, ids.second);
                } else {
                    num_migrated = 0;
                    usleep(1);
                }
                BGMigrationStatsUpdate(idx, false, num_migrated);
            }
            FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                        "should not hold any lock here!");
        }
        self->DestroyDIVFThread();
    }

    inline void BGMerge(Thread* self, uint64_t idx) {
        CHECK_NOT_NULLPTR(self, LOG_TAG_DIVFTREE);
        self->InitDIVFThread(attr.dimension);

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

        VectorID oldTarget = INVALID_VECTOR_ID;
        while (!end_signal.load(std::memory_order_acquire)) {
            self->LoopIncrement();
            FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                        "should not hold any lock here!");
            MergeTask nextTask;
            if (merge_tasks.PopHead(nextTask)) {
#ifdef EXCESS_LOGING
                DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
                        "BGMerge task " VECTORID_LOG_FMT,
                        VECTORID_LOG(nextTask.target));
#endif
                if (oldTarget == nextTask.target) {
                    msleep(1);
                }
                oldTarget = nextTask.target;
                bool succ = MergeCheck(nextTask.target);
                BGMergeStatsUpdate(idx, true, succ);
            } else {
                oldTarget = INVALID_VECTOR_ID;
                BGMergeStatsUpdate(idx, false, false);
            }
            FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                        "should not hold any lock here!");
        }
        self->DestroyDIVFThread();
    }

    inline void BGCompaction(Thread* self, uint64_t idx) {
        CHECK_NOT_NULLPTR(self, LOG_TAG_DIVFTREE);
        self->InitDIVFThread(attr.dimension);

        BufferManager* bufferMgr = BufferManager::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

        while (!end_signal.load(std::memory_order_acquire)) {
            self->LoopIncrement();
            FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                        "should not hold any lock here!");
            CompactionTask nextTask;
            if (compaction_tasks.PopHead(nextTask)) {
#ifdef EXCESS_LOGING
                DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
                        "BGCompaction task " VECTORID_LOG_FMT,
                        VECTORID_LOG(nextTask.target));
#endif
                bool succ = CompactionCheck(nextTask.target);
                BGCompactionStatsUpdate(idx, true, succ);
            } else {
                BGCompactionStatsUpdate(idx, false, false);
            }
            FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                        "should not hold any lock here!");
        }
        self->DestroyDIVFThread();
    }

#ifdef HANG_DETECTION
    inline void BGHangDetector(Thread* self) {
        CHECK_NOT_NULLPTR(self, LOG_TAG_HANG_DETECTOR);
        self->InitDIVFThread();

        while (!end_bghang_detector.load(std::memory_order_acquire)) {
            msleep(BG_HANG_DETECTOR_SLEEP_MS);
            Thread::all_threads_lock.lock();
            uint64_t num_threads = Thread::all_threads.size();
            self->LoopIncrement();
            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_HANG_DETECTOR,
                    "BGHangDetector checking %lu threads", num_threads);
            for (auto& thread : Thread::all_threads) {
                FatalAssert(thread != nullptr, LOG_TAG_HANG_DETECTOR, "null thread found!");
                if (thread->CheckHang()) {
                    DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_HANG_DETECTOR,
                            "Detected hang in thread id %lu",
                            thread->ID());
                }
            }
            Thread::all_threads_lock.unlock();
        }
        self->DestroyDIVFThread();
    }
#endif

    inline void StartBGThreads() {
        searchers.reserve(attr.num_searchers);
        for (size_t i = 0; i < attr.num_searchers; ++i) {
            searchers.emplace_back(new Thread(attr.random_base_perc));
            searchers.back()->StartMemberFunction(&DIVFTree::AsyncSearch, this, i);
        }

        bg_migrators.reserve(attr.num_migrators);
        for (size_t i = 0; i < attr.num_migrators; ++i) {
            bg_migrators.emplace_back(new Thread(attr.random_base_perc));
            bg_migrators.back()->StartMemberFunction(&DIVFTree::BGMigration, this, i);
        }

        bg_mergers.reserve(attr.num_mergers);
        for (size_t i = 0; i < attr.num_mergers; ++i) {
            bg_mergers.emplace_back(new Thread(attr.random_base_perc));
            bg_mergers.back()->StartMemberFunction(&DIVFTree::BGMerge, this, i);
        }

        bg_compactors.reserve(attr.num_compactors);
        for (size_t i = 0; i < attr.num_compactors; ++i) {
            bg_compactors.emplace_back(new Thread(attr.random_base_perc));
            bg_compactors.back()->StartMemberFunction(&DIVFTree::BGCompaction, this, i);
        }

#ifdef HANG_DETECTION
        bg_hang_detector = new Thread(attr.random_base_perc);
        bg_hang_detector->StartMemberFunction(&DIVFTree::BGHangDetector, this);
#endif
    }

    inline void DestroyBGThreads() {
        FatalAssert(end_signal.load(std::memory_order_acquire), LOG_TAG_DIVFTREE,
                    "We should be ending if we are here!");
        for (size_t i = 0; i < attr.num_compactors; ++i) {
            delete bg_compactors[i];
        }

        for (size_t i = 0; i < attr.num_mergers; ++i) {
            delete bg_mergers[i];
        }

        for (size_t i = 0; i < attr.num_migrators; ++i) {
            delete bg_migrators[i];
        }

        for (size_t i = 0; i < attr.num_searchers; ++i) {
            delete searchers[i];
        }

#ifdef HANG_DETECTION
        end_bghang_detector.store(true, std::memory_order_release);
        delete bg_hang_detector;
#endif
    }

TESTABLE;
};

};

#endif