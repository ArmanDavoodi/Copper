#ifndef DIVFTREE_IMPL_H_
#define DIVFTREE_IMPL_H_

#include "disaggregated/compute_node/divftree.h"
#include "disaggregated/compute_node/buffer.h"
#include "disaggregated/comm_layer.h"


// Todo: better logs and asserts -> style: <Function name>(self data(a=?), input data): msg, additonal variables if needed

namespace divftree {

// DIVFTreeVertex::DIVFTreeVertex(const DIVFTreeVertexAttributes& attributes);
// DIVFTreeVertex::DIVFTreeVertex(const DIVFTreeAttributes& attributes, VectorID id, DIVFTreeInterface* index);

// DIVFTreeVertex::~DIVFTreeVertex();

// RetStatus DIVFTreeVertex::BatchUpdate(const SortedList<std::pair<UpdateType, void*>, UpdateCMP>& updates);

/*
 *
 * for the vector metadata we have one byte for state(atomic) and 2 bytes for batch size/last offset
 * If a vector state is Invalid_something then it means that neighter the vector data nor the batch info is valid
 * If a vector state is pute Invalid, it means that the slot is empty
 * The only case where we insert vectors in atomic batches is when we have split.
 *  Note: this means that vertices inserted in a migration are not in the same batch!
 *        because they are logically independent insertions
 * If we have out-of-order messages causing the state of a vector to be invalid_something,
 *  then when the insertion comes we can insert the data and make the state valid(it could be valid outdated etc)
 *  and then depending on the state, they also insert the data in other vertices.
 * As long as there is a single Invalid or Invalid_something state in a batch, the whole batch is considered invalid
 * The reson for this is that if there is an invalid vector in a bathc, that means that we do not have the
 *  data of that vector and since the vectors in a batch are dependant, not reading them all can cause us
 *  to ignore some parts of the dataset.
 * Since we are first checking in a hash table that a vector has already been checked, we do not need to worry
 * about computation costs of reading an outdated vector
 */

enum BatchState {
    BATCH_STATE_VALID,
    BATCH_STATE_INVALID,
    BATCH_STATE_MID
};

template <bool is_leaf>
inline void _GetVectorMeta(Address meta, ClusterSizeType index, VectorState& state, VectorBatchMeta& batch_meta,
                           VectorID& target_id, Version& target_version, bool& batch_valid) {
    if constexpr (is_leaf) {
        VectorMetaData* vmd = reinterpret_cast<VectorMetaData*>(meta);
        state = vmd[index].state.load(std::memory_order_acquire);
        batch_valid = vmd[index].batch_valid.load(std::memory_order_acquire);
        batch_meta = vmd[index].batch_meta;
        target_id = vmd[index].id;
        target_version = 0;
    } else {
        CentroidMetaData* vmd = reinterpret_cast<CentroidMetaData*>(meta);
        state = vmd[index].state.load(std::memory_order_acquire);
        batch_valid = vmd[index].batch_valid.load(std::memory_order_acquire);
        batch_meta = vmd[index].batch_meta;
        target_id = vmd[index].id;
        target_version = vmd[index].version;
    }
}

template <bool is_leaf>
inline BatchState _IsBatchValid(Address meta, ClusterSizeType batch_start, ClusterSizeType& batch_end,
                                BufferManager* buffer) {
    VectorState state;
    VectorBatchMeta batch_meta;
    VectorID target_id;
    Version target_version;
    bool batch_valid;
    _GetVectorMeta<is_leaf>(meta, batch_start, state, batch_meta, target_id, target_version, batch_valid);

    if (!state.is_state_valid) {
        return BATCH_STATE_INVALID;
    }

    if (!batch_meta.is_batch_size) {
        /* we are in middle of a batch! */
        batch_start = batch_meta.batch_size_or_last_offset;
        _GetVectorMeta<is_leaf>(meta, batch_start, state, batch_meta, target_id, target_version, batch_valid);
        if (!state.is_state_valid) {
            /* the batch header is invalid so we cannot use its data to see where the batch ends */
            return BATCH_STATE_MID;
        }
    }

    FatalAssert(state.is_state_valid, LOG_TAG_DIVFTREE_VERTEX,
                "State should be valid here!");
    FatalAssert(batch_meta.is_batch_size == 1,
                LOG_TAG_DIVFTREE_VERTEX,
                "The last element in a batch must have batch size info");
    batch_end = batch_start - batch_meta.batch_size_or_last_offset;
    return batch_valid ? BATCH_STATE_VALID : BATCH_STATE_INVALID;
}

template <bool is_leaf>
inline void _SearchBatch(const VTYPE* query, size_t k, SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
                         BufferManager* buffer, VTYPE* data, Address meta, ClusterSizeType batch_start,
                         ClusterSizeType batch_end, uint16_t dim, DistanceType dtype,
                         ConcurrentHashTable<std::pair<VectorID, Version>, bool, VectorIDVersionPairCMP,
                                             VectorIDVersionPairHash>& seen,
                         std::unordered_set<VectorID, VectorIDHash>* in_cluster = nullptr) {
    for (ClusterSizeType i = batch_start; i > batch_end; --i) {
        VectorID target_id;
        if constexpr (is_leaf) {
            UNUSED_VARIABLE(in_cluster);
            UNUSED_VARIABLE(buffer);
            VectorMetaData* vmd = reinterpret_cast<VectorMetaData*>(meta);
            VectorState state = vmd[i].state.load(std::memory_order_acquire);
            FatalAssert(state.is_state_valid, LOG_TAG_DIVFTREE_VERTEX,
                        "State should be valid here!");
            FatalAssert(state.detail != VECTOR_STATE_OUTDATED,
                        LOG_TAG_DIVFTREE_VERTEX,
                        "a pure vector cannot become outdated!");
            if (state.detail == VECTOR_STATE_DELETED ||
                !seen.Emplace(std::make_pair(vmd[i].id, 0), true)) {
                continue;
            }
            target_id = vmd[i].id;
        } else {
            CentroidMetaData* vmd = reinterpret_cast<CentroidMetaData*>(meta);
            VectorState state = vmd[i].state.load(std::memory_order_acquire);
            FatalAssert(state.is_state_valid, LOG_TAG_DIVFTREE_VERTEX,
                        "State should be valid here!");
            if (in_cluster != nullptr) {
                in_cluster->emplace(vmd[i].id);
            }
            if (state.detail == VECTOR_STATE_DELETED ||
                !buffer->Exists(vmd[i].id, vmd[i].version) ||
                !seen.Emplace(std::make_pair(vmd[i].id, 0), true)) {
                continue;
            }

            if (in_cluster != nullptr && state.detail == VECTOR_STATE_OUTDATED &&
                (in_cluster->find(vmd[i].id) != in_cluster->end())) {
                continue;
            }
            target_id = vmd[i].id;
        }

        neighbours->Insert(ANNVectorInfo(Distance(query, &data[i * dim], dim, dtype), target_id));
        if (neighbours->Size() > k) {
            neighbours->PopBack();
        }
        FatalAssert(neighbours->Size() <= k, LOG_TAG_DIVFTREE_VERTEX,
                    "Neighbour list size exceeded k after insertion!");
    }
}

void DIVFTreeVertex::Search(const VTYPE* query, size_t k, SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
                            ConcurrentHashTable<std::pair<VectorID, Version>, bool, VectorIDVersionPairCMP,
                                                VectorIDVersionPairHash>& seen) {
    CHECK_NOT_NULLPTR(query, LOG_TAG_DIVFTREE_VERTEX);
    CHECK_NOT_NULLPTR(neighbours, LOG_TAG_DIVFTREE_VERTEX);
    FatalAssert(k > 0, LOG_TAG_DIVFTREE_VERTEX, "k must be greater than 0!");
    FatalAssert(neighbours->Size() <= k, LOG_TAG_DIVFTREE_VERTEX, "neighbour list size out of bounds!");
    const uint16_t dim = attr.index->GetAttributes().dimension;
    const DistanceType dtype = attr.index->GetAttributes().distanceAlg;
    BufferManager* buffer = BufferManager::GetInstance();
    CHECK_NOT_NULLPTR(buffer, LOG_TAG_DIVFTREE_VERTEX);
    VTYPE* data = cluster.Data(0, attr.centroid_id.IsLeaf(), attr.block_size, attr.cap, dim);
    Address meta = cluster.MetaData(0, attr.centroid_id.IsLeaf(), attr.block_size, attr.cap, dim);
    ClusterSizeType curr_size = size.load(std::memory_order_acquire);
    std::unordered_set<VectorID, VectorIDHash> in_cluster;
    std::vector<ClusterSizeType> invalid_batches;
    ClusterSizeType batch_start = curr_size - 1;
    ClusterSizeType batch_end = curr_size - 2;
    while(batch_start != (ClusterSizeType)(-1)) {
        FatalAssert(batch_start < curr_size, LOG_TAG_DIVFTREE_VERTEX,
                    "Batch start is out of bounds. batch_start=%hu, current_size=%hu",
                    batch_start, curr_size);
        BatchState batch_state;
        if (attr.centroid_id.IsLeaf()) {
            batch_state = _IsBatchValid<true>(meta, batch_start, batch_end, buffer);
        } else {
            batch_state = _IsBatchValid<false>(meta, batch_start, batch_end, buffer);
        }

        if (batch_state != BATCH_STATE_VALID) {
            while (invalid_batches.size() > 0 &&
                   invalid_batches.back() <= batch_start) {
                /* thse are in middle of the batch! */
                invalid_batches.pop_back();
            }
            invalid_batches.push_back(batch_start);
            batch_start = batch_end;
            batch_end = batch_start - 1;
            continue;
        }

        if (attr.centroid_id.IsLeaf()) {
            _SearchBatch<true>(query, k, neighbours, buffer, data, meta, batch_start, batch_end, dim, dtype, seen);
        } else {
            _SearchBatch<false>(query, k, neighbours, buffer, data, meta, batch_start, batch_end, dim, dtype, seen,
                                &in_cluster);
        }

        batch_start = batch_end;
        batch_end = batch_start - 1;
    }

    for (ClusterSizeType invalid_batch_start : invalid_batches) {
        ClusterSizeType invalid_batch_end = invalid_batch_start - 1;
        ClusterSizeType batch_start = invalid_batch_start;
        if (attr.centroid_id.IsLeaf() &&
            _IsBatchValid<true>(meta, batch_start, invalid_batch_end, buffer) == BATCH_STATE_VALID) {
            _SearchBatch<true>(query, k, neighbours, buffer, data, meta, batch_start,
                               invalid_batch_end, dim, dtype, seen);
        } else if (!attr.centroid_id.IsLeaf() &&
                   _IsBatchValid<false>(meta, batch_start, invalid_batch_end, buffer) ==
                       BATCH_STATE_VALID) {
            _SearchBatch<false>(query, k, neighbours, buffer, data, meta, batch_start,
                                invalid_batch_end, dim, dtype, seen);
        }
    }
}

// inline const DIVFTreeVertexAttributes& DIVFTreeVertex::GetAttributes() const;
// inline uint64_t DIVFTreeVertex::GetVisibleSize() const;
// inline VectorID DIVFTreeVertex::CentroidID() const;
// inline Version DIVFTreeVertex::VertexVersion() const;
// String DIVFTreeVertex::ToString(bool detailed = false) const;

// DIVFTree::DIVFTree(DIVFTreeAttributes attributes) {}
// DIVFTree::~DIVFTree() {}

RetStatus DIVFTree::Insert(const VTYPE* vec, VectorID& vec_id, uint8_t search_span,
                           bool create_completion_notification) {
    CHECK_NOT_NULLPTR(vec, LOG_TAG_DIVFTREE);
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
    CommLayer* commLayer = bufferMgr->GetCommLayer();
    CHECK_NOT_NULLPTR(commLayer, LOG_TAG_DIVFTREE);

    std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*> layers;
    VectorID target_leaf;
    Version target_version;
    Version root_version;
    BufferVertexEntry* leaf_entry = nullptr;

    while (true) {
        BufferVertexEntry* root = nullptr;
        rs = ReadAndPinRoot(root, root_version);
        FatalAssert(rs.IsOK(), LOG_TAG_DIVFTREE,
                    "Failed to read and pin root in Insert: %s", rs.Msg());
        CHECK_NOT_NULLPTR(root, LOG_TAG_DIVFTREE);

        if (root->selfId.IsLeaf()) {
            target_leaf = root->selfId;
            target_version = root_version;
            root->Unpin(root_version);
            break;
        }

        layers.reserve(root->selfId._level + 1);
        /* level 0 represents vectors and we do not need them */
        layers.emplace_back(nullptr);
        for (uint64_t i = 1; i <= root->selfId._level; ++i) {
            layers.emplace_back(
                new SortedList<ANNVectorInfo, SimilarityComparator>(attr.similarityComparator));
        }

        layers[root->selfId._level]->Insert(ANNVectorInfo(0, root->selfId, root_version));
        rs = ANNSearch(vec, 1, search_span, 1,
                        (uint8_t)(root->selfId._level), (uint8_t)VectorID::LEAF_LEVEL,
                        layers, root->Read(root_version));

        root->Unpin(root_version);
        FatalAssert(rs.IsOK(), LOG_TAG_DIVFTREE,
                    "ANNSearch failed in Insert: %s", rs.Msg());
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

    vec_id = GenerateNextVectorID(VectorID::VECTOR_LEVEL);
    CHECK_VECTORID_IS_VALID(vec_id, LOG_TAG_DIVFTREE);
    CHECK_VECTORID_IS_VECTOR(vec_id, LOG_TAG_DIVFTREE);
    if (create_completion_notification) {
        handleLock.Lock(SX_EXCLUSIVE);
        FatalAssert(completion_handles.find(vec_id) == completion_handles.end(), LOG_TAG_DIVFTREE,
                    "Completion handle for vector " VECTORID_LOG_FMT " already exists!",
                    VECTORID_LOG(vec_id));
        completion_handles[vec_id] = CompletionHandle{
            .type = UpdateType::INSERTION,
            .vector_id = vec_id,
            .completed = false,
            .status = RetStatus::Success()
        };
        handleLock.Unlock();
    }

    CommLayerMessage message;
    commLayer->BuildRequestMessage(MEMORY_NODE_ID, MessageType::CN_TO_MN_INSERT_REQUEST, message,
                                   InsertRequestMessage::Size(attr.dimension));

    InsertRequestMessage* insert_msg =
        reinterpret_cast<InsertRequestMessage*>(message.GetMessageBuffer());
    CHECK_NOT_NULLPTR(insert_msg, LOG_TAG_DIVFTREE);
    FatalAssert(insert_msg->type == MessageType::CN_TO_MN_INSERT_REQUEST, LOG_TAG_DIVFTREE,
                "Message type mismatch in InsertRequestMessage: expected %u, got %u",
                static_cast<uint8_t>(MessageType::CN_TO_MN_INSERT_REQUEST),
                static_cast<uint8_t>(insert_msg->type));
    insert_msg->target_leaf = target_leaf;
    insert_msg->target_version = target_version;
    insert_msg->vector = vec_id;
    memcpy(insert_msg->data, vec, attr.dimension * sizeof(VTYPE));
    rs = commLayer->SendMessage(message, false);
    FatalAssert(rs.IsOK(), LOG_TAG_DIVFTREE,
                "Failed to send CN_TO_MN_INSERT_REQUEST message for vector " VECTORID_LOG_FMT ": %s",
                VECTORID_LOG(vec_id), rs.Msg());

#ifdef EXCESS_LOGING
    DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
            "Insert END: Vector=%s, vector_id=" VECTORID_LOG_FMT ", rs=%s",
            VectorToString(vec, attr.dimension).ToCStr(), VECTORID_LOG(vec_id), RetStatusToString(rs));
#endif
    FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                "should not hold any lock here!");
    return rs;
}

RetStatus DIVFTree::Delete(VectorID vec_id, bool create_completion_notification) {
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
    CommLayer* commLayer = bufferMgr->GetCommLayer();
    CHECK_NOT_NULLPTR(commLayer, LOG_TAG_DIVFTREE);
    if (create_completion_notification) {
        handleLock.Lock(SX_EXCLUSIVE);
        auto it = completion_handles.find(vec_id);
        if (it != completion_handles.end()) {
            if (it->second.type == UpdateType::DELETION) {
                handleLock.Unlock();
                DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_DIVFTREE,
                        "Completion handle for vector " VECTORID_LOG_FMT " already exists for DELETE!",
                        VECTORID_LOG(vec_id));
                return RetStatus(RetStatus::DUPLICATE_DELETE);
            } else {
                FatalAssert(it->second.type == UpdateType::INSERTION, LOG_TAG_DIVFTREE,
                            "Completion handle for vector " VECTORID_LOG_FMT " has invalid type!",
                            VECTORID_LOG(vec_id));
                if (it->second.completed == false) {
                    /* the insert has not completed yet, so we can just remove the handle */
                    handleLock.Unlock();
                    DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_DIVFTREE,
                            "Cannot create DELETE completion handle for vector " VECTORID_LOG_FMT
                            " since INSERT has not completed yet!",
                            VECTORID_LOG(vec_id));
                    return RetStatus(RetStatus::INSERT_NOT_COMPLETED);
                } else {
                    /* the insert has completed, we can change the handle to DELETE */
                    if (it->second.status.IsOK()) {
                        /* only if the insert was successful we can change to delete */
                        /* todo: if for poll we see delete we return true for the completion */
                        it->second.type = UpdateType::DELETION;
                        it->second.completed = false;
                        FatalAssert(it->second.status.message == nullptr, LOG_TAG_DIVFTREE,
                                    "Completion handle for vector " VECTORID_LOG_FMT
                                    " has non-null message on changing from INSERT to DELETE!",
                                    VECTORID_LOG(vec_id));
                        handleLock.Unlock();
                    } else {
                        handleLock.Unlock();
                        DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_DIVFTREE,
                                "Cannot create DELETE completion handle for vector " VECTORID_LOG_FMT
                                " since INSERT failed with status: %u:%s",
                                VECTORID_LOG(vec_id), static_cast<uint32_t>(it->second.status.stat),
                                it->second.status.Msg());
                        char* fail_message = new char[256];
                        snprintf(fail_message, 256,
                                 "Vector not found due to failed INSERT");
                        return RetStatus(RetStatus::VECTOR_NOT_FOUND);
                    }
                }
            }
        } else {
            completion_handles[vec_id] = CompletionHandle{
                .type = UpdateType::DELETION,
                .vector_id = vec_id,
                .completed = false,
                .status = RetStatus::Success()
            };
            handleLock.Unlock();
        }
    }

    CommLayerMessage message;
    commLayer->BuildRequestMessage(MEMORY_NODE_ID, MessageType::CN_TO_MN_DELETE_REQUEST, message,
                                   DeleteRequestMessage::Size());
    DeleteRequestMessage* delete_msg =
        reinterpret_cast<DeleteRequestMessage*>(message.GetMessageBuffer());
    CHECK_NOT_NULLPTR(delete_msg, LOG_TAG_DIVFTREE);
    FatalAssert(delete_msg->type == MessageType::CN_TO_MN_DELETE_REQUEST, LOG_TAG_DIVFTREE,
                "Message type mismatch in DeleteRequestMessage: expected %u, got %u",
                static_cast<uint8_t>(MessageType::CN_TO_MN_DELETE_REQUEST),
                static_cast<uint8_t>(delete_msg->type));
    delete_msg->target_vector = vec_id;
    delete_msg->need_response = create_completion_notification;
    RetStatus rs = commLayer->SendMessage(message, false);
    FatalAssert(rs.IsOK(), LOG_TAG_DIVFTREE,
                "Failed to send CN_TO_MN_DELETE_REQUEST message for vector " VECTORID_LOG_FMT ": %s",
                VECTORID_LOG(vec_id), rs.Msg());

#ifdef EXCESS_LOGING
    DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
            "DELETE END: vector_id=" VECTORID_LOG_FMT,
            VECTORID_LOG(vec_id));
#endif
    FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                "should not hold any lock here!");
    return rs;
}

RetStatus DIVFTree::ApproximateKNearestNeighbours(const VTYPE* query, size_t k, uint8_t internal_node_search_span,
                                                  uint8_t leaf_node_search_span, SortType sort_type,
                                                  std::vector<ANNVectorInfo>& neighbours) {
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
    Version root_version;
    BufferVertexEntry* root = nullptr;
    rs = ReadAndPinRoot(root, root_version);
    CHECK_NOT_NULLPTR(root, LOG_TAG_DIVFTREE);
    FatalAssert(rs.IsOK(), LOG_TAG_DIVFTREE,
                "Failed to read and pin root in Insert: %s", rs.Msg());

    layers.reserve(root->selfId._level + 1);
    for (uint64_t i = 0; i <= root->selfId._level; ++i) {
        layers.emplace_back(new SortedList<ANNVectorInfo, SimilarityComparator>(attr.similarityComparator));
    }

    layers[root->selfId._level]->Insert(ANNVectorInfo(0, root->selfId, root_version));
    rs = ANNSearch(query, k, internal_node_search_span, leaf_node_search_span,
                    (uint8_t)(root->selfId._level), (uint8_t)VectorID::VECTOR_LEVEL, layers, root->Read(root_version));
    FatalAssert(rs.IsOK(), LOG_TAG_DIVFTREE,
                "ANNSearch failed in ApproximateKNearestNeighbours: %s", rs.Msg());
    root->Unpin(root_version);

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

    FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                "should not hold any lock here!");
    return rs;
}

size_t DIVFTree::ApproximateSize() const {
    return appr_size.load(std::memory_order_acquire);
}

const DIVFTreeAttributes& DIVFTree::GetAttributes() const {
    return attr;
}

// inline void DIVFTree::EndBGThreads();

// inline String DIVFTree::GetStatistics(std::string title_extention, bool clear_stats);
// inline void DIVFTree::StartStatsCollection();
// inline void DIVFTree::StopStatsCollection();
// inline void DIVFTree::ClearStats();

// inline VectorID DIVFTree::GenerateNextVectorID(uint8_t level);

// inline void DIVFTree::ClearStats(bool need_lock);
// void DIVFTree::BGMigrationStatsUpdate(uint64_t thread_index, bool completed_task, uint64_t num_migrated_vectors);
// void DIVFTree::BGMergeStatsUpdate(uint64_t thread_index, bool completed_task, bool cluster_merged);
// void DIVFTree::BGSearchStatsUpdate(uint64_t thread_index, bool completed_task);
// void DIVFTree::BGSearchStatsUpdateCreatedTask(uint64_t num_tasks);

// inline void DIVFTree::RoundRobinClustering(BufferVertexEntry* base, const ConstVectorBatch& batch,
//                                            BufferVertexEntry**& entries, ClusterSizeType marked_for_update);
// inline void DIVFTree::Clustering(BufferVertexEntry* base, const ConstVectorBatch& batch,
//                         BufferVertexEntry**& entries, ClusterSizeType marked_for_update);
// RetStatus DIVFTree::SplitAndInsert(VectorID target_id, Version target_version, uintptr_t target_remote_addr,
//                             uintptr_t* remote_addrs, const ConstVectorBatch& batch,
//                             ClusterSizeType marked_for_update);

// inline RetStatus DIVFTree::ReadAndCheckVersion(VectorID containerId, Version containerVersion,
//                                         BufferVertexEntry** entries, uint16_t max_entries, uint16_t& num_entries,
//                                         LockMode mode);
// RetStatus DIVFTree::Migrate(std::vector<MigrationInfo> targetBatch,
//                     VectorID src_id, VectorID dest_id,
//                     Version src_ver, Version dest_ver, uint64_t& num_migrated);
// ClusterSizeType DIVFTree::MigrationCheck(VectorID first_cluster, VectorID second_cluster);
// RetStatus DIVFTree::Merge(VectorID srcId, Version srcVersion,
//                 VectorID destId, Version destVersion);
// bool DIVFTree::MergeCheck(VectorID target, Version targetVersion, VectorID parent, Version parentVersion,
//                 uintptr_t target_remote_addr, uintptr_t parent_remote_addr);

void DIVFTree::SearchRoot(const VTYPE* query, size_t span,
                          std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers,
                          DIVFTreeVertex& pinned_root_version) {
    BufferManager* bufferMgr = BufferManager::GetInstance();
    FatalAssert(bufferMgr != nullptr, LOG_TAG_DIVFTREE, "BufferManager is not initialized.");
    CHECK_NOT_NULLPTR(&pinned_root_version, LOG_TAG_DIVFTREE);
    CHECK_VECTORID_IS_VALID(pinned_root_version.attr.centroid_id, LOG_TAG_DIVFTREE);
    CHECK_VECTORID_IS_CENTROID(pinned_root_version.attr.centroid_id, LOG_TAG_DIVFTREE);
    uint8_t level = pinned_root_version.attr.centroid_id._level;
    FatalAssert(layers[level]->Size() == 1 &&
                (*layers[level])[0].id == pinned_root_version.attr.centroid_id &&
                (*layers[level])[0].version == pinned_root_version.attr.version, LOG_TAG_DIVFTREE,
                "the highest level should only contain the pinned version of the root");
    FatalAssert(span > 0, LOG_TAG_DIVFTREE, "span cannot be 0");
    ConcurrentHashTable<std::pair<VectorID, Version>, bool, VectorIDVersionPairCMP,
                        VectorIDVersionPairHash> seen(pinned_root_version.attr.cap);
#ifdef EXCESS_LOGING
    DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
        "Searching for query=%s in root=%s, neighbours=(%p)%s", VectorToString(query, attr.dimension).ToCStr(),
        pinned_root_version.ToString(true).ToCStr(), layers[level - 1],
        layers[level - 1]->ToString<ANNVectorInfoToString>().ToCStr());
#endif
    pinned_root_version.Search(query, span, layers[level - 1], seen);
#ifdef EXCESS_LOGING
    DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
        "Searching for query=%s in root with id " VECTORID_LOG_FMT ", neighbours=(%p)%s",
        VectorToString(query, attr.dimension).ToCStr(), VECTORID_LOG(pinned_root_version.attr.centroid_id),
        layers[level - 1], layers[level - 1]->ToString<ANNVectorInfoToString>().ToCStr());
#endif
}

void DIVFTree::SearchVertex(VectorID id, Version version, const VTYPE* query, size_t span,
                            SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
                            ConcurrentHashTable<std::pair<VectorID, Version>, bool, VectorIDVersionPairCMP,
                                                VectorIDVersionPairHash>& seen) {
    CHECK_VECTORID_IS_VALID(id, LOG_TAG_DIVFTREE);
    CHECK_VECTORID_IS_CENTROID(id, LOG_TAG_DIVFTREE);
    CHECK_NOT_NULLPTR(query, LOG_TAG_DIVFTREE);
    CHECK_NOT_NULLPTR(neighbours, LOG_TAG_DIVFTREE);
    FatalAssert(span > 0, LOG_TAG_DIVFTREE, "span should be larget than 0!");

    BufferManager* bufferMgr = BufferManager::GetInstance();
    FatalAssert(bufferMgr != nullptr, LOG_TAG_DIVFTREE, "BufferManager is not initialized.");
    bool outdated;
    DIVFTreeVertex* vertex = nullptr;
    RetStatus rs = bufferMgr->ReadVertexIfAvailable(id, version, vertex, &outdated);
    FatalAssert(rs.IsOK(), LOG_TAG_DIVFTREE,
                "Failed to read vertex " VECTORID_LOG_FMT " version %u in SearchVertex: %s",
                VECTORID_LOG(id), version._raw, rs.Msg());
    CHECK_NOT_NULLPTR(vertex, LOG_TAG_DIVFTREE);
    if (!outdated) {
        if (vertex->NeedCompaction()) {
            bufferMgr->AddCompactionTaskIfNotExists(id);
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
    bufferMgr->UnpinVertex(id, version);
}

void DIVFTree::SearchLayer(const VTYPE* query, size_t span,
                           std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers, uint8_t level) {
    FatalAssert(level < layers.size() - 1, LOG_TAG_DIVFTREE, "level out of bounds!");
    FatalAssert((uint64_t)level > VectorID::VECTOR_LEVEL, LOG_TAG_DIVFTREE, "level out of bounds!");
    FatalAssert(layers[level - 1]->Empty(), LOG_TAG_DIVFTREE, "next level should be empty!");

#ifdef EXCESS_LOGING
    DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
        "SearchLayer BEGIN: query=%s, layer=%hhu, upper_layer=(%p)%s ",
        VectorToString(query, attr.dimension).ToCStr(), level-1, layers[level],
        layers[level]->ToString<ANNVectorInfoToString>().ToCStr());
#endif
    uint64_t taskId = threadSelf->GetNextTaskID();
    SearchTaskGenerator task_gen(threadSelf->ID(), taskId, layers[level]->Size(),
                                 query, span, &search_tasks,
                                 layers[level]->Size() *
                                 (((uint64_t)level == VectorID::LEAF_LEVEL) ? attr.leaf_max_size :
                                                                              attr.internal_max_size));
    BufferManager* bufferMgr = BufferManager::GetInstance();
    CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);
    bufferMgr->PrefetchAndPinVerticesForSearch(layers[level], task_gen);

#ifdef EXCESS_LOGING
    DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
        "SearchLayer: query=%s, layer=%hhu, taskId=%lu ",
        VectorToString(query, attr.dimension).ToCStr(), level-1, taskId);
#endif

    BGSearchStatsUpdateCreatedTask(layers[level]->Size());
    while (task_gen.num_completed.load(std::memory_order_acquire) < layers[level]->Size()) {
        (void)(ExecuteSearchTask(layers[level - 1]));
    }

    FatalAssert(task_gen.num_generated.load(std::memory_order_acquire) == layers[level]->Size(),
                    LOG_TAG_DIVFTREE, "not all tasks generated yet!");
    FatalAssert(task_gen.num_completed.load(std::memory_order_acquire) == layers[level]->Size(),
                    LOG_TAG_DIVFTREE, "not all tasks completed yet!");

    SANITY_CHECK({
        for (size_t i = 0; i < layers[level]->Size(); ++i) {
            FatalAssert(task_gen.task_set[i] != nullptr, LOG_TAG_DIVFTREE,
                        "task set entry cannot be null!");
            FatalAssert(task_gen.task_set[i]->master == threadSelf->ID(), LOG_TAG_DIVFTREE,
                        "task master id mismatch in task set entry!");
            FatalAssert(task_gen.task_set[i]->taskId == taskId, LOG_TAG_DIVFTREE,
                        "task id mismatch in task set entry!");
            FatalAssert(task_gen.task_set[i]->num_tasks == layers[level]->Size(), LOG_TAG_DIVFTREE,
                        "num tasks mismatch in task set entry!");
            FatalAssert(task_gen.task_set[i]->query == query, LOG_TAG_DIVFTREE,
                        "query pointer mismatch in task set entry!");
            FatalAssert(task_gen.task_set[i]->k == span, LOG_TAG_DIVFTREE,
                        "k/span mismatch in task set entry!");
            FatalAssert(task_gen.task_set[i]->num_completed == &task_gen.num_completed, LOG_TAG_DIVFTREE,
                        "num completed pointer mismatch in task set entry!");
            FatalAssert(task_gen.task_set[i]->seen == &task_gen.seen, LOG_TAG_DIVFTREE,
                        "seen pointer mismatch in task set entry!");
            FatalAssert(task_gen.task_set[i]->neighbours_list == &task_gen.neighbours_list, LOG_TAG_DIVFTREE,
                        "neighbours list pointer mismatch in task set entry!");
        }
    });

    /* todo: check if I need a better algorithm for this */
    for (auto& it : task_gen.neighbours_list) {
        FatalAssert(it.first != threadSelf->ID(), LOG_TAG_DIVFTREE,
                    "we should not have my own neighbours in the list!");
        FatalAssert(it.second != nullptr, LOG_TAG_DIVFTREE,
                    "neighbours list entry cannot be null!");
        layers[level - 1]->MergeWith(*(it.second), span, false);
        delete it.second;
        it.second = nullptr;
    }

#ifdef EXCESS_LOGING
    DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
        "SearchLayer END: query=%s, layer=%hhu, taskId=%lu, upper_layer=(%p)%s, lower_layer=(%p)%s",
        VectorToString(query, attr.dimension).ToCStr(), level-1, taskId, layers[level],
        layers[level]->ToString<ANNVectorInfoToString>().ToCStr(), layers[level-1],
        layers[level-1]->ToString<ANNVectorInfoToString>().ToCStr());
#endif
}

RetStatus DIVFTree::ANNSearch(const VTYPE* query, size_t k,
                              uint8_t internal_node_search_span, uint8_t leaf_node_search_span,
                              uint8_t start_level, uint8_t end_level,
                              std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers,
                              DIVFTreeVertex& pinned_root_version) {
    BufferManager* bufferMgr = BufferManager::GetInstance();
    FatalAssert(bufferMgr != nullptr, LOG_TAG_DIVFTREE, "BufferManager is not initialized.");
    /*
    * Since a version of the root is pinned, we basicaly have an MVCC snapshop of the
    * database based on that root version and it's children are not deleted unless they are unpinned or empty
    */
    FatalAssert(pinned_root_version.attr.centroid_id._level >= start_level, LOG_TAG_DIVFTREE,
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

            bufferMgr->GetCentroidsForMigrationCheck(current_level, firstId, secondId);

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
        if (pinned_root_version.attr.centroid_id._level == current_level) {
            FatalAssert(layers[current_level]->Size() == 1 &&
                        (*layers[current_level])[0].id == pinned_root_version.attr.centroid_id &&
                        (*layers[current_level])[0].version == pinned_root_version.attr.version, LOG_TAG_DIVFTREE,
                        "the highest level should only contain the pinned version of the root");
            SearchRoot(query, span, layers, pinned_root_version);
        } else {
            SearchLayer(query, span, layers, current_level);
        }

        if (layers[next_level]->Empty()) {
            layers[current_level]->Clear();
            if (current_level == pinned_root_version.attr.centroid_id._level) {
                return RetStatus::Fail();
            }
            ++current_level;
            continue;
        }
        --current_level;
    }

    return RetStatus::Success();
}

inline RetStatus DIVFTree::ExecuteSearchTask(SortedList<ANNVectorInfo, SimilarityComparator>* neighbours) {
    RetStatus rs = RetStatus::Success();
    BufferManager* bufferMgr = BufferManager::GetInstance();
    CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

    SearchTask* nextTask = nullptr;
    FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                "should not hold any lock here!");
    SortedList<ANNVectorInfo, SimilarityComparator>* neighbour_list = nullptr;
    if (search_tasks.PopHead(nextTask)) {
        CHECK_NOT_NULLPTR(nextTask, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_VALID(nextTask->target, LOG_TAG_DIVFTREE);
        FatalAssert(nextTask->master != INVALID_DIVF_THREAD_ID, LOG_TAG_DIVFTREE,
                    "master id cannot be invalid!");
        if (neighbours != nullptr && nextTask->master == threadSelf->ID()) {
            neighbour_list = neighbours;
        } else {
            auto handle = nextTask->neighbours_list->Get(threadSelf->ID());
            if (handle.IsValid()) {
                neighbour_list = handle.Value();
            } else {
                neighbour_list =
                    new SortedList<ANNVectorInfo, SimilarityComparator>(attr.similarityComparator, nextTask->k);
                bool inserted =
                    nextTask->neighbours_list->Insert(threadSelf->ID(), neighbour_list, handle.GetHash(), false, false);
                FatalAssert(inserted, LOG_TAG_DIVFTREE,
                            "failed to insert neighbour list in task %lu",
                            nextTask->taskId);
                UNUSED_VARIABLE(inserted);
            }
            nextTask->neighbours_list->Release(handle);
        }
        CHECK_NOT_NULLPTR(neighbour_list, LOG_TAG_DIVFTREE);

#ifdef EXCESS_LOGING
    DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
            "BGSearch BEGIN: taskId=%lu", nextTask->taskId);
#endif
        SearchVertex(nextTask->target, nextTask->version, nextTask->query, nextTask->k,
                     neighbour_list, *(nextTask->seen));
#ifdef EXCESS_LOGING
    DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_DIVFTREE,
            "BGSearch END: taskId=%lu", nextTask->taskId);
#endif
        nextTask->num_completed->fetch_add(1);
        rs = RetStatus::Success();
    } else {
        rs = RetStatus::Fail();
    }
    FatalAssert(threadSelf->SanityCheckNumLocksHeldByMe() == 0, LOG_TAG_THREAD,
                "should not hold any lock here!");
    return rs;
}
// inline void DIVFTree::AsyncSearchAndComm(Thread* self, uint64_t idx);
// inline void DIVFTree::BGMigration(Thread* self, uint64_t idx);
// inline void DIVFTree::BGMerge(Thread* self, uint64_t idx);

// inline void DIVFTree::StartBGThreads();
// inline void DIVFTree::DestroyBGThreads();
};

#endif