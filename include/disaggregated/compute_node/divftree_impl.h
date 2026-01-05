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

inline BatchState _GetBatchState(VectorMetaData* vmd, ClusterSizeType batch_start, ClusterSizeType& batch_end) {

    if (!vmd[batch_start].state.load(std::memory_order_acquire).is_state_valid) {
        return BATCH_STATE_INVALID;
    }

    if (!vmd[batch_start].batch_meta.is_batch_size) {
        /* we are in middle of a batch! */
        batch_start = vmd[batch_start].batch_meta.batch_size_or_last_offset;
        if (!vmd[batch_start].state.load(std::memory_order_acquire).is_state_valid) {
            /* the batch header is invalid so we cannot use its data to see where the batch ends */
            return BATCH_STATE_MID;
        }
    }

    batch_end = batch_start - vmd[batch_start].batch_meta.batch_size_or_last_offset;
    return vmd[batch_start].batch_valid.load(std::memory_order_acquire) ? BATCH_STATE_VALID : BATCH_STATE_INVALID;
}

inline void _SearchVectorBatch(const VTYPE* query, size_t k, SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
                               DataArray data, VectorMetaData* vmd, ClusterSizeType batch_start,
                               ClusterSizeType batch_end, DistanceType dtype,
                               ConcurrentHashTable<std::pair<VectorID, Version>, bool, VectorIDVersionPairCMP,
                                                   VectorIDVersionPairHash>& seen) {
    for (ClusterSizeType i = batch_start; i > batch_end; --i) {
        VectorState state = vmd[i].state.load(std::memory_order_acquire);
        FatalAssert(state.is_state_valid, LOG_TAG_DIVFTREE_VERTEX,
                    "State should be valid here!");
        FatalAssert(state.detail != VECTOR_STATE_OUTDATED,
                    LOG_TAG_DIVFTREE_VERTEX,
                    "a pure vector cannot become outdated!");
        VectorData& vdata = data[i];
        FatalAssert(vdata.id.IsVector(), LOG_TAG_DIVFTREE_VERTEX,
                    "Data ID should be a vector ID here!");
        if (state.detail == VECTOR_STATE_DELETED ||
            !seen.Emplace(std::make_pair(vdata.id, 0), true)) {
            continue;
        }

        neighbours->Insert(ANNVectorInfo(Distance(query, &(vdata.data[0]), data.dim, dtype), vdata.id));
        if (neighbours->Size() > k) {
            neighbours->PopBack();
        }
        FatalAssert(neighbours->Size() <= k, LOG_TAG_DIVFTREE_VERTEX,
                    "Neighbour list size exceeded k after insertion!");
    }
}

inline void _SearchCentroidBatch(const VTYPE* query, size_t k, SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
                               BufferManager* buffer, CentroidArray data, VectorMetaData* vmd, ClusterSizeType batch_start,
                               ClusterSizeType batch_end, DistanceType dtype,
                               ConcurrentHashTable<std::pair<VectorID, Version>, bool, VectorIDVersionPairCMP,
                                                   VectorIDVersionPairHash>& seen,
                               std::unordered_set<VectorID, VectorIDHash>* in_cluster = nullptr) {
    for (ClusterSizeType i = batch_start; i > batch_end; --i) {
        VectorState state = vmd[i].state.load(std::memory_order_acquire);
        FatalAssert(state.is_state_valid, LOG_TAG_DIVFTREE_VERTEX,
                    "State should be valid here!");
        CentroidData& cdata = data[i];
        FatalAssert(cdata.id.IsCentroid(), LOG_TAG_DIVFTREE_VERTEX,
                    "Data ID should be a centroid ID here!");
        FatalAssert(cdata.id.IsCentroid(), LOG_TAG_DIVFTREE_VERTEX,
                    "Data ID should be a centroid ID here!");
        if (in_cluster != nullptr) {
            in_cluster->emplace(cdata.id);
        }
        if (state.detail == VECTOR_STATE_DELETED ||
            !buffer->Exists(cdata.id, cdata.version) ||
            !seen.Emplace(std::make_pair(cdata.id, cdata.version), true)) {
            continue;
        }

        if (in_cluster != nullptr && state.detail == VECTOR_STATE_OUTDATED &&
            (in_cluster->find(cdata.id) != in_cluster->end())) {
            continue;
        }

        neighbours->Insert(ANNVectorInfo(Distance(query, &(cdata.data[0]), data.dim, dtype), cdata.id, cdata.version));
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
    Address data = cluster.Data(0, attr.centroid_id.IsLeaf(), attr.block_size, attr.cap, dim);
    VectorMetaData* vmd = cluster.MetaData(0, attr.centroid_id.IsLeaf(), attr.block_size, attr.cap, dim);
    ClusterSizeType curr_size = size.load(std::memory_order_acquire);
    std::unordered_set<VectorID, VectorIDHash> in_cluster;
    std::vector<ClusterSizeType> invalid_batches;
    ClusterSizeType batch_start = curr_size - 1;
    ClusterSizeType batch_end = curr_size - 2;
    while(batch_start != (ClusterSizeType)(-1)) {
        FatalAssert(batch_start < curr_size, LOG_TAG_DIVFTREE_VERTEX,
                    "Batch start is out of bounds. batch_start=%hu, current_size=%hu",
                    batch_start, curr_size);

        if (_GetBatchState(vmd, batch_start, batch_end) != BATCH_STATE_VALID) {
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
            _SearchVectorBatch(query, k, neighbours, DataArray(data, dim), vmd, batch_start, batch_end, dtype, seen);
        } else {
            _SearchCentroidBatch(query, k, neighbours, buffer, CentroidArray(data, dim), vmd,
                                 batch_start, batch_end, dtype, seen, &in_cluster);
        }

        batch_start = batch_end;
        batch_end = batch_start - 1;
    }

    for (ClusterSizeType invalid_batch_start : invalid_batches) {
        ClusterSizeType invalid_batch_end = invalid_batch_start - 1;
        ClusterSizeType batch_start = invalid_batch_start;
        if (attr.centroid_id.IsLeaf() && _GetBatchState(vmd, batch_start, invalid_batch_end) == BATCH_STATE_VALID) {
            _SearchVectorBatch(query, k, neighbours, DataArray(data, dim), vmd, batch_start, invalid_batch_end,
                               dtype, seen);
        } else if (!attr.centroid_id.IsLeaf() &&
                   _GetBatchState(vmd, batch_start, invalid_batch_end) == BATCH_STATE_VALID) {
            _SearchCentroidBatch(query, k, neighbours, buffer, CentroidArray(data, dim), vmd, batch_start,
                                 invalid_batch_end, dtype, seen);
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

        layers.reserve(root->selfId.Level() + 1);
        /* level 0 represents vectors and we do not need them */
        layers.emplace_back(nullptr);
        for (uint64_t i = 1; i <= root->selfId.Level(); ++i) {
            layers.emplace_back(
                new SortedList<ANNVectorInfo, SimilarityComparator>(attr.similarityComparator));
        }

        layers[root->selfId.Level()]->Insert(ANNVectorInfo(0, root->selfId, root_version));
        rs = ANNSearch(vec, 1, search_span, 1,
                        (uint8_t)(root->selfId.Level()), (uint8_t)VectorID::LEAF_LEVEL,
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

    vec_id = GenerateNextVectorID();
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

    layers.reserve(root->selfId.Level() + 1);
    for (uint64_t i = 0; i <= root->selfId.Level(); ++i) {
        layers.emplace_back(new SortedList<ANNVectorInfo, SimilarityComparator>(attr.similarityComparator));
    }

    layers[root->selfId.Level()]->Insert(ANNVectorInfo(0, root->selfId, root_version));
    rs = ANNSearch(query, k, internal_node_search_span, leaf_node_search_span,
                    (uint8_t)(root->selfId.Level()), (uint8_t)VectorID::VECTOR_LEVEL, layers,
                    root->Read(root_version));
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

// inline VectorID DIVFTree::GenerateNextVectorID();

// inline void DIVFTree::ClearStats(bool need_lock);
// void DIVFTree::BGMigrationStatsUpdate(uint64_t thread_index, bool completed_task, uint64_t num_migrated_vectors);
// void DIVFTree::BGMergeStatsUpdate(uint64_t thread_index, bool completed_task, bool cluster_merged);
// void DIVFTree::BGSearchStatsUpdate(uint64_t thread_index, bool completed_task);
// void DIVFTree::BGSearchStatsUpdateCreatedTask(uint64_t num_tasks);

inline void DIVFTree::InsertBatch(VectorID target_id, Version target_version,
                                  ClusterSizeType insert_offset, ConstVectorBatch batch) {
    FatalAssert(batch.size > 0, LOG_TAG_DIVFTREE,
                "Batch size must be greater than 0!");
    FatalAssert(insert_offset + batch.size <= attr.internal_max_size, LOG_TAG_DIVFTREE,
                "Insert offset + batch size exceeds max cluster size!");
    CHECK_VECTORID_IS_VALID(target_id, LOG_TAG_DIVFTREE);
    CHECK_VECTORID_IS_INTERNAL(target_id, LOG_TAG_DIVFTREE);
    CHECK_NOT_NULLPTR(batch.data, LOG_TAG_DIVFTREE);

    BufferManager* bufferMgr = BufferManager::GetInstance();
    CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

    BufferVertexEntry* entry = bufferMgr->GetBufferEntry(target_id);
    if (entry == nullptr) {
        /* should we ignore this case and assume it is deleted or it is not yet available? */
        /* if I ignore it and it is not available yet, I will not have enough info on its size */
        /* unless when I am reading a cluster for the first time, I first read its size */
        return;
    }
    VertexData* vertex_data =
        entry->ReadVersionIfInCache(target_version, CacheUpdateInfo(insert_offset, batch), true);
    if (vertex_data == nullptr) {
        return;
    }
    DIVFTreeVertex& vertex = vertex_data->vertex;

    VectorMetaData* vmd = vertex.cluster.MetaData(0, target_id.IsLeaf(), attr.internal_blck_size,
                                                  attr.internal_max_size, attr.dimension);
    Address data = vertex.cluster.Data(0, target_id.IsLeaf(), attr.internal_blck_size,
                                       attr.internal_max_size, attr.dimension);
    CHECK_NOT_NULLPTR(vmd, LOG_TAG_DIVFTREE);
    CHECK_NOT_NULLPTR(data, LOG_TAG_DIVFTREE);

    const size_t VECTOR_SIZE = target_id.IsLeaf() ?
                                  VectorData::Size(attr.dimension) :
                                  CentroidData::Size(attr.dimension);

    memcpy(data + insert_offset * VECTOR_SIZE,
           batch.data,
           batch.size * VECTOR_SIZE);
    std::vector<ClusterSizeType> potential_out_of_order_updates;
    potential_out_of_order_updates.reserve(batch.size);
    for (ClusterSizeType i = 0; i < batch.size; ++i) {
        const ClusterSizeType batch_off = insert_offset + i;
        FatalAssert(!(vmd[batch_off].batch_valid.load(std::memory_order_acquire)),
                    LOG_TAG_DIVFTREE,
                    "The target batch metadata should not be valid when inserting an atomic batch!");
        SANITY_CHECK({
            if (target_id.IsLeaf()) {
                VectorData& vdata = DataArray(data, attr.dimension)[batch_off];
                CHECK_VECTORID_IS_VALID(vdata.id, LOG_TAG_DIVFTREE);
                CHECK_VECTORID_IS_VECTOR(vdata.id, LOG_TAG_DIVFTREE);
            } else {
                CentroidData& cdata = CentroidArray(data, attr.dimension)[batch_off];
                CHECK_VECTORID_IS_VALID(cdata.id, LOG_TAG_DIVFTREE);
                CHECK_VECTORID_IS_CENTROID(cdata.id, LOG_TAG_DIVFTREE);
            }
        });
        vmd[batch_off].batch_meta.is_batch_size = 1;
        vmd[batch_off].batch_meta.batch_size_or_last_offset = 1;
        VectorState new_state = ChangeVectorState(vmd[batch_off].state, VECTOR_STATE_NORMAL);
        FatalAssert(new_state.is_state_valid, LOG_TAG_DIVFTREE,
                    "New state should be valid here!");
        vmd[batch_off].batch_valid.store(true, std::memory_order_release);
        if (new_state.detail != VECTOR_STATE_NORMAL) {
            potential_out_of_order_updates.emplace_back(batch_off);
        }
    }
    vertex.UpdateApproximateSize(insert_offset + batch.size);

    if (potential_out_of_order_updates.empty()) {
        entry->Unpin(target_version);
        vertex_data = nullptr;
        entry = nullptr;
        return;
    }

    vertex_data->clusterLock.Lock(SX_EXCLUSIVE);
    if (vertex_data->outofOrderUpdates == nullptr) {
        vertex_data->clusterLock.Unlock();
        entry->Unpin(target_version);
        vertex_data = nullptr;
        entry = nullptr;
        return;
    }

    std::unordered_map<std::pair<VectorID, Version>,
                       std::vector<std::pair<ClusterSizeType, ClusterSizeType>>,
                       VectorIDVersionPairHash> out_of_order_migrations;
    std::vector<ClusterSizeType> out_of_order_deletions;
    std::vector<ClusterSizeType> out_of_order_outdates;
    ValidateBatches(target_id, target_version, potential_out_of_order_updates.size(),
                    potential_out_of_order_updates.data(), false, out_of_order_migrations,
                    out_of_order_deletions, out_of_order_outdates, true);
}

inline void DIVFTree::InsertAtomicBatch(VectorID target_id, Version target_version,
                                        ClusterSizeType insert_offset, ClusterSizeType mark_outdated_offset,
                                        ConstVectorBatch batch) {
    FatalAssert(batch.size > 1, LOG_TAG_DIVFTREE,
                "Batch size must be greater than 1!");
    FatalAssert(insert_offset + batch.size <= attr.internal_max_size, LOG_TAG_DIVFTREE,
                "Insert offset + batch size exceeds max cluster size!");
    FatalAssert(mark_outdated_offset < insert_offset, LOG_TAG_DIVFTREE,
                "Mark outdated offset must be less than insert offset!");
    CHECK_VECTORID_IS_VALID(target_id, LOG_TAG_DIVFTREE);
    CHECK_VECTORID_IS_INTERNAL(target_id, LOG_TAG_DIVFTREE);
    CHECK_NOT_NULLPTR(batch.data, LOG_TAG_DIVFTREE);

    BufferManager* bufferMgr = BufferManager::GetInstance();
    CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

    BufferVertexEntry* entry = bufferMgr->GetBufferEntry(target_id);
    if (entry == nullptr) {
        /* should we ignore this case and assume it is deleted or it is not yet available? */
        /* if I ignore it and it is not available yet, I will not have enough info on its size */
        /* unless when I am reading a cluster for the first time, I first read its size */
        return;
    }

    VertexData* vertex_data =
        entry->ReadVersionIfInCache(target_version, CacheUpdateInfo(insert_offset, batch, mark_outdated_offset), true);
    /* this should be handled by read */
    // if (vertex == nullptr) {
    //     if (entry->currentVersion >= target_version) {
    //         /* the vertex is deleted */
    //         entry->headerLock.Unlock();
    //         return;
    //     }

    //     entry->headerLock.Unlock();
    //     entry->headerLock.Lock(SX_EXCLUSIVE);
    //     vertex = entry->Read(target_version, true);
    //     if (vertex == nullptr) {
    //         if (entry->currentVersion >= target_version) {
    //             /* the vertex is deleted */
    //             entry->headerLock.Unlock();
    //             return;
    //         }

    //         Version old_version = entry->currentVersion;
    //         entry->currentVersion = target_version;
    //         entry->liveVersions.insert(...)
    //     }
    // }
    if (vertex_data == nullptr) {
        return;
    }
    DIVFTreeVertex& vertex = vertex_data->vertex;

    /* step 1: Insert batch data */
    ClusterSizeType current_batch_meta_offset = insert_offset + batch.size - 1;
    VectorMetaData* vmd = vertex.cluster.MetaData(0, false, attr.internal_blck_size,
                                                  attr.internal_max_size, attr.dimension);
    CentroidArray cluster_data(vertex.cluster.Data(0, false, attr.internal_blck_size, attr.internal_max_size,
                                              attr.dimension), attr.dimension);
    CHECK_NOT_NULLPTR(vmd, LOG_TAG_DIVFTREE);
    FatalAssert(!(vmd[current_batch_meta_offset].batch_valid.load(std::memory_order_acquire)),
                LOG_TAG_DIVFTREE,
                "The target batch metadata should not be valid when inserting an atomic batch!");
    memcpy(&(cluster_data[insert_offset]),
           batch.data,
           batch.size * CentroidData::Size(attr.dimension));

    /* step 2: set batch metadata */
    for (ClusterSizeType i = 0; i < batch.size; ++i) {
        const ClusterSizeType batch_off = insert_offset + i;
        FatalAssert(!(vmd[batch_off].batch_valid.load(std::memory_order_acquire)),
                    LOG_TAG_DIVFTREE,
                    "The target batch metadata should not be valid when inserting an atomic batch!");
        CHECK_VECTORID_IS_VALID(cluster_data[batch_off].id, LOG_TAG_DIVFTREE);
        CHECK_VECTORID_IS_CENTROID(cluster_data[batch_off].id, LOG_TAG_DIVFTREE);
        vmd[batch_off].batch_meta.is_batch_size = 0;
        vmd[batch_off].batch_meta.batch_size_or_last_offset = current_batch_meta_offset;
        VectorState new_state = ChangeVectorState(vmd[batch_off].state, VECTOR_STATE_NORMAL);
        FatalAssert(new_state.is_state_valid, LOG_TAG_DIVFTREE,
                    "New state should be valid here!");
    }

    std::unordered_map<std::pair<VectorID, Version>,
                       std::vector<std::pair<ClusterSizeType, ClusterSizeType>>,
                       VectorIDVersionPairHash> out_of_order_migrations;
    std::vector<ClusterSizeType> out_of_order_deletions;
    std::vector<ClusterSizeType> out_of_order_outdates;

    out_of_order_migrations.reserve(batch.size);
    out_of_order_deletions.reserve(batch.size);
    out_of_order_outdates.reserve(batch.size);

    /* step 3: mark outdated offset */
    bool cluster_locked = false;
    VectorState new_state =
        ChangeVectorState(vmd[mark_outdated_offset].state, VECTOR_STATE_OUTDATED);
    FatalAssert(new_state.detail == VECTOR_STATE_OUTDATED, LOG_TAG_DIVFTREE,
                "The new state detail should be outdated here!");
    if (!new_state.is_state_valid) {
        cluster_locked = true;
        vertex_data->clusterLock.Lock(SX_EXCLUSIVE);
        new_state = vmd[mark_outdated_offset].state.load(std::memory_order_acquire);
        FatalAssert(new_state.detail == VECTOR_STATE_OUTDATED, LOG_TAG_DIVFTREE,
                "The new state detail should be outdated here!");
        if (!new_state.is_state_valid) {
            if (vertex_data->outofOrderUpdates == nullptr) {
                vertex_data->outofOrderUpdates = new std::unordered_map<ClusterSizeType, OutOfOrderUpdateInfo>();
            }
            FatalAssert(vertex_data->outofOrderUpdates->find(mark_outdated_offset) ==
                        vertex_data->outofOrderUpdates->end(),
                        LOG_TAG_DIVFTREE,
                        "There should not be an existing out-of-order update for this offset!");
            vertex_data->outofOrderUpdates->emplace(mark_outdated_offset, current_batch_meta_offset);
            vertex_data->clusterLock.Unlock();
            entry->Unpin(target_version);
            vertex_data = nullptr;
            entry = nullptr;
            return;
        }
    }

    ClusterSizeType outdated_batch_offset;
    if (vmd[mark_outdated_offset].batch_meta.is_batch_size) {
        outdated_batch_offset = mark_outdated_offset;
    } else {
        outdated_batch_offset = vmd[mark_outdated_offset].batch_meta.batch_size_or_last_offset;
    }
    FatalAssert(outdated_batch_offset >= mark_outdated_offset, LOG_TAG_DIVFTREE,
                "Outdated batch offset should be greater than or equal to mark outdated offset!");

    if ((!vmd[outdated_batch_offset].state.load(std::memory_order_acquire).is_state_valid) ||
        (!vmd[outdated_batch_offset].batch_valid.load(std::memory_order_acquire))) {

        bool outdated_batch_valid = false;
        if (!cluster_locked) {
            cluster_locked = true;
            vertex_data->clusterLock.Lock(SX_EXCLUSIVE);
            if ((vmd[outdated_batch_offset].state.load(std::memory_order_acquire).is_state_valid) &&
                (vmd[outdated_batch_offset].batch_valid.load(std::memory_order_acquire))) {
                outdated_batch_valid = true;
            }
        }

        FatalAssert(cluster_locked, LOG_TAG_DIVFTREE,
                    "Cluster lock should be held here!");
        if (!outdated_batch_valid) {
            if (vertex_data->outofOrderUpdates == nullptr) {
                vertex_data->outofOrderUpdates = new std::unordered_map<ClusterSizeType, OutOfOrderUpdateInfo>();
            }
            FatalAssert(vertex_data->outofOrderUpdates->find(mark_outdated_offset) ==
                            vertex_data->outofOrderUpdates->end(),
                        LOG_TAG_DIVFTREE,
                        "There should not be an existing out-of-order update for this offset!");
            vertex_data->outofOrderUpdates->emplace(mark_outdated_offset, current_batch_meta_offset);
            vertex_data->clusterLock.Unlock();
            entry->Unpin(target_version);
            vertex_data = nullptr;
            entry = nullptr;
            return;
        }
    }

    ValidateBatches(target_id, target_version, 1, &outdated_batch_offset, cluster_locked,
                  out_of_order_migrations, out_of_order_deletions, out_of_order_outdates);
    threadSelf->SanityCheckLockNotHeldByMe(&vertex_data->clusterLock);
    return;
}

inline void DIVFTree::MigrateVectors(VectorID src_id, Version src_version,
                                     VectorID dest_id, Version dest_version,
                                     ClusterSizeType num_vectors, const ClusterSizeType* offsets,
                                     ClusterSizeType dest_insert_offset) {
    CHECK_VECTORID_IS_VALID(src_id, LOG_TAG_DIVFTREE);
    CHECK_VECTORID_IS_CENTROID(src_id, LOG_TAG_DIVFTREE);
    CHECK_VECTORID_IS_VALID(dest_id, LOG_TAG_DIVFTREE);
    CHECK_VECTORID_IS_CENTROID(dest_id, LOG_TAG_DIVFTREE);
    FatalAssert(dest_id != src_id, LOG_TAG_DIVFTREE,
                "Source and destination IDs must be different!");
    FatalAssert(dest_id.Level() == src_id.Level(), LOG_TAG_DIVFTREE,
                "Source and destination levels must be the same!");
    FatalAssert(num_vectors > 0, LOG_TAG_DIVFTREE,
                "Number of vectors to migrate must be greater than 0!");
    CHECK_NOT_NULLPTR(offsets, LOG_TAG_DIVFTREE);

    BufferManager* bufferMgr = BufferManager::GetInstance();
    CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

    BufferVertexEntry* src_entry = bufferMgr->GetBufferEntry(src_id);
    VertexData* src_vertex_data = nullptr;
    ClusterCacheState src_cache_state = ClusterCacheState::DELETED;
    if (src_entry != nullptr) {
        src_vertex_data =
            src_entry->ReadAndLockVersion(src_version, true);
    }

    if (src_vertex_data != nullptr) {
        src_cache_state = src_vertex_data->cacheState.load(std::memory_order_acquire);
    }

    if (src_cache_state == ClusterCacheState::REMOTE_READ_IN_PROGRESS) {
        /* handle this after the read is done */
        if (src_vertex_data->pendingUpdates == nullptr) {
            src_vertex_data->pendingUpdates = new std::vector<CacheUpdateInfo>();
        }
        src_vertex_data->pendingUpdates->emplace_back(true, dest_id, dest_version,
                                                      num_vectors, offsets,
                                                      dest_insert_offset);
        src_vertex_data->clusterLock.Unlock();
        src_entry->Unpin(src_version);
        return;
    }

    if (src_vertex_data != nullptr) {
        src_vertex_data->clusterLock.Unlock();
    }

    BufferVertexEntry* dest_entry = bufferMgr->GetBufferEntry(dest_id);
    VertexData* dest_vertex_data = nullptr;
    ClusterCacheState dest_cache_state = ClusterCacheState::DELETED;
    if (dest_entry != nullptr) {
        dest_vertex_data =
            dest_entry->ReadAndLockVersion(dest_version, true);
    }

    if (dest_vertex_data != nullptr) {
        dest_cache_state = dest_vertex_data->cacheState.load(std::memory_order_acquire);
        FatalAssert(dest_cache_state != ClusterCacheState::DELETED, LOG_TAG_DIVFTREE,
                    "Destination vertex cluster should not be deleted here!");
        if (dest_cache_state == ClusterCacheState::UNCACHED) {
            dest_vertex_data->vertex.UpdateApproximateSize(num_vectors + dest_insert_offset);
        }
        dest_vertex_data->clusterLock.Unlock();
    }

    ClusterSizeType* dest_offsets = nullptr;
    VectorBatch batch;
    batch.size = 0;
    const size_t VECTOR_SIZE = src_id.IsLeaf() ? VectorData::Size(attr.dimension) :
                                                 CentroidData::Size(attr.dimension);

    if (src_cache_state == ClusterCacheState::CACHED_COOLING ||
        src_cache_state == ClusterCacheState::CACHED_HOT) {
        if (dest_cache_state == ClusterCacheState::CACHED_COOLING ||
            dest_cache_state == ClusterCacheState::CACHED_HOT ||
            dest_cache_state == ClusterCacheState::REMOTE_READ_IN_PROGRESS) {
            /* both source and destination are cached */
            batch.data = new char[num_vectors * VECTOR_SIZE];
        } else {
            /* only source is cached */
            batch.data = nullptr;
        }
        DIVFTreeVertex& src_vertex = src_vertex_data->vertex;
        FatalAssert(src_vertex.cluster.IsAssigned(), LOG_TAG_DIVFTREE,
                    "Source vertex cluster should be valid here!");
        VectorMetaData* src_vmd =
            src_vertex.cluster.MetaData(0, src_id.IsLeaf(),
                                        src_id.IsLeaf() ? attr.leaf_max_size : attr.internal_max_size,
                                        src_id.IsLeaf() ? attr.leaf_max_size : attr.internal_max_size,
                                        attr.dimension);
        Address src_data = src_vertex.cluster.Data(0, src_id.IsLeaf(),
                                                   src_id.IsLeaf() ? attr.leaf_blck_size : attr.internal_blck_size,
                                                   src_id.IsLeaf() ? attr.leaf_max_size : attr.internal_max_size,
                                                   attr.dimension);
        bool need_reiterate = false;
        for (ClusterSizeType i = 0; i < num_vectors; ++i) {
            const ClusterSizeType src_offset = offsets[i];

            VectorState new_state = ChangeVectorState(src_vmd[src_offset].state, VECTOR_STATE_MIGRATED);
            if (!new_state.is_state_valid) {
                need_reiterate = true;
            }
        }

        if (batch.data != nullptr) {
            if (need_reiterate) {
                dest_offsets = new ClusterSizeType[num_vectors];
                src_vertex_data->clusterLock.Lock(SX_EXCLUSIVE);
            }

            for (ClusterSizeType i = 0; i < num_vectors; ++i) {
                const ClusterSizeType src_offset = offsets[i];
                VectorState new_state = src_vmd[src_offset].state.load(std::memory_order_acquire);
                if (!new_state.is_state_valid) {
                    FatalAssert(need_reiterate, LOG_TAG_DIVFTREE,
                                "Need reiterate should be true here!");
                    if (src_vertex_data->outofOrderUpdates == nullptr) {
                        src_vertex_data->outofOrderUpdates =
                            new std::unordered_map<ClusterSizeType, OutOfOrderUpdateInfo>();
                    }
                    FatalAssert(src_vertex_data->outofOrderUpdates->find(src_offset) ==
                                src_vertex_data->outofOrderUpdates->end(), LOG_TAG_DIVFTREE,
                                "There should not be an existing out-of-order update for this offset!");
                    src_vertex_data->outofOrderUpdates->emplace(src_offset, dest_id, dest_version,
                                                                dest_insert_offset + i);
                } else {
                    memcpy(batch.data + (batch.size * VECTOR_SIZE),
                           src_data + (src_offset * VECTOR_SIZE),
                           VECTOR_SIZE);
                    if (need_reiterate) {
                        dest_offsets[batch.size] = dest_insert_offset + i;
                    }
                    batch.size += 1;
                }
            }

            if (need_reiterate) {
                src_vertex_data->clusterLock.Unlock();
            }
        }
    }

    if (src_vertex_data != nullptr) {
        threadSelf->SanityCheckLockNotHeldByMe(&src_vertex_data->clusterLock);
        src_entry->Unpin(src_version);
        src_vertex_data = nullptr;
        src_entry = nullptr;
    }

    if (dest_cache_state == ClusterCacheState::UNCACHED) {
        FatalAssert(batch.size == 0, LOG_TAG_DIVFTREE,
                    "Batch size should be 0 when destination is uncached!");
        FatalAssert(batch.data == nullptr, LOG_TAG_DIVFTREE,
                    "Batch data should be null when destination is uncached!");
        FatalAssert(dest_vertex_data != nullptr, LOG_TAG_DIVFTREE,
                    "Destination vertex data should not be null when destination is uncached!");
        dest_entry->Unpin(dest_version);
        dest_vertex_data = nullptr;
        dest_entry = nullptr;
        return;
    } else if (dest_cache_state == ClusterCacheState::DELETED) {
        FatalAssert(batch.size == 0, LOG_TAG_DIVFTREE,
                    "Batch size should be 0 when destination is uncached!");
        FatalAssert(batch.data == nullptr, LOG_TAG_DIVFTREE,
                    "Batch data should be null when destination is uncached!");
        FatalAssert(dest_vertex_data == nullptr, LOG_TAG_DIVFTREE,
                    "Destination vertex data should be null when destination is deleted!");
        dest_entry = nullptr;
        return;
    } else if (dest_cache_state == ClusterCacheState::REMOTE_READ_IN_PROGRESS) {
        dest_vertex_data->clusterLock.Lock(SX_EXCLUSIVE);
        dest_cache_state = dest_vertex_data->cacheState.load(std::memory_order_acquire);
        FatalAssert(dest_cache_state != ClusterCacheState::DELETED, LOG_TAG_DIVFTREE,
                    "Destination vertex cluster should not be deleted here!");
        FatalAssert(dest_cache_state != ClusterCacheState::UNCACHED, LOG_TAG_DIVFTREE,
                    "Destination vertex cluster should not be uncached here!");
    }

    if (batch.data == nullptr) {
        if (dest_cache_state == ClusterCacheState::REMOTE_READ_IN_PROGRESS) {
            /* handle this after the read is done */
            if (dest_vertex_data->pendingUpdates == nullptr) {
                dest_vertex_data->pendingUpdates = new std::vector<CacheUpdateInfo>();
            }
            dest_vertex_data->pendingUpdates->emplace_back(false, src_id, src_version,
                                                           num_vectors, offsets,
                                                           dest_insert_offset);
            dest_vertex_data->clusterLock.Unlock();
            dest_entry->Unpin(dest_version);
            return;
        }

        FatalAssert(dest_cache_state == ClusterCacheState::CACHED_COOLING ||
                    dest_cache_state == ClusterCacheState::CACHED_HOT, LOG_TAG_DIVFTREE,
                    "Destination vertex cluster should be cached here!");
        threadSelf->SanityCheckLockNotHeldByMe(&dest_vertex_data->clusterLock);
        bufferMgr->ReadVectorsFromRemote(dest_id, dest_version, dest_insert_offset, num_vectors);
        /* we should keep it pinned until the read is done */
        return;
    }

    if (dest_offsets != nullptr) {
        if (dest_cache_state == ClusterCacheState::REMOTE_READ_IN_PROGRESS) {
            /* handle this after the read is done */
            if (dest_vertex_data->pendingUpdates == nullptr) {
                dest_vertex_data->pendingUpdates = new std::vector<CacheUpdateInfo>();
            }
            dest_vertex_data->pendingUpdates->emplace_back(batch, dest_offsets);
            dest_vertex_data->clusterLock.Unlock();
            dest_entry->Unpin(dest_version);
            return;
        }

        FatalAssert(dest_cache_state == ClusterCacheState::CACHED_COOLING ||
                    dest_cache_state == ClusterCacheState::CACHED_HOT, LOG_TAG_DIVFTREE,
                    "Destination vertex cluster should be cached here!");
        threadSelf->SanityCheckLockNotHeldByMe(&dest_vertex_data->clusterLock);
        InsertBatch(dest_id, dest_version, dest_offsets, batch);
        delete[] dest_offsets;
        dest_offsets = nullptr;
        if (batch.data != nullptr) {
            delete[] batch.data;
            batch.data = nullptr;
        }
        dest_entry->Unpin(dest_version);
        return;
    }

    if (dest_cache_state == ClusterCacheState::REMOTE_READ_IN_PROGRESS) {
        /* handle this after the read is done */
        if (dest_vertex_data->pendingUpdates == nullptr) {
            dest_vertex_data->pendingUpdates = new std::vector<CacheUpdateInfo>();
        }
        dest_vertex_data->pendingUpdates->emplace_back(dest_insert_offset, batch);
        dest_vertex_data->clusterLock.Unlock();
        dest_entry->Unpin(dest_version);
        return;
    }

    FatalAssert(dest_cache_state == ClusterCacheState::CACHED_COOLING ||
                dest_cache_state == ClusterCacheState::CACHED_HOT, LOG_TAG_DIVFTREE,
                "Destination vertex cluster should be cached here!");
    threadSelf->SanityCheckLockNotHeldByMe(&dest_vertex_data->clusterLock);
    InsertBatch(dest_id, dest_version, dest_insert_offset, batch);
    if (batch.data != nullptr) {
        delete[] batch.data;
        batch.data = nullptr;
    }
    dest_entry->Unpin(dest_version);
    return;
}

inline void DIVFTree::MigrateOutOfOrder(VectorID src_id, Version src_version,
                                        VectorID dest_id, Version dest_version,
                                        const std::vector<std::pair<ClusterSizeType, ClusterSizeType>>& offset_pairs) {
    CHECK_VECTORID_IS_VALID(src_id, LOG_TAG_DIVFTREE);
    CHECK_VECTORID_IS_INTERNAL(src_id, LOG_TAG_DIVFTREE);
    CHECK_VECTORID_IS_VALID(dest_id, LOG_TAG_DIVFTREE);
    CHECK_VECTORID_IS_INTERNAL(dest_id, LOG_TAG_DIVFTREE);
    FatalAssert(dest_id != src_id, LOG_TAG_DIVFTREE,
                "Source and destination IDs must be different!");
    FatalAssert(dest_id.Level() == src_id.Level(), LOG_TAG_DIVFTREE,
                "Source and destination levels must be the same!");
    FatalAssert(!offset_pairs.empty(), LOG_TAG_DIVFTREE,
                "Offset pairs must not be empty!");

    BufferManager* bufferMgr = BufferManager::GetInstance();
    CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

    BufferVertexEntry* dest_entry = bufferMgr->GetBufferEntry(dest_id);
    if (dest_entry == nullptr) {
        /* should we ignore this case and assume it is deleted or it is not yet available? */
        /* if I ignore it and it is not available yet, I will not have enough info on its size */
        /* unless when I am reading a cluster for the first time, I first read its size */
        return;
    }

    VertexData* dest_vertex_data =
        dest_entry->ReadVersionIfInCache(dest_version, CacheUpdateInfo(src_id, src_version, &offset_pairs), true);
    if (dest_vertex_data == nullptr) {
        return;
    }
    DIVFTreeVertex& dest_vertex = dest_vertex_data->vertex;
    VectorMetaData* dest_vmd =
        dest_vertex.cluster.MetaData(0, false, attr.internal_blck_size, attr.internal_max_size, attr.dimension);
    Address dest_cluster_data = dest_vertex.cluster.Data(0, false, attr.internal_blck_size,
                                                         attr.internal_max_size, attr.dimension);

    BufferVertexEntry* src_entry = bufferMgr->GetBufferEntry(src_id);
    CHECK_NOT_NULLPTR(src_entry, LOG_TAG_DIVFTREE);

    VertexData* src_vertex_data = src_entry->ReadVersionIfInCache(src_version, false);
    CHECK_NOT_NULLPTR(src_vertex_data, LOG_TAG_DIVFTREE);
    DIVFTreeVertex& src_vertex = src_vertex_data->vertex;
    VectorMetaData* src_vmd =
        src_vertex.cluster.MetaData(0, false, attr.internal_blck_size, attr.internal_max_size, attr.dimension);
    Address src_cluster_data = src_vertex.cluster.Data(0, false, attr.internal_blck_size,
                                                       attr.internal_max_size, attr.dimension);
    const size_t VECTOR_SIZE = CentroidData::Size(attr.dimension);

    std::vector<ClusterSizeType> batch_offsets;
    batch_offsets.reserve(offset_pairs.size());
    for (const auto& offset_pair : offset_pairs) {
        FatalAssert(src_vmd[offset_pair.first].state.load(std::memory_order_acquire).is_state_valid,
                    LOG_TAG_DIVFTREE,
                    "Source vector state should be valid here!");
        FatalAssert(src_vmd[offset_pair.first].state.load(std::memory_order_acquire).detail == VECTOR_STATE_MIGRATED,
                    LOG_TAG_DIVFTREE,
                    "Source vector state should be migrated here!");
        memcpy(dest_cluster_data + (offset_pair.second * VECTOR_SIZE),
               src_cluster_data + (offset_pair.first * VECTOR_SIZE),
               VECTOR_SIZE);
        dest_vmd[offset_pair.second].batch_meta.is_batch_size = 1;
        dest_vmd[offset_pair.second].batch_meta.batch_size_or_last_offset = 1;
        FatalAssert(!dest_vmd[offset_pair.second].state.load(std::memory_order_acquire).is_state_valid,
                    LOG_TAG_DIVFTREE,
                    "Destination vector state should not be valid here!");
        VectorState new_state = ChangeVectorState(dest_vmd[offset_pair.second].state, VECTOR_STATE_NORMAL);
        FatalAssert(new_state.is_state_valid, LOG_TAG_DIVFTREE,
                    "New state should be valid here!");
        if (new_state.detail != VECTOR_STATE_NORMAL) {
            batch_offsets.emplace_back(offset_pair.second);
        }
        dest_vmd[offset_pair.second].batch_valid.store(true, std::memory_order_release);
    }

    std::unordered_map<std::pair<VectorID, Version>,
                       std::vector<std::pair<ClusterSizeType, ClusterSizeType>>,
                       VectorIDVersionPairHash> out_of_order_migrations;
    std::vector<ClusterSizeType> out_of_order_deletions;
    std::vector<ClusterSizeType> out_of_order_outdates;
    ValidateBatches(dest_id, dest_version, static_cast<ClusterSizeType>(batch_offsets.size()),
                    batch_offsets.data(), false,
                    out_of_order_migrations, out_of_order_deletions, out_of_order_outdates, true);
    threadSelf->SanityCheckLockNotHeldByMe(&dest_vertex_data->clusterLock);
    return;
}

inline void DIVFTree::ValidateBatches(VectorID target_id, Version target_version, ClusterSizeType num_batches,
                                      ClusterSizeType* batch_meta_offsets, bool cluster_locked,
                                      std::unordered_map<std::pair<VectorID, Version>,
                                                         std::vector<std::pair<ClusterSizeType, ClusterSizeType>>,
                                                         VectorIDVersionPairHash>& out_of_order_migrations,
                                      std::vector<ClusterSizeType>& out_of_order_deletions,
                                      std::vector<ClusterSizeType>& out_of_order_outdates, bool batchs_are_valid) {
    BufferManager* bufferMgr = BufferManager::GetInstance();
    CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_DIVFTREE);

    BufferVertexEntry* entry = bufferMgr->GetBufferEntry(target_id);
    CHECK_NOT_NULLPTR(entry, LOG_TAG_DIVFTREE);

    VertexData* vertex_data = entry->ReadVersionIfInCache(target_version, false);
    CHECK_NOT_NULLPTR(vertex_data, LOG_TAG_DIVFTREE);
    DIVFTreeVertex& vertex = vertex_data->vertex;
    VectorMetaData* vmd =
        vertex.cluster.MetaData(0, target_id.IsLeaf(), attr.internal_blck_size, attr.internal_max_size, attr.dimension);

    if (!cluster_locked) {
        vertex_data->clusterLock.Lock(SX_EXCLUSIVE);
    }

    if (batchs_are_valid && (vertex_data->outofOrderUpdates == nullptr || vertex_data->outofOrderUpdates->empty())) {
        /* all batches are already valid and there is no out-of-order updates */
        vertex_data->clusterLock.Unlock();
        entry->Unpin(target_version);
        vertex_data = nullptr;
        entry = nullptr;
        return;
    }

    bool first_iteration = true;
    while (true) {
        for (ClusterSizeType batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            const ClusterSizeType batch_meta_offset = batch_meta_offsets[batch_idx];
            FatalAssert(vmd[batch_meta_offset].state.load(std::memory_order_acquire).is_state_valid,
                        LOG_TAG_DIVFTREE,
                        "Vector state should be valid here!");
            FatalAssert(vmd[batch_meta_offset].batch_meta.is_batch_size == 1,
                        LOG_TAG_DIVFTREE,
                        "Batch metadata should indicate batch size here!");
            const ClusterSizeType batch_size = vmd[batch_meta_offset].batch_meta.batch_size_or_last_offset;
            const ClusterSizeType insert_offset = batch_meta_offset - batch_size + 1;
            FatalAssert((insert_offset == batch_meta_offset) ||
                        (vmd[insert_offset].batch_meta.is_batch_size == 0),
                        LOG_TAG_DIVFTREE,
                        "Insert offset batch metadata should indicate last offset here!");
            FatalAssert((insert_offset == batch_meta_offset) ||
                        (vmd[insert_offset].batch_meta.batch_size_or_last_offset == batch_meta_offset),
                        LOG_TAG_DIVFTREE,
                        "Insert offset batch metadata should point to current batch meta offset here!");
            if (batchs_are_valid && first_iteration) {
                FatalAssert(vmd[batch_meta_offset].batch_valid.load(std::memory_order_acquire),
                        LOG_TAG_DIVFTREE,
                        "Batch metadata should be valid here!");
            } else {
                FatalAssert(!vmd[batch_meta_offset].batch_valid.load(std::memory_order_acquire),
                        LOG_TAG_DIVFTREE,
                        "Batch metadata should not be valid here!");
                vmd[batch_meta_offset].batch_valid.store(true, std::memory_order_release);
            }

            if (vertex_data->outofOrderUpdates == nullptr) {
                FatalAssert(!batchs_are_valid || !first_iteration, LOG_TAG_DIVFTREE,
                        "If batchs are not valid, we should not have entered this loop!");
                continue;
            }

            for (ClusterSizeType i = 0; i < batch_size; ++i) {
                const ClusterSizeType batch_off = insert_offset + i;
                FatalAssert(vmd[batch_off].state.load(std::memory_order_acquire).is_state_valid,
                            LOG_TAG_DIVFTREE,
                            "Vector state should be valid here!");
                auto it = vertex_data->outofOrderUpdates->find(batch_off);
                if (it == vertex_data->outofOrderUpdates->end()) {
                    FatalAssert(vmd[batch_off].state.load(std::memory_order_acquire).detail == VECTOR_STATE_NORMAL,
                                LOG_TAG_DIVFTREE,
                                "Vector state should be normal for in-order updates!");
                    continue;
                }

                switch (it->second.type) {
                    case OutOfOrderUpdateType::OUT_OF_ORDER_DELETION:
                        FatalAssert(vmd[batch_off].state.load(std::memory_order_acquire).detail == VECTOR_STATE_DELETED,
                                    LOG_TAG_DIVFTREE,
                                    "Vector state should be deleted for out-of-order deletion!");
                        out_of_order_deletions.emplace_back(batch_off);
                        break;
                    case OutOfOrderUpdateType::OUT_OF_ORDER_MIGRATION:
                        FatalAssert(vmd[batch_off].state.load(std::memory_order_acquire).detail == VECTOR_STATE_MIGRATED,
                                    LOG_TAG_DIVFTREE,
                                    "Vector state should be moved for out-of-order migration!");
                        std::pair<VectorID, Version> target_container =
                            std::make_pair(it->second.address.containerId, it->second.address.containerVersion);
                        if (out_of_order_migrations.find(target_container) == out_of_order_migrations.end()) {
                            auto res =
                                out_of_order_migrations.emplace(target_container,
                                                                std::vector<std::pair<ClusterSizeType, ClusterSizeType>>());
                            res.first->second.reserve(batch_size);
                            res.first->second.emplace_back(batch_off, it->second.address.offset);
                        } else {
                            out_of_order_migrations[target_container].emplace_back(batch_off, it->second.address.offset);
                        }
                        break;
                    case OutOfOrderUpdateType::OUT_OF_ORDER_OUTDATED:
                        FatalAssert(vmd[batch_off].state.load(std::memory_order_acquire).detail == VECTOR_STATE_OUTDATED,
                                    LOG_TAG_DIVFTREE,
                                    "Vector state should be outdated for out-of-order outdate!");
                        out_of_order_outdates.emplace_back(it->second.insertion_batch_offset);
                        break;
                    default:
                        FatalAssert(false, LOG_TAG_DIVFTREE,
                                    "Unknown out-of-order update type!");
                }

                vertex_data->outofOrderUpdates->erase(it);
                if (vertex_data->outofOrderUpdates->empty()) {
                    delete vertex_data->outofOrderUpdates;
                    vertex_data->outofOrderUpdates = nullptr;
                }
            }
        }

        first_iteration = false;
        if (!out_of_order_outdates.empty()) {
            batch_meta_offsets[0] = out_of_order_outdates.back();
            num_batches = 1;
            out_of_order_outdates.pop_back();
        } else {
            break;
        }
    }

    vertex_data->clusterLock.Unlock();
    if (!out_of_order_deletions.empty()) {
        DeleteVectors(target_id, target_version, out_of_order_deletions.size(), out_of_order_deletions.data());
    }
    for (const auto& migration : out_of_order_migrations) {
        MigrateOutOfOrder(target_id, target_version,
                        migration.first.first, migration.first.second,
                        migration.second);
    }
    entry->Unpin(target_version);
    vertex_data = nullptr;
    entry = nullptr;
    return;
}

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
    uint8_t level = pinned_root_version.attr.centroid_id.Level();
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
    FatalAssert(pinned_root_version.attr.centroid_id.Level() >= start_level, LOG_TAG_DIVFTREE,
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
        if (pinned_root_version.attr.centroid_id.Level() == current_level) {
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
            if (current_level == pinned_root_version.attr.centroid_id.Level()) {
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