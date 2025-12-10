#ifndef DIVFTREE_H_
#define DIVFTREE_H_

#include "compute_node/divftree.h"
#include "compute_node/buffer.h"
#include "comm_layer.h"


// Todo: better logs and asserts -> style: <Function name>(self data(a=?), input data): msg, additonal variables if needed

namespace divftree {

// DIVFTreeVertex::DIVFTreeVertex(const DIVFTreeVertexAttributes& attributes);
// DIVFTreeVertex::DIVFTreeVertex(const DIVFTreeAttributes& attributes, VectorID id, DIVFTreeInterface* index);

// DIVFTreeVertex::~DIVFTreeVertex();

// RetStatus DIVFTreeVertex::BatchUpdate(const SortedList<std::pair<UpdateType, void*>, UpdateCMP>& updates);

// inline RetStatus DIVFTreeVertex::ChangeVectorState(ClusterSizeType targetOffset, VectorState targetState);
// inline RetStatus DIVFTreeVertex::ChangeVectorState(VectorMetaData* targetMeta, VectorState targetState);
// inline RetStatus DIVFTreeVertex::ChangeVectorState(CentroidMetaData* targetMeta, VectorState targetState);

// void DIVFTreeVertex::Search(const VTYPE* query, size_t k, SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
//             std::unordered_set<std::pair<VectorID, Version>, VectorIDVersionPairHash>& seen);

// inline const DIVFTreeVertexAttributes& DIVFTreeVertex::GetAttributes() const;
// inline uint64_t DIVFTreeVertex::GetVisibleSize() const;
// inline VectorID DIVFTreeVertex::CentroidID() const;
// inline Version DIVFTreeVertex::VertexVersion() const;
// String DIVFTreeVertex::ToString(bool detailed = false) const;

// DIVFTree::DIVFTree(DIVFTreeAttributes attributes);
// DIVFTree::~DIVFTree();

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
        BufferVertexEntry* root = ReadAndPinRoot(root_version);
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
            .type = UpdateType::INSERT,
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
            if (it->second.type == UpdateType::DELETE) {
                handleLock.Unlock();
                DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_DIVFTREE,
                        "Completion handle for vector " VECTORID_LOG_FMT " already exists for DELETE!",
                        VECTORID_LOG(vec_id));
                return RetStatus{
                    .stat = RetStatus::DUPLICATE_DELETE,
                    .message = nullptr
                };
            } else {
                FatalAssert(it->second.type == UpdateType::INSERT, LOG_TAG_DIVFTREE,
                            "Completion handle for vector " VECTORID_LOG_FMT " has invalid type!",
                            VECTORID_LOG(vec_id));
                if (it->second.completed == false) {
                    /* the insert has not completed yet, so we can just remove the handle */
                    handleLock.Unlock();
                    DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_DIVFTREE,
                            "Cannot create DELETE completion handle for vector " VECTORID_LOG_FMT
                            " since INSERT has not completed yet!",
                            VECTORID_LOG(vec_id));
                    return RetStatus{
                        .stat = RetStatus::INSERT_NOT_COMPLETED,
                        .message = nullptr
                    };
                } else {
                    /* the insert has completed, we can change the handle to DELETE */
                    if (it->second.status.IsOK()) {
                        /* only if the insert was successful we can change to delete */
                        /* todo: if for poll we see delete we return true for the completion */
                        it->second.type = UpdateType::DELETE;
                        it->second.completed = false;
                        FatalAssert(it->second.message == nullptr, LOG_TAG_DIVFTREE,
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
                        return RetStatus{
                            .stat = RetStatus::VECTOR_NOT_FOUND,
                            .message = fail_message
                        };
                    }
                }
            }
        } else {
            completion_handles[vec_id] = CompletionHandle{
                .type = UpdateType::DELETE,
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

// RetStatus DIVFTree::ApproximateKNearestNeighbours(const VTYPE* query, size_t k, uint8_t internal_node_search_span,
//                                                   uint8_t leaf_node_search_span, SortType sort_type,
//                                                   std::vector<ANNVectorInfo>& neighbours) {

// }

// size_t DIVFTree::Size() const;
// const DIVFTreeAttributes& DIVFTree::GetAttributes() const;

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

// void DIVFTree::SearchRoot(const VTYPE* query, size_t span,
//                 std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers,
//                 DIVFTreeVertex& pinned_root_version);
// void DIVFTree::SearchVertex(VectorID id, Version version, const VTYPE* query, size_t span,
//                     SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
//                     std::unordered_set<std::pair<VectorID, Version>, VectorIDVersionPairHash>& seen);
// void DIVFTree::SearchLayer(const VTYPE* query, size_t span,
//                     std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers, uint8_t level);
// RetStatus DIVFTree::ANNSearch(const VTYPE* query, size_t k, uint8_t internal_node_search_span, uint8_t leaf_node_search_span,
//                     uint8_t start_level, uint8_t end_level,
//                     std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers,
//                     DIVFTreeVertex& pinned_root_version);

// inline void DIVFTree::AsyncSearchAndComm(Thread* self, uint64_t idx);
// inline void DIVFTree::BGMigration(Thread* self, uint64_t idx);
// inline void DIVFTree::BGMerge(Thread* self, uint64_t idx);

// inline void DIVFTree::StartBGThreads();
// inline void DIVFTree::DestroyBGThreads();
};

#endif