#ifndef DIVFTREE_H_
#define DIVFTREE_H_

#include "common.h"
#include "disaggregated/vector_utils.h"
#include "disaggregated/comm_layer.h"
#include "distance.h"

#include "utils/synchronization.h"
#include "utils/concurrent_datastructures.h"

#include "interface/divftree.h"

#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <vector>

// Todo: better logs and asserts -> style: <Function name>(self data(a=?), input data): msg, additonal variables if needed

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
    std::atomic<size_t>* num_completed;

    ConcurrentHashTable<std::pair<VectorID, Version>, bool, VectorIDVersionPairCMP, VectorIDVersionPairHash>* seen;
    ConcurrentHashTable<DIVFThreadID, SortedList<ANNVectorInfo, SimilarityComparator>*,
                        DIVFThreadIDCMP, DIVFThreadIDHash>* neighbours_list;

    SearchTask() = default;
    SearchTask(DIVFThreadID req_thread, uint64_t task_id, uint64_t nt, VectorID id, Version ver, const VTYPE* q,
               size_t span, std::atomic<size_t>* nc,
            ConcurrentHashTable<std::pair<VectorID, Version>, bool, VectorIDVersionPairCMP, VectorIDVersionPairHash>* s,
            ConcurrentHashTable<DIVFThreadID, SortedList<ANNVectorInfo, SimilarityComparator>*,
                                DIVFThreadIDCMP, DIVFThreadIDHash>* nl) :
        master(req_thread), taskId(task_id), num_tasks(nt),
        target(id), version(ver), query(q), k(span),
        num_completed(nc), seen(s), neighbours_list(nl) {
        CHECK_VECTORID_IS_VALID(id, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(q, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(nc, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(s, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(nl, LOG_TAG_DIVFTREE);
        FatalAssert(span > 0, LOG_TAG_DIVFTREE, "k cannot be 0!");
    }
};

struct SearchTaskGenerator {
    const DIVFThreadID master;
    const uint64_t taskId;
    const uint64_t num_tasks;
    const VTYPE* query;
    const size_t span;
    BlockingQueue<SearchTask*>* taskQueue;

    SearchTask** task_set;
    ConcurrentHashTable<std::pair<VectorID, Version>, bool, VectorIDVersionPairCMP, VectorIDVersionPairHash> seen;
    ConcurrentHashTable<DIVFThreadID, SortedList<ANNVectorInfo, SimilarityComparator>*,
                        DIVFThreadIDCMP, DIVFThreadIDHash> neighbours_list;

    std::atomic<uint64_t> num_generated;
    std::atomic<size_t> num_completed;

    SearchTaskGenerator(DIVFThreadID m, uint64_t t_id, uint64_t nt, const VTYPE* q, size_t s,
                        BlockingQueue<SearchTask*>* tq, size_t num_excpected_vectors_in_search) :
        master(m), taskId(t_id), num_tasks(nt), query(q), span(s), taskQueue(tq),
        task_set(new SearchTask*[nt]), seen(num_excpected_vectors_in_search),
        neighbours_list(), num_generated(0), num_completed(0) {
        CHECK_NOT_NULLPTR(q, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(tq, LOG_TAG_DIVFTREE);
        FatalAssert(span > 0, LOG_TAG_DIVFTREE, "k cannot be 0!");
        FatalAssert(nt > 0, LOG_TAG_DIVFTREE, "num_tasks cannot be 0!");
    }

    ~SearchTaskGenerator() {
        for (uint64_t i = 0; i < num_tasks; ++i) {
            delete task_set[i];
        }
        delete[] task_set;
        seen.Clear();
        neighbours_list.Clear();
    }

    void GenerateTask(VectorID target, Version version) {
        FatalAssert(num_generated.load(std::memory_order_acquire) < num_tasks, LOG_TAG_DIVFTREE,
                    "All tasks have already been generated!");
        FatalAssert(target.IsValid(), LOG_TAG_DIVFTREE,
                    "Target VectorID is not valid!");
        FatalAssert(target.IsCentroid(), LOG_TAG_DIVFTREE,
                    "Target VectorID is not a centroid!");
        uint64_t idx = num_generated.fetch_add(1);
        FatalAssert(idx < num_tasks, LOG_TAG_DIVFTREE,
                    "All tasks have already been generated!");
        FatalAssert(task_set[idx] == nullptr, LOG_TAG_DIVFTREE,
                    "Task slot %lu is already occupied!", idx);
        task_set[idx] =
            new SearchTask(master, taskId, num_tasks, target, version, query, span, &num_completed,
                           &seen, &neighbours_list);
        taskQueue->Push(task_set[idx]);
    }

    void GenerateTask(std::vector<std::pair<VectorID, Version>> targets) {
        FatalAssert(!targets.empty(), LOG_TAG_DIVFTREE,
                    "Targets list is empty!");
        FatalAssert(num_generated.load(std::memory_order_acquire) + targets.size() <= num_tasks, LOG_TAG_DIVFTREE,
                    "All tasks have already been generated!");
        uint64_t idx = num_generated.fetch_add(targets.size());
        FatalAssert(idx + targets.size() <= num_tasks, LOG_TAG_DIVFTREE,
                    "All tasks have already been generated!");
        for (size_t i = 0; i < targets.size(); ++i) {
            FatalAssert(targets[i].first.IsValid(), LOG_TAG_DIVFTREE,
                        "Target VectorID is not valid!");
            FatalAssert(targets[i].first.IsCentroid(), LOG_TAG_DIVFTREE,
                        "Target VectorID is not a centroid!");
            FatalAssert(task_set[idx + i] == nullptr, LOG_TAG_DIVFTREE,
                        "Task slot %lu is already occupied!", idx + i);
            task_set[idx + i] =
                new SearchTask(master, taskId, num_tasks, targets[i].first, targets[i].second, query, span,
                               &num_completed, &seen, &neighbours_list);
        }
        taskQueue->BatchPush(&task_set[idx], targets.size());
    }

    void GenerateTask(std::vector<std::vector<std::pair<VectorID, Version>>> targets) {
        FatalAssert(!targets.empty(), LOG_TAG_DIVFTREE,
                    "Targets list is empty!");
        size_t total_size = 0;
        for (const auto& t : targets) {
            FatalAssert(!t.empty(), LOG_TAG_DIVFTREE,
                        "One of the target sub-lists is empty!");
            total_size += t.size();
        }
        uint64_t idx = num_generated.fetch_add(total_size);
        FatalAssert(idx + total_size <= num_tasks, LOG_TAG_DIVFTREE,
                    "All tasks have already been generated!");
        for (const auto& t : targets) {
            for (size_t i = 0; i < t.size(); ++i) {
                FatalAssert(t[i].first.IsValid(), LOG_TAG_DIVFTREE,
                            "Target VectorID is not valid!");
                FatalAssert(t[i].first.IsCentroid(), LOG_TAG_DIVFTREE,
                            "Target VectorID is not a centroid!");
                FatalAssert(task_set[idx + i] == nullptr, LOG_TAG_DIVFTREE,
                            "Task slot %lu is already occupied!", idx + i);
                task_set[idx + i] =
                    new SearchTask(master, taskId, num_tasks, t[i].first, t[i].second, query, span,
                                   &num_completed, &seen, &neighbours_list);
            }
        }
        taskQueue->BatchPush(&task_set[idx], targets.size());
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

struct InsertionBatchInfo {
    ConstVectorBatch batch;
    ClusterSizeType insert_offset;
    ClusterSizeType mark_outdated_offset;
};

struct DeletionInfo {
    ClusterSizeType delete_offset;
};

enum class UpdateType : uint8_t {
    INSERTION,
    DELETION
};

struct UpdateCMP {
    inline static ClusterSizeType GetOffset(const UpdateType& type, const void* info) {
        switch (type) {
            case UpdateType::INSERTION:
                return reinterpret_cast<const InsertionBatchInfo*>(info)->insert_offset;
            case UpdateType::DELETION:
                return reinterpret_cast<const DeletionInfo*>(info)->delete_offset;
            default:
                FatalAssert(false, LOG_TAG_BASIC, "Unknown UpdateType: %u", static_cast<uint8_t>(type));
                return INVALID_OFFSET; /* to suppress compiler warning */
        }
    }

    inline int operator()(const std::pair<UpdateType, void*>& a, const std::pair<UpdateType, void*>& b) const {
        ClusterSizeType a_offset = GetOffset(a.first, a.second);
        ClusterSizeType b_offset = GetOffset(b.first, b.second);
        if (a_offset < b_offset) {
            return -1;
        } else if (a_offset > b_offset) {
            return 1;
        } else {
            return 0;
        }
    }
};

class DIVFTreeVertex {
public:
    DIVFTreeVertex() = delete;
    DIVFTreeVertex(DIVFTreeVertex&) = delete;
    DIVFTreeVertex(const DIVFTreeVertex&) = delete;
    DIVFTreeVertex(DIVFTreeVertex&&) = delete;

    DIVFTreeVertex(const DIVFTreeVertexAttributes& attributes);
    DIVFTreeVertex(const DIVFTreeAttributes& attributes, VectorID id, DIVFTreeInterface* index);

    ~DIVFTreeVertex();

    RetStatus BatchUpdate(const SortedList<std::pair<UpdateType, void*>, UpdateCMP>& updates);

    /*
     * if id.level is vertex/cluster and targetState is Invalid then the cluster should be locked in exclusive mode
     * otherwise if id.level is vertex/cluster then target should be locked in shared mode
     */

    void Search(const VTYPE* query, size_t k, SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
                ConcurrentHashTable<std::pair<VectorID, Version>, bool, VectorIDVersionPairCMP,
                                    VectorIDVersionPairHash>& seen);

    inline bool NeedCompaction() const;

    inline const DIVFTreeVertexAttributes& GetAttributes() const;
    inline ClusterSizeType GetVisibleSize() const;
    inline VectorID CentroidID() const;
    inline Version VertexVersion() const;
    String ToString(bool detailed = false) const;
    inline void UpdateApproximateSize(uint64_t new_size);


protected:
    const DIVFTreeVertexAttributes attr;
    std::atomic<uint64_t> unpinCount;
    std::atomic<uint64_t> size;
    alignas(CACHE_LINE_SIZE) Cluster cluster;

TESTABLE;
friend class DIVFTree;
};

struct BufferVertexEntry;

struct CompletionHandle {
    UpdateType type;
    VectorID vector_id;
    bool completed;
    RetStatus status;
};

/*
    Insert AtomicBatch (for split -> cause another vector to become outdated):
        step 1:
        lock(S) header

        step 2:
        if (version not available):
            step 2.1:
            upgrade header to X
            if version is newer than currentVersion:
                change currentVersion to new version
                deduct currentVersionPin from the old currentVersion pin if exists
                add new version with pin = currentVersionPin to live versions
                deduct currentVersionPin from old version pin if exists
            else:
                add new version with pin = 0 to live versions
            set its size to what was given in the message
            if pin is not 0:
                set state to REMOTE_READ_IN_PROGRESS
                issue RDMA read for the whole cluster
            else :
                set state to UNCACHED
            unlock header
            return

        step 3:
        v <- read version(no pin needed)
        increment the pin
        unlock header
        // since these are updates we are not accessing them so we should not make them hot!

        step 4
        set batch state to invalid
        write vector data + id + version + batch meta
        change vector states to valid
        if ( there are out of order updates ):
            for migrations -> save the offsets in a list
            for outdated vectors -> save the offset in another list

        step 5:
        p_off <- go to the vector that needs to be outdated
        change vector state to outdated
        if (vector state is invalid):
            Lock(X) clusterLock
            if outOforder == nullptr -> creat it(do not pin vertex)
            add p_off.offset to outOfOrderUpdates with type outdated
            if (state is outdated_invalid):
                Unlock clusterLock
                unpin vertex
                return
            remove p_off.offset from outOfOrderUpdates
            if outOforder.size == 0:
                delete outOfOrderUpdates
            Unlock clusterLock

        step 6:
        p_batch <- get its batch info
        if the p_batch is invalid:
            Lock(X) clusterLock
            if p_batch.state is still invalid:
                if outOforder == nullptr -> creat it(do not pin vertex)
                add <p_batch.offset, c_batch.offset> to outOfOrderUpdates with type outdated
                Unlock clusterLock
                unpin vertex
                return
            Unlock clusterLock

        set c_batch state to valid
        Lock(X) clusterLock
        if outOforder == nullptr:
            Unlock clusterLock
            unpin vertex
            return
        for each e in migration list:
            if outOfOrder does not contain e.offset:
                continue
            save the target address in a temp variable
        for each e in outdated list:
            if outOfOrder does not contain e.offset:
                continue
            get its batch offset, set it to valid
            check if there are other batches that are chained to it or its elements and set them all to valid
        check if anything depends on the validity of c_batch and if yes:
            validate them and their dependents recursively
        if outOforder is empty:
            delete outOfOrderUpdates
        Unlock clusterLock

        for migrations -> Insert them to their respective clusters if they are cached
        unpin vertex
        return



    Insert Vector (for simple insert to leaves or for migration):
        step 1:
        lock(S) header

        if (version not available):
            step 2.1:
            upgrade header to X
            if version is newer than currentVersion:
                change currentVersion to new version
                deduct currentVersionPin from the old currentVersion pin if exists
                add new version with pin = currentVersionPin to live versions
                deduct currentVersionPin from old version pin if exists
            else:
                add new version with pin = 0 to live versions
            set its size to what was given in the message
            if pin is not 0:
                set state to REMOTE_READ_IN_PROGRESS
                issue RDMA read for the whole cluster
            else :
                set state to UNCACHED
            unlock header
            return

        step 3:
        v <- read version(no pin needed)
        increment the pin
        unlock header
        // since these are updates we are not accessing them so we should not make them hot!

        step 4
        write vector data + id + version + batch meta(should be valid with size 1)
        change vector states to valid
        if ( there are out of order updates ):
            for migrations -> save the offsets in a list
            for outdated vectors -> save the offset in another list

        step 5:
        Lock(X) clusterLock
        if outOforder == nullptr:
            Unlock clusterLock
            unpin vertex
            return

        for each e in migration list:
            if outOfOrder does not contain e.offset:
                continue
            save the target address in a temp variable
        for each e in outdated list:
            if outOfOrder does not contain e.offset:
                continue
            get its batch offset, set it to valid
            check if there are other batches that are chained to it or its elements and set them all to valid
        if outOforder is empty:
            delete outOfOrderUpdates
        Unlock clusterLock

        for migrations -> Insert them to their respective clusters if they are cached
        unpin vertex
        return

    Migrate Vector:
        step 1:
        lock(S) target header

        step 2:
        if (target version not available):
            step 2.1:
            upgrade header to X
            if version is newer than currentVersion:
                change currentVersion to new version
                deduct currentVersionPin from the old currentVersion pin if exists
                add new version with pin = currentVersionPin to live versions
                deduct currentVersionPin from old version pin if exists
            else:
                add new version with pin = 0 to live versions
            set its size to what was given in the message
            if pin is not 0:
                set state to REMOTE_READ_IN_PROGRESS
                issue RDMA read for the whole cluster
            else :
                set state to UNCACHED
            unlock target header
        else:
            tv <- read version(no pin needed)
            increment the pin
            unlock target header
            // since these are updates we are not accessing them so we should not make them hot!

        lock(S) src header
        if (src version not available):
            step 2.2:
            upgrade header to X
            if version is newer than currentVersion:
                change currentVersion to new version
                deduct currentVersionPin from the old currentVersion pin if exists
                add new version with pin = currentVersionPin to live versions
                deduct currentVersionPin from old version pin if exists
            else:
                add new version with pin = 0 to live versions
            set its size to what was given in the message
            if pin is not 0:
                set state to REMOTE_READ_IN_PROGRESS
                issue RDMA read for the whole cluster
            else :
                set state to UNCACHED
            unlock src header
        else:
            sv <- read version(no pin needed)
            increment the pin
            unlock src header
            // since these are updates we are not accessing them so we should not make them hot!

        step 3:
        if (neither sv nor tv is in CACHED_HOT state):
            return

        step 4:
        if sv is cached:
            read offsets to be migrated
            change all their state to migrated

            if tv is cached:
                lock(X) src clusterLock
                if outOforder == nullptr -> creat it(do not pin vertex) only if needed
                for all with invalid state:
                    if state is now valid ignore
                    add their offsets to outOfOrderUpdates with type migration
                unlock src clusterLock

                put all with valid state in a list with their addresses -> during previous steps

        step 5:
        if tv is cached:
            if sv was cached:
                Insert all of the migrated vectors using the list created before
                unpin both sv and tv
                return
            else:
                assert all of them should have invalid states(it can be outdated invalid etc)
                make their batch states valid
                set batch sizes to 1
                use request message data to get migration offset and size and issue RDMA read to read
                    only the vector data and not the metadata!
                we should keep tv pinned and let the RDMA read handler unpin it
                return
        if sv is cached:
            unpin sv
        return

    Delete Vector
        step 1:
        lock(S) header

        step 2:
        if (version not available):
            step 2.1:
            upgrade header to X
            if version is newer than currentVersion:
                change currentVersion to new version
                deduct currentVersionPin from the old currentVersion pin if exists
                add new version with pin = currentVersionPin to live versions
                deduct currentVersionPin from old version pin if exists
            else:
                add new version with pin = 0 to live versions
            if pin is not 0:
                set state to REMOTE_READ_IN_PROGRESS
                issue RDMA read for the whole cluster
            else :
                set state to UNCACHED
            unlock header
            return

        step 3:
        v <- read version(no pin needed)
        increment the pin
        unlock header
        // since these are updates we are not accessing them so we should not make them hot!

        step 4
        p_off <- go to the vector that needs to be deleted
        change vector state to deleted
        if (vector state is invalid):
            Lock(X) clusterLock
            if state is still invalid:
                if outOforder == nullptr -> creat it(do not pin vertex)
                add p_off.offset to outOfOrderUpdates with type deleted
                Unlock clusterLock
                unpin vertex
                return
            Unlock clusterLock

        unpin vertex
        read vector id and version and change the buffer state for that cluster
        return

    ------

    these should only change buffer entry data
    Vertex Split
    Vertex Merge
    Vertex Pruned
    Vertex Compacted

*/

class DIVFTree {
public:
    DIVFTree(DIVFTreeAttributes attributes);
    ~DIVFTree();

    /* todo: for now since it is single node, the create_completion_notification here does not do anything and returning
       from this function indicates that the vector is inserted. But we need to change
       this in the multi node environment */
    RetStatus Insert(const VTYPE* vec, VectorID& vec_id, uint8_t search_span,
                     bool create_completion_notification = false);

    /* todo: for now since it is single node, the create_completion_notification here does not do anything and returning
       from this function indicates that the vector is deleted. But we need to change
       this in the multi node environment */
    RetStatus Delete(VectorID vec_id, bool create_completion_notification = false);

    RetStatus ApproximateKNearestNeighbours(const VTYPE* query,
                                            size_t k, uint8_t internal_node_search_span, uint8_t leaf_node_search_span,
                                            SortType sort_type, std::vector<ANNVectorInfo>& neighbours);

    size_t ApproximateSize() const;

    const DIVFTreeAttributes& GetAttributes() const;

    inline void EndBGThreads();

    inline String GetStatistics(std::string title_extention = "", bool clear_stats = false);

    inline void StartStatsCollection();

    inline void StopStatsCollection();

    inline void ClearStats() ;


    // inline String ToString(bool detailed = false) const;

protected:
    DIVFTreeAttributes attr;
    std::atomic<uint64_t> appr_size;
    std::atomic<bool> end_signal;
    std::unordered_map<VectorID, CompletionHandle, VectorIDHash> completion_handles;
    SXSpinLock handleLock;
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
    RetStatus ReadAndPinRoot(BufferVertexEntry*& root_entry, Version& root_version);

    inline VectorID GenerateNextVectorID(uint8_t level);

    inline void ClearStats(bool need_lock);

    void BGMigrationStatsUpdate(uint64_t thread_index, bool completed_task, uint64_t num_migrated_vectors);

    void BGMergeStatsUpdate(uint64_t thread_index, bool completed_task, bool cluster_merged);

    void BGSearchStatsUpdate(uint64_t thread_index, bool completed_task);

    void BGSearchStatsUpdateCreatedTask(uint64_t num_tasks);

    inline void InsertBatch(VectorID target_id, Version target_version,
                            ClusterSizeType insert_offset, ConstVectorBatch batch);
    inline void InsertBatch(VectorID target_id, Version target_version,
                            ClusterSizeType* insert_offsets, ConstVectorBatch batch);
    /* used for split/compaction/expansion */
    /* todo: maybe return a bool to indicate whether we need to delete the batch or what or maybe also pass the message*/
    inline void InsertAtomicBatch(VectorID target_id, Version target_version,
                                  ClusterSizeType insert_offset, ClusterSizeType mark_outdated_offset,
                                  ConstVectorBatch batch);

    inline void MigrateVectors(VectorID src_id, Version src_version,
                               VectorID dest_id, Version dest_version,
                               ClusterSizeType num_vectors, const ClusterSizeType* offsets,
                               ClusterSizeType dest_insert_offset);

    inline void MigrateOutOfOrder(VectorID src_id, Version src_version,
                                  VectorID dest_id, Version dest_version,
                                  const std::vector<std::pair<ClusterSizeType, ClusterSizeType>>& offset_pairs);

    inline void DeleteVectors(VectorID container_id, Version container_version,
                              ClusterSizeType num_vectors, const ClusterSizeType* offsets);

    inline void ValidateBatches(VectorID target_id, Version target_version, ClusterSizeType num_batches,
                                ClusterSizeType* batch_meta_offsets, bool cluster_locked,
                                std::unordered_map<std::pair<VectorID, Version>,
                                                   std::vector<std::pair<ClusterSizeType, ClusterSizeType>>,
                                                   VectorIDVersionPairHash>& out_of_order_migrations,
                                std::vector<ClusterSizeType>& out_of_order_deletions,
                                std::vector<ClusterSizeType>& out_of_order_outdates);

    inline void RoundRobinClustering(BufferVertexEntry* base, const ConstVectorBatch& batch,
                                     BufferVertexEntry**& entries, ClusterSizeType marked_for_update);

    /*
     * will only fill in the raw centroid vectors to
     * the centroids batch and allocates memory for version and ids but does not fill them
     */
    inline void Clustering(BufferVertexEntry* base, const ConstVectorBatch& batch,
                           BufferVertexEntry**& entries, ClusterSizeType marked_for_update);

    RetStatus SplitAndInsert(VectorID target_id, Version target_version, uintptr_t target_remote_addr,
                             uintptr_t* remote_addrs, const ConstVectorBatch& batch,
                             ClusterSizeType marked_for_update);

    inline RetStatus ReadAndCheckVersion(VectorID containerId, Version containerVersion,
                                         BufferVertexEntry** entries, uint16_t max_entries, uint16_t& num_entries,
                                         LockMode mode);

    /* Todo: recheck to see if everything works fine */
    /* todo: need to refactor */
    RetStatus Migrate(std::vector<MigrationInfo> targetBatch,
                      VectorID src_id, VectorID dest_id,
                      Version src_ver, Version dest_ver, uint64_t& num_migrated);

    ClusterSizeType MigrationCheck(VectorID first_cluster, VectorID second_cluster);

    /* todo: need to refactor */
    RetStatus Merge(VectorID srcId, Version srcVersion,
                    VectorID destId, Version destVersion);

    bool MergeCheck(VectorID target, Version targetVersion, VectorID parent, Version parentVersion,
                    uintptr_t target_remote_addr, uintptr_t parent_remote_addr);

    void SearchRoot(const VTYPE* query, size_t span,
                    std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers,
                    DIVFTreeVertex& pinned_root_version);

    void SearchVertex(VectorID id, Version version, const VTYPE* query, size_t span,
                      SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
                      ConcurrentHashTable<std::pair<VectorID, Version>, bool, VectorIDVersionPairCMP,
                                          VectorIDVersionPairHash>& seen);

    /* todo: use multiple threads for searching each layer -> what if we use a single pool for all searches?
       if there are few threads, they will do the search layer themselves but if there are free threads they
       can help each other */
    void SearchLayer(const VTYPE* query, size_t span,
                     std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers, uint8_t level);

    RetStatus ANNSearch(const VTYPE* query, size_t k, uint8_t internal_node_search_span, uint8_t leaf_node_search_span,
                        uint8_t start_level, uint8_t end_level,
                        std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers,
                        DIVFTreeVertex& pinned_root_version);

    /* use threadSelf->ID() and check if a task is generated by me, I should use my own neighbour list -> bgthreads use nullptr*/
    inline RetStatus ExecuteSearchTask(SortedList<ANNVectorInfo, SimilarityComparator>* neighbours);
    inline void AsyncSearchAndComm(Thread* self, uint64_t idx);

    inline void BGMigration(Thread* self, uint64_t idx);
    inline void BGMerge(Thread* self, uint64_t idx);

    inline void StartBGThreads();
    inline void DestroyBGThreads();

TESTABLE;
};

};

#endif