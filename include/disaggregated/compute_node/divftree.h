#ifndef DIVFTREE_H_
#define DIVFTREE_H_

#include "common.h"
#include "disaggregated/vector_utils.h"
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