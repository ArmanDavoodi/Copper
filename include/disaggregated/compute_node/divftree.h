#ifndef DIVFTREE_H_
#define DIVFTREE_H_

#include "common.h"
#include "vector_utils.h"
#include "distance.h"

#include "utils/synchronization.h"
#include "utils/concurrent_datastructures.h"

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

struct SearchTaskGenerator {
    const DIVFThreadID master;
    const uint64_t taskId;
    const uint64_t num_tasks;
    const VTYPE* query;
    const size_t span;

    std::atomic<bool>* taken;
    std::atomic<bool>* done;
    SortedList<ANNVectorInfo, SimilarityComparator>* neighbours;
    std::atomic<size_t>* viewed;
    std::atomic<bool>* shared_data;
    SearchTask** task_set;

    BlockingQueue<SearchTask*>* taskQueue;
    uint64_t num_generated;

    SearchTaskGenerator(DIVFThreadID m, uint64_t t_id, uint64_t nt, const VTYPE* q, size_t s,
                        std::atomic<bool>* tk, std::atomic<bool>* dn,
                        SortedList<ANNVectorInfo, SimilarityComparator>* nb,
                        std::atomic<size_t>* vd, std::atomic<bool>* sd, SearchTask** ts,
                        BlockingQueue<SearchTask*>* tq) :
        master(m), taskId(t_id), num_tasks(nt), query(q), span(s),
        taken(tk), done(dn), neighbours(nb), viewed(vd), shared_data(sd), task_set(ts), taskQueue(tq),
        num_generated(0) {
        CHECK_NOT_NULLPTR(q, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(tk, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(dn, LOG_TAG_DIVFTREE);
        CHECK_NOT_NULLPTR(tq, LOG_TAG_DIVFTREE);
        FatalAssert(span > 0, LOG_TAG_DIVFTREE, "k cannot be 0!");
        FatalAssert(nt > 0, LOG_TAG_DIVFTREE, "num_tasks cannot be 0!");
    }

    bool GenerateTask(VectorID target, Version version) {
        FatalAssert(num_generated < num_tasks, LOG_TAG_DIVFTREE,
                    "All tasks have already been generated!");
        FatalAssert(target.IsValid(), LOG_TAG_DIVFTREE,
                    "Target VectorID is not valid!");
        FatalAssert(target.IsCentroid(), LOG_TAG_DIVFTREE,
                    "Target VectorID is not a centroid!");
        FatalAssert(task_set[num_generated] == nullptr, LOG_TAG_DIVFTREE,
                    "Task slot {} is already occupied!", num_generated);
        task_set[num_generated] =
            new SearchTask(taskId, num_tasks, target, version, query, span, taken, done, viewed, shared_data, task_set);
        task_set[num_generated]->master = master;
        task_set[num_generated]->neighbours = neighbours;
        taskQueue->Push(task_set[num_generated]);
        ++num_generated;
        return num_generated == num_tasks;
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
    inline ClusterSizeType GetOffset(const UpdateType& type, const void* info) const {
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

     /*
        possible:
            Insertion: invalid -> valid:
                exp could be: (in other words, vectorstate should be invalid)
                    invalid -> Succ
                    migrated_inv -> has to insert to new place instead
                    outdated_inv -> can be ignored as at this point we have the new version
                    deleted_inv -> can be ignored
                exp cannot be:
                    valid -> double insertion in the same offset of the same version of the same cluster
                    migrated -> double insertion ...
                    outdated -> double insertion ...
                    deleted -> double insertion ...
            Deletion: valid -> invalid
                exp could be: (in other words, vectorstate should be normal)
                    valid -> Succ
                    invalid -> not inserted yet -> set to deleted_inv
                exp cannot be:
                    migrated -> mismatch with MN because if MN has told us that this is migrated it should have happend
                                and we cannot fail.
                    migrated_inv -> same reason as above
                    outdated -> same reason as above
                    outdated_inv -> same reason as above
                    deleted -> double deletion
                    deleted_inv -> double deletion
            Migration: valid -> migrated
                exp could be: (in other words, vectorstate should be normal)
                    valid -> Succ
                    invalid -> not inserted yet -> set to migrated_inv
                exp cannot be:
                    migrated -> double migration
                    migrated_inv -> double migration
                    outdated -> ...
                    outdated_inv -> ...
                    deleted -> ...
                    deleted_inv -> ...
            MarkOutdated: valid -> outdated
                Note: this is only possible when container is not leaf
                exp could be: (in other words, vectorstate should be normal)
                    valid -> Succ
                    invalid -> not inserted yet -> set to outdated_inv
                exp cannot be:
                    migrated -> ...
                    migrated_inv -> ...
                    outdated -> ...
                    outdated_inv -> ...
                    deleted -> ...
                    deleted_inv -> ...
        impossible:
            MigrationAbortion: migrated -> valid : this should happen only in MN side
     */
    inline RetStatus ChangeVectorState(ClusterSizeType targetOffset, VectorState targetState);
    inline RetStatus ChangeVectorState(VectorMetaData* targetMeta, VectorState targetState);
    inline RetStatus ChangeVectorState(CentroidMetaData* targetMeta, VectorState targetState);

    void Search(const VTYPE* query, size_t k, SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
                std::unordered_set<std::pair<VectorID, Version>, VectorIDVersionPairHash>& seen);

    inline const DIVFTreeVertexAttributes& GetAttributes() const;
    inline uint64_t GetVisibleSize() const;
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

class DIVFTree : public DIVFTreeInterface {
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

    size_t Size() const;

    const DIVFTreeAttributes& GetAttributes() const;

    inline void EndBGThreads();

    inline String GetStatistics(std::string title_extention = "", bool clear_stats = false);

    inline void StartStatsCollection();

    inline void StopStatsCollection();

    inline void ClearStats() ;


    // inline String ToString(bool detailed = false) const;

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

    inline void ClearStats(bool need_lock);

    void BGMigrationStatsUpdate(uint64_t thread_index, bool completed_task, uint64_t num_migrated_vectors);

    void BGMergeStatsUpdate(uint64_t thread_index, bool completed_task, bool cluster_merged);

    void BGCompactionStatsUpdate(uint64_t thread_index, bool completed_task, bool cluster_compacted);

    void BGSearchStatsUpdate(uint64_t thread_index, bool completed_task);

    void BGSearchStatsUpdateCreatedTask(uint64_t num_tasks);

    inline void RoundRobinClustering(const DIVFTreeVertex* base, const ConstVectorBatch& batch,
                                     BufferVertexEntry**& entries, DIVFTreeVertex**& clusters, VectorBatch& centroids,
                                     uint16_t marked_for_update = INVALID_OFFSET);

    /*
     * will only fill in the raw centroid vectors to
     * the centroids batch and allocates memory for version and ids but does not fill them
     */
    inline void Clustering(const DIVFTreeVertex* base, const ConstVectorBatch& batch,
                           BufferVertexEntry**& entries, DIVFTreeVertex**& clusters, VectorBatch& centroids,
                           uint16_t marked_for_update = INVALID_OFFSET);

    inline BufferVertexEntry* ExpandTree(VectorID expRootId);

    RetStatus SplitAndInsert(BufferVertexEntry* container_entry, const ConstVectorBatch& batch,
                             uint16_t marked_for_update = INVALID_OFFSET);

    inline RetStatus ReadAndCheckVersion(VectorID containerId, Version containerVersion,
                                         BufferVertexEntry** entries, uint16_t max_entries, uint16_t& num_entries,
                                         LockMode mode);

    /* Todo: recheck to see if everything works fine */
    /* todo: need to refactor */
    RetStatus Migrate(std::vector<MigrationInfo> targetBatch,
                      VectorID src_id, VectorID dest_id,
                      Version src_ver, Version dest_ver, uint64_t& num_migrated);

    uint64_t MigrationCheck(VectorID first_cluster, VectorID second_cluster);

    /* todo: need to refactor */
    RetStatus Merge(VectorID srcId, Version srcVersion,
                    VectorID destId, Version destVersion);

    bool MergeCheck(VectorID target);

    void SearchRoot(const VTYPE* query, size_t span,
                    std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers,
                    DIVFTreeVertex* pinned_root_version);

    void SearchVertex(VectorID id, Version version, const VTYPE* query, size_t span,
                      SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
                      std::unordered_set<std::pair<VectorID, Version>, VectorIDVersionPairHash>& seen);

    /* todo: use multiple threads for searching each layer -> what if we use a single pool for all searches?
       if there are few threads, they will do the search layer themselves but if there are free threads they
       can help each other */
    void SearchLayer(const VTYPE* query, size_t span,
                     std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers, uint8_t level);

    RetStatus ANNSearch(const VTYPE* query, size_t k, uint8_t internal_node_search_span, uint8_t leaf_node_search_span,
                        uint8_t start_level, uint8_t end_level,
                        std::vector<SortedList<ANNVectorInfo, SimilarityComparator>*>& layers,
                        DIVFTreeVertex* pinned_root_version);

    inline void AsyncSearch(Thread* self, uint64_t idx);

    inline void BGMigration(Thread* self, uint64_t idx);
    inline void BGMerge(Thread* self, uint64_t idx);

    inline void BGCompaction(Thread* self, uint64_t idx);

    inline void StartBGThreads();
    inline void DestroyBGThreads();

TESTABLE;
};

};

#endif