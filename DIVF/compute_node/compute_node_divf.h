#ifndef CN_DIVF_H_
#define CN_DIVF_H_


#include "common.h"
#include "vector_utils.h"
#include "distance.h"

#include "utils/thread.h"
#include "utils/sorted_list.h"
#include "utils/concurrent_datastructures.h"
#include "utils/vector_directory.h"


#include "DIVF/compute_node/buffer.h"


namespace divftree {
struct DIVFIndexAttr {
    size_t num_user_threads;
    size_t pool_size;
    size_t page_size;
    IndexMeta index_meta;
};

class DIVFIndex {
public:
    DIVFIndex(const DIVFIndexAttr& attr) : index_attr(attr) {
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                "Creating DIVFIndex with page size %zu, pool size %zu, "
                "and %zu user threads.", index_attr.page_size,
                index_attr.pool_size, index_attr.num_user_threads);

        RetStatus rs = BufferMgr::Init(
            index_attr.page_size,
            index_attr.pool_size,
            &search_task_queue,
            index_attr.num_user_threads,
            index_attr.index_meta
        );
        FatalAssert(rs.IsOK(), LOG_TAG_BASIC,
                    "Failed to initialize BufferMgr in DIVFIndex constructor: %s",
                    rs.Msg());
    }

    ~DIVFIndex() {
        RetStatus rs = BufferMgr::Destroy();
        FatalAssert(rs.IsOK(), LOG_TAG_BASIC,
                    "Failed to destroy BufferMgr in DIVFIndex destructor: %s",
                    rs.Msg());
    }

    RetStatus ANNSearch(const VTYPE* query, uint32_t k, uint32_t internal_nprobe, uint32_t leaf_nprobe,
                        std::vector<std::pair<DTYPE, IVFVectorID>>& neighbours) {
        if (query == nullptr || k == 0 || internal_nprobe == 0 || leaf_nprobe == 0) {
            FatalAssert(false, LOG_TAG_BASIC, "Invalid arguments to IVFIndex::ANNSearch()");
            return RetStatus::Fail("Invalid arguments to IVFIndex::ANNSearch()");
        }

        if (index_attr.index_meta.top_centroids.empty() || index_attr.index_meta.num_points == 0) {
            FatalAssert(false, LOG_TAG_BASIC, "Index is not built yet!");
            return RetStatus::Fail("Index is not built yet!");
        }

        bool is_leaf = index_attr.index_meta.top_centroids[0].id.IsLeaf();
        uint32_t max_nprobe = std::max(k, std::max(leaf_nprobe, internal_nprobe)) + 1;
        uint32_t nprobe = is_leaf ? leaf_nprobe : internal_nprobe;
        if (nprobe > index_attr.index_meta.top_centroids.size()) {
            DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BASIC,
                    "nprobe (%zu) is greater than the number of clusters (%zu). Reducing nprobe to %zu.",
                    nprobe, index_attr.index_meta.top_centroids.size(), index_attr.index_meta.top_centroids.size());
            nprobe = index_attr.index_meta.top_centroids.size();
        }

        if (neighbours.size() > 0) {
            DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BASIC,
                    "Output neighbours vector is not empty. Clearing previous contents.");
            neighbours.clear();
        }

        SortedList<std::pair<DTYPE, IVFVectorID>, L2DTYPEIDPairCMP> topk_list(L2DTYPEIDPairCMP(), std::move(neighbours));
        SortedList<std::pair<DTYPE, VectorID>, L2DTYPEIDPairCMP> closest_centroids(L2DTYPEIDPairCMP(), max_nprobe);
        SortedList<std::pair<DTYPE, IVFVectorID>, L2DTYPEIDPairCMP> tmp_top_vectors(L2DTYPEIDPairCMP(), max_nprobe);
        SortedList<std::pair<DTYPE, VectorID>, L2DTYPEIDPairCMP> tmp_centroids(L2DTYPEIDPairCMP(), max_nprobe);
        std::vector<VectorID> cluster_ids;
        cluster_ids.reserve(max_nprobe);
        for (size_t c = 0; c < index_attr.index_meta.top_centroids.size(); c++) {
            closest_centroids.Insert(std::make_pair(Distance(query, index_attr.index_meta.top_centroids[c].data,
                                                             DIMENSION, DistanceType::L2),
                                                    index_attr.index_meta.top_centroids[c].id));
            if (closest_centroids.Size() > nprobe) {
                closest_centroids.PopBack();
            }
        }

        FatalAssert(closest_centroids.Size() == std::min((size_t)nprobe, index_attr.index_meta.top_centroids.size()),
                    LOG_TAG_BASIC,
                    "Size of closest centroids list should be equal to nprobe or num_centroids, whichever is smaller.");
        if (closest_centroids.Size() == 0) {
            DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BASIC,
                    "No centroids found during ANNSearch(). This should not happen if the index is built correctly.");
            return RetStatus::Fail("No centroids found during ANNSearch()");
        }

        std::atomic<uint32_t> tasks_completed = 0;
        SXLock neighbour_list_lock;
        BufferMgr* bufferMgr = BufferMgr::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_BASIC);
        RetStatus status = RetStatus::Success();

        while (!closest_centroids.Empty()) {
            tasks_completed.store(0, std::memory_order_relaxed);
            is_leaf = closest_centroids[0].second.IsLeaf();
            uint8_t level = closest_centroids[0].second._level;
            FatalAssert(level > VectorID::VECTOR_LEVEL, LOG_TAG_BASIC,
                        "Closest centroids should be internal vertices, but found level %u",
                        level);
            nprobe = is_leaf ? k : (level == VectorID::LEAF_LEVEL + 1 ? leaf_nprobe : internal_nprobe);
            cluster_ids.clear();
            for (size_t i = 0; i < closest_centroids.Size(); ++i) {
                FatalAssert((i == 0) || (closest_centroids[i].first >= closest_centroids[i - 1].first), LOG_TAG_BASIC,
                            "SortedList is not sorted from most similar to least similar");
                cluster_ids.push_back(closest_centroids[i].second);
            }
            closest_centroids.Clear();
            tmp_centroids.Clear();
            tmp_top_vectors.Clear();


            IVFSearchTaskFactory task_factory{
                .query_vector = query,
                .num_sibling_tasks = 0,
                .top_k = nprobe,
                .is_leaf = is_leaf,
                .num_tasks_completed = &tasks_completed,
                .neighbour_list_lock = &neighbour_list_lock,
                .top_centroids = nullptr,
            };

            if (is_leaf) {
                task_factory.top_vectors = &topk_list;
            } else {
                task_factory.top_centroids = &closest_centroids;
            }

            bufferMgr->PrefetchClustersForSearch(cluster_ids.data(), cluster_ids.size(), &task_factory);
            FatalAssert(status.IsOK(), LOG_TAG_BASIC,
                        "Failed to prefetch clusters in DIVFIndex::ANNSearch(): %s",
                        status.Msg());
            FatalAssert(task_factory.num_sibling_tasks > 0, LOG_TAG_BASIC,
                        "No search tasks were created in DIVFIndex::ANNSearch()");
            size_t num_triggered_polls = 0;
            size_t num_empty_queue_polls = 0;
            size_t num_iterations = 0;
            while (tasks_completed.load(std::memory_order_acquire) < task_factory.num_sibling_tasks) {
                ++num_iterations;
                IVFSearchTask* task = nullptr;
                if (threadSelf->UniformRange64(0, (index_attr.num_user_threads * poll_rate) - 1) == 0) {
                    ++num_triggered_polls;
                    status = bufferMgr->PollRemoteReads();
                    FatalAssert(status.IsOK(), LOG_TAG_BASIC,
                                "Failed to poll remote reads in DIVFIndex::ANNSearch(): %s",
                                status.Msg());
                }

                if (search_task_queue.PopHead(task)) {
                    FatalAssert(task != nullptr, LOG_TAG_BASIC,
                                "Received null task from search task queue in DIVFIndex::ANNSearch()");
                    RetStatus task_status = RetStatus::Success();
                    if (task->is_leaf) {
                        task_status = ProcessIVFSearchTask<ClusterType::Leaf>(task, tmp_top_vectors);
                    } else {
                        task_status = ProcessIVFSearchTask<ClusterType::Internal>(task, tmp_centroids);
                    }

                    FatalAssert(task_status.IsOK(), LOG_TAG_BASIC,
                                "Failed to process IVF search task in DIVFIndex::ANNSearch(): %s",
                                task_status.Msg());
                    bufferMgr->UnpinCluster(task->cluster_id);
                    delete task;
                } else {
                    ++num_empty_queue_polls;
                    status = bufferMgr->PollRemoteReads();
                    FatalAssert(status.IsOK(), LOG_TAG_BASIC,
                                "Failed to poll remote reads in DIVFIndex::ANNSearch(): %s",
                                status.Msg());
                }
            }

            threadSelf->UpdateSearchStats(task_factory.num_sibling_tasks, num_iterations,
                                          num_triggered_polls, num_empty_queue_polls);
        }

        topk_list.Extract(neighbours);

        threadSelf->IncrementNumQueries();
        return RetStatus::Success();
    }

    String GetStats(MemoryStatsNode*& m_stat_list, bool reset_after_fetch = false) {
        BufferMgr* bufferMgr = BufferMgr::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_BASIC);
        RDMA_Manager* rdmaMgr = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdmaMgr, LOG_TAG_BASIC);
        String stats = rdmaMgr->GetStats(reset_after_fetch);
        stats += String("\n----------------\n");
        stats += bufferMgr->GetStats(m_stat_list, reset_after_fetch);
        return stats;
    }

protected:
    static constexpr size_t poll_rate = 10;
    DIVFIndexAttr index_attr;
    BlockingQueue<IVFSearchTask*> search_task_queue;

    template<ClusterType CT>
    RetStatus ProcessIVFSearchTask(IVFSearchTask* task,
                                   SortedList<std::pair<DTYPE, typename ClusterTraits<CT>::IDType>,
                                   L2DTYPEIDPairCMP>& temp_list) {
        if (task == nullptr) {
            return RetStatus::Fail("Null task provided to ProcessIVFSearchTask()");
        }

        size_t num_vectors = task->cluster_partition_num_elements;
        FatalAssert((CT == ClusterType::Leaf) == (task->is_leaf), LOG_TAG_BASIC,
                    "Cluster type does not match task leaf status in ProcessIVFSearchTask()");
        const typename ClusterTraits<CT>::ElementType* data_ptr =
            reinterpret_cast<const typename ClusterTraits<CT>::ElementType*>(task->cluster_partition_address);

        for (size_t i = 0; i < num_vectors; ++i) {
            DTYPE dist = Distance(task->query_vector, data_ptr[i].data, DIMENSION, DistanceType::L2);

            temp_list.Insert(std::make_pair(dist, data_ptr[i].id));
            if (temp_list.Size() > task->top_k) {
                temp_list.PopBack();
            }
        }

        task->neighbour_list_lock->Lock(SX_EXCLUSIVE);
        if constexpr (CT == ClusterType::Leaf) {
            task->top_vectors->MergeWith(temp_list, task->top_k, true);
        } else {
            task->top_centroids->MergeWith(temp_list, task->top_k, true);
        }
        task->neighbour_list_lock->Unlock();
        temp_list.Clear();
        size_t num_total_processes = task->num_sibling_tasks;
        size_t num_tasks_completed = task->num_tasks_completed->fetch_add(1) + 1;
        UNUSED_VARIABLE(num_total_processes);
        UNUSED_VARIABLE(num_tasks_completed);
        FatalAssert(num_tasks_completed <= num_total_processes, LOG_TAG_BASIC,
                    "More tasks completed than total sibling tasks in ProcessIVFSearchTask()");
        return RetStatus::Success();
    }
};

};

#endif