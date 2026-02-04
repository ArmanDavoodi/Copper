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
    uint16_t dimension;
    size_t num_user_threads;
    size_t pool_size;
    size_t page_size;
};

class DIVFIndex {
public:
    DIVFIndex(const DIVFIndexAttr& attr) : index_attr(attr) {
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                "Creating DIVFIndex with dimension %hu, page size %zu, pool size %zu, "
                "and %zu user threads.",
                index_attr.dimension, index_attr.page_size,
                index_attr.pool_size, index_attr.num_user_threads);

        RetStatus rs = BufferMgr::Init(
            index_attr.page_size,
            index_attr.pool_size,
            &search_task_queue,
            index_attr.dimension,
            index_attr.num_user_threads,
            centroids,
            centroid_data,
            num_centroids
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

    RetStatus ANNSearch(const VTYPE* query, size_t k, size_t nprobe,
                        std::vector<std::pair<DTYPE, IVFVectorID>>& neighbours) {
        if (query == nullptr || k == 0 || nprobe == 0) {
            FatalAssert(false, LOG_TAG_BASIC, "Invalid arguments to IVFIndex::ANNSearch()");
            return RetStatus::Fail("Invalid arguments to IVFIndex::ANNSearch()");
        }

        if (num_centroids == 0) {
            FatalAssert(false, LOG_TAG_BASIC, "Index is not built yet!");
            return RetStatus::Fail("Index is not built yet!");
        }

        if (nprobe > num_centroids) {
            DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BASIC,
                    "nprobe (%zu) is greater than the number of clusters (%zu). Reducing nprobe to %zu.",
                    nprobe, num_centroids, num_centroids);
            nprobe = num_centroids;
        }

        if (neighbours.size() > 0) {
            DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BASIC,
                    "Output neighbours vector is not empty. Clearing previous contents.");
            neighbours.clear();
        }

        SortedList<std::pair<DTYPE, IVFVectorID>, L2DTYPEIDPairCMP> topk_list(L2DTYPEIDPairCMP(), std::move(neighbours));
        SortedList<std::pair<DTYPE, VectorID>, L2DTYPEIDPairCMP> closest_centroids(L2DTYPEIDPairCMP(), nprobe);
        std::vector<VectorID> cluster_ids;
        cluster_ids.reserve(nprobe);
        for (size_t c = 0; c < num_centroids; c++) {
            closest_centroids.Insert(std::make_pair(Distance(query, &centroid_data[c * index_attr.dimension],
                                                             index_attr.dimension, DistanceType::L2),
                                                    centroids[c].centroid_id));
            if (closest_centroids.Size() > nprobe) {
                closest_centroids.PopBack();
            }
        }

        for (const auto& cent : closest_centroids) {
            cluster_ids.push_back(cent.second);
        }

        std::atomic<size_t> tasks_completed = 0;
        SXLock neighbour_list_lock;

        IVFSearchTaskFactory task_factory{
            .query_vector = query,
            .num_sibling_tasks = 0,
            .top_k = k,
            .is_leaf = true,
            .num_tasks_completed = &tasks_completed,
            .neighbour_list_lock = &neighbour_list_lock,
            .top_vectors = &topk_list,
        };

        BufferMgr* bufferMgr = BufferMgr::GetInstance();
        CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_BASIC);
        RetStatus status =
            bufferMgr->PrefetchClustersForSearch(cluster_ids.data(), cluster_ids.size(), &task_factory);
        FatalAssert(status.IsOK(), LOG_TAG_BASIC,
                    "Failed to prefetch clusters in DIVFIndex::ANNSearch(): %s",
                    status.Msg());
        FatalAssert(task_factory.num_sibling_tasks > 0, LOG_TAG_BASIC,
                    "No search tasks were created in DIVFIndex::ANNSearch()");
        SortedList<std::pair<DTYPE, IVFVectorID>, L2DTYPEIDPairCMP> temp_list(L2DTYPEIDPairCMP(), nprobe);
        while (tasks_completed.load(std::memory_order_acquire) < task_factory.num_sibling_tasks) {
            IVFSearchTask* task = nullptr;
            if (threadSelf->UniformRange64(0, (index_attr.num_user_threads * poll_rate) - 1) == 0) {
                status = bufferMgr->PollRemoteReads();
                FatalAssert(status.IsOK(), LOG_TAG_BASIC,
                            "Failed to poll remote reads in DIVFIndex::ANNSearch(): %s",
                            status.Msg());
            }

            if (search_task_queue.PopHead(task)) {
                FatalAssert(task != nullptr, LOG_TAG_BASIC,
                            "Received null task from search task queue in DIVFIndex::ANNSearch()");
                RetStatus task_status = ProcessIVFSearchTask(task, temp_list);
                FatalAssert(task_status.IsOK(), LOG_TAG_BASIC,
                            "Failed to process IVF search task in DIVFIndex::ANNSearch(): %s",
                            task_status.Msg());
                bufferMgr->UnpinCluster(task->cluster_id);
                delete task;
            } else {
                status = bufferMgr->PollRemoteReads();
                FatalAssert(status.IsOK(), LOG_TAG_BASIC,
                            "Failed to poll remote reads in DIVFIndex::ANNSearch(): %s",
                            status.Msg());
            }
        }

        topk_list.Extract(neighbours);
        return RetStatus::Success();
    }

protected:
    static constexpr size_t poll_rate = 10;
    const DIVFIndexAttr index_attr;
    ClusterMeta* centroids = nullptr;
    VTYPE* centroid_data = nullptr;
    size_t num_centroids = 0;
    BlockingQueue<IVFSearchTask*> search_task_queue;

    RetStatus ProcessIVFSearchTask(IVFSearchTask* task,
                                   SortedList<std::pair<DTYPE, IVFVectorID>, L2DTYPEIDPairCMP>& temp_list) {
        if (task == nullptr) {
            return RetStatus::Fail("Null task provided to ProcessIVFSearchTask()");
        }

        size_t vector_size = (task->is_leaf ? sizeof(IVFVectorID) : sizeof(VectorID)) +
                             (sizeof(VTYPE) * index_attr.dimension);
        size_t num_vectors = task->cluster_partition_num_elements;
        const uint8_t* data_ptr = reinterpret_cast<const uint8_t*>(task->cluster_partition_address);

        for (size_t i = 0; i < num_vectors; ++i) {
            const uint8_t* vector_offset = data_ptr + (i * vector_size);
            IVFVectorID vid = *(reinterpret_cast<const IVFVectorID*>(vector_offset));
            const VTYPE* vector_data = reinterpret_cast<const VTYPE*>(vector_offset + (task->is_leaf ? sizeof(IVFVectorID) : sizeof(VectorID)));
            DTYPE dist = Distance(task->query_vector, vector_data, index_attr.dimension, DistanceType::L2);

            if (task->is_leaf) {
                temp_list.Insert(std::make_pair(dist, vid));
                if (temp_list.Size() > task->top_k) {
                    temp_list.PopBack();
                }
            } else {
                FatalAssert(false, LOG_TAG_BASIC,
                            "ProcessIVFSearchTask called for non-leaf task, which is not supported.");
                return RetStatus::Fail("Non-leaf tasks are not supported in ProcessIVFSearchTask()");
            }
        }

        task->neighbour_list_lock->Lock(SX_EXCLUSIVE);
        task->top_vectors->MergeWith(temp_list, task->top_k, true);
        task->neighbour_list_lock->Unlock();
        temp_list.Clear();

        task->num_tasks_completed->fetch_add(1, std::memory_order_release);
        return RetStatus::Success();
    }
};

};

#endif