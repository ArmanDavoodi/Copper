#ifndef MN_DIVF_H_
#define MN_DIVF_H_


#include "common.h"
#include "vector_utils.h"
#include "distance.h"

#include "utils/thread.h"
#include "utils/sorted_list.h"
#include "utils/concurrent_datastructures.h"
#include "utils/vector_directory.h"
#include "utils/single_page_memory_pool.h"
#include "utils/rdma_manager.h"


namespace divftree {

struct IVFCluster {
    VectorID centroid_id = INVALID_VECTOR_ID;
    union {
        VTYPE* centroid = nullptr;
        MVTYPE* centroid_tmp;
    };
    size_t num_points = 0;
    char* data = nullptr; /* data points stored in a flat array */
};


class IVFIndex {
public:
    IVFIndex(uint16_t dim, size_t max_vectors, size_t num_locks) :
             dim(dim), size(0), vectorDirectory(max_vectors, dim, num_locks) {
        if (dim == 0) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "IVFIndex dimension cannot be zero!");
        }

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                "IVFIndex created with dimension %hu, max vectors %zu, and %zu locks.",
                dim, max_vectors, num_locks);
    }

    ~IVFIndex() {}


    RetStatus Build(const VTYPE* data, size_t num_points, size_t num_clusters, bool insert_duplicates,
                    IVFVectorID* out_vector_ids, size_t max_iterations, size_t num_threads = 0) {
        if (data == nullptr || num_points == 0 || num_clusters < 2 ||
            num_clusters > num_points || clusters.size() != 0 || out_vector_ids == nullptr ||
            num_threads > num_points) {
            FatalAssert(false, LOG_TAG_BASIC, "Invalid arguments to IVFIndex::Build()");
            return RetStatus::Fail("Invalid arguments to IVFIndex::Build()");
        }

        if (num_threads == 0) {
            num_threads = std::min((size_t)(std::thread::hardware_concurrency()), num_points / 8);
            if (num_threads == 0) {
                num_threads = 1;
            }
        }

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                "Starting IVFIndex::Build() with %zu data points, %zu clusters, %zu max iterations, "
                "%s duplicate insertion, and %zu threads.",
                num_points, num_clusters, max_iterations,
                insert_duplicates ? "allowing" : "disallowing", num_threads);

        clusters.resize(num_clusters);
        DIVF_MEMSET(out_vector_ids, UINT8_MAX, num_points * sizeof(IVFVectorID));
        size_t data_seen = 0;
        bool duplicate = false;
        bool* valid = new bool[num_points];
        SXSpinLock* cluster_build_locks = new SXSpinLock[num_clusters];
        MVTYPE* temp_storage = new MVTYPE[dim * clusters.size() * num_threads];
        size_t* cluster_sizes = new size_t[clusters.size() * num_threads];

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Choosing the first centroids from the existing data points...");
        for (size_t c = 0; c < num_clusters; c++) {
            FatalAssert(data_seen < num_points, LOG_TAG_BASIC,
                        "Not enough unique data points to initialize centroids!");
            FatalAssert(clusters[c].centroid_tmp == nullptr, LOG_TAG_BASIC,
                        "Centroid temporary storage should be null at this point!");
            FatalAssert(out_vector_ids[data_seen] == INVALID_IVF_VECTOR_ID, LOG_TAG_BASIC,
                        "Output vector IDs should be invalid at this point!");
            out_vector_ids[data_seen] =
                vectorDirectory.Insert(data + (data_seen * dim), insert_duplicates, &duplicate);
            if (duplicate) {
                FatalAssert(insert_duplicates == (out_vector_ids[data_seen] != INVALID_IVF_VECTOR_ID),
                            LOG_TAG_BASIC, "Duplicate found when insert_duplicates is false!");
                valid[data_seen] = false;
                data_seen++;
                c--;
                duplicate = false;
                continue;
            }

            valid[data_seen] = true;
            IVFVectorInfo* info = vectorDirectory.Find(out_vector_ids[data_seen]);
            CHECK_NOT_NULLPTR(info, LOG_TAG_BASIC);

            clusters[c].centroid_id = VectorID::AsID(c);
            clusters[c].centroid_tmp = new MVTYPE[dim];
            for (size_t d = 0; d < dim; d++) {
                clusters[c].centroid_tmp[d] = static_cast<MVTYPE>(data[(data_seen * dim) + d]);
            }
            clusters[c].num_points = 1;
            clusters[c].data = nullptr;
            info->centroid_id = clusters[c].centroid_id;
            info->vector = const_cast<VTYPE*>(data + (data_seen * dim));

            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BASIC, "%zu-th vector was chosen as the %zu-th centroid.", data_seen, c);
            data_seen++;
        }

        std::vector<Thread*> builder_threads;
        std::atomic<size_t>* current_size = new std::atomic<size_t>[clusters.size()];
        for (size_t c = 0; c < clusters.size(); c++) {
            current_size[c].store(0, std::memory_order_relaxed);
        }

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Starting clustering process...");
        if (num_threads == 1) {
            SequentialBuild(data, data_seen, num_points, valid,
                            insert_duplicates, out_vector_ids, temp_storage, cluster_sizes, current_size,
                            max_iterations);
        } else {
            builder_threads.reserve(num_threads - 1);
            uint64_t thread_range = (num_points / num_threads);
            size_t thread_step = std::max(1lu, thread_range / 8lu);
            std::atomic<size_t> seen_idx(data_seen);
            std::atomic<bool> converged(true);
            std::barrier<> sync_point(num_threads);
            for (size_t t = 1; t < num_threads; t++) {
                builder_threads.emplace_back(new Thread(100));
                Thread* thrd = builder_threads.back();
                thrd->StartMemberFunction(&IVFIndex::ParallelBuilder, this, data, &seen_idx, num_points,
                                         thread_step, valid, insert_duplicates,
                                         out_vector_ids, cluster_build_locks,
                                         &(temp_storage[t * dim * clusters.size()]),
                                         &(cluster_sizes[t * clusters.size()]), &sync_point, &converged,
                                         current_size, max_iterations);
            }
            ParallelBuild(data, seen_idx, num_points, thread_step,
                          valid, insert_duplicates, out_vector_ids, cluster_build_locks,
                          temp_storage, cluster_sizes, true, &sync_point, &converged, current_size, max_iterations);

            for (Thread* t : builder_threads) {
                delete t;
            }
            builder_threads.clear();
        }

        delete[] valid;
        delete[] cluster_build_locks;
        delete[] temp_storage;
        delete[] cluster_sizes;
        delete[] current_size;

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "IVFIndex::Build() completed successfully with %zu unique vectors and"
                "%lu total size.", vectorDirectory.Size(true), vectorDirectory.Size());
        size = vectorDirectory.Size();
        return RetStatus::Success();
    }

    RetStatus ANNSearch(const VTYPE* query, size_t k, size_t nprobe,
                        std::vector<std::pair<DTYPE, IVFVectorID>>& neighbours) {
        if (query == nullptr || k == 0 || nprobe == 0) {
            FatalAssert(false, LOG_TAG_BASIC, "Invalid arguments to IVFIndex::ANNSearch()");
            return RetStatus::Fail("Invalid arguments to IVFIndex::ANNSearch()");
        }

        if (clusters.empty() || size == 0) {
            FatalAssert(false, LOG_TAG_BASIC, "Index is not built yet!");
            return RetStatus::Fail("Index is not built yet!");
        }

        if (nprobe > clusters.size()) {
            DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BASIC,
                    "nprobe (%zu) is greater than the number of clusters (%zu). Reducing nprobe to %zu.",
                    nprobe, clusters.size(), clusters.size());
            nprobe = clusters.size();
        }

        if (neighbours.size() > 0) {
            DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BASIC,
                    "Output neighbours vector is not empty. Clearing previous contents.");
            neighbours.clear();
        }

        SortedList<std::pair<DTYPE, IVFVectorID>, L2DTYPEIDPairCMP> topk_list(L2DTYPEIDPairCMP(), std::move(neighbours));
        SortedList<std::pair<DTYPE, VectorID>, L2DTYPEIDPairCMP> closest_centroids(L2DTYPEIDPairCMP(), nprobe);

        for (size_t c = 0; c < clusters.size(); c++) {
            closest_centroids.Insert(std::make_pair(Distance(query, clusters[c].centroid, dim, DistanceType::L2),
                                                    clusters[c].centroid_id));
            if (closest_centroids.Size() > nprobe) {
                closest_centroids.PopBack();
            }
        }

        for (const auto& cent : closest_centroids) {
            size_t cent_idx = cent.second._val;
            FatalAssert(cent_idx < clusters.size(), LOG_TAG_BASIC,
                        "Invalid centroid index found during ANNSearch()");
            size_t num_points = clusters[cent_idx].num_points;
            if (num_points == 0 || clusters[cent_idx].data == nullptr) {
                continue;
            }

            for (size_t p = 0; p < num_points; p++) {
                void* v_off =
                    reinterpret_cast<void*>(clusters[cent_idx].data +
                        (p * (sizeof(IVFVectorID) + (dim * sizeof(VTYPE)))));
                IVFVectorID vid = *(reinterpret_cast<IVFVectorID*>(v_off));
                VTYPE* vector = reinterpret_cast<VTYPE*>(v_off + sizeof(IVFVectorID));
                DTYPE dist = Distance(query, vector, dim, DistanceType::L2);
                topk_list.Insert(std::make_pair(dist, vid));
                if (topk_list.Size() > k) {
                    topk_list.PopBack();
                }
            }
        }

        topk_list.Extract(neighbours);
        return RetStatus::Success();
    }

protected:
    const uint16_t dim;
    size_t size;
    std::vector<IVFCluster> clusters;
    SXSpinLock vector_directory_lock;
    VectorDirectory vectorDirectory;

    inline void ClearClusterData() {
        FatalAssert(clusters.size() >= 2, LOG_TAG_BASIC,
                    "There should be at least 2 clusters to clear data!");
        for (size_t c = 0; c < clusters.size(); c++) {
            CHECK_NOT_NULLPTR(clusters[c].centroid_tmp, LOG_TAG_BASIC);
            DIVF_MEMSET(clusters[c].centroid_tmp, 0, sizeof(MVTYPE) * dim);
            clusters[c].num_points = 0;
        }
    }

    inline void PartialFirstAssignments(const VTYPE* data, size_t start_idx, size_t end_idx,
                                        bool* is_valid, bool insert_duplicates, IVFVectorID* out_vector_ids) {
        FatalAssert(start_idx < end_idx, LOG_TAG_BASIC,
                    "Invalid start and end indices for PartialFirstAssignments()");
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BASIC, "First Assignments from %zu to %zu...", start_idx, end_idx);
        bool duplicate = false;
        for (size_t i = start_idx; i < end_idx; i++) {
            FatalAssert(out_vector_ids[i] == INVALID_IVF_VECTOR_ID, LOG_TAG_BASIC,
                        "Output vector IDs should be invalid at this point!");
            uint64_t vector_hash = vectorDirectory.GetVectorHash(data + (i * dim));
            vectorDirectory.LockVector(vector_hash, SX_EXCLUSIVE);
            out_vector_ids[i] = vectorDirectory.Insert(vector_hash, data + (i * dim), insert_duplicates, &duplicate);
            FatalAssert((out_vector_ids[i] != INVALID_IVF_VECTOR_ID) || (insert_duplicates && duplicate), LOG_TAG_BASIC,
                        "Duplicate found when insert_duplicates is false!");
            FatalAssert((out_vector_ids[i] == INVALID_IVF_VECTOR_ID) || (out_vector_ids[i].vector_hash == vector_hash),
                        LOG_TAG_BASIC, "Inserted vector hash does not match!");
            if (duplicate) {
                is_valid[i] = false;
                duplicate = false;
                vectorDirectory.UnlockVector(vector_hash);
                continue;
            }
            is_valid[i] = true;
            IVFVectorInfo* info = vectorDirectory.Find(out_vector_ids[i]);
            CHECK_NOT_NULLPTR(info, LOG_TAG_BASIC);
            info->vector = const_cast<VTYPE*>(data + (i * dim));
            vectorDirectory.UnlockVector(vector_hash);

            size_t closest_idx = 0;
            DTYPE closest_dist = Distance(data + (i * dim), clusters[0].centroid_tmp, dim, DistanceType::L2);
            FatalAssert(closest_dist > 0, LOG_TAG_BASIC,
                        "Distance computation returned 0!");
            for (size_t c = 1; c < clusters.size(); c++) {
                DTYPE dist = Distance(data + (i * dim), clusters[c].centroid_tmp, dim, DistanceType::L2);
                FatalAssert(dist > 0, LOG_TAG_BASIC,
                            "Distance computation returned 0!");
                if (MoreSimilar(dist, closest_dist, DistanceType::L2) > 0) {
                    closest_dist = dist;
                    closest_idx = c;
                }
            }

            info->centroid_id = clusters[closest_idx].centroid_id;
        }
    }

    inline void PartialAssignments(const VTYPE* data, size_t start_idx, size_t end_idx,
                                   const bool* is_valid, const IVFVectorID* out_vector_ids,
                                   std::atomic<bool>* converged) {
        FatalAssert(start_idx < end_idx, LOG_TAG_BASIC,
                    "Invalid start and end indices for PartialAssignments()");
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BASIC, "Assignments from %zu to %zu...", start_idx, end_idx);
        for (size_t i = start_idx; i < end_idx; i++) {
            if (!is_valid[i]) {
                continue;
            }

            IVFVectorInfo* info = vectorDirectory.Find(out_vector_ids[i]);
            CHECK_NOT_NULLPTR(info, LOG_TAG_BASIC);
            size_t old_cent = info->centroid_id._val;
            size_t closest_idx = 0;
            DTYPE closest_dist = Distance(data + (i * dim), clusters[0].centroid_tmp, dim, DistanceType::L2);
            FatalAssert(closest_dist > 0, LOG_TAG_BASIC,
                        "Distance computation returned 0!");
            for (size_t c = 1; c < clusters.size(); c++) {
                DTYPE dist = Distance(data + (i * dim), clusters[c].centroid_tmp, dim, DistanceType::L2);
                FatalAssert(dist > 0, LOG_TAG_BASIC,
                            "Distance computation returned 0!");
                if (MoreSimilar(dist, closest_dist, DistanceType::L2) > 0) {
                    closest_dist = dist;
                    closest_idx = c;
                }
            }

            if (old_cent != closest_idx) {
                converged->store(false, std::memory_order_release);
            }
            info->centroid_id = clusters[closest_idx].centroid_id;
        }
    }

    inline void PartialUpdateCentroids(const VTYPE* data, size_t start_idx, size_t end_idx, const bool* is_valid,
                                       const IVFVectorID* out_vector_ids, SXSpinLock* cluster_locks,
                                       MVTYPE* temp_storage, size_t* cluster_sizes) {
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BASIC, "Updating Centroids by checking vectors %zu to %zu...",
                start_idx, end_idx);
        DIVF_MEMSET(temp_storage, 0, sizeof(MVTYPE) * dim * clusters.size());
        DIVF_MEMSET(cluster_sizes, 0, sizeof(size_t) * clusters.size());

        for (size_t i = start_idx; i < end_idx; i++) {
            if (!is_valid[i]) {
                continue;
            }

            IVFVectorInfo* info = vectorDirectory.Find(out_vector_ids[i]);
            CHECK_NOT_NULLPTR(info, LOG_TAG_BASIC);
            VectorID centroid_id = info->centroid_id;

            size_t c_idx = centroid_id._val;
            cluster_sizes[c_idx]++;
            for (size_t d = 0; d < dim; d++) {
                temp_storage[(c_idx * dim) + d] += static_cast<MVTYPE>(data[(i * dim) + d]);
            }
        }

        size_t checked = 0;
        while(checked < clusters.size()) {
            checked = 0;
            for (size_t c = 0; c < clusters.size(); c++) {
                if (cluster_sizes[c] == 0) {
                    ++checked;
                    continue;
                }

                if (cluster_locks != nullptr && !cluster_locks[c].TryLock(SX_EXCLUSIVE)) {
                    continue;
                }

                clusters[c].num_points += cluster_sizes[c];
                for (size_t d = 0; d < dim; d++) {
                    clusters[c].centroid_tmp[d] += temp_storage[(c * dim) + d];
                }
                if (cluster_locks != nullptr) {
                    cluster_locks[c].Unlock();
                }
                cluster_sizes[c] = 0;
                ++checked;
            }
            FatalAssert(checked <= clusters.size(), LOG_TAG_BASIC,
                        "Checked clusters exceeded total number of clusters!");
            if (checked < clusters.size()) {
                DIVFTREE_YIELD();
            }
        }
    }

    inline void TakeTask(size_t& start_idx, size_t& size, size_t step_size, size_t end_idx,
                         std::atomic<size_t>& seen_idx) {
        size = step_size;
        start_idx = seen_idx.fetch_add(size);
        if (start_idx >= end_idx) {
            size = 0;
        } else if (end_idx - start_idx < size) {
            size = end_idx - start_idx;
        }
    }

    inline void ParallelFirstIteration(const VTYPE* data, std::atomic<size_t>& seen_idx, size_t end_idx,
                                       size_t step_size, bool* is_valid, bool insert_duplicates,
                                       IVFVectorID* out_vector_ids, SXSpinLock* cluster_build_locks,
                                       MVTYPE* temp_storage, size_t* cluster_sizes,
                                       bool is_master_thread, std::barrier<>* sync_point) {
        size_t beg, size;
        while (true) {
            TakeTask(beg, size, step_size, end_idx, seen_idx);
            if (size == 0) {
                break;
            }
            FatalAssert(beg + size <= end_idx, LOG_TAG_BASIC,
                        "Invalid task taken from the queue!");
            FatalAssert(size > 0, LOG_TAG_BASIC,
                        "Invalid task size taken from the queue!");
            FatalAssert(beg + size > beg, LOG_TAG_BASIC,
                        "Invalid task range taken from the queue!");
            PartialFirstAssignments(data, beg, beg + size, is_valid, insert_duplicates, out_vector_ids);
        }
        BARRIER(*sync_point, "ParallelFirstIter -> Assignment Phase Completed");
        if (is_master_thread) {
            ClearClusterData();
            seen_idx.store(0, std::memory_order_release);
        }
        BARRIER(*sync_point, "ParallelFirstIter -> Master Cleared Cluster Data");

        while (true) {
            TakeTask(beg, size, step_size, end_idx, seen_idx);
            if (size == 0) {
                break;
            }
            FatalAssert(beg + size <= end_idx, LOG_TAG_BASIC,
                        "Invalid task taken from the queue!");
            FatalAssert(size > 0, LOG_TAG_BASIC,
                        "Invalid task size taken from the queue!");
            FatalAssert(beg + size > beg, LOG_TAG_BASIC,
                        "Invalid task range taken from the queue!");
            PartialUpdateCentroids(data, beg, beg + size, is_valid, out_vector_ids,
                                   cluster_build_locks,
                                   temp_storage, cluster_sizes);
        }
        BARRIER(*sync_point, "ParallelFirstIter -> Iteration Completed");
    }

    inline void ParallelIteration(const VTYPE* data, std::atomic<size_t>& seen_idx, size_t end_idx,
                                  size_t step_size, const bool* is_valid,
                                  const IVFVectorID* out_vector_ids, SXSpinLock* cluster_build_locks,
                                  MVTYPE* temp_storage, size_t* cluster_sizes,
                                  bool is_master_thread, std::barrier<>* sync_point, std::atomic<bool>* converged) {
        size_t beg, size;
        while (true) {
            TakeTask(beg, size, step_size, end_idx, seen_idx);
            if (size == 0) {
                break;
            }
            FatalAssert(beg + size <= end_idx, LOG_TAG_BASIC,
                        "Invalid task taken from the queue!");
            FatalAssert(size > 0, LOG_TAG_BASIC,
                        "Invalid task size taken from the queue!");
            FatalAssert(beg + size > beg, LOG_TAG_BASIC,
                        "Invalid task range taken from the queue!");
            PartialAssignments(data, beg, beg + size, is_valid, out_vector_ids, converged);
        }

        BARRIER(*sync_point, "ParallelIter -> Assignment Phase Completed");
        if (converged->load(std::memory_order_acquire)) {
            return;
        }

        if (is_master_thread) {
            ClearClusterData();
            seen_idx.store(0, std::memory_order_release);
        }
        BARRIER(*sync_point, "ParallelIter -> Master Cleared Cluster Data");

        while (true) {
            TakeTask(beg, size, step_size, end_idx, seen_idx);
            if (size == 0) {
                break;
            }
            FatalAssert(beg + size <= end_idx, LOG_TAG_BASIC,
                        "Invalid task taken from the queue!");
            FatalAssert(size > 0, LOG_TAG_BASIC,
                        "Invalid task size taken from the queue!");
            FatalAssert(beg + size > beg, LOG_TAG_BASIC,
                        "Invalid task range taken from the queue!");
            PartialUpdateCentroids(data, beg, beg + size, is_valid, out_vector_ids,
                                   cluster_build_locks,
                                   temp_storage, cluster_sizes);
        }
        BARRIER(*sync_point, "ParallelIter -> Iteration Completed");
    }

    inline void PartialStore(const VTYPE* data, size_t start_idx, size_t end_idx,
                             const bool* is_valid, const IVFVectorID* out_vector_ids,
                             std::atomic<size_t>* current_size) {
        FatalAssert(start_idx < end_idx, LOG_TAG_BASIC,
                    "Invalid start and end indices for PartialStore()");
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Storing vectors from %zu to %zu...", start_idx, end_idx);
        for (size_t i = start_idx; i < end_idx; i++) {
            if (!is_valid[i]) {
                continue;
            }

            IVFVectorInfo* info = vectorDirectory.Find(out_vector_ids[i]);
            CHECK_NOT_NULLPTR(info, LOG_TAG_BASIC);
            size_t c_idx = info->centroid_id._val;
            info->offset = current_size[c_idx].fetch_add(1);
            FatalAssert(info->offset < clusters[c_idx].num_points, LOG_TAG_BASIC,
                        "Cluster data offset exceeded allocated size!");
            FatalAssert(info->vector == (data + (i * dim)), LOG_TAG_BASIC,
                        "Vector pointer mismatch during store!");
            void* add =
                reinterpret_cast<void*>(clusters[c_idx].data) +
                (info->offset * (sizeof(IVFVectorID) + (dim * sizeof(VTYPE))));
            info->vector = reinterpret_cast<VTYPE*>(add + sizeof(IVFVectorID));

            DIVF_MEMCOPY(add, &out_vector_ids[i], sizeof(IVFVectorID));
            DIVF_MEMCOPY(info->vector, data + (i * dim), sizeof(VTYPE) * dim);
        }
    }

    inline void ParallelStore(const VTYPE* data, std::atomic<size_t>& seen_idx, size_t end_idx,
                              size_t step_size, const bool* is_valid, const IVFVectorID* out_vector_ids,
                              std::atomic<size_t>* current_size) {
        size_t beg, size;
        while (true) {
            TakeTask(beg, size, step_size, end_idx, seen_idx);
            if (size == 0) {
                break;
            }
            FatalAssert(beg + size <= end_idx, LOG_TAG_BASIC,
                        "Invalid task taken from the queue!");
            FatalAssert(size > 0, LOG_TAG_BASIC,
                        "Invalid task size taken from the queue!");
            FatalAssert(beg + size > beg, LOG_TAG_BASIC,
                        "Invalid task range taken from the queue!");
            PartialStore(data, beg, beg + size, is_valid, out_vector_ids, current_size);
        }
    }

    inline void SequentialBuild(const VTYPE* data, size_t seen_idx, size_t end_idx,
                                bool* is_valid, bool insert_duplicates,
                                IVFVectorID* out_vector_ids, MVTYPE* temp_storage, size_t* cluster_sizes,
                                std::atomic<size_t>* current_size, size_t max_iterations) {

        PartialFirstAssignments(data, seen_idx, end_idx, is_valid, insert_duplicates, out_vector_ids);
        ClearClusterData();
        PartialUpdateCentroids(data, 0, end_idx, is_valid, out_vector_ids,
                               nullptr, temp_storage, cluster_sizes);
        for (size_t c = 0; c < clusters.size(); c++) {
            FatalAssert(clusters[c].num_points > 0, LOG_TAG_BASIC,
                        "Cluster has no points assigned to it!");
            String centroid_data = "[";
            for (size_t d = 0; d < dim; d++) {
                clusters[c].centroid_tmp[d] =
                    static_cast<MVTYPE>(clusters[c].centroid_tmp[d] / static_cast<MVTYPE>(clusters[c].num_points));
                centroid_data += String(MVTYPE_FMT "%s", clusters[c].centroid_tmp[d],
                                       (d + 1 == dim) ? "]" : ", ");
            }
            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BASIC,
                    "Centroid %zu has %zu points assigned to it. Centroid data: %s",
                    c, clusters[c].num_points, centroid_data.ToCStr());
        }

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "First iteration completed.");

        bool wait_until_converged = (max_iterations == 0);
        std::atomic<bool> converged = true;
        for (size_t i = 0; wait_until_converged || i < max_iterations - 1; i++) {
            converged.store(true, std::memory_order_relaxed);
            PartialAssignments(data, 0, end_idx, is_valid, out_vector_ids, &converged);
            if (converged.load(std::memory_order_relaxed)) {
                break;
            }
            ClearClusterData();
            PartialUpdateCentroids(data, 0, end_idx, is_valid, out_vector_ids,
                                   nullptr, temp_storage, cluster_sizes);

            for (size_t c = 0; c < clusters.size(); c++) {
                FatalAssert(clusters[c].num_points > 0, LOG_TAG_BASIC,
                            "Cluster has no points assigned to it!");
                for (size_t d = 0; d < dim; d++) {
                    clusters[c].centroid_tmp[d] =
                        static_cast<MVTYPE>(clusters[c].centroid_tmp[d] / static_cast<MVTYPE>(clusters[c].num_points));
                }
            }
            if (wait_until_converged) {
                DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "%zu%s iteration completed. converged: %s",
                        i + 2, (i == 0 ? "ed" : "th"), converged.load(std::memory_order_relaxed) ? "true" : "false");
            } else {
                DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "%zu/%zu iteration completed. converged: %s",
                        i + 2, max_iterations, converged.load(std::memory_order_relaxed) ? "true" : "false");
            }
        }


        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Clustering completed. Finalizing centroids and storing data...");

        for (size_t c = 0; c < clusters.size(); c++) {
            FatalAssert(clusters[c].num_points > 0, LOG_TAG_BASIC,
                        "Cluster has no points assigned to it!");
            VTYPE* final_centroid = new VTYPE[dim];
            String centroid_data = "[";
            for (size_t d = 0; d < dim; d++) {
                if (converged.load(std::memory_order_relaxed)) {
                    final_centroid[d] = static_cast<VTYPE>(clusters[c].centroid_tmp[d]);
                } else {
                    final_centroid[d] = static_cast<VTYPE>(clusters[c].centroid_tmp[d] /
                                                            static_cast<MVTYPE>(clusters[c].num_points));
                }
                centroid_data += String(VTYPE_FMT "%s", final_centroid[d],
                                       (d + 1 == dim) ? "]" : ", ");
            }
            delete[] clusters[c].centroid_tmp;
            clusters[c].centroid = final_centroid;
            clusters[c].data = new char[clusters[c].num_points * ((dim * sizeof(VTYPE)) + sizeof(IVFVectorID))];
            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                    "Finalized centroid %zu with %zu points. Centroid data: %s",
                    c, clusters[c].num_points, centroid_data.ToCStr());
        }


        PartialStore(data, 0, end_idx, is_valid, out_vector_ids, current_size);
        SANITY_CHECK({
            for (size_t c = 0; c < clusters.size(); c++) {
                FatalAssert(current_size[c].load(std::memory_order_relaxed) == clusters[c].num_points, LOG_TAG_BASIC,
                            "Cluster size mismatch after data storage!");
            }
        });
    }

    inline void ParallelBuild(const VTYPE* data, std::atomic<size_t>& seen_idx, size_t end_idx,
                              size_t step_size, bool* is_valid, bool insert_duplicates,
                              IVFVectorID* out_vector_ids, SXSpinLock* cluster_build_locks,
                              MVTYPE* temp_storage, size_t* cluster_sizes,
                              bool is_master_thread, std::barrier<>* sync_point, std::atomic<bool>* converged,
                              std::atomic<size_t>* current_size,
                              size_t max_iterations) {
        ParallelFirstIteration(data, seen_idx, end_idx, step_size, is_valid,
                               insert_duplicates, out_vector_ids, cluster_build_locks,
                               temp_storage, cluster_sizes, is_master_thread, sync_point);
        if (is_master_thread) {
            seen_idx.store(0, std::memory_order_release);
            for (size_t c = 0; c < clusters.size(); c++) {
                FatalAssert(clusters[c].num_points > 0, LOG_TAG_BASIC,
                            "Cluster has no points assigned to it!");
                String centroid_data = "[";
                for (size_t d = 0; d < dim; d++) {
                    clusters[c].centroid_tmp[d] =
                        static_cast<MVTYPE>(clusters[c].centroid_tmp[d] / static_cast<MVTYPE>(clusters[c].num_points));
                    centroid_data += String(MVTYPE_FMT "%s", clusters[c].centroid_tmp[d],
                                           (d + 1 == dim) ? "]" : ", ");
                }
                DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BASIC,
                        "Centroid %zu has %zu points assigned to it. Centroid data: %s",
                        c, clusters[c].num_points, centroid_data.ToCStr());
            }
            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "First iteration completed.");
            converged->store(true, std::memory_order_release);
        }
        BARRIER(*sync_point, "ParallelBuild -> First Iteration Completed + Master Reset SeenIdx + Starting Iterations...");
        bool wait_until_converged = (max_iterations == 0);
        bool conv = false;
        for (size_t i = 0; wait_until_converged || i < max_iterations - 1; i++) {
            ParallelIteration(data, seen_idx, end_idx, step_size, is_valid,
                              out_vector_ids, cluster_build_locks,
                              temp_storage, cluster_sizes, is_master_thread, sync_point, converged);

            if (converged->load(std::memory_order_acquire)) {
                conv = true;
                break;
            }

            BARRIER(*sync_point, "ParallelIter -> Not Converged");

            if (is_master_thread) {
                seen_idx.store(0, std::memory_order_release);
                for (size_t c = 0; c < clusters.size(); c++) {
                    FatalAssert(clusters[c].num_points > 0, LOG_TAG_BASIC,
                                "Cluster has no points assigned to it!");
                    for (size_t d = 0; d < dim; d++) {
                        clusters[c].centroid_tmp[d] =
                            static_cast<MVTYPE>(clusters[c].centroid_tmp[d] / static_cast<MVTYPE>(clusters[c].num_points));
                    }
                }

                if (wait_until_converged) {
                    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "%zu%s iteration completed. converged: %s",
                            i + 2, (i == 0 ? "ed" : "th"), converged->load(std::memory_order_acquire) ? "true" : "false");
                } else {
                    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "%zu/%zu iteration completed. converged: %s",
                            i + 2, max_iterations, converged->load(std::memory_order_acquire) ? "true" : "false");
                }
                converged->store(true, std::memory_order_release);
            }
            BARRIER(*sync_point, String("ParallelBuild -> iteration %zu/%zu Completed, Iterating till convergence: %s",
                                        i + 2, max_iterations,
                                        wait_until_converged ? "true" : "false").ToCStr());
        }

        if (is_master_thread) {
            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Clustering completed. Finalizing centroids and storing data...");

            seen_idx.store(0, std::memory_order_release);
            for (size_t c = 0; c < clusters.size(); c++) {
                FatalAssert(clusters[c].num_points > 0, LOG_TAG_BASIC,
                            "Cluster has no points assigned to it!");
                VTYPE* final_centroid = new VTYPE[dim];
                String centroid_data("(conv:%s)[", conv ? "t" : "f");
                for (size_t d = 0; d < dim; d++) {
                    if (!conv) {
                        final_centroid[d] = static_cast<VTYPE>(clusters[c].centroid_tmp[d]);
                    } else {
                        final_centroid[d] = static_cast<VTYPE>(clusters[c].centroid_tmp[d] /
                                                               static_cast<MVTYPE>(clusters[c].num_points));
                    }
                    centroid_data += String(VTYPE_FMT "(" MVTYPE_FMT ")%s", final_centroid[d],
                                            clusters[c].centroid_tmp[d], (d + 1 == dim) ? "]" : ", ");
                }
                delete[] clusters[c].centroid_tmp;
                clusters[c].centroid = final_centroid;
                clusters[c].data = new char[clusters[c].num_points * ((dim * sizeof(VTYPE)) + sizeof(IVFVectorID))];
                DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                        "Finalized centroid %zu with %zu points. Centroid data: %s",
                        c, clusters[c].num_points, centroid_data.ToCStr());
            }
        }

        BARRIER(*sync_point, "ParallelBuild -> KMeans Completed, Starting Data Storage...");
        ParallelStore(data, seen_idx, end_idx, step_size, is_valid, out_vector_ids,
                      current_size);

        BARRIER(*sync_point, "ParallelBuild -> Data Storage Completed");
        SANITY_CHECK({
            if (is_master_thread) {
                for (size_t c = 0; c < clusters.size(); c++) {
                    FatalAssert(current_size[c] == clusters[c].num_points, LOG_TAG_BASIC,
                                "Cluster size mismatch after data storage!");
                }
            }
        });
    }

    inline void ParallelBuilder(Thread* self, const VTYPE* data, std::atomic<size_t>* seen_idx, size_t end_idx,
                                size_t step_size, bool* is_valid, bool insert_duplicates,
                                IVFVectorID* out_vector_ids, SXSpinLock* cluster_build_locks,
                                MVTYPE* temp_storage, size_t* cluster_sizes, std::barrier<>* sync_point,
                                std::atomic<bool>* converged, std::atomic<size_t>* current_size,
                                size_t max_iterations) {
        CHECK_NOT_NULLPTR(self, LOG_TAG_DIVFTREE);
        self->InitDIVFThread();
        ParallelBuild(data, *seen_idx, end_idx, step_size, is_valid,
                      insert_duplicates, out_vector_ids, cluster_build_locks,
                      temp_storage, cluster_sizes, false, sync_point, converged,
                      current_size, max_iterations);
        self->DestroyDIVFThread();
    }
};

};

#endif