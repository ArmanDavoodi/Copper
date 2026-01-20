#ifndef IVF_H_
#define IVF_H_


#include "common.h"
#include "vector_utils.h"
#include "distance.h"

#include "utils/thread.h"
#include "utils/sorted_list.h"
#include "utils/concurrent_datastructures.h"
#include "utils/vector_directory.h"


namespace divftree {

struct IVFCluster {
    VectorID centroid_id;
    union {
        VTYPE* centroid;
        MVTYPE* centroid_tmp;
    };
    size_t num_points;
    char* data; /* data points stored in a flat array */
};

struct BuilderTask {
    size_t start_idx;
    size_t end_idx;
};


class IVFIndex {
public:
    IVFIndex(uint16_t dim, size_t max_vectors, size_t num_locks) :
             dim(dim), size(0), vectorDirectory(max_vectors, dim, num_locks) {
        if (dim == 0) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "IVFIndex dimension cannot be zero!");
        }
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

        clusters.resize(num_clusters);
        ClearClusterData();
        memset(out_vector_ids, UINT8_MAX, num_points * sizeof(IVFVectorID));
        size_t data_seen = 0;
        bool duplicate = false;
        DTYPE* distances = new DTYPE[num_clusters * num_threads];
        bool* valid = new bool[num_clusters];
        SXSpinLock* cluster_build_locks = new SXSpinLock[num_clusters];
        MVTYPE* temp_storage = new MVTYPE[dim * clusters.size() * num_threads];
        size_t* cluster_sizes = new size_t[clusters.size() * num_threads];
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
            data_seen++;
        }

        std::vector<Thread> builder_threads;
        if (num_threads == 1) {
            SequentialBuild(data, data_seen, num_points, distances, valid,
                            insert_duplicates, out_vector_ids, temp_storage, cluster_sizes,
                            max_iterations);
        } else {
            builder_threads.reserve(num_threads - 1);
            uint64_t thread_range = (num_points / num_threads);
            
        }
    }

    RetStatus ANNSearch(const VTYPE* query, size_t k,
                        std::vector<std::pair<DTYPE, VectorID>>& neighbours) {
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_NOT_IMPLEMENTED, "IVFIndex ANNSearch() not implemented!");
        return RetStatus::Fail(nullptr);
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
            memset(clusters[c].centroid_tmp, 0, sizeof(MVTYPE) * dim);
            clusters[c].num_points = 0;
        }
    }

    inline void PartialFirstAssignments(const VTYPE* data, size_t start_idx, size_t end_idx, DTYPE* distances,
                                        bool* is_valid, bool insert_duplicates, IVFVectorID* out_vector_ids) {
        FatalAssert(start_idx < end_idx, LOG_TAG_BASIC,
                    "Invalid start and end indices for PartialFirstAssignments()");
        bool duplicate = false;
        for (size_t i = start_idx; i < end_idx; i++) {
            FatalAssert(out_vector_ids[i] == INVALID_IVF_VECTOR_ID, LOG_TAG_BASIC,
                        "Output vector IDs should be invalid at this point!");
            uint64_t vector_hash = vectorDirectory.GetVectorHash(data + (i * dim));
            vectorDirectory.LockVector(vector_hash, SX_EXCLUSIVE);
            out_vector_ids[i] = vectorDirectory.Insert(vector_hash, data + (i * dim), insert_duplicates, &duplicate);
            vectorDirectory.UnlockVector(vector_hash);
            FatalAssert((out_vector_ids[i] != INVALID_IVF_VECTOR_ID) || (insert_duplicates && duplicate), LOG_TAG_BASIC,
                        "Duplicate found when insert_duplicates is false!");
            FatalAssert((out_vector_ids[i] == INVALID_IVF_VECTOR_ID) || (out_vector_ids[i].vector_hash == vector_hash),
                        LOG_TAG_BASIC, "Inserted vector hash does not match!");
            if (duplicate) {
                is_valid[i] = false;
                duplicate = false;
                continue;
            }
            is_valid[i] = true;

            size_t closest_idx = 0;
            distances[i] = Distance(data + (i * dim), clusters[0].centroid_tmp, dim, DistanceType::L2);
            FatalAssert(distances[i] > 0, LOG_TAG_BASIC,
                        "Distance computation returned 0!");
            for (size_t c = 1; c < clusters.size(); c++) {
                DTYPE dist = Distance(data + (i * dim), clusters[c].centroid_tmp, dim, DistanceType::L2);
                FatalAssert(dist > 0, LOG_TAG_BASIC,
                            "Distance computation returned 0!");
                if (MoreSimilar(dist, distances[i], DistanceType::L2)) {
                    distances[i] = dist;
                    closest_idx = c;
                }
            }

            IVFVectorInfo* info = vectorDirectory.Find(out_vector_ids[i]);
            CHECK_NOT_NULLPTR(info, LOG_TAG_BASIC);
            info->centroid_id = clusters[closest_idx].centroid_id;
        }
    }

    inline void PartialAssignments(const VTYPE* data, size_t start_idx, size_t end_idx, DTYPE* distances,
                                   const bool* is_valid, const IVFVectorID* out_vector_ids,
                                   std::atomic<bool>* converged) {
        FatalAssert(start_idx < end_idx, LOG_TAG_BASIC,
                    "Invalid start and end indices for PartialAssignments()");
        for (size_t i = start_idx; i < end_idx; i++) {
            if (!is_valid[i]) {
                continue;
            }

            IVFVectorInfo* info = vectorDirectory.Find(out_vector_ids[i]);
            CHECK_NOT_NULLPTR(info, LOG_TAG_BASIC);
            size_t old_cent = info->centroid_id._val;
            size_t closest_idx = 0;
            distances[i] = Distance(data + (i * dim), clusters[0].centroid_tmp, dim, DistanceType::L2);
            FatalAssert(distances[i] > 0, LOG_TAG_BASIC,
                        "Distance computation returned 0!");
            for (size_t c = 1; c < clusters.size(); c++) {
                DTYPE dist = Distance(data + (i * dim), clusters[c].centroid_tmp, dim, DistanceType::L2);
                FatalAssert(dist > 0, LOG_TAG_BASIC,
                            "Distance computation returned 0!");
                if (MoreSimilar(dist, distances[i], DistanceType::L2)) {
                    distances[i] = dist;
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
        memset(temp_storage, 0, sizeof(MVTYPE) * dim * clusters.size());
        memset(cluster_sizes, 0, sizeof(size_t) * clusters.size());

        for (size_t i = start_idx; i < end_idx; i++) {
            if (!is_valid[i]) {
                continue;
            }

            IVFVectorInfo* info = vectorDirectory.Find(out_vector_ids[i]);
            CHECK_NOT_NULLPTR(info, LOG_TAG_BASIC);
            VectorID centroid_id = info->centroid_id;
            FatalAssert(centroid_id.IsValid() && centroid_id.IsCentroid(), LOG_TAG_BASIC,
                        "Invalid centroid ID found during UpdateCentroids()!");

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
        FatalAssert(start_idx < end_idx, LOG_TAG_BASIC,
                    "Invalid start index taken from task queue!");
        FatalAssert(size > 0, LOG_TAG_BASIC,
                    "Invalid size taken from task queue!");
    }

    inline void ParallelFirstIteration(const VTYPE* data, std::atomic<size_t>& seen_idx, size_t end_idx,
                                       size_t step_size, DTYPE* distances, bool* is_valid, bool insert_duplicates,
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
            PartialFirstAssignments(data, beg, beg + size, distances, is_valid, insert_duplicates, out_vector_ids);
        }
        sync_point->arrive_and_wait();
        if (is_master_thread) {
            ClearClusterData();
            seen_idx.store(0, std::memory_order_release);
        }
        sync_point->arrive_and_wait();

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
        sync_point->arrive_and_wait();
    }

    inline void ParallelIteration(const VTYPE* data, std::atomic<size_t>& seen_idx, size_t end_idx,
                                  size_t step_size, DTYPE* distances, const bool* is_valid,
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
            PartialAssignments(data, beg, beg + size, distances, is_valid, out_vector_ids, converged);
        }

        sync_point->arrive_and_wait();
        if (converged->load(std::memory_order_acquire)) {
            return;
        }

        if (is_master_thread) {
            ClearClusterData();
            seen_idx.store(0, std::memory_order_release);
        }
        sync_point->arrive_and_wait();

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
        sync_point->arrive_and_wait();
    }

    inline void PartialStore(const VTYPE* data, size_t start_idx, size_t end_idx,
                             const bool* is_valid, const IVFVectorID* out_vector_ids,
                             std::atomic<size_t>* current_size) {
        FatalAssert(start_idx < end_idx, LOG_TAG_BASIC,
                    "Invalid start and end indices for PartialStore()");
        for (size_t i = start_idx; i < end_idx; i++) {
            if (!is_valid[i]) {
                continue;
            }

            IVFVectorInfo* info = vectorDirectory.Find(out_vector_ids[i]);
            CHECK_NOT_NULLPTR(info, LOG_TAG_BASIC);
            size_t c_idx = info->centroid_id._val;
            FatalAssert(info->vector == nullptr, LOG_TAG_BASIC,
                        "Vector pointer should be null at this point!");
            info->offset = current_size[c_idx].fetch_add(1);
            FatalAssert(info->offset < clusters[c_idx].num_points, LOG_TAG_BASIC,
                        "Cluster data offset exceeded allocated size!");
            info->vector = reinterpret_cast<VTYPE*>(clusters[c_idx].data) + (info->offset * dim);
            memcpy(info->vector, data + (i * dim), sizeof(VTYPE) * dim);
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
                                DTYPE* distances, bool* is_valid, bool insert_duplicates,
                                IVFVectorID* out_vector_ids, MVTYPE* temp_storage, size_t* cluster_sizes,
                                std::atomic<size_t>* current_size, size_t max_iterations) {

        PartialFirstAssignments(data, seen_idx, end_idx, distances, is_valid, insert_duplicates, out_vector_ids);
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

        bool wait_until_converged = (max_iterations == 0);
        std::atomic<bool> converged = true;
        for (size_t i = 0; wait_until_converged || i < max_iterations - 1; i++) {
            converged.store(true, std::memory_order_relaxed);
            PartialAssignments(data, 0, end_idx, distances, is_valid, out_vector_ids, &converged);
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
        }

        for (size_t c = 0; c < clusters.size(); c++) {
            FatalAssert(clusters[c].num_points > 0, LOG_TAG_BASIC,
                        "Cluster has no points assigned to it!");
            VTYPE* final_centroid = new VTYPE[dim];
            for (size_t d = 0; d < dim; d++) {
                if (converged.load(std::memory_order_relaxed)) {
                    final_centroid[d] = static_cast<VTYPE>(clusters[c].centroid_tmp[d]);
                } else {
                    final_centroid[d] = static_cast<VTYPE>(clusters[c].centroid_tmp[d] /
                                                            static_cast<MVTYPE>(clusters[c].num_points));
                }
            }
            delete[] clusters[c].centroid_tmp;
            clusters[c].centroid = final_centroid;
            clusters[c].data = new char[clusters[c].num_points * dim * sizeof(VTYPE)];
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
                              size_t step_size, DTYPE* distances, bool* is_valid, bool insert_duplicates,
                              IVFVectorID* out_vector_ids, SXSpinLock* cluster_build_locks,
                              MVTYPE* temp_storage, size_t* cluster_sizes,
                              bool is_master_thread, std::barrier<>* sync_point, std::atomic<bool>* converged,
                              std::atomic<size_t>* current_size,
                              size_t max_iterations) {
        ParallelFirstIteration(data, seen_idx, end_idx, step_size, distances, is_valid,
                               insert_duplicates, out_vector_ids, cluster_build_locks,
                               temp_storage, cluster_sizes, is_master_thread, sync_point);
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
        }
        sync_point->arrive_and_wait();
        converged->store(true, std::memory_order_release);
        bool wait_until_converged = (max_iterations == 0);
        bool conv = false;
        for (size_t i = 0; wait_until_converged || i < max_iterations - 1; i++) {
            ParallelIteration(data, seen_idx, end_idx, step_size, distances, is_valid,
                              out_vector_ids, cluster_build_locks,
                              temp_storage, cluster_sizes, is_master_thread, sync_point, converged);

            if (converged->load(std::memory_order_acquire)) {
                conv = true;
                break;
            }

            if (is_master_thread) {
                seen_idx.store(0, std::memory_order_release);
                converged->store(true, std::memory_order_release);
                for (size_t c = 0; c < clusters.size(); c++) {
                    FatalAssert(clusters[c].num_points > 0, LOG_TAG_BASIC,
                                "Cluster has no points assigned to it!");
                    for (size_t d = 0; d < dim; d++) {
                        clusters[c].centroid_tmp[d] =
                            static_cast<MVTYPE>(clusters[c].centroid_tmp[d] / static_cast<MVTYPE>(clusters[c].num_points));
                    }
                }
            }
            sync_point->arrive_and_wait();
        }

        if (is_master_thread) {
            seen_idx.store(0, std::memory_order_release);
            for (size_t c = 0; c < clusters.size(); c++) {
                FatalAssert(clusters[c].num_points > 0, LOG_TAG_BASIC,
                            "Cluster has no points assigned to it!");
                VTYPE* final_centroid = new VTYPE[dim];
                for (size_t d = 0; d < dim; d++) {
                    if (conv) {
                        final_centroid[d] = static_cast<VTYPE>(clusters[c].centroid_tmp[d]);
                    } else {
                        final_centroid[d] = static_cast<VTYPE>(clusters[c].centroid_tmp[d] /
                                                               static_cast<MVTYPE>(clusters[c].num_points));
                    }
                }
                delete[] clusters[c].centroid_tmp;
                clusters[c].centroid = final_centroid;
                clusters[c].data = new char[clusters[c].num_points * dim * sizeof(VTYPE)];
            }
        }

        sync_point->arrive_and_wait();
        ParallelStore(data, seen_idx, end_idx, step_size, is_valid, out_vector_ids,
                      current_size);

        sync_point->arrive_and_wait();
        SANITY_CHECK({
            if (is_master_thread) {
                for (size_t c = 0; c < clusters.size(); c++) {
                    FatalAssert(current_size[c] == clusters[c].num_points, LOG_TAG_BASIC,
                                "Cluster size mismatch after data storage!");
                }
            }
        });
    }

    inline void ParallelBuilder(Thread* self, const VTYPE* data, std::atomic<size_t>& seen_idx, size_t end_idx,
                                size_t step_size, DTYPE* distances, bool* is_valid, bool insert_duplicates,
                                IVFVectorID* out_vector_ids, SXSpinLock* cluster_build_locks,
                                MVTYPE* temp_storage, size_t* cluster_sizes, std::barrier<>* sync_point,
                                std::atomic<bool>* converged, std::atomic<size_t>* current_size,
                                size_t max_iterations) {
        CHECK_NOT_NULLPTR(self, LOG_TAG_DIVFTREE);
        self->InitDIVFThread();
        ParallelBuild(data, seen_idx, end_idx, step_size, distances, is_valid,
                      insert_duplicates, out_vector_ids, cluster_build_locks,
                      temp_storage, cluster_sizes, false, sync_point, converged,
                      current_size, max_iterations);
        self->DestroyDIVFThread();
    }
};

};

#endif