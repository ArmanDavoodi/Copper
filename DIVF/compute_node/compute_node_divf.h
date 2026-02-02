#ifndef CN_DIVF_H_
#define CN_DIVF_H_


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


class DIVFIndex {
public:
    DIVFIndex(uint16_t dim, uint8_t self_node_idx) :
             dim(dim), size(0) {
        if (dim == 0) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "IVFIndex dimension cannot be zero!");
        }

        
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                "IVFIndex created with dimension %hu, max vectors %zu, and %zu locks.",
                dim, max_vectors, num_locks);
    }

    ~DIVFIndex() {}


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

};

};

#endif