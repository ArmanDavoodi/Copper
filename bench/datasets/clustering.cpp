#include <iostream>
#include <cstdint>
#include <string>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <omp.h>
#include <sys/mman.h>
#include <vector>

#define UINT8 1
#define UINT16 2
#define FLOAT 3
#define UINT32 4
#define UINT64 5
#define DOUBLE 6

namespace divftree {
#if defined(VECTOR_TYPE)
    #if VECTOR_TYPE == UINT8
        using VTYPE = uint8_t;
        #define VTYPE_FMT "%hhu"
        #pragma message("DTYPE = UINT8")
    #elif VECTOR_TYPE == UINT16
        using VTYPE = uint16_t;
        #define VTYPE_FMT "%hu"
        #pragma message("DTYPE = UINT16")
    #elif VECTOR_TYPE == FLOAT
        using VTYPE = float;
        #define VTYPE_FMT "%0.2f"
        #pragma message("DTYPE = FLOAT")
    #else
        #error UNDEFINED VECTOR_TYPE!
    #endif
#else
using VTYPE = uint8_t;
#define VTYPE_FMT "%hhu"
#error VECTOR_TYPE not found!
#endif

#if defined(CENTROID_TYPE)
    #if CENTROID_TYPE == UINT8
        using CTYPE = uint8_t;
        #define CTYPE_FMT "%hhu"
        using MVTYPE = double;
        #define MVTYPE_FMT "%0.4f"
        #pragma message("CTYPE = UINT8")
    #elif CENTROID_TYPE == UINT16
        using CTYPE = uint16_t;
        #define CTYPE_FMT "%hu"
        using MVTYPE = double;
        #define MVTYPE_FMT "%0.4f"
        #pragma message("CTYPE = UINT16")
    #elif CENTROID_TYPE == FLOAT
        using CTYPE = float;
        #define CTYPE_FMT "%0.2f"
        using MVTYPE = double;
        #define MVTYPE_FMT "%0.4f"
        #pragma message("CTYPE = FLOAT")
    #else
        #error UNDEFINED CENTROID_TYPE!
    #endif
#else
using CTYPE = uint8_t;
#define CTYPE_FMT "%hhu"
using MVTYPE = double;
#define MVTYPE_FMT "%lu"
#error CENTROID_TYPE not found!
#endif

#if defined(DISTANCE_TYPE)
    #if DISTANCE_TYPE == UINT8
        using DTYPE = uint8_t;
        #define DTYPE_FMT "%hhu"
        #pragma message("DTYPE = UINT8")
    #elif DISTANCE_TYPE == UINT16
        using DTYPE = uint16_t;
        #define DTYPE_FMT "%hu"
        #pragma message("DTYPE = UINT16")
    #elif DISTANCE_TYPE == UINT32
        using DTYPE = uint32_t;
        #define DTYPE_FMT "%u"
        #pragma message("DTYPE = UINT32")
    #elif DISTANCE_TYPE == UINT64
        using DTYPE = uint64_t;
        #define DTYPE_FMT "%lu"
        #pragma message("DTYPE = UINT64")
    #elif DISTANCE_TYPE == FLOAT
        using DTYPE = float;
        #define DTYPE_FMT "%0.2f"
        #pragma message("DTYPE = FLOAT")
    #elif DISTANCE_TYPE == DOUBLE
        using DTYPE = double;
        #define DTYPE_FMT "%0.4f"
        #pragma message("DTYPE = DOUBLE")
    #else
        #error UNDEFINED DISTANCE_TYPE!
    #endif
#else
using DTYPE = uint8_t;
#define DTYPE_FMT "%hhu"
#error DISTANCE_TYPE not found!
#endif
}

#ifndef DIMENSION
#define DIMENSION 128
#error DIMENSION not defined!
#endif

#include "common.h"
#include "utils/concurrent_datastructures.h"

template<typename T1, typename T2>
inline divftree::DTYPE L2Squared(const T1* a, const T2* b) {
    divftree:: DTYPE dist = 0;
    for (uint16_t i = 0; i < DIMENSION; i++) {
        divftree::DTYPE diff = static_cast<divftree::DTYPE>(a[i]) - static_cast<divftree::DTYPE>(b[i]);
        dist += diff * diff;
    }
    return dist;
}

inline bool MoreSimilar(const divftree::DTYPE& dist1, const divftree::DTYPE& dist2) {
    return dist1 < dist2;
}

struct __attribute__((packed)) VectorData {
    divftree::IVFVectorID id;
    uint32_t num_duplicates;
    divftree::VTYPE data[DIMENSION];
};

struct CentroidData {
    divftree::VectorID id;
    divftree::CTYPE data[DIMENSION];
};

struct ClusterData {
    divftree::VectorID id;
    uint32_t num_points;
    std::atomic<uint32_t> num_total_points;
    uint32_t offset;

    divftree::MVTYPE data[DIMENSION];

    // ClusterData() = default;
    // ClusterData(const ClusterData& other) {
    //     id = other.id;
    //     num_points = other.num_points;
    //     num_total_points.store(other.num_total_points.load(std::memory_order_acquire), std::memory_order_release);
    //     offset = other.offset;
    //     memcpy(data, other.data, sizeof(divftree::MVTYPE) * DIMENSION);
    // }
    // ClusterData& operator=(const ClusterData& other) {
    //     if (this != &other) {
    //         id = other.id;
    //         num_points = other.num_points;
    //         num_total_points.store(other.num_total_points.load(std::memory_order_acquire), std::memory_order_release);
    //         offset = other.offset;
    //         memcpy(data, other.data, sizeof(divftree::MVTYPE) * DIMENSION);
    //     }
    //     return *this;
    // }
};

struct ClusterMetaData {
    std::atomic<uint32_t> next_index; // used for parallel update of the assigned_vector_indices
    std::vector<uint32_t> assigned_vector_indices; // indices of vectors assigned to this cluster, used for updating the centroid in parallel

    divftree::SXSpinLock lock; // protects the cluster data when updating them in parallel

    // ClusterMetaData() = default;
    // ClusterMetaData(const ClusterMetaData& other) {
    //     next_index.store(other.next_index.load(std::memory_order_acquire), std::memory_order_release);
    //     assigned_vector_indices = other.assigned_vector_indices;
    // }
    // ClusterMetaData& operator=(const ClusterMetaData& other) {
    //     if (this != &other) {
    //         next_index.store(other.next_index.load(std::memory_order_acquire), std::memory_order_release);
    //         assigned_vector_indices = other.assigned_vector_indices;
    //     }
    //     return *this;
    // }
    // ClusterMetaData(ClusterMetaData&& other) {
    //     next_index.store(other.next_index.load(std::memory_order_acquire), std::memory_order_release);
    //     assigned_vector_indices = std::move(other.assigned_vector_indices);
    // }
    // ClusterMetaData& operator=(ClusterMetaData&& other) {
    //     if (this != &other) {
    //         next_index.store(other.next_index.load(std::memory_order_acquire), std::memory_order_release);
    //         assigned_vector_indices = std::move(other.assigned_vector_indices);
    //     }
    //     return *this;
    // }
};

struct ClusterManager {
    const uint8_t level;
    std::vector<std::pair<ClusterData*, ClusterMetaData*>> clusters;
    divftree::SXLock lock; // protects the cluster manager when adding new clusters

    ClusterManager(uint8_t lev) : level(lev) {}
    ~ClusterManager() {
        for (auto& cluster : clusters) {
            delete cluster.first;
            delete cluster.second;
        }
    }

    uint32_t AddClusters(uint32_t num_clusters, bool clear_data) {
        lock.Lock(divftree::LockMode::SX_EXCLUSIVE);
        size_t old_size = clusters.size();
        divftree::VectorID next_id;
        if (!clusters.empty()) {
            next_id = clusters.back().first->id;
            next_id._val += 1;
        } else {
            next_id._val = 0;
            next_id._level = level;
            next_id._creator_node_id = 0;
        }

        clusters.resize(old_size + num_clusters);
        for (size_t i = old_size; i < clusters.size(); i++) {
            clusters[i].first = new ClusterData();
            clusters[i].second = new ClusterMetaData();
            clusters[i].first->id = next_id;
            next_id._val += 1;
            clusters[i].first->num_points = 0;
            clusters[i].first->offset = 0;
            clusters[i].first->num_total_points.store(0, std::memory_order_release);
            if (clear_data) {
                memset(clusters[i].first->data, 0, sizeof(divftree::MVTYPE) * DIMENSION);
            }

            clusters[i].second->next_index.store(0, std::memory_order_release);
        }
        lock.Unlock();
        return old_size;
    }

    inline std::pair<ClusterData*, ClusterMetaData*> At(uint32_t idx) {
        lock.Lock(divftree::LockMode::SX_SHARED);
        std::pair<ClusterData*, ClusterMetaData*> res = clusters[idx];
        lock.Unlock();
        return res;
    }

    inline ClusterData& operator[](uint32_t idx) {
        lock.Lock(divftree::LockMode::SX_SHARED);
        ClusterData* res = clusters[idx].first;
        lock.Unlock();
        return *res;
    }

    inline uint32_t SetOffsets() {
        lock.Lock(divftree::LockMode::SX_SHARED);
        uint32_t num_seen = 0;
        for (uint32_t i = 0; i < clusters.size(); i++) {
            std::pair<ClusterData*, ClusterMetaData*> cluster_info = clusters[i];
            ClusterData& cluster_data = *(cluster_info.first);
            ClusterMetaData& cluster_meta = *(cluster_info.second);
            FatalAssert(cluster_data.num_points > 0, LOG_TAG_BASIC, "Cluster has no points assigned!");
            cluster_data.offset = num_seen;
            num_seen += cluster_data.num_points;
            cluster_meta.next_index.store(0, std::memory_order_release);
        }
        lock.Unlock();
        return num_seen;
    }
};

template<typename DataType>
void kmeans(DataType* data, uint32_t* const valid_indices, uint32_t* assignments, uint32_t num_points, uint32_t num_clusters,
            uint32_t max_iterations, size_t num_threads, ClusterManager& cluster_manager,
            uint32_t* const valid_cluster_indices) {
    std::atomic<bool> converged = false;
    
    std::cout << "Initializing centroids using " << num_threads << " threads...\n";
    #pragma omp parallel num_threads(num_threads) \
        shared(data, valid_indices, assignments, num_points, num_clusters, max_iterations,\
               cluster_manager, valid_cluster_indices, converged)
    {
        divftree::Thread* st = divftree::threadSelf;
        bool created = false;
        if (st == nullptr) {
            created = true;
            st = new divftree::Thread(100);
            st->InitDIVFThread(true);
        }

        /* choose first centroids */
        #pragma omp for schedule(static)
        for (uint32_t i = 0; i < num_clusters; i++) {
            uint32_t idx = (valid_indices == nullptr ? i : valid_indices[i]);
            uint32_t c = (valid_cluster_indices == nullptr ? i : valid_cluster_indices[i]);
            std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c);
            ClusterData& cluster_data = *(cluster_info.first);
            for (uint16_t d = 0; d < DIMENSION; d++) {
                cluster_data.data[d] = static_cast<divftree::MVTYPE>(data[idx].data[d]);
            }
        }

        divftree::MVTYPE* cluster_sums = new divftree::MVTYPE[num_clusters * DIMENSION];
        uint32_t* cluster_counts = new uint32_t[num_clusters];
        memset(cluster_sums, 0, sizeof(divftree::MVTYPE) * num_clusters * DIMENSION);
        memset(cluster_counts, 0, sizeof(uint32_t) * num_clusters);

        for (uint32_t iter = 0;
            ((iter < max_iterations && max_iterations != 0) || !converged.load(std::memory_order_acquire));
            ++iter) {

            #pragma omp single
            {
                if (iter > 0) {
                    converged.store(true, std::memory_order_release);
                }
            }

            /* assignment */
            #pragma omp for schedule(static)
            for (uint32_t idx = 0; idx < num_points; idx++) {
                uint32_t i = (valid_indices == nullptr ? idx : valid_indices[idx]);
                uint32_t c = (valid_cluster_indices == nullptr ? 0 : valid_cluster_indices[0]);
                uint32_t best_cluster = 0;
                divftree::DTYPE best_dist = L2Squared(data[i].data, cluster_manager.At(c).first->data);
                for (uint32_t c_idx = 1; c_idx < num_clusters; c_idx++) {
                    c = (valid_cluster_indices == nullptr ? c_idx : valid_cluster_indices[c_idx]);
                    divftree::DTYPE dist = L2Squared(data[i].data, cluster_manager.At(c).first->data);
                    if (MoreSimilar(dist, best_dist)) {
                        best_dist = dist;
                        best_cluster = c_idx;
                    }
                }

                c = (valid_cluster_indices == nullptr ? best_cluster : valid_cluster_indices[best_cluster]);
                if (assignments[i] != c) {
                    converged.store(false, std::memory_order_release);
                    assignments[i] = c;
                }

                ++(cluster_counts[best_cluster]);
                for (uint16_t d = 0; d < DIMENSION; d++) {
                    cluster_sums[best_cluster * DIMENSION + d] += static_cast<divftree::MVTYPE>(data[i].data[d]);
                }
            }

            if (converged.load(std::memory_order_acquire)) {
                #pragma omp single
                {
                    if (max_iterations == 0) {
                        std::cout << "Iteration " << iter+1 << " completed. Converged: Yes\n";
                    } else {
                        std::cout << "Iteration " << iter+1 << "/" << max_iterations << " completed. Converged: Yes\n";
                    }
                }
                break;
            }

            #pragma omp barrier

            #pragma omp for schedule(static)
            for (uint32_t i = 0; i < num_clusters; i++) {
                uint32_t c = (valid_cluster_indices == nullptr ? i : valid_cluster_indices[i]);
                std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c);
                ClusterData& cluster_data = *(cluster_info.first);
                cluster_data.num_points = 0;
                memset(cluster_data.data, 0, sizeof(divftree::MVTYPE) * DIMENSION);
            }

            uint32_t tn = (uint32_t)omp_get_thread_num();
            uint32_t step_size = (uint32_t)std::max(1lu, (size_t)num_clusters / num_threads);
            tn = tn * step_size;
            for (uint32_t i = 0; i < num_clusters; i++) {
                uint32_t c_idx = (tn + i) % num_clusters;
                uint32_t c = (valid_cluster_indices == nullptr ? c_idx : valid_cluster_indices[c_idx]);
                std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c);
                ClusterData& cluster_data = *(cluster_info.first);
                ClusterMetaData& cluster_meta = *(cluster_info.second);
                cluster_meta.lock.Lock(divftree::LockMode::SX_EXCLUSIVE);
                cluster_data.num_points += cluster_counts[c_idx];
                cluster_counts[c_idx] = 0;
                for (uint16_t d = 0; d < DIMENSION; d++) {
                    cluster_data.data[d] += cluster_sums[c_idx * DIMENSION + d];
                    cluster_sums[c_idx * DIMENSION + d] = 0;
                }
                cluster_meta.lock.Unlock();
            }

            #pragma omp barrier

            #pragma omp for schedule(static)
            for (uint32_t i = 0; i < num_clusters; i++) {
                uint32_t c = (valid_cluster_indices == nullptr ? i : valid_cluster_indices[i]);
                std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c);
                ClusterData& cluster_data = *(cluster_info.first);
                FatalAssert(cluster_data.num_points > 0, LOG_TAG_BASIC, "Cluster has no points assigned!");
                for (uint16_t d = 0; d < DIMENSION; d++) {
                    cluster_data.data[d] /= static_cast<divftree::MVTYPE>(cluster_data.num_points);
                }
            }

            #pragma omp single
            {
                FatalAssert(!converged.load(std::memory_order_acquire), LOG_TAG_BASIC, "Should not be converged at this point!");
                if (max_iterations == 0) {
                    std::cout << "Iteration " << iter+1 << " completed. Converged: No\n";
                } else {
                    std::cout << "Iteration " << iter+1 << "/" << max_iterations << " completed. Converged: No\n";
                }
            }
        }

        delete[] cluster_sums;
        delete[] cluster_counts;
        if (created) {
            st->DestroyDIVFThread();
        }
    }
}

template<typename DataType>
void kmeans(DataType* data, uint32_t* const valid_indices, uint32_t* assignments, uint32_t num_points, uint32_t num_clusters,
            uint32_t max_iterations, ClusterManager& cluster_manager, uint32_t* const valid_cluster_indices) {
    bool converged = false;

    /* choose the first centroids */
    for (uint32_t i = 0; i < num_clusters; i++) {
        uint32_t idx = (valid_indices == nullptr ? i : valid_indices[i]);
        uint32_t c = (valid_cluster_indices == nullptr ? i : valid_cluster_indices[i]);
        std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c);
        ClusterData& cluster_data = *(cluster_info.first);
        cluster_data.num_points = 0;
        for (uint16_t d = 0; d < DIMENSION; d++) {
            cluster_data.data[d] = static_cast<divftree::MVTYPE>(data[idx].data[d]);
        }
    }

    divftree::MVTYPE* cluster_sums = new divftree::MVTYPE[num_clusters * DIMENSION];
    uint32_t* cluster_counts = new uint32_t[num_clusters];
    memset(cluster_sums, 0, sizeof(divftree::MVTYPE) * num_clusters * DIMENSION);
    memset(cluster_counts, 0, sizeof(uint32_t) * num_clusters);

    for (uint32_t iter = 0; ((iter < max_iterations && max_iterations != 0) || !converged); ++iter) {
        if (iter > 0) {
            converged = true;
        }

        /* assignment */
        for (uint32_t idx = 0; idx < num_points; idx++) {
            uint32_t i = (valid_indices == nullptr ? idx : valid_indices[idx]);
            uint32_t c = (valid_cluster_indices == nullptr ? 0 : valid_cluster_indices[0]);
            uint32_t best_cluster = 0;
            divftree::DTYPE best_dist = L2Squared(data[i].data, cluster_manager.At(c).first->data);
            for (uint32_t c_idx = 1; c_idx < num_clusters; c_idx++) {
                c = (valid_cluster_indices == nullptr ? c_idx : valid_cluster_indices[c_idx]);
                divftree::DTYPE dist = L2Squared(data[i].data, cluster_manager.At(c).first->data);
                if (MoreSimilar(dist, best_dist)) {
                    best_dist = dist;
                    best_cluster = c_idx;
                }
            }

            c = (valid_cluster_indices == nullptr ? best_cluster : valid_cluster_indices[best_cluster]);
            ClusterData& cluster_data = *(cluster_manager.At(c).first);
            if (assignments[i] != c) {
                converged = false;
                assignments[i] = c;
            }

            ++(cluster_counts[best_cluster]);
            for (uint16_t d = 0; d < DIMENSION; d++) {
                cluster_sums[best_cluster * DIMENSION + d] += static_cast<divftree::MVTYPE>(data[i].data[d]);
            }
        }

        if (converged) {
            // if (max_iterations == 0) {
            //     std::cout << "Iteration " << iter+1 << " completed. Converged: Yes\n";
            // } else {
            //     std::cout << "Iteration " << iter+1 << "/" << max_iterations << " completed. Converged: Yes\n";
            // }
            break;
        }

        /* centroid computation */
        for (uint32_t i = 0; i < num_clusters; i++) {
            uint32_t c = (valid_cluster_indices == nullptr ? i : valid_cluster_indices[i]);
            std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c);
            ClusterData& cluster_data = *(cluster_info.first);
            cluster_data.num_points = cluster_counts[i];
            cluster_counts[i] = 0;
            for (uint16_t d = 0; d < DIMENSION; d++) {
                cluster_data.data[d] = cluster_sums[i * DIMENSION + d] / static_cast<divftree::MVTYPE>(cluster_data.num_points);
                cluster_sums[i * DIMENSION + d] = 0;
            }
        }

        FatalAssert(!converged, LOG_TAG_BASIC, "Should not be converged at this point!");

        // if (max_iterations == 0) {
        //     std::cout << "Iteration " << iter+1 << " completed. Converged: No\n";
        // } else {
        //     std::cout << "Iteration " << iter+1 << "/" << max_iterations << " completed. Converged: No\n";
        // }
    }

    delete[] cluster_sums;
    delete[] cluster_counts;
}

void kmeans_simple(VectorData*& data, uint32_t num_points, uint32_t num_clusters, uint32_t max_iterations,
                   size_t num_threads, ClusterManager& cluster_manager) {
    cluster_manager.AddClusters(num_clusters, false);

    uint32_t* assignments = new uint32_t[num_points];
    memset(assignments, 0, sizeof(uint32_t) * num_points);

    kmeans(data, nullptr, assignments, num_points, num_clusters, max_iterations, num_threads,
           cluster_manager, nullptr);

    uint32_t num_seen = cluster_manager.SetOffsets();
    FatalAssert(num_seen == num_points, LOG_TAG_BASIC,
                "Total number of assigned points should be equal to num_points!");

    std::cout << "Allocating memory for reordered vector data...\n";
    VectorData* reordered_data =
        (VectorData*)mmap64(nullptr, (size_t)num_points * sizeof(VectorData),
                            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (reordered_data == MAP_FAILED) {
        std::cerr << "Error allocating memory for reordered vector data. errno: " << errno << ", msg: " << strerror(errno) << "\n";
        FatalAssert(false, LOG_TAG_BASIC, "Error allocating memory for reordered vector data.");
    }

    std::cout << "Reordering vector data based on cluster assignments using " << num_threads << " threads...\n";
    #pragma omp parallel num_threads(num_threads) shared(cluster_manager, assignments, num_points, reordered_data, data)
    {
        divftree::Thread* st = divftree::threadSelf;
        bool created = false;
        if (st == nullptr) {
            created = true;
            st = new divftree::Thread(100);
            st->InitDIVFThread(true);
        }

        #pragma omp for schedule(static)
        for (uint32_t i = 0; i < num_points; i++) {
            uint32_t c = assignments[i];

            std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c);
            ClusterData& cluster_data = *(cluster_info.first);
            ClusterMetaData& cluster_meta_data = *(cluster_info.second);

            cluster_data.num_total_points.fetch_add(data[i].num_duplicates + 1);
            uint32_t idx = cluster_meta_data.next_index.fetch_add(1) + cluster_data.offset;
            FatalAssert(idx < num_points, LOG_TAG_BASIC, "Index out of bounds in reordered data!");
            memcpy(&reordered_data[idx], &data[i], sizeof(VectorData));
        }
        if (created) {
            st->DestroyDIVFThread();
        }
    }
    delete[] assignments;

    munmap(data, (size_t)num_points * sizeof(VectorData));
    data = reordered_data;
}

inline uint32_t GetNumClusters(uint32_t num_points, uint32_t cluster_cap, bool strict) {
    FatalAssert(cluster_cap > 0, LOG_TAG_BASIC, "Cluster capacity must be greater than 0!");
    FatalAssert(num_points > cluster_cap, LOG_TAG_BASIC, "Number of points must be greater than 0!");

    if (strict) {
        return std::max(2u, num_points / cluster_cap);
    }

    uint32_t num_clusters = num_points / (cluster_cap * 2);
    if (num_clusters < 2) {
        num_clusters = std::max(2u, num_points / cluster_cap);
    }
    return num_clusters;
}

struct SplitTask {
    uint32_t target_cluster;
    uint32_t new_cluster_offsets;
    uint32_t num_clusters;
};

template<typename DataType>
void kmeans_capped(DataType*& data, uint32_t num_points, uint32_t cluster_cap, uint32_t max_iterations,
                   size_t num_threads, ClusterManager& cluster_manager, bool strict) {
    size_t num_clusters = GetNumClusters(num_points, cluster_cap, strict);
    cluster_manager.AddClusters(num_clusters, false);

    uint32_t* assignments = new uint32_t[num_points];
    memset(assignments, 0, sizeof(uint32_t) * num_points);

    std::cout << "Running initial k-means with " << num_clusters << " clusters...\n";
    kmeans(data, nullptr, assignments, num_points, num_clusters, max_iterations, num_threads,
           cluster_manager, nullptr);

    divftree::BlockingQueue<SplitTask> split_tasks(num_clusters);

    uint32_t num_seen = 0;
    uint32_t num_new_clusters_needed = 0;
    for (uint32_t i = 0; i < num_clusters; i++) {
        std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(i);
        ClusterData& cluster_data = *(cluster_info.first);
        ClusterMetaData& cluster_meta_data = *(cluster_info.second);
        FatalAssert(cluster_data.num_points > 0, LOG_TAG_BASIC, "Cluster has no points assigned!");
        if (cluster_data.num_points > cluster_cap) {
            uint32_t num_clusters_needed = GetNumClusters(cluster_data.num_points, cluster_cap, strict);
            SplitTask st;
            st.target_cluster = i;
            st.new_cluster_offsets = num_new_clusters_needed + num_clusters;
            st.num_clusters = num_clusters_needed;
            split_tasks.Push(std::move(st));
            num_new_clusters_needed += num_clusters_needed - 1;
            cluster_meta_data.assigned_vector_indices.resize(cluster_data.num_points);
            cluster_meta_data.next_index.store(0, std::memory_order_release);
        }
        num_seen += cluster_data.num_points;
    }
    FatalAssert(num_seen == num_points, LOG_TAG_BASIC,
                "Total number of assigned points should be equal to num_points!");

    if (num_new_clusters_needed > 0) {
        std::cout << "Adding " << num_new_clusters_needed << " new clusters for splitting(" << split_tasks.Size() <<
                     " tasks)...\n";
        cluster_manager.AddClusters(num_new_clusters_needed, false);
    }
    std::atomic<size_t> num_tasks = split_tasks.Size();
    DataType* reordered_data = nullptr;

    std::cout << "Splitting clusters that exceed the capacity using " << num_threads << " threads...\n";
    #pragma omp parallel num_threads(num_threads) shared(cluster_manager, assignments, num_points,\
                                                            data, split_tasks, num_tasks)
    {
        divftree::Thread* st = divftree::threadSelf;
        bool created = false;
        if (st == nullptr) {
            created = true;
            st = new divftree::Thread(100);
            st->InitDIVFThread(true);
        }

        if (num_new_clusters_needed > 0) {
            #pragma omp for schedule(guided)
            for (uint32_t i = 0; i < num_points; i++) {
                uint32_t c = assignments[i];
                std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c);
                ClusterData& cluster_data = *(cluster_info.first);
                ClusterMetaData& cluster_meta_data = *(cluster_info.second);
                if (cluster_data.num_points <= cluster_cap) {
                    continue;
                }

                uint32_t idx = cluster_meta_data.next_index.fetch_add(1);
                FatalAssert(idx < cluster_data.num_points, LOG_TAG_BASIC, "Index out of bounds in assigned_vector_indices!");
                cluster_meta_data.assigned_vector_indices[idx] = i;
            }

            SplitTask task;
            bool has_task = false;
            while (num_tasks.load(std::memory_order_acquire) > 0) {
                if (!has_task) {
                    if (!split_tasks.PopHead(task)) {
                        continue;
                    }
                }
                has_task = false;

                uint32_t* new_valid_cluster_indices = new uint32_t[task.num_clusters];
                for (uint32_t i = 0; i < task.num_clusters; i++) {
                    uint32_t c = (i == 0 ? task.target_cluster : task.new_cluster_offsets + i - 1);
                    new_valid_cluster_indices[i] = c;
                }

                std::pair<ClusterData*, ClusterMetaData*> target_cluster_info = cluster_manager.At(task.target_cluster);
                ClusterData& target_cluster_data = *(target_cluster_info.first);
                ClusterMetaData& target_cluster_meta_data = *(target_cluster_info.second);
                FatalAssert(target_cluster_meta_data.assigned_vector_indices.size() == target_cluster_data.num_points, LOG_TAG_BASIC,
                            "Size of assigned_vector_indices should be equal to the number of points in the cluster!");

                DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Splitting cluster %u with %u points into %u clusters...\n",
                        task.target_cluster, target_cluster_data.num_points, task.num_clusters);
                uint32_t old_size = target_cluster_data.num_points;
                kmeans(data, target_cluster_meta_data.assigned_vector_indices.data(), assignments, target_cluster_data.num_points,
                       task.num_clusters, max_iterations, cluster_manager, new_valid_cluster_indices);

                uint32_t num_points_seen = 0;
                uint32_t num_new_clusters_needed_for_task = 0;
                std::vector<SplitTask> new_split_tasks;
                new_split_tasks.clear();
                new_split_tasks.reserve(task.num_clusters);
                for (uint32_t i = 0; i < task.num_clusters; i++) {
                    uint32_t c = new_valid_cluster_indices[i];
                    std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c);
                    ClusterData& cluster_data = *(cluster_info.first);
                    ClusterMetaData& cluster_meta_data = *(cluster_info.second);
                    FatalAssert(cluster_data.num_points > 0, LOG_TAG_BASIC, "Cluster has no points assigned!");
                    if (cluster_data.num_points > cluster_cap) {
                        uint32_t num_clusters_needed = GetNumClusters(cluster_data.num_points, cluster_cap, strict);
                        SplitTask st;
                        st.target_cluster = c;
                        st.new_cluster_offsets = num_new_clusters_needed_for_task;
                        st.num_clusters = num_clusters_needed;
                        new_split_tasks.push_back(st);
                        num_new_clusters_needed_for_task += num_clusters_needed - 1;
                        if (c != task.target_cluster) {
                            cluster_meta_data.assigned_vector_indices.resize(cluster_data.num_points);
                            cluster_meta_data.next_index.store(0, std::memory_order_release);
                        }
                    }
                    num_points_seen += cluster_data.num_points;
                }
                FatalAssert(num_points_seen == old_size, LOG_TAG_BASIC,
                            "Total number of assigned points should be equal to num_points!");

                if (!new_split_tasks.empty()) {
                    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Adding %u new clusters for further splitting(%u tasks)...\n",
                            num_new_clusters_needed_for_task, new_split_tasks.size());
                    uint32_t off = cluster_manager.AddClusters(num_new_clusters_needed_for_task, false);
                    for (auto& st : new_split_tasks) {
                        st.new_cluster_offsets += off;
                    }

                    FatalAssert(target_cluster_meta_data.assigned_vector_indices.size() == old_size, LOG_TAG_BASIC,
                                "Size of assigned_vector_indices should be equal to the number of points in the cluster!");
                    for (uint32_t p = 0; p < target_cluster_meta_data.assigned_vector_indices.size(); p++) {
                        uint32_t idx = target_cluster_meta_data.assigned_vector_indices[p];
                        uint32_t c = assignments[idx];
                        std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c);
                        ClusterData& cluster_data = *(cluster_info.first);
                        ClusterMetaData& cluster_meta_data = *(cluster_info.second);

                        if ((target_cluster_data.num_points > cluster_cap) && (c != task.target_cluster)) {
                            target_cluster_meta_data.assigned_vector_indices[p] = target_cluster_meta_data.assigned_vector_indices.back();
                            target_cluster_meta_data.assigned_vector_indices.pop_back();
                            p--;
                        }

                        if (cluster_data.num_points <= cluster_cap) {
                            continue;
                        }

                        if (c != task.target_cluster) {
                            uint32_t vidx = cluster_meta_data.next_index.fetch_add(1);
                            FatalAssert(vidx < cluster_data.num_points, LOG_TAG_BASIC, "Index out of bounds in assigned_vector_indices!");
                            cluster_meta_data.assigned_vector_indices[vidx] = idx;
                        }
                    }

                    task = new_split_tasks[0];
                    has_task = true;
                    if (new_split_tasks.size() > 1) {
                        split_tasks.BatchPush(&new_split_tasks[1], new_split_tasks.size() - 1);
                        num_tasks.fetch_add(new_split_tasks.size() - 1);
                    }
                } else {
                    num_tasks.fetch_sub(1);
                }

                delete[] new_valid_cluster_indices;
            }
        }

        #pragma omp single
        {
            uint32_t num_seen = cluster_manager.SetOffsets();
            FatalAssert(num_seen == num_points, LOG_TAG_BASIC,
                        "Total number of assigned points should be equal to num_points!");

            std::cout << "Allocating memory for reordered vector data...\n";
            reordered_data =
                (DataType*)mmap64(nullptr, (size_t)num_points * sizeof(DataType),
                                    PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (reordered_data == MAP_FAILED) {
                std::cerr << "Error allocating memory for reordered vector data. errno: " << errno << ", msg: " << strerror(errno) << "\n";
                FatalAssert(false, LOG_TAG_BASIC, "Error allocating memory for reordered vector data.");
            }

            std::cout << "Reordering vector data based on cluster assignments...\n";
        }

        #pragma omp for schedule(static)
        for (uint32_t i = 0; i < num_points; i++) {
            uint32_t c = assignments[i];
            std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c);
            ClusterData& cluster_data = *(cluster_info.first);
            ClusterMetaData& cluster_meta_data = *(cluster_info.second);
            if constexpr (std::is_same<DataType, VectorData>::value) {
                cluster_data.num_total_points.fetch_add(data[i].num_duplicates + 1);
            }
            uint32_t idx = cluster_meta_data.next_index.fetch_add(1) + cluster_data.offset;
            FatalAssert(idx < num_points, LOG_TAG_BASIC, "Index out of bounds in reordered data!");
            memcpy(&reordered_data[idx], &data[i], sizeof(DataType));
        }

        if (created) {
            st->DestroyDIVFThread();
        }
    }

    delete[] assignments;

    munmap(data, (size_t)num_points * sizeof(DataType));
    data = reordered_data;

    std::cout << "Clustering completed. Final number of clusters: " << cluster_manager.clusters.size() << "\n";
    std::cout << " Memory utilization: " <<
                 ((double)num_points * (double)100) / ((double)cluster_manager.clusters.size() * (double)cluster_cap) <<
                 "%\n";
}

enum class AlgorithmType : uint8_t {
    KMEANS,
    KMEANS_CAPPED,
    KMEANS_HIERARCHICAL
};


/*
input file format:
    uint32_t num_total_points
    uint32_t num_unique_points
    uint16_t dimension
    For each unique vector:
        IVFVectorID id
        uint32_t num_duplicates
        VTYPE[dimension] vector_data
*/
struct Args {
    std::string input_file;
    std::string output_file;
    size_t num_threads;
    uint32_t num_points_to_use;
    uint32_t max_iterations;
    AlgorithmType algorithm;

    FILE* input_fp;
    FILE* output_fp;

    uint16_t dimension;
    uint32_t num_total_points;

    uint32_t num_total_points_used = 0;

    union {
        struct {
            uint32_t num_clusters;
        } kmeans_args;

        struct {
            uint32_t cluster_cap;
            bool strict;
        } kmeans_capped_args;
    };

    void print() {
        std::cout << "Input file: " << input_file << "\n";
        std::cout << "Output file: " << output_file << "\n";
        std::cout << "Num threads: " << num_threads << "\n";
        std::cout << "Num points to use: " << num_points_to_use << "/" << num_total_points << "\n";
        std::cout << "Max iterations: " << max_iterations << "\n";
        std::cout << "Algorithm: ";
        switch (algorithm) {
            case AlgorithmType::KMEANS:
                std::cout << "KMEANS\n";
                std::cout << "Num clusters: " << kmeans_args.num_clusters << "\n";
                break;
            case AlgorithmType::KMEANS_CAPPED:
                std::cout << "KMEANS_CAPPED\n";
                std::cout << "Cluster capacity: " << kmeans_capped_args.cluster_cap << "\n";
                std::cout << "Strict mode: " << (kmeans_capped_args.strict ? "True" : "False") << "\n";
                break;
            case AlgorithmType::KMEANS_HIERARCHICAL:
                std::cout << "KMEANS_HIERARCHICAL\n";
                break;
        }
    }
};

inline void PrintUsage(int argc, char* argv[]) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <num_threads> <num_points> "
                 "<max_iter> <algorithm> (<num_clusters> | <cluster_cap> <compute_num_clusters_strict>)\n";
    std::cerr << "\t<input_file>: Path to the input binary file containing vector data.\n";
    std::cerr << "\t<output_file>: Path to the output binary file to write clustered vector data.\n";
    std::cerr << "\t<num_threads>: Number of threads to use for clustering. If set to 0 "
                 "or more than number of hardware threads, max number of hardware threads will be used instead\n";
    std::cerr << "\t<num_points>: Number of points to use from the input file."
                 " Must be a non-zero number, less than or equal to the total number of points in the file.\n";
    std::cerr << "\t<max_iter>: Maximum number of iterations for the clustering algorithm. Set to 0 for no limit.\n";
    std::cerr << "\t<algorithm>: Clustering algorithm to use. "
                 "Must be one of the following: 'kmeans', 'kmeans_capped', 'kmeans_hierarchical'.\n";
    std::cerr << "\t<num_clusters>: (Only for kmeans) Number of clusters to form. "
                 "Must be a positive integer greater than one and less than or equal to num_points.\n";
    std::cerr << "\t<cluster_cap>: (Only for kmeans_capped) Maximum number of points allowed in each cluster. "
                 "Must be a positive integer greater than one and less than num_points.\n";
    std::cerr << "\t<compute_num_clusters_strict>: (Only for kmeans_capped) "
                 "Must be either 't' for true and 'f' for false. "
                 "If set to true, the number of clusters will be computed as max(2, floor(num_points / cluster_cap)). "
                 "If set to false, the number of clusters will be computed as floor(num_points / (2 * cluster_cap)) "
                 "with a minimum of max(2, floor(num_points / cluster_cap)). "
                 "This is to minimize the number of clusters with size=cap/2, "
                 "which can lead to better memory utilization.\n";

    std::cerr << "Recived " << argc - 1 << " arguments:\n";
    for (int i = 1; i < argc; i++) {
        std::cerr << "\tArgument " << i << ": " << argv[i] << "\n";
    }
}

inline void ParseArgs(int argc, char* argv[], Args& args) {
    if (argc != 8 && argc != 9) {
        PrintUsage(argc, argv);
        FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments!");
        exit(EXIT_FAILURE);
    }

    if (strcmp(argv[6], "kmeans") == 0) {
        if (argc != 8) {
            PrintUsage(argc, argv);
            FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments for kmeans!");
            exit(EXIT_FAILURE);
        }
        args.algorithm = AlgorithmType::KMEANS;
    } else if (strcmp(argv[6], "kmeans_capped") == 0) {
        if (argc != 9) {
            PrintUsage(argc, argv);
            FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments for kmeans_capped!");
            exit(EXIT_FAILURE);
        }
        args.algorithm = AlgorithmType::KMEANS_CAPPED;
    } else if (strcmp(argv[6], "kmeans_hierarchical") == 0) {
        FatalAssert(false, LOG_TAG_NOT_IMPLEMENTED, "Not yet implemented!");
        exit(EXIT_FAILURE);
        args.algorithm = AlgorithmType::KMEANS_HIERARCHICAL;
    } else {
        PrintUsage(argc, argv);
        FatalAssert(false, LOG_TAG_BASIC, "Invalid algorithm type!");
        exit(EXIT_FAILURE);
    }

    args.input_file = argv[1];

    if (!std::filesystem::exists(args.input_file)) {
        std::cerr << "Error: Input file '" << args.input_file << "' does not exist.\n";
        FatalAssert(false, LOG_TAG_BASIC, "Input file does not exist!");
        exit(EXIT_FAILURE);
    }

    if (!std::filesystem::is_regular_file(args.input_file)) {
        std::cerr << "Error: Input file '" << args.input_file << "' is not a regular file.\n";
        FatalAssert(false, LOG_TAG_BASIC, "Input file is not a regular file!");
        exit(EXIT_FAILURE);
    }

    args.input_fp = fopen(args.input_file.c_str(), "rb");
    if (!args.input_fp) {
        std::cerr << "Error: Failed to open input file '" << args.input_file << "'. errno: " << errno << ", msg: " << strerror(errno) << "\n";
        FatalAssert(false, LOG_TAG_BASIC, "Failed to open input file!");
        exit(EXIT_FAILURE);
    }

    size_t ret = fread(&args.num_total_points, sizeof(uint32_t), 1, args.input_fp);
    if (ret != 1) {
        std::cerr << "Error reading num_total_points from file.\n";
        fclose(args.input_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error reading num_total_points from file!");
        exit(EXIT_FAILURE);
    }

    if (args.num_total_points == 0) {
        std::cerr << "Error: num_total_points must be greater than 0.\n";
        fclose(args.input_fp);
        FatalAssert(false, LOG_TAG_BASIC, "num_total_points must be greater than 0!");
        exit(EXIT_FAILURE);
    }

    uint32_t num_unique_points;
    ret = fread(&num_unique_points, sizeof(uint32_t), 1, args.input_fp);
    if (ret != 1) {
        std::cerr << "Error reading num_unique_points from file.\n";
        fclose(args.input_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error reading num_unique_points from file!");
        exit(EXIT_FAILURE);
    }

    if (num_unique_points == 0 || num_unique_points > args.num_total_points) {
        std::cerr << "Error: num_unique_points must be greater than 0 and less than or equal to num_total_points.\n";
        fclose(args.input_fp);
        FatalAssert(false, LOG_TAG_BASIC, "num_unique_points must be greater than 0 and less than or equal to num_total_points!");
        exit(EXIT_FAILURE);
    }

    ret = fread(&args.dimension, sizeof(uint16_t), 1, args.input_fp);
    if (ret != 1) {
        std::cerr << "Error reading dimension from file.\n";
        fclose(args.input_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error reading dimension from file!");
        exit(EXIT_FAILURE);
    }

    if (args.dimension != DIMENSION) {
        std::cerr << "Error: Dimension in file (" << args.dimension << ") does not match expected dimension (" << DIMENSION << ").\n";
        fclose(args.input_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Dimension in file does not match expected dimension!");
        exit(EXIT_FAILURE);
    }

    args.output_file = argv[2];
    if (strcmp(args.output_file.c_str(), args.input_file.c_str()) == 0) {
        std::cerr << "Error: Output file path cannot be the same as input file path.\n";
        fclose(args.input_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Output file path cannot be the same as input file path!");
        exit(EXIT_FAILURE);
    }

    if (args.output_file.empty()) {
        std::cerr << "Error: Output file path cannot be empty.\n";
        fclose(args.input_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Output file path cannot be empty!");
        exit(EXIT_FAILURE);
    }

    if (std::filesystem::exists(args.output_file)) {
        if (!std::filesystem::is_regular_file(args.output_file)) {
            std::cerr << "Error: Output file '" << args.output_file << "' exists but is not a regular file.\n";
            fclose(args.input_fp);
            FatalAssert(false, LOG_TAG_BASIC, "Output file is not a regular file!");
            exit(EXIT_FAILURE);
        }
        std::cerr << "Warning: Output file '" << args.output_file << "' already exists and will be overwritten.\n";
    } else {
        // Extract the directory part
        std::filesystem::path out_path_obj(args.output_file);
        std::filesystem::path out_dir = out_path_obj.parent_path();

        // Create directories if they don't exist
        if (!out_dir.empty() && !std::filesystem::exists(out_dir)) {
            std::cout << "Output directory '" << out_dir.string() << "' does not exist. Creating directories...\n";
            std::filesystem::create_directories(out_dir);
        }
    }

    args.output_fp = fopen(args.output_file.c_str(), "wb");
    if (!args.output_fp) {
        std::cerr << "Error: Failed to open output file '" << args.output_file << "'. errno: " << errno << ", msg: " << strerror(errno) << "\n";
        fclose(args.input_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Failed to open output file!");
        exit(EXIT_FAILURE);
    }

    args.num_threads = std::stoul(argv[3]);
    size_t max_num_threads = omp_get_max_threads();
    if (args.num_threads == 0 || args.num_threads > max_num_threads) {
        std::cerr << "Warning, num_threads(" << args.num_threads << ") must be between 1 and " << max_num_threads << ". Using " << max_num_threads << " threads.\n";
        args.num_threads = max_num_threads;
    }

    args.num_points_to_use = std::stoul(argv[4]);
    if (args.num_points_to_use == 0 || args.num_points_to_use > num_unique_points) {
        std::cerr << "Error: num_points_to_use must be greater than 0 and less than or equal to num_unique_points (" << num_unique_points << ").\n";
        fclose(args.input_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "num_points_to_use must be greater than 0 and less than or equal to num_unique_points!");
        exit(EXIT_FAILURE);
    }

    args.max_iterations = std::stoul(argv[5]);

    if (args.algorithm == AlgorithmType::KMEANS) {
        args.kmeans_args.num_clusters = std::stoul(argv[7]);
        if (args.kmeans_args.num_clusters < 2 || args.kmeans_args.num_clusters > args.num_points_to_use) {
            std::cerr << "Error: num_clusters must be greater than 1 and less than or equal to num_points_to_use (" << args.num_points_to_use << ").\n";
            fclose(args.input_fp);
            fclose(args.output_fp);
            FatalAssert(false, LOG_TAG_BASIC, "num_clusters must be greater than 1 and less than or equal to num_points_to_use!");
            exit(EXIT_FAILURE);
        }
    } else if (args.algorithm == AlgorithmType::KMEANS_CAPPED) {
        args.kmeans_capped_args.cluster_cap = std::stoul(argv[7]);
        if (args.kmeans_capped_args.cluster_cap < 2 || args.kmeans_capped_args.cluster_cap >= args.num_points_to_use) {
            std::cerr << "Error: cluster_cap must be greater than 1 and less than num_points_to_use (" << args.num_points_to_use << ").\n";
            fclose(args.input_fp);
            fclose(args.output_fp);
            FatalAssert(false, LOG_TAG_BASIC, "cluster_cap must be greater than 1 and less than num_points_to_use!");
            exit(EXIT_FAILURE);
        }

        char strict_arg = argv[8][0];
        if (strict_arg == 't' || strict_arg == 'T') {
            args.kmeans_capped_args.strict = true;
        } else if (strict_arg == 'f' || strict_arg == 'F') {
            args.kmeans_capped_args.strict = false;
        } else {
            std::cerr << "Error: compute_num_clusters_strict argument must be either 't' for true or 'f' for false.\n";
            fclose(args.input_fp);
            fclose(args.output_fp);
            FatalAssert(false, LOG_TAG_BASIC, "compute_num_clusters_strict argument must be either 't' for true or 'f' for false!");
            exit(EXIT_FAILURE);
        }
    } else {
        FatalAssert(false, LOG_TAG_NOT_IMPLEMENTED, "Hierarchical k-means is not yet implemented!");
        exit(EXIT_FAILURE);
    }
}


/*
output file format:
    uint32_t num_total_points
    uint32_t num_unique_points
    uint16_t dimension
    uint8_t num_levels (for hierarchical k-means, set to 1 for flat k-means and k-means capped)
    for each level starting from root to leaf:
        uint32_t num_clusters
        for each cluster in the level (clusters with the same parent cluster are stored contiguously in the file -> should be already sorted in clustermanager):
            VectorID cluster_id
            uint32_t num_points_in_cluster
            uint32_t num_unique_points_in_cluster (only for leaf clusters, does not exists for non-leaf clusters)
            uint32_t cluster_vector_offset (offset in the level where the cluster vector data starts -> it is the number of data points before and not the size it self)
            CTYPE[dimension] centroid_vector

    For each unique vector: -> vectors in the same cluster are stored contiguously in the file, so we can read them sequentially for each cluster. should be already sorted in the data
        IVFVectorID id
        uint32_t num_duplicates
        VTYPE[dimension] vector_data
*/
void write_output(VectorData* data, ClusterManager* cluster_managers, uint8_t num_levels,
                  uint32_t num_points, uint32_t num_unique_points, FILE* output_fp) {
    std::cout << "Writing output to file...\n";
    size_t ret = fwrite(&num_points, sizeof(uint32_t), 1, output_fp);
    if (ret != 1) {
        std::cerr << "Error writing num_total_points to output file.\n";
        fclose(output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error writing num_total_points to output file!");
        exit(EXIT_FAILURE);
    }

    ret = fwrite(&num_unique_points, sizeof(uint32_t), 1, output_fp);
    if (ret != 1) {
        std::cerr << "Error writing num_unique_points to output file.\n";
        fclose(output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error writing num_unique_points to output file!");
        exit(EXIT_FAILURE);
    }

    uint16_t dimension_to_write = DIMENSION;
    ret = fwrite(&dimension_to_write, sizeof(uint16_t), 1, output_fp);
    if (ret != 1) {
        std::cerr << "Error writing dimension to output file.\n";
        fclose(output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error writing dimension to output file!");
        exit(EXIT_FAILURE);
    }

    ret = fwrite(&num_levels, sizeof(uint8_t), 1, output_fp);
    if (ret != 1) {
        std::cerr << "Error writing num_levels to output file.\n";
        fclose(output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error writing num_levels to output file!");
        exit(EXIT_FAILURE);
    }

    /* level 0 are the vectors so they are skipped here! num levels refers to the number of levels above the raw vectors */
    for (uint8_t level = num_levels; level > 0; level--) {
        ClusterManager& cluster_manager = cluster_managers[level - 1];
        uint32_t num_clusters = cluster_manager.clusters.size();
        ret = fwrite(&num_clusters, sizeof(uint32_t), 1, output_fp);
        if (ret != 1) {
            std::cerr << "Error writing num_clusters for level " << (uint32_t)level << " to output file.\n";
            fclose(output_fp);
            FatalAssert(false, LOG_TAG_BASIC, "Error writing num_clusters to output file!");
            exit(EXIT_FAILURE);
        }

        for (const auto& cluster : cluster_manager.clusters) {
            ClusterData& cluster_data = *(cluster.first);
            ClusterMetaData& cluster_meta_data = *(cluster.second);

            ret = fwrite(&cluster_data.id, sizeof(divftree::VectorID), 1, output_fp);
            if (ret != 1) {
                std::cerr << "Error writing cluster_id for cluster " << cluster_data.id._id << " in level " << (uint32_t)level << " to output file.\n";
                fclose(output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing cluster_id to output file!");
                exit(EXIT_FAILURE);
            }

            ret = fwrite(&cluster_data.num_points, sizeof(uint32_t), 1, output_fp);
            if (ret != 1) {
                std::cerr << "Error writing num_points for cluster " << cluster_data.id._id << " in level " << (uint32_t)level << " to output file.\n";
                fclose(output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing num_points to output file!");
                exit(EXIT_FAILURE);
            }

            if (level == 1) {
                uint32_t num_unique_points_in_cluster = cluster_data.num_total_points.load(std::memory_order_acquire);
                FatalAssert(num_unique_points_in_cluster >= cluster_data.num_points, LOG_TAG_BASIC,
                            "num_unique_points_in_cluster should be greater than or equal to num_points in the cluster!");
                ret = fwrite(&num_unique_points_in_cluster, sizeof(uint32_t), 1, output_fp);
                if (ret != 1) {
                    std::cerr << "Error writing num_unique_points_in_cluster for cluster " << cluster_data.id._id << " in level " << (uint32_t)level << " to output file.\n";
                    fclose(output_fp);
                    FatalAssert(false, LOG_TAG_BASIC, "Error writing num_unique_points_in_cluster to output file!");
                    exit(EXIT_FAILURE);
                }
            }

            ret = fwrite(&cluster_data.offset, sizeof(uint32_t), 1, output_fp);
            if (ret != 1) {
                std::cerr << "Error writing cluster_vector_offset for cluster " << cluster_data.id._id << " in level " << (uint32_t)level << " to output file.\n";
                fclose(output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing cluster_vector_offset to output file!");
                exit(EXIT_FAILURE);
            }

            ret = fwrite(cluster_data.data, sizeof(cluster_data.data), 1, output_fp);
            if (ret != 1) {
                std::cerr << "Error writing centroid_vector for cluster " << cluster_data.id._id << " in level " << (uint32_t)level << " to output file.\n";
                fclose(output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing centroid_vector to output file!");
                exit(EXIT_FAILURE);
            }
        }
    }

    ret = fwrite(data, 1, sizeof(VectorData) * (size_t)num_unique_points, output_fp);
    if (ret != sizeof(VectorData) * (size_t)num_unique_points) {
        std::cerr << "Error writing vector data to output file.\n";
        fclose(output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error writing vector data to output file!");
        exit(EXIT_FAILURE);
    }
    fclose(output_fp);
}

void build_ivf_index(VectorData* data, Args& args) {
    std::cout << "Building IVF index using k-means clustering...\n";
    ClusterManager cluster_manager(1);

    kmeans_simple(data, args.num_points_to_use, args.kmeans_args.num_clusters, args.max_iterations,
                   args.num_threads, cluster_manager);

    write_output(data, &cluster_manager, 1, args.num_total_points_used, args.num_points_to_use, args.output_fp);
}

void build_capped_ivf_index(VectorData* data, Args& args) {
    std::cout << "Building IVF index with using capped k-means clustering...\n";
    ClusterManager cluster_manager(1);
    kmeans_capped(data, args.num_points_to_use, args.kmeans_capped_args.cluster_cap, args.max_iterations,
                  args.num_threads, cluster_manager, args.kmeans_capped_args.strict);
    write_output(data, &cluster_manager, 1, args.num_total_points_used, args.num_points_to_use, args.output_fp);
}

void build_divftree_index(VectorData* data, Args& args) {
    FatalAssert(false, LOG_TAG_NOT_IMPLEMENTED, "DIVF-Tree index building is not yet implemented!");
}

int main(int argc, char* argv[]) {
    divftree::Thread st(100);
    st.InitDIVFThread(true);

    Args args;
    ParseArgs(argc, argv, args);
    args.print();

    std::cout << "Allocating enough memory for vector data...\n";
    VectorData* data = (VectorData*)mmap64(nullptr, (size_t)args.num_points_to_use * sizeof(VectorData),
                                           PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (data == MAP_FAILED) {
        std::cerr << "Error allocating memory for vector data. errno: " << errno << ", msg: " << strerror(errno) << "\n";
        fclose(args.input_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error allocating memory for vector data!");
        exit(EXIT_FAILURE);
    }

    std::cout << "Reading vector data from file...\n";
    size_t ret = fread(data, 1, (size_t)args.num_points_to_use * sizeof(VectorData), args.input_fp);
    fclose(args.input_fp);
    if (ret != (size_t)args.num_points_to_use * sizeof(VectorData)) {
        std::cerr << "Error reading data from file.\n";
        fclose(args.output_fp);
        munmap(data, (size_t)args.num_points_to_use * sizeof(VectorData));
        FatalAssert(false, LOG_TAG_BASIC, "Error reading data from file!");
        exit(EXIT_FAILURE);
    }

    std::cout << "Successfully read " << args.num_points_to_use << " vectors from file.\n";
    std::cout << "Some final checks on the data before clustering...\n";
    for (uint32_t i = 0; i < args.num_points_to_use; i++) {
        args.num_total_points_used += (data[i].num_duplicates + 1);
    }

    FatalAssert(args.num_total_points_used >= args.num_points_to_use, LOG_TAG_BASIC,
                "Total number of points represented by the unique vectors (num_total_points_used) should be greater than or equal to num_points_to_use!");

    switch (args.algorithm) {
        case AlgorithmType::KMEANS:
            build_ivf_index(data, args);
            break;
        case AlgorithmType::KMEANS_CAPPED:
            build_capped_ivf_index(data, args);
            break;
        case AlgorithmType::KMEANS_HIERARCHICAL:
            build_divftree_index(data, args);
            break;
    }

    std::cout << "Clustering and index building completed successfully. freeing memory...\n";
    munmap(data, (size_t)args.num_points_to_use * sizeof(VectorData));
    st.DestroyDIVFThread();
}