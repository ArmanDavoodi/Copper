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
using MVTYPE = float;
#define MVTYPE_FMT "%2.f"
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
inline divftree::DTYPE L2Squared(const T1* __restrict__ a, const T2* __restrict__ b) {
    divftree:: DTYPE dist = 0;
    #pragma GCC ivdep
    for (int i = 0; i < DIMENSION; i++) {
        divftree::DTYPE diff = static_cast<divftree::DTYPE>(a[i]) - static_cast<divftree::DTYPE>(b[i]);
        dist += diff * diff;
    }
    return dist;
}

template<typename T1, typename T2>
inline void VCpy(T1* __restrict__ dest, const T2* __restrict__ src) {
    if constexpr (std::is_same_v<T1, T2>) {
        memcpy(dest, src, sizeof(T1) * DIMENSION);
    } else {
        #pragma GCC ivdep
        for (int i = 0; i < DIMENSION; i++) {
            dest[i] = static_cast<T1>(src[i]);
        }
    }
}

template<typename T1, typename T2>
inline void VSet(T1* __restrict__ dest, T2 val) {
     if constexpr (std::is_same_v<T1, T2>) {
        memset(dest, val, sizeof(T1) * DIMENSION);
    } else {
        #pragma GCC ivdep
        for (int i = 0; i < DIMENSION; i++) {
            dest[i] = static_cast<T1>(val);
        }
    }
}

template<typename T1, typename T2>
inline void VAdd(T1* __restrict__ dest, const T2* __restrict__ v) {
    #pragma GCC ivdep
    for (int i = 0; i < DIMENSION; i++) {
        dest[i] += static_cast<T1>(v[i]);
    }
}

template<typename DestT, typename T1, typename T2>
inline void VAdd(DestT* __restrict__ dest, const T1* __restrict__ v1, const T2* __restrict__ v2) {
    #pragma GCC ivdep
    for (int i = 0; i < DIMENSION; i++) {
        dest[i] += static_cast<DestT>(v1[i]) + static_cast<DestT>(v2[i]);
    }
}

template<typename T1, typename T2>
inline void VDivInPlace(T1* __restrict__ vec, T2 value) {
    using CalcT = std::common_type_t<T1, T2>;
    CalcT inv = CalcT(1) / static_cast<CalcT>(value);

    #pragma GCC ivdep
    for (int i = 0; i < DIMENSION; i++) {
        vec[i] = static_cast<T1>(static_cast<CalcT>(vec[i]) * inv);
    }
}

template<typename T1, typename T2, typename ScalarT>
inline void VCpyNormalize(T1* __restrict__ v1, const T2* __restrict__ v2, ScalarT value) {
    using CalcT = std::common_type_t<T1, T2, ScalarT>;
    CalcT inv = CalcT(1) / static_cast<CalcT>(value);

    #pragma GCC ivdep
    for (int i = 0; i < DIMENSION; i++) {
        v1[i] = static_cast<T1>(static_cast<CalcT>(v2[i]) * inv);
    }
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

    uint32_t num_duplicates = 0; /* just so that it matches the structure of VectorData */

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
    uint32_t subtree_size = 0;

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
    ClusterManager(ClusterManager&& other) : level(other.level), clusters(std::move(other.clusters)) {}

    ~ClusterManager() {
        for (auto& cluster : clusters) {
            if (cluster.first != nullptr) {
                delete cluster.first;
            }
            if (cluster.second != nullptr) {
                delete cluster.second;
            }
        }
    }

    inline ClusterManager& operator=(ClusterManager&& other) {
        if (this != &other) {
            FatalAssert(level == other.level, LOG_TAG_BASIC, "Cluster managers should be of the same level!");
            FatalAssert(clusters.size() == other.clusters.size(), LOG_TAG_BASIC, "Cluster managers should have the same number of clusters!");
            clusters = std::move(other.clusters);
        }
        return *this;
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

    inline std::pair<ClusterData*, ClusterMetaData*> At(uint32_t idx, bool lock_cluster) {
        if (!lock_cluster) {
            return clusters[idx];
        }

        lock.Lock(divftree::LockMode::SX_SHARED);
        std::pair<ClusterData*, ClusterMetaData*> res = clusters[idx];
        lock.Unlock();
        return res;
    }

    inline ClusterData& operator[](uint32_t idx) {
        lock.Lock(divftree::LockMode::SX_SHARED);
        ClusterData* res = clusters[idx].first;
        FatalAssert(res != nullptr, LOG_TAG_BASIC, "cluster is empty!");
        lock.Unlock();
        return *res;
    }

    inline uint32_t SetOffsets() {
        lock.Lock(divftree::LockMode::SX_SHARED);
        uint32_t num_seen = 0;
        for (uint32_t i = 0; i < clusters.size(); i++) {
            std::pair<ClusterData*, ClusterMetaData*> cluster_info = clusters[i];
            if (cluster_info.first == nullptr || cluster_info.second == nullptr) {
                FatalAssert(cluster_info.first == nullptr && cluster_info.second == nullptr, LOG_TAG_BASIC, "both should be null!");
                continue;
            }
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

    inline uint32_t RemoveEmptyClusters() {
        lock.Lock(divftree::LockMode::SX_EXCLUSIVE);
        uint32_t removed = 0;
        for (uint32_t i = 0; i < clusters.size(); i++) {
            if (clusters[i].first == nullptr || clusters[i].second == nullptr) {
                FatalAssert(clusters[i].first == nullptr && clusters[i].second == nullptr, LOG_TAG_BASIC, "both should be null!");
                ++removed;
                continue;
            }
            FatalAssert(i >= removed, LOG_TAG_BASIC, "i cannot be greater than remove");
            if (removed > 0) {
                clusters[i - removed] = clusters[i];
            }
        }
        clusters.resize(clusters.size() - removed);
        uint32_t new_size = clusters.size();
        lock.Unlock();
        return new_size;
    }

    inline void SetEmpty(uint32_t idx) {
        FatalAssert(idx < clusters.size(), LOG_TAG_BASIC, "idx out of bounds!");
        lock.Lock(divftree::LockMode::SX_SHARED);
        FatalAssert(clusters[idx].first != nullptr, LOG_TAG_BASIC, "Already empty!");
        FatalAssert(clusters[idx].second != nullptr, LOG_TAG_BASIC, "Already empty!");
        FatalAssert(clusters[idx].first->num_points == 0, LOG_TAG_BASIC, "cluster should be empty!");
        delete clusters[idx].first;
        clusters[idx].first = nullptr;
        delete clusters[idx].second;
        clusters[idx].second = nullptr;
        lock.Unlock();
    }

    inline bool GetIfNotEmpty(uint32_t idx, bool lock_cluster, std::pair<ClusterData*, ClusterMetaData*>& out) {
        if (!lock_cluster) {
            if (clusters[idx].first == nullptr || clusters[idx].second == nullptr) {
                FatalAssert(clusters[idx].first == nullptr && clusters[idx].second == nullptr, LOG_TAG_BASIC, "both should be null!");
                return false;
            }
            out = clusters[idx];
            return true;
        }

        lock.Lock(divftree::LockMode::SX_SHARED);
        if (clusters[idx].first == nullptr || clusters[idx].second == nullptr) {
            FatalAssert(clusters[idx].first == nullptr && clusters[idx].second == nullptr, LOG_TAG_BASIC, "both should be null!");
            lock.Unlock();
            return false;
        }
        out = clusters[idx];
        lock.Unlock();
        return true;
    }
};

template<typename InternalStructure>
class DataSet;

template<>
class DataSet<ClusterManager> {
public:
    DataSet(ClusterManager& cm) : cluster_manager(cm), img(nullptr) {}

    ~DataSet() {
        FatalAssert(img == nullptr, LOG_TAG_BASIC, "Image should be deleted before the dataset is destroyed!");
    }

    uint32_t Size() const {
        return (uint32_t)cluster_manager.clusters.size();
    }

    inline uint32_t DataWeight(uint32_t idx, bool uniform) {
        FatalAssert(cluster_manager.clusters.size() > idx, LOG_TAG_BASIC, "Index out of bounds!");
        if (uniform) {
            FatalAssert(cluster_manager.clusters[idx].second->subtree_size == cluster_manager.clusters[idx].first->num_points, LOG_TAG_BASIC,
                    "subtree size should be equal to num_points!");
            return 1;
        }
        FatalAssert(cluster_manager.clusters[idx].second->subtree_size >= cluster_manager.clusters[idx].first->num_points, LOG_TAG_BASIC,
                    "subtree size should be equal to or grater than num_points!");
        // FatalAssert(cluster_manager.clusters[idx].second->subtree_size > cluster_manager.clusters[idx].first->num_points || cluster_manager.clusters[idx].first->id._level == divftree::VectorID::LEAF_LEVEL, LOG_TAG_BASIC,
        //             "subtree size should be greater than num_points(unless this is a leaf or an internal node with a single leaf child that has a single element inside!)");
        return cluster_manager.clusters[idx].second->subtree_size;
    }

    inline ClusterData& operator[](uint32_t idx) {
        FatalAssert(cluster_manager.clusters.size() > idx, LOG_TAG_BASIC, "Index out of bounds!");
        return *cluster_manager.clusters[idx].first;
    }

    void CreateEmptyImage() {
        FatalAssert(img == nullptr, LOG_TAG_BASIC, "Image already exists!");
        img = new ClusterManager(cluster_manager.level);
        img->clusters.resize(cluster_manager.clusters.size(), {nullptr, nullptr});
    }

    /* it only copies the pointers */
    void CpyToImg(uint32_t src_idx, uint32_t dst_idx) {
        CHECK_NOT_NULLPTR(img, LOG_TAG_BASIC);
        FatalAssert(cluster_manager.level == img->level, LOG_TAG_BASIC, "Cluster managers should be of the same level!");
        FatalAssert(cluster_manager.clusters.size() > src_idx, LOG_TAG_BASIC, "Index out of bounds!");
        FatalAssert(img->clusters.size() > dst_idx, LOG_TAG_BASIC, "Index out of bounds!");
        FatalAssert(img->clusters.size() == cluster_manager.clusters.size(), LOG_TAG_BASIC, "Cluster managers should have the same number of clusters!");
        FatalAssert(img->clusters[dst_idx].first == nullptr, LOG_TAG_BASIC, "Destination cluster should be empty!");
        FatalAssert(img->clusters[dst_idx].second == nullptr, LOG_TAG_BASIC, "Destination cluster should be empty!");
        CHECK_NOT_NULLPTR(cluster_manager.clusters[src_idx].first, LOG_TAG_BASIC);
        CHECK_NOT_NULLPTR(cluster_manager.clusters[src_idx].second, LOG_TAG_BASIC);
        img->clusters[dst_idx] = cluster_manager.clusters[src_idx];
    }

    inline void ReplaceWithImg() {
        CHECK_NOT_NULLPTR(img, LOG_TAG_BASIC);
        FatalAssert(cluster_manager.level == img->level, LOG_TAG_BASIC, "Cluster managers should be of the same level!");
        FatalAssert(cluster_manager.clusters.size() == img->clusters.size(), LOG_TAG_BASIC, "Cluster managers should have the same number of clusters!");
        cluster_manager = std::move(*img);
        delete img;
        img = nullptr;
    }

protected:
    ClusterManager& cluster_manager;
    ClusterManager* img;
};

template<>
class DataSet<VectorData> {
public:
    DataSet(VectorData*& vd, uint32_t n_points) : vec_data(vd), img(nullptr), num_points(n_points) {
        CHECK_NOT_NULLPTR(vec_data, LOG_TAG_BASIC);
        FatalAssert(num_points > 0, LOG_TAG_BASIC, "Number of points should be greater than 0!");
    }

    ~DataSet() {
        FatalAssert(img == nullptr, LOG_TAG_BASIC, "Image should be deleted before the dataset is destroyed!");
    }

    uint32_t Size() const {
        return num_points;
    }

    inline uint32_t DataWeight(uint32_t idx, bool uniform) {
        UNUSED_VARIABLE(uniform);
        UNUSED_VARIABLE(idx);
        FatalAssert(num_points > idx, LOG_TAG_BASIC, "Index out of bounds!");
        return 1;
    }

    inline VectorData& operator[](uint32_t idx) {
        CHECK_NOT_NULLPTR(vec_data, LOG_TAG_BASIC);
        FatalAssert(num_points > idx, LOG_TAG_BASIC, "Index out of bounds!");
        return vec_data[idx];
    }

    void CreateEmptyImage() {
        img = reinterpret_cast<VectorData*>(mmap64(nullptr, (size_t)num_points * sizeof(VectorData),
                                            PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (img == MAP_FAILED) {
            std::cerr << "Error allocating memory for reordered vector data. errno: " << errno << ", msg: " << strerror(errno) << "\n";
            FatalAssert(false, LOG_TAG_BASIC, "Error allocating memory for reordered vector data.");
        }
    }

    /* it only copies the pointers */
    void CpyToImg(uint32_t src_idx, uint32_t dst_idx) {
        CHECK_NOT_NULLPTR(vec_data, LOG_TAG_BASIC);
        CHECK_NOT_NULLPTR(img, LOG_TAG_BASIC);
        FatalAssert(num_points > dst_idx, LOG_TAG_BASIC, "Index out of bounds!");
        FatalAssert(num_points > src_idx, LOG_TAG_BASIC, "Index out of bounds!");
        memcpy(img + dst_idx, vec_data + src_idx, sizeof(VectorData));
    }

    inline void ReplaceWithImg() {
        CHECK_NOT_NULLPTR(img, LOG_TAG_BASIC);
        CHECK_NOT_NULLPTR(vec_data, LOG_TAG_BASIC);
        munmap(vec_data, (size_t)num_points * sizeof(VectorData));
        vec_data = img;
        img = nullptr;
    }

    inline VectorData* Data() {
        CHECK_NOT_NULLPTR(vec_data, LOG_TAG_BASIC);
        return vec_data;
    }

protected:
    VectorData*& vec_data;
    VectorData* img;
    uint32_t num_points;
};

template<typename DataSetInternal>
void kmeans(DataSet<DataSetInternal>& data, uint32_t* const valid_indices, uint32_t* assignments,
            uint32_t num_points, uint32_t num_clusters,
            uint32_t max_iterations, size_t num_threads, ClusterManager& cluster_manager,
            uint32_t* const valid_cluster_indices, bool weighted_kmeans = false) {
    std::atomic<bool> converged = false;
    std::atomic<uint32_t> first_non_empty_cluster = 0;

    std::cout << "Initializing centroids using " << num_threads << " threads...\n" << std::flush;
    #pragma omp parallel num_threads(num_threads) \
        shared(data, valid_indices, assignments, num_points, num_clusters, max_iterations,\
               cluster_manager, valid_cluster_indices, converged, first_non_empty_cluster)
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
            std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c, false);
            ClusterData& cluster_data = *(cluster_info.first);
            // divftree::String centroid_str = divftree::String("Initializing centroid %u with vector index %u: data=[", c, idx);
            VCpy(cluster_data.data, data[idx].data);
            // for (uint16_t d = 0; d < DIMENSION; d++) {
            //     cluster_data.data[d] = static_cast<divftree::MVTYPE>(data[idx].data[d]);
            //     // centroid_str += divftree::String(VTYPE_FMT "%s", data[idx].data[d], (d == DIMENSION - 1) ? "]" : ", ");
            // }
            // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "%s", centroid_str.ToCStr());
        }

        divftree::MVTYPE* cluster_sums = new divftree::MVTYPE[num_clusters * DIMENSION];
        uint32_t* cluster_weights = new uint32_t[num_clusters];
        uint32_t* cluster_counts = new uint32_t[num_clusters];
        memset(cluster_sums, 0, sizeof(divftree::MVTYPE) * num_clusters * DIMENSION);
        memset(cluster_weights, 0, sizeof(uint32_t) * num_clusters);
        memset(cluster_counts, 0, sizeof(uint32_t) * num_clusters);

        for (uint32_t iter = 0; iter < max_iterations || max_iterations == 0; ++iter) {

            #pragma omp barrier

            #pragma omp single
            {
                if (iter > 0) {
                    converged.store(true, std::memory_order_release);
                    uint32_t c_idx = first_non_empty_cluster.load(std::memory_order_relaxed);
                    for (; c_idx < num_clusters; ++c_idx) {
                        uint32_t c = (valid_cluster_indices == nullptr ? c_idx : valid_cluster_indices[c_idx]);
                        std::pair<ClusterData*, ClusterMetaData*> cluster_info;
                        if (cluster_manager.GetIfNotEmpty(c, false, cluster_info)) {
                            break;
                        }
                    }
                    FatalAssert(c_idx != num_clusters, LOG_TAG_BASIC, "All clusters are empty!");
                    FatalAssert(c_idx == 0 || num_clusters > 2, LOG_TAG_BASIC, "2means should not return empty clsuters if initial centroids are chosen from base vectors");
                    first_non_empty_cluster.store(c_idx, std::memory_order_release);
                }
            }
            /* assignment */
            #pragma omp for schedule(static)
            for (uint32_t idx = 0; idx < num_points; idx++) {
                uint32_t fc = first_non_empty_cluster.load(std::memory_order_acquire);
                uint32_t i = (valid_indices == nullptr ? idx : valid_indices[idx]);
                uint32_t c = (valid_cluster_indices == nullptr ? fc : valid_cluster_indices[fc]);
                uint32_t best_cluster = fc;
                std::pair<ClusterData*, ClusterMetaData*> cluster_info;
                if (!cluster_manager.GetIfNotEmpty(c, false, cluster_info)) {
                    FatalAssert(false, LOG_TAG_BASIC, "The first non empty index cannot be empty!");
                    continue;
                }
                divftree::DTYPE best_dist = L2Squared(data[i].data, cluster_info.first->data);
                // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                //     "Distance of vector index %u to cluster %u is " DTYPE_FMT, i, c, best_dist);
                for (uint32_t c_idx = fc + 1; c_idx < num_clusters; c_idx++) {
                    c = (valid_cluster_indices == nullptr ? c_idx : valid_cluster_indices[c_idx]);
                    if (!cluster_manager.GetIfNotEmpty(c, false, cluster_info)) {
                        continue;
                    }
                    divftree::DTYPE dist = L2Squared(data[i].data, cluster_info.first->data);
                    if (MoreSimilar(dist, best_dist)) {
                        best_dist = dist;
                        best_cluster = c_idx;
                    }
                    // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                    //     "Distance of vector index %u to cluster %u is " DTYPE_FMT, i, c, dist);
                }

                c = (valid_cluster_indices == nullptr ? best_cluster : valid_cluster_indices[best_cluster]);
                if (assignments[i] != c) {
                    // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                    //         "Vector index %u was reassigned to cluster %u (distance: " DTYPE_FMT ") from cluster %u, ",
                    //         i, c, best_dist, assignments[i]);
                    converged.store(false, std::memory_order_release);
                    assignments[i] = c;
                } /* else {
                    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                            "Vector index %u remains in cluster %u (distance: " DTYPE_FMT ")",
                            i, c, best_dist);
                } */

                ++(cluster_counts[best_cluster]);
                cluster_weights[best_cluster] += data.DataWeight(i, !weighted_kmeans);
                if (weighted_kmeans) {
                    for (uint16_t d = 0; d < DIMENSION; d++) {
                        cluster_sums[best_cluster * DIMENSION + d] +=
                            static_cast<divftree::MVTYPE>(data[i].data[d]) *
                            static_cast<divftree::MVTYPE>(data.DataWeight(i, false));
                    }
                } else {
                    VAdd(cluster_sums + best_cluster * DIMENSION, data[i].data);
                }
            }

            // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
            //         "Assignment step of iteration %u completed. Checking for convergence %s...",
            //         iter+1, converged.load(std::memory_order_acquire) ? "Yes" : "No");
            if (converged.load(std::memory_order_acquire)) {
                #pragma omp single
                {
                    if (max_iterations == 0) {
                        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Iteration %u completed. Converged: Yes", iter+1);
                    } else {
                        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Iteration %u/%u completed. Converged: Yes", iter+1, max_iterations);
                    }
                }
                // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                //     "Assignment step of iteration %u completed. converged", iter+1);
                break;
            }

            // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
            //         "Assignment step of iteration %u completed. not converged", iter+1);
            #pragma omp barrier

            // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
            //         "start centroid computation step for iteration %u", iter+1);

            #pragma omp for schedule(static)
            for (uint32_t i = 0; i < num_clusters; i++) {
                uint32_t c = (valid_cluster_indices == nullptr ? i : valid_cluster_indices[i]);
                std::pair<ClusterData*, ClusterMetaData*> cluster_info;
                if (!cluster_manager.GetIfNotEmpty(c, false, cluster_info)) {
                    FatalAssert(num_clusters > 2, LOG_TAG_BASIC, "2means should not return empty clusters when initial centroids are chosen from the actual vectors");
                    continue;
                }
                ClusterData& cluster_data = *(cluster_info.first);
                ClusterMetaData& cluster_meta = *(cluster_info.second);
                cluster_data.num_points = 0;
                cluster_meta.subtree_size = 0;
                memset(cluster_data.data, 0, sizeof(divftree::MVTYPE) * DIMENSION);
            }

            uint32_t tn = (uint32_t)omp_get_thread_num();
            uint32_t step_size = (uint32_t)std::max(1lu, (size_t)num_clusters / num_threads);
            tn = tn * step_size;
            for (uint32_t i = 0; i < num_clusters; i++) {
                uint32_t c_idx = (tn + i) % num_clusters;
                uint32_t c = (valid_cluster_indices == nullptr ? c_idx : valid_cluster_indices[c_idx]);
                std::pair<ClusterData*, ClusterMetaData*> cluster_info;
                if (!cluster_manager.GetIfNotEmpty(c, false, cluster_info)) {
                    FatalAssert(num_clusters > 2, LOG_TAG_BASIC, "2means should not return empty clusters when initial centroids are chosen from the actual vectors");
                    continue;
                }
                ClusterData& cluster_data = *(cluster_info.first);
                ClusterMetaData& cluster_meta = *(cluster_info.second);
                cluster_meta.lock.Lock(divftree::LockMode::SX_EXCLUSIVE);
                cluster_meta.subtree_size += cluster_weights[c_idx];
                cluster_data.num_points += cluster_counts[c_idx];
                cluster_counts[c_idx] = 0;
                cluster_weights[c_idx] = 0;
                VAdd(cluster_data.data, cluster_sums + c_idx * DIMENSION);
                memset(cluster_sums + c_idx * DIMENSION, 0, sizeof(divftree::MVTYPE) * DIMENSION);
                cluster_meta.lock.Unlock();
            }

            #pragma omp barrier

            #pragma omp for schedule(static)
            for (uint32_t i = 0; i < num_clusters; i++) {
                uint32_t c = (valid_cluster_indices == nullptr ? i : valid_cluster_indices[i]);
                std::pair<ClusterData*, ClusterMetaData*> cluster_info;
                if (!cluster_manager.GetIfNotEmpty(c, false, cluster_info)) {
                    FatalAssert(num_clusters > 2, LOG_TAG_BASIC, "2means should not return empty clusters when initial centroids are chosen from the actual vectors");
                    continue;
                }
                if (cluster_info.first->num_points == 0) {
                    FatalAssert(num_clusters > 2, LOG_TAG_BASIC, "2means should not return empty clusters when initial centroids are chosen from the actual vectors");
                    FatalAssert(cluster_info.second->subtree_size == 0, LOG_TAG_BASIC, "subtree size should also be 0");
                    cluster_manager.SetEmpty(c);
                    continue;
                }
                ClusterData& cluster_data = *(cluster_info.first);
                ClusterMetaData& cluster_meta = *(cluster_info.second);

                FatalAssert(cluster_data.num_points > 0, LOG_TAG_BASIC, "Cluster has no points assigned!");
                // divftree::String cluster_str = divftree::String("Updating centroid for cluster %u: num_points=%u, data=[", c, cluster_data.num_points);
                FatalAssert(weighted_kmeans || (cluster_data.num_points == cluster_meta.subtree_size), LOG_TAG_BASIC,
                            "if we are taking uniform avg, num_points should be the same as subtree-size here");
                VDivInPlace(cluster_data.data, static_cast<divftree::MVTYPE>(cluster_meta.subtree_size));
                // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "%s", cluster_str.ToCStr());
            }

            #pragma omp single
            {
                FatalAssert(!converged.load(std::memory_order_acquire), LOG_TAG_BASIC, "Should not be converged at this point!");
                if (max_iterations == 0) {
                    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Iteration %u completed. Converged: No", iter+1);
                } else {
                    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Iteration %u/%u completed. Converged: No", iter+1, max_iterations);
                }
            }
        }

        delete[] cluster_sums;
        delete[] cluster_weights;
        delete[] cluster_counts;
        if (created) {
            st->DestroyDIVFThread();
        }
    }
}

template<typename DataSetInternal>
void kmeans(DataSet<DataSetInternal>& data, uint32_t* const valid_indices, uint32_t* assignments, uint32_t num_points, uint32_t num_clusters,
            uint32_t max_iterations, ClusterManager& cluster_manager, uint32_t* const valid_cluster_indices, bool weighted_kmeans = false) {
    bool converged = false;

    /* choose the first centroids */
    for (uint32_t i = 0; i < num_clusters; i++) {
        uint32_t idx = (valid_indices == nullptr ? i : valid_indices[i]);
        uint32_t c = (valid_cluster_indices == nullptr ? i : valid_cluster_indices[i]);
        std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c, true);
        ClusterData& cluster_data = *(cluster_info.first);
        cluster_data.num_points = 0;
        // divftree::String centroid_str = divftree::String("Initializing centroid %u with vector index %u: data=[", c, idx);
        VCpy(cluster_data.data, data[idx].data);
        // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "%s", centroid_str.ToCStr());
    }

    divftree::MVTYPE* cluster_sums = new divftree::MVTYPE[num_clusters * DIMENSION];
    uint32_t* cluster_weights = new uint32_t[num_clusters];
    uint32_t* cluster_counts = new uint32_t[num_clusters];
    memset(cluster_sums, 0, sizeof(divftree::MVTYPE) * num_clusters * DIMENSION);
    memset(cluster_weights, 0, sizeof(uint32_t) * num_clusters);
    memset(cluster_counts, 0, sizeof(uint32_t) * num_clusters);
    uint32_t fc = 0;

    for (uint32_t iter = 0; iter < max_iterations || max_iterations == 0; ++iter) {
        if (iter > 0) {
            converged = true;
            for (; fc < num_clusters; ++fc) {
                uint32_t c = (valid_cluster_indices == nullptr ? fc : valid_cluster_indices[fc]);
                std::pair<ClusterData*, ClusterMetaData*> cluster_info;
                if (cluster_manager.GetIfNotEmpty(c, true, cluster_info)) {
                    break;
                }
            }
            FatalAssert(fc < num_clusters, LOG_TAG_BASIC, "All clusters are empty!");
            FatalAssert(fc == 0 || num_clusters > 2, LOG_TAG_BASIC, "2means should not return empty clsuters if initial centroids are chosen from base vectors");
        }

        /* assignment */
        for (uint32_t idx = 0; idx < num_points; idx++) {
            FatalAssert(fc < num_clusters, LOG_TAG_BASIC, "All clusters are empty!");
            FatalAssert(fc == 0 || num_clusters > 2, LOG_TAG_BASIC, "cannot have empty clusters for 2means when initial centroids are chosen from the base vectors");
            uint32_t i = (valid_indices == nullptr ? idx : valid_indices[idx]);
            uint32_t c = (valid_cluster_indices == nullptr ? fc : valid_cluster_indices[fc]);
            uint32_t best_cluster = fc;
            divftree::DTYPE best_dist = L2Squared(data[i].data, cluster_manager.At(c, true).first->data);
            // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
            //         "Distance of vector index %u to cluster %u is " DTYPE_FMT, i, c, best_dist);
            for (uint32_t c_idx = fc + 1; c_idx < num_clusters; c_idx++) {
                c = (valid_cluster_indices == nullptr ? c_idx : valid_cluster_indices[c_idx]);
                std::pair<ClusterData*, ClusterMetaData*> cluster_info;
                if (!cluster_manager.GetIfNotEmpty(c, true, cluster_info)) {
                    FatalAssert(num_clusters > 2, LOG_TAG_BASIC, "cannot have empty clsuters for 2 means in our alg");
                    continue;
                }
                divftree::DTYPE dist = L2Squared(data[i].data, cluster_info.first->data);
                // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                //         "Distance of vector index %u to cluster %u is " DTYPE_FMT, i, c, dist);
                if (MoreSimilar(dist, best_dist)) {
                    best_dist = dist;
                    best_cluster = c_idx;
                }
            }

            c = (valid_cluster_indices == nullptr ? best_cluster : valid_cluster_indices[best_cluster]);
            ClusterData& cluster_data = *(cluster_manager.At(c, true).first);
            if (assignments[i] != c) {
                // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                //         "Vector index %u was reassigned to cluster %u (distance: " DTYPE_FMT ") from cluster %u, ",
                //         i, c, best_dist, assignments[i]);
                converged = false;
                assignments[i] = c;
            } /* else {
                DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                        "Vector index %u remains in cluster %u (distance: " DTYPE_FMT ")", i, c, best_dist);
            } */

            ++(cluster_counts[best_cluster]);
            cluster_weights[best_cluster] += data.DataWeight(i, !weighted_kmeans);
            if (weighted_kmeans) {
                for (uint16_t d = 0; d < DIMENSION; d++) {
                    cluster_sums[best_cluster * DIMENSION + d] +=
                        static_cast<divftree::MVTYPE>(data[i].data[d]) *
                        static_cast<divftree::MVTYPE>(data.DataWeight(i, !weighted_kmeans));
                }
            } else {
                VAdd(cluster_sums + best_cluster * DIMENSION, data[i].data);
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
            std::pair<ClusterData*, ClusterMetaData*> cluster_info;
            if (!cluster_manager.GetIfNotEmpty(c, true, cluster_info)) {
                FatalAssert(num_clusters > 2, LOG_TAG_BASIC, "cannot have empty clsuters for 2 means in our alg");
                continue;
            }
            cluster_info.first->num_points = 0;
            if (cluster_counts[i] == 0) {
                FatalAssert(num_clusters > 2, LOG_TAG_BASIC, "cannot have empty clsuters for 2 means in our alg");
                cluster_manager.SetEmpty(c);
                continue;
            }

            ClusterData& cluster_data = *(cluster_info.first);
            ClusterMetaData& cluster_meta = *(cluster_info.second);
            cluster_data.num_points = cluster_counts[i];
            cluster_meta.subtree_size = cluster_weights[i];
            cluster_counts[i] = 0;
            cluster_weights[i] = 0;
            FatalAssert(weighted_kmeans || (cluster_data.num_points == cluster_meta.subtree_size), LOG_TAG_BASIC,
                        "if we are taking uniform avg, num_points should be the same as subtree-size here");
            // divftree::String cluster_str = divftree::String("Updating centroid for cluster %u: num_points=%u, data=[", c, cluster_data.num_points);
            // for (uint16_t d = 0; d < DIMENSION; d++) {
            //     cluster_data.data[d] = cluster_sums[i * DIMENSION + d] / static_cast<divftree::MVTYPE>(cluster_meta.subtree_size);
            //     cluster_sums[i * DIMENSION + d] = 0;
            //     // cluster_str += divftree::String(MVTYPE_FMT "%s", cluster_data.data[d], (d == DIMENSION - 1) ? "]" : ", ");
            // }
            VCpyNormalize(cluster_data.data, cluster_sums + i * DIMENSION, cluster_meta.subtree_size);
            memset(cluster_sums + i * DIMENSION, 0, sizeof(divftree::MVTYPE) * DIMENSION);
            // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "%s", cluster_str.ToCStr());
        }

        FatalAssert(!converged, LOG_TAG_BASIC, "Should not be converged at this point!");

        // if (max_iterations == 0) {
        //     std::cout << "Iteration " << iter+1 << " completed. Converged: No\n";
        // } else {
        //     std::cout << "Iteration " << iter+1 << "/" << max_iterations << " completed. Converged: No\n";
        // }
    }

    delete[] cluster_sums;
    delete[] cluster_weights;
    delete[] cluster_counts;
}

void kmeans_simple(DataSet<VectorData>& data, uint32_t num_points, uint32_t num_clusters, uint32_t max_iterations,
                   size_t num_threads, ClusterManager& cluster_manager) {
    cluster_manager.AddClusters(num_clusters, false);

    uint32_t* assignments = new uint32_t[num_points];
    memset(assignments, 0, sizeof(uint32_t) * num_points);

    kmeans(data, nullptr, assignments, num_points, num_clusters, max_iterations, num_threads,
           cluster_manager, nullptr);
    uint32_t num_seen = cluster_manager.SetOffsets();
    FatalAssert(num_seen == num_points, LOG_TAG_BASIC,
                "Total number of assigned points should be equal to num_points!");

    std::cout << "Allocating memory for reordered vector data...\n" << std::flush;
    data.CreateEmptyImage();

    std::cout << "Reordering vector data based on cluster assignments using " << num_threads << " threads...\n" << std::flush;
    #pragma omp parallel num_threads(num_threads) shared(cluster_manager, assignments, num_points, data)
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

            std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c, false);
            FatalAssert(cluster_info.first->num_points > 0, LOG_TAG_BASIC, "Cluster should have points assigned!");
            ClusterData& cluster_data = *(cluster_info.first);
            ClusterMetaData& cluster_meta_data = *(cluster_info.second);

            cluster_data.num_total_points.fetch_add(data[i].num_duplicates + 1);
            uint32_t idx = cluster_meta_data.next_index.fetch_add(1) + cluster_data.offset;
            FatalAssert(idx < num_points, LOG_TAG_BASIC, "Index out of bounds in reordered data!");
            data.CpyToImg(i, idx);
        }
        if (created) {
            st->DestroyDIVFThread();
        }
    }
    delete[] assignments;

    uint32_t new_num_clusters = cluster_manager.RemoveEmptyClusters();
    if (num_clusters != new_num_clusters) {
        FatalAssert(new_num_clusters < num_clusters, LOG_TAG_BASIC, "new num clusters cannot be greater than old num clusters");
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BASIC, "kmeans-simple: %u clusters where empty -> new num clusters = %u", new_num_clusters - num_clusters, new_num_clusters);
    }

    data.ReplaceWithImg();
}

inline uint32_t GetSampleSize(uint32_t num_points, uint32_t min_sample_size, uint32_t vector_per_cluster, uint32_t num_clusters, size_t num_threads, uint32_t& block_size) {
    min_sample_size = (uint32_t)(std::min((size_t)num_points, std::max((size_t)min_sample_size, (size_t)min_sample_size * (size_t)(num_threads * 0.75))));
    uint32_t sample_size;
    if ((uint64_t)(vector_per_cluster) * (uint64_t)num_clusters > (uint64_t)(num_points)) {
        sample_size = std::min(num_points, min_sample_size);
    } else {
        sample_size = std::min(num_points, std::max(min_sample_size, vector_per_cluster * num_clusters));
    }
    block_size = num_points / sample_size;
    sample_size = num_points / block_size;
    return sample_size;
}

template<typename DataSetInternal>
void kmeans_sampled_general(DataSet<DataSetInternal>& data, uint32_t num_points,
                            uint32_t min_sample_size, uint32_t vector_per_cluster, uint32_t num_clusters,
                            uint32_t max_iterations, size_t num_threads, ClusterManager& cluster_manager,
                            uint32_t* const valid_indices, uint32_t* assignments,
                            uint32_t* const valid_cluster_indices, std::mt19937& gen, bool weighted_kmeans = false) {
    uint32_t block_size = 0;
    uint32_t sample_size = GetSampleSize(num_points, min_sample_size, vector_per_cluster, num_clusters, num_threads, block_size);
    FatalAssert(sample_size > 0, LOG_TAG_BASIC, "Sample size should be greater than 0!");
    FatalAssert(sample_size <= num_points, LOG_TAG_BASIC, "Sample size should be less than or equal to the number of points!");
    FatalAssert(sample_size >= num_clusters, LOG_TAG_BASIC, "Sample size should be greater than or equal to the number of clusters!");
    FatalAssert(block_size > 0, LOG_TAG_BASIC, "Block size should be greater than 0!");
    FatalAssert(block_size <= num_points, LOG_TAG_BASIC, "Block size should be less than or equal to the number of points!");
    FatalAssert(block_size <= sample_size, LOG_TAG_BASIC, "Block size should be less than or equal to the sample size!");
    FatalAssert(num_points / block_size == sample_size, LOG_TAG_BASIC, "num_points should be divisible by block_size to get the correct sample size!");

    if (sample_size == num_points) {
        FatalAssert(block_size == 1, LOG_TAG_BASIC, "If sample size is equal to num points, block size should be 1!");
        if (num_threads == 1) {
            kmeans(data, valid_indices, assignments, num_points, num_clusters, max_iterations,
                   cluster_manager, valid_cluster_indices, weighted_kmeans);
        } else {
            kmeans(data, valid_indices, assignments, num_points, num_clusters, max_iterations, num_threads,
                cluster_manager, valid_cluster_indices, weighted_kmeans);
        }
        return;
    }
    FatalAssert(block_size > 1, LOG_TAG_BASIC, "Block size should be greater than 1 for sampled k-means!");

    std::uniform_int_distribution<uint32_t> rand_num(0, block_size - 1);
    uint32_t* sample_indices = new uint32_t[sample_size];

    if (num_threads > 1) {
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
                "Sampling %u points for k-means initialization with block size %u using %zu threads...",
                sample_size, block_size, num_threads);
    }

    for (uint32_t i = 0; i < sample_size; i++) {
        for (uint32_t j = 0; j < block_size; j++) {
            uint32_t idx = (valid_indices == nullptr ? (i * block_size) + j : valid_indices[(i * block_size) + j]);
            assignments[idx] = (uint32_t)-1; // mark as unassigned for the initial k-means
        }
        uint32_t idx = (i * block_size) + rand_num(gen);
        FatalAssert(idx < num_points, LOG_TAG_BASIC, "Sample index out of bounds!");
        sample_indices[i] = (valid_indices == nullptr ? idx : valid_indices[idx]);
    }

    if (sample_size * block_size != num_points) {
        FatalAssert(sample_size * block_size < num_points, LOG_TAG_BASIC, "Sample size times block size should be less than or equal to num points!");
        for (uint32_t i = sample_size * block_size; i < num_points; i++) {
            uint32_t idx = (valid_indices == nullptr ? i : valid_indices[i]);
            assignments[idx] = (uint32_t)-1; // mark as unassigned for the initial k-means
        }
    }

    if (num_threads == 1) {
        kmeans(data, sample_indices, assignments, sample_size, num_clusters, max_iterations,
               cluster_manager, valid_cluster_indices, weighted_kmeans);
    } else {
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Sampling done");
        kmeans(data, sample_indices, assignments, sample_size, num_clusters, max_iterations, num_threads,
            cluster_manager, valid_cluster_indices, weighted_kmeans);
    }
    delete[] sample_indices;

    uint32_t fc = 0;
    if (num_clusters > 2) {
        for (; fc < num_clusters; ++fc) {
            std::pair<ClusterData*, ClusterMetaData*> cluster_info;
            uint32_t c = (valid_cluster_indices == nullptr ? fc : valid_cluster_indices[fc]);
            if (cluster_manager.GetIfNotEmpty(c, num_threads == 1, cluster_info)) {
                break;
            }
        }
    }

    if (num_threads == 1) {
        for (uint32_t i = 0; i < num_points; i++) {
            uint32_t idx = (valid_indices == nullptr ? i : valid_indices[i]);
            if (assignments[idx] != (uint32_t)-1) {
                continue; // already assigned in the initial k-means
            }
            std::pair<ClusterData*, ClusterMetaData*> cluster_info;
            uint32_t c = (valid_cluster_indices == nullptr ? fc : valid_cluster_indices[fc]);
            uint32_t best_cluster = c;
            divftree::DTYPE best_dist = L2Squared(data[idx].data, cluster_manager.At(c, true).first->data);
            for (uint32_t c_idx = fc + 1; c_idx < num_clusters; c_idx++) {
                uint32_t c = (valid_cluster_indices == nullptr ? c_idx : valid_cluster_indices[c_idx]);
                if (!cluster_manager.GetIfNotEmpty(c, true, cluster_info)) {
                    continue;
                }
                divftree::DTYPE dist = L2Squared(data[idx].data, cluster_info.first->data);
                if (MoreSimilar(dist, best_dist)) {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            assignments[idx] = best_cluster;
            ClusterData& cluster_data = *(cluster_manager.At(best_cluster, true).first);
            ClusterMetaData& cluster_meta_data = *(cluster_manager.At(best_cluster, true).second);
            ++(cluster_data.num_points); // update num_points for the assigned cluster
            cluster_meta_data.subtree_size += data.DataWeight(idx, !weighted_kmeans); // update subtree size for the assigned cluster
        }
        return;
    }


    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
            "Assigning remaining points to clusters for sampled k-means using %zu threads...", num_threads);
    #pragma omp parallel num_threads(num_threads)
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
            uint32_t idx = (valid_indices == nullptr ? i : valid_indices[i]);
            if (assignments[idx] != (uint32_t)-1) {
                continue; // already assigned in the initial k-means
            }
            std::pair<ClusterData*, ClusterMetaData*> cluster_info;
            uint32_t c = (valid_cluster_indices == nullptr ? fc : valid_cluster_indices[fc]);
            uint32_t best_cluster = c;
            divftree::DTYPE best_dist = L2Squared(data[idx].data, cluster_manager.At(c, false).first->data);
            for (uint32_t c_idx = fc + 1; c_idx < num_clusters; c_idx++) {
                uint32_t c = (valid_cluster_indices == nullptr ? c_idx : valid_cluster_indices[c_idx]);
                if (!cluster_manager.GetIfNotEmpty(c, false, cluster_info)) {
                    continue;
                }
                divftree::DTYPE dist = L2Squared(data[idx].data, cluster_info.first->data);
                if (MoreSimilar(dist, best_dist)) {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            assignments[idx] = best_cluster;
            ClusterData& cluster_data = *(cluster_manager.At(best_cluster, false).first);
            ClusterMetaData& cluster_meta_data = *(cluster_manager.At(best_cluster, false).second);
            cluster_meta_data.lock.Lock(divftree::LockMode::SX_EXCLUSIVE);
            ++(cluster_data.num_points); // update num_points for the assigned cluster
            cluster_meta_data.subtree_size += data.DataWeight(idx, !weighted_kmeans); // update subtree size for the assigned cluster
            cluster_meta_data.lock.Unlock();
        }

        if (created) {
            st->DestroyDIVFThread();
        }
    }
}

void kmeans_sampled(DataSet<VectorData>& data, uint32_t num_points,
                    uint32_t min_sample_size, uint32_t vector_per_cluster, uint32_t num_clusters,
                    uint32_t max_iterations, size_t num_threads, ClusterManager& cluster_manager) {
    cluster_manager.AddClusters(num_clusters, false);

    uint32_t* assignments = new uint32_t[num_points];
    memset(assignments, -1, sizeof(uint32_t) * num_points);

    std::mt19937 gen(std::random_device{}());

    kmeans_sampled_general(data, num_points, min_sample_size, vector_per_cluster, num_clusters, max_iterations,
                            num_threads, cluster_manager, nullptr, assignments, nullptr, gen);

    std::cout << "Setting cluster offsets for reordering...\n" << std::flush;
    uint32_t num_seen = cluster_manager.SetOffsets();
    FatalAssert(num_seen == num_points, LOG_TAG_BASIC,
                "Total number of assigned points should be equal to num_points!");

    std::cout << "Allocating memory for reordered vector data...\n" << std::flush;
    data.CreateEmptyImage();

    #pragma omp parallel num_threads(num_threads) shared(cluster_manager, assignments, num_points, data)
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

            std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c, false);
            FatalAssert(cluster_info.first->num_points > 0, LOG_TAG_BASIC, "Cluster should have points assigned!");
            ClusterData& cluster_data = *(cluster_info.first);
            ClusterMetaData& cluster_meta_data = *(cluster_info.second);

            cluster_data.num_total_points.fetch_add(data[i].num_duplicates + 1);
            uint32_t idx = cluster_meta_data.next_index.fetch_add(1) + cluster_data.offset;
            FatalAssert(idx < num_points, LOG_TAG_BASIC, "Index out of bounds in reordered data!");
            data.CpyToImg(i, idx);
        }
        if (created) {
            st->DestroyDIVFThread();
        }
    }
    delete[] assignments;

    uint32_t new_num_clusters = cluster_manager.RemoveEmptyClusters();
    if (num_clusters != new_num_clusters) {
        FatalAssert(new_num_clusters < num_clusters, LOG_TAG_BASIC, "new num clusters cannot be greater than old num clusters");
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BASIC, "kmeans-sampled: %u clusters where empty -> new num clusters = %u", new_num_clusters - num_clusters, new_num_clusters);
    }

    data.ReplaceWithImg();
}

enum class NumClusterCompAlg : uint8_t {
    SIMPLE,
    SIMPLE_COEFFICIENT,
    MAX,
    MAX_SAMPLED
};

struct ClusteringConfig {
    NumClusterCompAlg num_cluster_comp_alg;
    union {
        uint32_t coefficient;
        struct {
            uint32_t max_num_clusters;
            uint32_t min_sample_size;
            uint32_t num_points_per_cluster;
        };
    };

    std::string ToString() const {
        switch (num_cluster_comp_alg) {
            case NumClusterCompAlg::SIMPLE:
                return "SIMPLE";
            case NumClusterCompAlg::SIMPLE_COEFFICIENT:
                return "SIMPLE_COEFFICIENT(coefficient=" + std::to_string(coefficient) + ")";
            case NumClusterCompAlg::MAX:
                return "MAX(max_num_clusters=" + std::to_string(max_num_clusters) + ")";
            case NumClusterCompAlg::MAX_SAMPLED:
                return "MAX_SAMPLED(max_num_clusters=" + std::to_string(max_num_clusters) + ", min_sample_size=" + std::to_string(min_sample_size) + ", num_points_per_cluster=" + std::to_string(num_points_per_cluster) + ")";
            default:
                return "UNKNOWN";
        }
    }
};

inline uint32_t GetNumClusters(uint32_t num_points, uint32_t cluster_cap, ClusteringConfig config) {
    FatalAssert(cluster_cap > 0, LOG_TAG_BASIC, "Cluster capacity must be greater than 0!");
    FatalAssert(num_points > cluster_cap, LOG_TAG_BASIC, "Number of points must be greater than 0!");

    if (config.num_cluster_comp_alg == NumClusterCompAlg::MAX || config.num_cluster_comp_alg == NumClusterCompAlg::MAX_SAMPLED) {
        return std::max(2u, std::min(config.max_num_clusters, num_points / cluster_cap));
    }

    if (config.num_cluster_comp_alg == NumClusterCompAlg::SIMPLE_COEFFICIENT) {
        uint32_t num_clusters = num_points / (cluster_cap * config.coefficient);
        if (num_clusters >= 2) {
            return num_clusters;
        }
    }

    return std::max(2u, num_points / cluster_cap);
}

struct SplitTask {
    uint32_t target_cluster;
    uint32_t new_cluster_offsets;
    uint32_t num_clusters;
};

template<typename DataSetInternal>
void kmeans_capped(DataSet<DataSetInternal>& data, uint32_t num_points, uint32_t cluster_cap, uint32_t max_iterations,
                   size_t num_threads, ClusterManager& cluster_manager, ClusteringConfig config,
                   bool weighted_kmeans = false) {
    size_t num_clusters = GetNumClusters(num_points, cluster_cap, config);
    cluster_manager.AddClusters(num_clusters, false);

    uint32_t* assignments = new uint32_t[num_points];
    memset(assignments, 0, sizeof(uint32_t) * num_points);

    std::cout << "Running initial k-means with " << num_clusters << " clusters...\n" << std::flush;
    if (config.num_cluster_comp_alg == NumClusterCompAlg::MAX_SAMPLED) {
        std::mt19937 gen(std::random_device{}());
        kmeans_sampled_general(data, num_points, config.min_sample_size, config.num_points_per_cluster, num_clusters,
                               max_iterations, num_threads, cluster_manager, nullptr, assignments, nullptr, gen, weighted_kmeans);
    } else {
        kmeans(data, nullptr, assignments, num_points, num_clusters, max_iterations, num_threads,
               cluster_manager, nullptr, weighted_kmeans);
    }

    divftree::BlockingQueue<SplitTask> split_tasks(num_clusters);

    uint32_t num_seen = 0;
    uint32_t num_new_clusters_needed = 0;
    for (uint32_t i = 0; i < num_clusters; i++) {
        std::pair<ClusterData*, ClusterMetaData*> cluster_info;
        if (!cluster_manager.GetIfNotEmpty(i, false, cluster_info)) {
            FatalAssert(num_clusters > 2, LOG_TAG_BASIC, "2means should not return empty clusters when initial centroids are chosen from the actual vectors");
            continue;
        }
        ClusterData& cluster_data = *(cluster_info.first);
        ClusterMetaData& cluster_meta_data = *(cluster_info.second);
        FatalAssert(cluster_data.num_points > 0, LOG_TAG_BASIC, "Cluster has no points assigned!");
        if (cluster_data.num_points > cluster_cap) {
            uint32_t num_clusters_needed = GetNumClusters(cluster_data.num_points, cluster_cap, config);
            SplitTask st;
            st.target_cluster = i;
            st.new_cluster_offsets = num_new_clusters_needed + num_clusters;
            st.num_clusters = num_clusters_needed;
            split_tasks.Push(std::move(st));
            num_new_clusters_needed += num_clusters_needed - 1;
            cluster_meta_data.assigned_vector_indices.resize(cluster_data.num_points);
            cluster_meta_data.next_index.store(0, std::memory_order_release);
            // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Cluster %u has %u points, needs to be split into %u clusters\n",
            //         i, cluster_data.num_points, num_clusters_needed);
        } /* else {
            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Cluster %u has %u points, does not need to be split\n",
                    i, cluster_data.num_points);
        } */
        num_seen += cluster_data.num_points;
    }
    FatalAssert(num_seen == num_points, LOG_TAG_BASIC,
                "Total number of assigned points should be equal to num_points!");

    if (num_new_clusters_needed > 0) {
        std::cout << "Adding " << num_new_clusters_needed << " new clusters for splitting(" << split_tasks.Size() <<
                     " tasks)...\n" << std::flush;
        cluster_manager.AddClusters(num_new_clusters_needed, false);
    }
    std::atomic<size_t> num_tasks = split_tasks.Size();

    std::cout << "Splitting clusters that exceed the capacity using " << num_threads << " threads...\n" << std::flush;
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
            std::mt19937 gen(std::random_device{}());

            #pragma omp for schedule(guided)
            for (uint32_t i = 0; i < num_points; i++) {
                uint32_t c = assignments[i];
                std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c, false);
                FatalAssert(cluster_info.first->num_points > 0, LOG_TAG_BASIC, "Cluster should have points assigned!");
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

                std::pair<ClusterData*, ClusterMetaData*> target_cluster_info = cluster_manager.At(task.target_cluster, true);
                ClusterData& target_cluster_data = *(target_cluster_info.first);
                ClusterMetaData& target_cluster_meta_data = *(target_cluster_info.second);
                FatalAssert(target_cluster_meta_data.assigned_vector_indices.size() == target_cluster_data.num_points, LOG_TAG_BASIC,
                            "Size of assigned_vector_indices should be equal to the number of points in the cluster!");

                DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Splitting cluster %u with %u points into %u clusters...",
                        task.target_cluster, target_cluster_data.num_points, task.num_clusters);
                uint32_t old_size = target_cluster_data.num_points;
                if (config.num_cluster_comp_alg == NumClusterCompAlg::MAX_SAMPLED) {
                    kmeans_sampled_general(data, target_cluster_data.num_points, config.min_sample_size, config.num_points_per_cluster, task.num_clusters,
                                           max_iterations, 1, cluster_manager, target_cluster_meta_data.assigned_vector_indices.data(),
                                           assignments, new_valid_cluster_indices, gen, weighted_kmeans);
                } else {
                    kmeans(data, target_cluster_meta_data.assigned_vector_indices.data(), assignments, target_cluster_data.num_points,
                        task.num_clusters, max_iterations, cluster_manager, new_valid_cluster_indices, weighted_kmeans);
                }

                uint32_t num_points_seen = 0;
                uint32_t num_new_clusters_needed_for_task = 0;
                std::vector<SplitTask> new_split_tasks;
                new_split_tasks.clear();
                new_split_tasks.reserve(task.num_clusters);
                for (uint32_t i = 0; i < task.num_clusters; i++) {
                    uint32_t c = new_valid_cluster_indices[i];
                    std::pair<ClusterData*, ClusterMetaData*> cluster_info;
                    if (!cluster_manager.GetIfNotEmpty(c, true, cluster_info)) {
                        FatalAssert(task.num_clusters > 2, LOG_TAG_BASIC, "2means should not return empty clusters when initial centroids are chosen from the actual vectors");
                        continue;
                    }

                    ClusterData& cluster_data = *(cluster_info.first);
                    ClusterMetaData& cluster_meta_data = *(cluster_info.second);
                    FatalAssert(cluster_data.num_points > 0, LOG_TAG_BASIC, "Cluster has no points assigned!");
                    if (cluster_data.num_points > cluster_cap) {
                        uint32_t num_clusters_needed = GetNumClusters(cluster_data.num_points, cluster_cap, config);
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
                        // DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Cluster %u has %u points, needs to be split into %u clusters\n",
                        //         c, cluster_data.num_points, num_clusters_needed);
                    } /* else {
                        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Cluster %u has %u points, does not need to be split\n",
                                c, cluster_data.num_points);
                    } */
                    num_points_seen += cluster_data.num_points;
                }
                FatalAssert(num_points_seen == old_size, LOG_TAG_BASIC,
                            "Total number of assigned points should be equal to num_points!");

                if (!new_split_tasks.empty()) {
                    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Adding %u new clusters for further splitting(%u tasks)...",
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
                        std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c, true);
                        ClusterData& cluster_data = *(cluster_info.first);
                        ClusterMetaData& cluster_meta_data = *(cluster_info.second);
                        FatalAssert(cluster_data.num_points > 0, LOG_TAG_BASIC, "Cluster should have points assigned!");

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

        #pragma omp barrier

        #pragma omp single
        {
            uint32_t num_seen = cluster_manager.SetOffsets();
            FatalAssert(num_seen == num_points, LOG_TAG_BASIC,
                        "Total number of assigned points should be equal to num_points!");

            std::cout << "Allocating memory for reordered vector data...\n" << std::flush;
            data.CreateEmptyImage();
            std::cout << "Reordering vector data based on cluster assignments...\n" << std::flush;
        }

        #pragma omp for schedule(static)
        for (uint32_t i = 0; i < num_points; i++) {
            uint32_t c = assignments[i];
            std::pair<ClusterData*, ClusterMetaData*> cluster_info = cluster_manager.At(c, false);
            ClusterData& cluster_data = *(cluster_info.first);
            ClusterMetaData& cluster_meta_data = *(cluster_info.second);
            FatalAssert(cluster_data.num_points > 0, LOG_TAG_BASIC, "Cluster should have points assigned!");
            if constexpr (std::is_same<DataSetInternal, VectorData>::value) {
                cluster_data.num_total_points.fetch_add(data[i].num_duplicates + 1);
            }
            uint32_t idx = cluster_meta_data.next_index.fetch_add(1) + cluster_data.offset;
            FatalAssert(idx < num_points, LOG_TAG_BASIC, "Index out of bounds in reordered data!");
            data.CpyToImg(i, idx);
        }

        if (created) {
            st->DestroyDIVFThread();
        }
    }
    num_clusters = cluster_manager.RemoveEmptyClusters();

    delete[] assignments;

    data.ReplaceWithImg();

    std::cout << "Clustering completed. Final number of clusters: " << cluster_manager.clusters.size() << "\n" << std::flush;
    std::cout << " Memory utilization: " <<
                 ((double)num_points * (double)100) / ((double)cluster_manager.clusters.size() * (double)cluster_cap) <<
                 "%\n" << std::flush;
}

enum class AlgorithmType : uint8_t {
    KMEANS,
    KMEANS_SAMPLED,
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
            uint32_t min_sample_size;
            uint32_t num_point_per_cluster;
            uint32_t num_clusters;
        } kmeans_sampled_args;

        struct {
            uint32_t cluster_cap;
            ClusteringConfig conf;
        } kmeans_capped_args;

        struct {
            uint32_t leaf_cap;
            uint32_t internal_cap;
            bool weighted_kmeans;
            ClusteringConfig conf;
        } divftree_kmeans_capped_args;
    };

    void print() {
        std::cout << "Input file: " << input_file << "\n" << std::flush;
        std::cout << "Output file: " << output_file << "\n" << std::flush;
        std::cout << "Num threads: " << num_threads << "\n" << std::flush;
        std::cout << "Num points to use: " << num_points_to_use << "/" << num_total_points << "\n" << std::flush;
        std::cout << "Max iterations: " << max_iterations << "\n" << std::flush;
        std::cout << "Algorithm: " << std::flush;
        switch (algorithm) {
            case AlgorithmType::KMEANS:
                std::cout << "KMEANS\n" << std::flush;
                std::cout << "Num clusters: " << kmeans_args.num_clusters << "\n" << std::flush;
                break;
            case AlgorithmType::KMEANS_SAMPLED:
                std::cout << "KMEANS_SAMPLED\n" << std::flush;
                std::cout << "Num clusters: " << kmeans_sampled_args.num_clusters << "\n" << std::flush;
                std::cout << "Min sample size: " << kmeans_sampled_args.min_sample_size << "\n" << std::flush;
                std::cout << "Points per cluster: " << kmeans_sampled_args.num_point_per_cluster << "\n" << std::flush;
                break;
            case AlgorithmType::KMEANS_CAPPED:
                std::cout << "KMEANS_CAPPED\n" << std::flush;
                std::cout << "Cluster capacity: " << kmeans_capped_args.cluster_cap << "\n" << std::flush;
                std::cout << "Config: " << kmeans_capped_args.conf.ToString() << "\n" << std::flush;
                break;
            case AlgorithmType::KMEANS_HIERARCHICAL:
                std::cout << "KMEANS_HIERARCHICAL\n" << std::flush;
                std::cout << "Leaf cluster capacity: " << divftree_kmeans_capped_args.leaf_cap << "\n" << std::flush;
                std::cout << "Internal cluster capacity: " << divftree_kmeans_capped_args.internal_cap << "\n" << std::flush;
                std::cout << "Use Weighted Kmeans: " << (divftree_kmeans_capped_args.weighted_kmeans ? "T" : "F")
                          << "\n" << std::flush;
                std::cout << "Config: " << divftree_kmeans_capped_args.conf.ToString() << "\n" << std::flush;
                break;
            default:
                std::cout << "UNKNOWN\n" << std::flush;
        }
    }
};

inline void PrintUsage(int argc, char* argv[]) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <num_threads> <num_points> "
                 "<max_iter> <algorithm> (<num_clusters> | <num_clusters> <min_sample_size> <num_point_per_cluster> | (<cluster_cap> | <leaf_cap> <internal_cap> <weighted_kmeans>) <clustering_alg> [<max_num_clusters> | <coefficent> | <max_num_clusters> <min_sample_size> <num_point_per_cluster>])\n";
    std::cerr << "\t<input_file>: Path to the input binary file containing vector data.\n";
    std::cerr << "\t<output_file>: Path to the output binary file to write clustered vector data.\n";
    std::cerr << "\t<num_threads>: Number of threads to use for clustering. If set to 0 "
                 "or more than number of hardware threads, max number of hardware threads will be used instead\n";
    std::cerr << "\t<num_points>: Number of points to use from the input file."
                 " Must be a non-zero number, less than or equal to the total number of points in the file.\n";
    std::cerr << "\t<max_iter>: Maximum number of iterations for the clustering algorithm. Set to 0 for no limit.\n";
    std::cerr << "\t<algorithm>: Clustering algorithm to use. "
                 "Must be one of the following: 'kmeans', 'kmeans_sampled', 'kmeans_capped', 'kmeans_hierarchical'.\n";
    std::cerr << "\t<num_clusters>: (Only for kmeans or kmeans_sampled) Number of clusters to form. "
                 "Must be a positive integer greater than one and less than or equal to num_points.\n";
    std::cerr << "\t<min_sample_size>: (Only for kmeans_sampled or other methods with 'max_sampled' type) Minimum number of points to sample from the input file. "
                 "Must be a positive integer greater than 1 and less than or equal to num_points.\n";
    std::cerr << "\t<num_point_per_cluster>: (Only for kmeans_sampled or other methods with 'max_sampled' type) Number of points to sample per cluster in the initial k-means. "
                 "Must be a positive integer greater than 1 and less than num_points.\n";
    std::cerr << "\t<cluster_cap>: (Only for kmeans_capped) Maximum number of points allowed in each cluster. "
                 "Must be a positive integer greater than one and less than num_points.\n";
    std::cerr << "\t<leaf_cap>: (Only for kmeans_hierarchical) Maximum number of points allowed in each leaf cluster."
                 "Must be a positive integer greater than one and less than num_points.\n";
    std::cerr << "\t<internal_cap>: (Only for kmeans_hierarchical) Maximum number of points allowed in each internal cluster."
                 "Must be a positive integer greater than one and less than or equal to <leaf_cap>.\n";
    std::cerr << "\t<weighted_kmeans>: (Only for kmeans_hierarchical) If set to 1, will use the number of raw vectors per subtree while computing the internal node centroids."
                 "Must be either 1 or 0.\n";
    std::cerr << "\t<clustering_alg>: (Only for kmeans_capped or kmeans_hierarchical) Clustering algorithm to use. "
                 "Must be one of the following: 'simple', 'simple_coefficient', 'max', 'max_sampled'.\n";
    std::cerr << "\t<max_num_clusters>: (Only for kmeans_capped or kmeans_hierarchical with 'max' algorithm) Maximum number of clusters to form per each call to kmeans."
                 "Must be a positive integer greater than one and less than or equal to num_points.\n";
    std::cerr << "\t<coefficient>: (Only for kmeans_capped or kmeans_hierarchical with 'simple_coefficient' algorithm) Coefficient to use in computing number of clusters. "
                 "Must be a positive integer greater than one and less than num_points.\n";

    std::cerr << "Recived " << argc - 1 << " arguments:\n";
    for (int i = 1; i < argc; i++) {
        std::cerr << "\tArgument " << i << ": " << argv[i] << "\n";
    }
}

inline void ParseArgs(int argc, char* argv[], Args& args) {
    if (argc < 8 || argc > 14) {
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
    } else if (strcmp(argv[6], "kmeans_sampled") == 0) {
        if (argc != 10) {
            PrintUsage(argc, argv);
            FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments for kmeans_sampled!");
            exit(EXIT_FAILURE);
        }
        uint32_t n_clusters = std::stoul(argv[7]);
        if (n_clusters <= 1) {
            PrintUsage(argc, argv);
            FatalAssert(false, LOG_TAG_BASIC, "Number of clusters must be greater than 1 and less than or equal to num_points for kmeans_sampled!");
            exit(EXIT_FAILURE);
        }
        args.kmeans_sampled_args.num_clusters = n_clusters;
        uint32_t min_sample_size = std::stoul(argv[8]);
        if (min_sample_size <= 1 || n_clusters >= min_sample_size) {
            PrintUsage(argc, argv);
            FatalAssert(false, LOG_TAG_BASIC, "Min sample size must be greater than 1 and less than or equal to num_points for kmeans_sampled!");
            exit(EXIT_FAILURE);
        }

        uint32_t num_point_per_cluster = std::stoul(argv[9]);
        if (num_point_per_cluster <= 1) {
            PrintUsage(argc, argv);
            FatalAssert(false, LOG_TAG_BASIC, "Number of points per cluster must be greater than 1 and less than num_points for kmeans_sampled!");
            exit(EXIT_FAILURE);
        }
        args.kmeans_sampled_args.min_sample_size = min_sample_size;
        args.kmeans_sampled_args.num_point_per_cluster = num_point_per_cluster;
        args.algorithm = AlgorithmType::KMEANS_SAMPLED;
    } else if (strcmp(argv[6], "kmeans_capped") == 0) {
        if (argc < 9) {
            PrintUsage(argc, argv);
            FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments for kmeans_capped!");
            exit(EXIT_FAILURE);
        }
        if (strcmp(argv[8], "simple") == 0) {
            if (argc != 9) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments for kmeans_capped with simple!");
                exit(EXIT_FAILURE);
            }
            args.kmeans_capped_args.conf.num_cluster_comp_alg = NumClusterCompAlg::SIMPLE;
        } else if (strcmp(argv[8], "simple_coefficient") == 0) {
            if (argc != 10) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments for kmeans_capped with simple_coefficient!");
                exit(EXIT_FAILURE);
            }
            uint32_t coefficient = std::stoul(argv[9]);
            if (coefficient <= 1) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Coefficient must be greater than 1 for simple_coefficient algorithm!");
                exit(EXIT_FAILURE);
            }
            args.kmeans_capped_args.conf.num_cluster_comp_alg = NumClusterCompAlg::SIMPLE_COEFFICIENT;
            args.kmeans_capped_args.conf.coefficient = coefficient;
        } else if (strcmp(argv[8], "max") == 0) {
            if (argc != 10) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments for kmeans_capped with max!");
                exit(EXIT_FAILURE);
            }
            uint32_t max_num_clusters = std::stoul(argv[9]);
            if (max_num_clusters <= 1) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Max number of clusters must be greater than 1 for max algorithm!");
                exit(EXIT_FAILURE);
            }
            args.kmeans_capped_args.conf.num_cluster_comp_alg = NumClusterCompAlg::MAX;
            args.kmeans_capped_args.conf.max_num_clusters = max_num_clusters;
        } else if (strcmp(argv[8], "max_sampled") == 0) {
            if (argc != 12) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments for kmeans_capped with max_sampled!");
                exit(EXIT_FAILURE);
            }
            uint32_t max_num_clusters = std::stoul(argv[9]);
            if (max_num_clusters <= 1) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Max number of clusters must be greater than 1 for max_sampled algorithm!");
                exit(EXIT_FAILURE);
            }
            uint32_t min_sample_size = std::stoul(argv[10]);
            if (min_sample_size <= 1) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Min sample size must be greater than 1 for max_sampled algorithm!");
                exit(EXIT_FAILURE);
            }
            uint32_t num_point_per_cluster = std::stoul(argv[11]);
            if (num_point_per_cluster <= 1) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Number of points per cluster must be greater than 1 and less than num_points for max_sampled algorithm!");
                exit(EXIT_FAILURE);
            }
            args.kmeans_capped_args.conf.num_cluster_comp_alg = NumClusterCompAlg::MAX_SAMPLED;
            args.kmeans_capped_args.conf.max_num_clusters = max_num_clusters;
            args.kmeans_capped_args.conf.min_sample_size = min_sample_size;
            args.kmeans_capped_args.conf.num_points_per_cluster = num_point_per_cluster;
        } else {
            PrintUsage(argc, argv);
            FatalAssert(false, LOG_TAG_BASIC, "Invalid clustering algorithm for kmeans_capped!");
            exit(EXIT_FAILURE);
        }
        args.algorithm = AlgorithmType::KMEANS_CAPPED;
    } else if (strcmp(argv[6], "kmeans_hierarchical") == 0) {
        if (argc < 11) {
            PrintUsage(argc, argv);
            FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments for kmeans_hierarchical!");
            exit(EXIT_FAILURE);
        }

        uint32_t w = std::stoul(argv[9]);
        if (w != 0 && w != 1) {
            PrintUsage(argc, argv);
            FatalAssert(false, LOG_TAG_BASIC, "Invalid value for whether to use weighted kmeans or not!");
            exit(EXIT_FAILURE);
        }
        args.divftree_kmeans_capped_args.weighted_kmeans = (w != 0);

        if (strcmp(argv[10], "simple") == 0) {
            if (argc != 11) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments for kmeans_hierarchical with simple!");
                exit(EXIT_FAILURE);
            }
            args.divftree_kmeans_capped_args.conf.num_cluster_comp_alg = NumClusterCompAlg::SIMPLE;
        } else if (strcmp(argv[10], "simple_coefficient") == 0) {
            if (argc != 12) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments for kmeans_hierarchical with simple_coefficient!");
                exit(EXIT_FAILURE);
            }
            uint32_t coefficient = std::stoul(argv[11]);
            if (coefficient <= 1) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Coefficient must be greater than 1 for simple_coefficient algorithm!");
                exit(EXIT_FAILURE);
            }
            args.divftree_kmeans_capped_args.conf.num_cluster_comp_alg = NumClusterCompAlg::SIMPLE_COEFFICIENT;
            args.divftree_kmeans_capped_args.conf.coefficient = coefficient;
        } else if (strcmp(argv[10], "max") == 0) {
            if (argc != 12) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments for kmeans_hierarchical with max!");
                exit(EXIT_FAILURE);
            }
            uint32_t max_num_clusters = std::stoul(argv[11]);
            if (max_num_clusters <= 1) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Max number of clusters must be greater than 1 for max algorithm!");
                exit(EXIT_FAILURE);
            }
            args.divftree_kmeans_capped_args.conf.num_cluster_comp_alg = NumClusterCompAlg::MAX;
            args.divftree_kmeans_capped_args.conf.max_num_clusters = max_num_clusters;
        } else if (strcmp(argv[10], "max_sampled") == 0) {
            if (argc != 14) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments for kmeans_hierarchical with max_sampled!");
                exit(EXIT_FAILURE);
            }
            uint32_t max_num_clusters = std::stoul(argv[11]);
            if (max_num_clusters <= 1) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Max number of clusters must be greater than 1 for max_sampled algorithm!");
                exit(EXIT_FAILURE);
            }
            uint32_t min_sample_size = std::stoul(argv[12]);
            if (min_sample_size <= 1) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Min sample size must be greater than 1 for max_sampled algorithm!");
                exit(EXIT_FAILURE);
            }
            uint32_t num_point_per_cluster = std::stoul(argv[13]);
            if (num_point_per_cluster <= 1) {
                PrintUsage(argc, argv);
                FatalAssert(false, LOG_TAG_BASIC, "Number of points per cluster must be greater than 1 and less than num_points for max_sampled algorithm!");
                exit(EXIT_FAILURE);
            }
            args.divftree_kmeans_capped_args.conf.num_cluster_comp_alg = NumClusterCompAlg::MAX_SAMPLED;
            args.divftree_kmeans_capped_args.conf.max_num_clusters = max_num_clusters;
            args.divftree_kmeans_capped_args.conf.min_sample_size = min_sample_size;
            args.divftree_kmeans_capped_args.conf.num_points_per_cluster = num_point_per_cluster;
        } else {
            PrintUsage(argc, argv);
            FatalAssert(false, LOG_TAG_BASIC, "Invalid clustering algorithm for kmeans_hierarchical!");
            exit(EXIT_FAILURE);
        }
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

    if (args.algorithm == AlgorithmType::KMEANS_SAMPLED) {
        if (args.kmeans_sampled_args.min_sample_size >= num_unique_points) {
            std::cerr << "Error: min_sample_size must be less than num_unique_points (" << num_unique_points << ").\n";
            fclose(args.input_fp);
            FatalAssert(false, LOG_TAG_BASIC, "min_sample_size must be less than num_unique_points!");
            exit(EXIT_FAILURE);
        }

        if (args.kmeans_sampled_args.num_point_per_cluster >= num_unique_points) {
            std::cerr << "Error: num_point_per_cluster must be less than num_unique_points (" << num_unique_points << ").\n";
            fclose(args.input_fp);
            FatalAssert(false, LOG_TAG_BASIC, "num_point_per_cluster must be less than num_unique_points!");
            exit(EXIT_FAILURE);
        }
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
            std::cout << "Output directory '" << out_dir.string() << "' does not exist. Creating directories...\n" << std::flush;
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
    if (args.num_points_to_use == 0) {
        std::cerr << "Warning: num_points_to_use is set to 0, which is invalid. Using num_unique_points (" << num_unique_points << ") instead.\n";
        args.num_points_to_use = num_unique_points;
    } else if (args.num_points_to_use > num_unique_points) {
        std::cerr << "Error: num_points_to_use must be less than or equal to num_unique_points (" << num_unique_points << ").\n";
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
        if (args.kmeans_capped_args.conf.num_cluster_comp_alg == NumClusterCompAlg::MAX_SAMPLED) {
            if (args.kmeans_capped_args.conf.min_sample_size >= args.num_points_to_use) {
                std::cerr << "Error: min_sample_size must be less than num_points_to_use (" << args.num_points_to_use << ").\n";
                fclose(args.input_fp);
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "min_sample_size must be less than num_points_to_use!");
                exit(EXIT_FAILURE);
            }
            if (args.kmeans_capped_args.conf.num_points_per_cluster >= args.num_points_to_use) {
                std::cerr << "Error: num_points_per_cluster must be less than num_points_to_use (" << args.num_points_to_use << ").\n";
                fclose(args.input_fp);
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "num_points_per_cluster must be less than num_points_to_use!");
                exit(EXIT_FAILURE);
            }
        }
        args.kmeans_capped_args.cluster_cap = std::stoul(argv[7]);
        if (args.kmeans_capped_args.cluster_cap < 2 || args.kmeans_capped_args.cluster_cap >= args.num_points_to_use) {
            std::cerr << "Error: cluster_cap must be greater than 1 and less than num_points_to_use (" << args.num_points_to_use << ").\n";
            fclose(args.input_fp);
            fclose(args.output_fp);
            FatalAssert(false, LOG_TAG_BASIC, "cluster_cap must be greater than 1 and less than num_points_to_use!");
            exit(EXIT_FAILURE);
        }
    } else if (args.algorithm != AlgorithmType::KMEANS_SAMPLED) {
        FatalAssert(args.algorithm == AlgorithmType::KMEANS_HIERARCHICAL, LOG_TAG_BASIC, "Invalid algorithm type!");
        args.divftree_kmeans_capped_args.leaf_cap = std::stoul(argv[7]);
        args.divftree_kmeans_capped_args.internal_cap = std::stoul(argv[8]);
        if (args.divftree_kmeans_capped_args.leaf_cap < 2 || args.divftree_kmeans_capped_args.leaf_cap >= args.num_points_to_use) {
            std::cerr << "Error: leaf_cap must be greater than 1 and less than num_points_to_use (" << args.num_points_to_use << ").\n";
            fclose(args.input_fp);
            fclose(args.output_fp);
            FatalAssert(false, LOG_TAG_BASIC, "leaf_cap must be greater than 1 and less than num_points_to_use!");
            exit(EXIT_FAILURE);
        }
        if (args.divftree_kmeans_capped_args.internal_cap < 2 /* || args.divftree_kmeans_capped_args.internal_cap > args.divftree_kmeans_capped_args.leaf_cap */) {
            std::cerr << "Error: internal_cap must be greater than 1 and less than or equal to leaf_cap (" << args.divftree_kmeans_capped_args.leaf_cap << ").\n";
            fclose(args.input_fp);
            fclose(args.output_fp);
            FatalAssert(false, LOG_TAG_BASIC, "internal_cap must be greater than 1 and less than or equal to leaf_cap!");
            exit(EXIT_FAILURE);
        }

        if (args.divftree_kmeans_capped_args.conf.num_cluster_comp_alg == NumClusterCompAlg::MAX_SAMPLED) {
            if (args.divftree_kmeans_capped_args.conf.min_sample_size >= args.num_points_to_use) {
                std::cerr << "Error: min_sample_size must be less than num_points_to_use (" << args.num_points_to_use << ").\n";
                fclose(args.input_fp);
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "min_sample_size must be less than num_points_to_use!");
                exit(EXIT_FAILURE);
            }
            if (args.divftree_kmeans_capped_args.conf.num_points_per_cluster >= args.num_points_to_use) {
                std::cerr << "Error: num_points_per_cluster must be less than num_points_to_use (" << args.num_points_to_use << ").\n";
                fclose(args.input_fp);
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "num_points_per_cluster must be less than num_points_to_use!");
                exit(EXIT_FAILURE);
            }
        }
    }
}


/*
output file format:
    IndexType idx_type
    uint32_t cluster_cap (only for kmeans-capped)
    uint32_t leaf_cluster_cap (only for hierarchical k-means)
    uint32_t internal_cluster_cap (only for hierarchical k-means)
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
                  Args& args) {
    std::cout << "Writing output to file...\n" << std::flush;
    divftree::IndexType idx_type;
    size_t ret = 0;
    switch (args.algorithm) {
        case AlgorithmType::KMEANS:
        case AlgorithmType::KMEANS_SAMPLED:
            idx_type = divftree::IndexType::IVF_FLAT;
            ret = fwrite(&idx_type, sizeof(divftree::IndexType), 1, args.output_fp);
            if (ret != 1) {
                std::cerr << "Error writing index type to output file.\n";
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing index type to output file!");
                exit(EXIT_FAILURE);
            }
            break;
        case AlgorithmType::KMEANS_CAPPED:
            idx_type = divftree::IndexType::IVF_CAPPED;
            ret = fwrite(&idx_type, sizeof(divftree::IndexType), 1, args.output_fp);
            if (ret != 1) {
                std::cerr << "Error writing index type to output file.\n";
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing index type to output file!");
                exit(EXIT_FAILURE);
            }
            ret = fwrite(&args.kmeans_capped_args.cluster_cap, sizeof(uint32_t), 1, args.output_fp);
            if (ret != 1) {
                std::cerr << "Error writing cluster_cap to output file.\n";
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing cluster_cap to output file!");
                exit(EXIT_FAILURE);
            }
            break;
        case AlgorithmType::KMEANS_HIERARCHICAL:
            idx_type = divftree::IndexType::IVF_TREE;
            ret = fwrite(&idx_type, sizeof(divftree::IndexType), 1, args.output_fp);
            if (ret != 1) {
                std::cerr << "Error writing index type to output file.\n";
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing index type to output file!");
                exit(EXIT_FAILURE);
            }
            ret = fwrite(&args.divftree_kmeans_capped_args.leaf_cap, sizeof(uint32_t), 1, args.output_fp);
            if (ret != 1) {
                std::cerr << "Error writing leaf_cluster_cap to output file.\n";
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing leaf_cluster_cap to output file!");
                exit(EXIT_FAILURE);
            }
            ret = fwrite(&args.divftree_kmeans_capped_args.internal_cap, sizeof(uint32_t), 1, args.output_fp);
            if (ret != 1) {
                std::cerr << "Error writing internal_cluster_cap to output file.\n";
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing internal_cluster_cap to output file!");
                exit(EXIT_FAILURE);
            }
            break;
        default:
            FatalAssert(false, LOG_TAG_BASIC, "Invalid algorithm type!");
            exit(EXIT_FAILURE);
    }

    ret = fwrite(&args.num_total_points_used, sizeof(uint32_t), 1, args.output_fp);
    if (ret != 1) {
        std::cerr << "Error writing num_total_points to output file.\n";
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error writing num_total_points to output file!");
        exit(EXIT_FAILURE);
    }

    ret = fwrite(&args.num_points_to_use, sizeof(uint32_t), 1, args.output_fp);
    if (ret != 1) {
        std::cerr << "Error writing num_unique_points to output file.\n";
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error writing num_unique_points to output file!");
        exit(EXIT_FAILURE);
    }

    uint16_t dimension_to_write = DIMENSION;
    ret = fwrite(&dimension_to_write, sizeof(uint16_t), 1, args.output_fp);
    if (ret != 1) {
        std::cerr << "Error writing dimension to output file.\n";
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error writing dimension to output file!");
        exit(EXIT_FAILURE);
    }

    ret = fwrite(&num_levels, sizeof(uint8_t), 1, args.output_fp);
    if (ret != 1) {
        std::cerr << "Error writing num_levels to output file.\n";
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error writing num_levels to output file!");
        exit(EXIT_FAILURE);
    }

    std::cout << "num_total_points: " << args.num_total_points_used << "\n" << std::flush;
    std::cout << "num_unique_points: " << args.num_points_to_use << "\n" << std::flush;
    std::cout << "dimension: " << dimension_to_write << "\n" << std::flush;
    std::cout << "num_levels: " << (uint32_t)num_levels << "\n" << std::flush;

    /* level 0 are the vectors so they are skipped here! num levels refers to the number of levels above the raw vectors */
    for (uint8_t level = num_levels; level > 0; level--) {
        ClusterManager& cluster_manager = cluster_managers[level - 1];
        uint32_t num_clusters = cluster_manager.clusters.size();
        // std::cout << "\tWriting level " << (uint32_t)level << " with " << num_clusters << " clusters...\n";
        ret = fwrite(&num_clusters, sizeof(uint32_t), 1, args.output_fp);
        if (ret != 1) {
            std::cerr << "Error writing num_clusters for level " << (uint32_t)level << " to output file.\n";
            fclose(args.output_fp);
            FatalAssert(false, LOG_TAG_BASIC, "Error writing num_clusters to output file!");
            exit(EXIT_FAILURE);
        }

        // uint32_t cluster_count = 0;
        for (const auto& cluster : cluster_manager.clusters) {
            ClusterData& cluster_data = *(cluster.first);
            ClusterMetaData& cluster_meta_data = *(cluster.second);
            uint32_t num_points_in_cluster = cluster_data.num_points;
            uint32_t num_total_points_in_cluster = cluster_data.num_total_points.load(std::memory_order_acquire);

            // std::cout << "\t\tWriting cluster " << cluster_count++ << "/" << num_clusters << " with id {level="
            //           << (uint64_t)cluster_data.id._level << ", val=" << (uint64_t)cluster_data.id._val
            //           << ", id=" << (uint64_t)cluster_data.id._id
            //           << "}, num_points: " << num_points_in_cluster
            //           << ", num_total_points: " << num_total_points_in_cluster
            //           << ", offset: " << cluster_data.offset
            //           << ", data=[";
            ret = fwrite(&cluster_data.id, sizeof(divftree::VectorID), 1, args.output_fp);
            if (ret != 1) {
                std::cerr << "Error writing cluster_id for cluster " << cluster_data.id._id << " in level " << (uint32_t)level << " to output file.\n";
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing cluster_id to output file!");
                exit(EXIT_FAILURE);
            }

            ret = fwrite(&num_points_in_cluster, sizeof(uint32_t), 1, args.output_fp);
            if (ret != 1) {
                std::cerr << "Error writing num_points for cluster " << cluster_data.id._id << " in level " << (uint32_t)level << " to output file.\n";
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing num_points to output file!");
                exit(EXIT_FAILURE);
            }

            if (level == 1) {
                FatalAssert(num_total_points_in_cluster >= cluster_data.num_points, LOG_TAG_BASIC,
                            "num_total_points_in_cluster should be greater than or equal to num_points in the cluster!");
                ret = fwrite(&num_total_points_in_cluster, sizeof(uint32_t), 1, args.output_fp);
                if (ret != 1) {
                    std::cerr << "Error writing num_total_points_in_cluster for cluster " << cluster_data.id._id << " in level " << (uint32_t)level << " to output file.\n";
                    fclose(args.output_fp);
                    FatalAssert(false, LOG_TAG_BASIC, "Error writing num_total_points_in_cluster to output file!");
                    exit(EXIT_FAILURE);
                }
            }

            ret = fwrite(&cluster_data.offset, sizeof(uint32_t), 1, args.output_fp);
            if (ret != 1) {
                std::cerr << "Error writing cluster_vector_offset for cluster " << cluster_data.id._id << " in level " << (uint32_t)level << " to output file.\n";
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing cluster_vector_offset to output file!");
                exit(EXIT_FAILURE);
            }

            if constexpr (std::is_same<divftree::CTYPE, divftree::MVTYPE>::value) {
                ret = fwrite(cluster_data.data, sizeof(cluster_data.data), 1, args.output_fp);
                if (ret != 1) {
                    std::cerr << "Error writing centroid_vector for cluster " << cluster_data.id._id << " in level " << (uint32_t)level << " to output file.\n";
                    fclose(args.output_fp);
                    FatalAssert(false, LOG_TAG_BASIC, "Error writing centroid_vector to output file!");
                    exit(EXIT_FAILURE);
                }
            } else {
                for (uint16_t d = 0; d < DIMENSION; d++) {
                    divftree::CTYPE val_to_write = static_cast<divftree::CTYPE>(cluster_data.data[d]);
                    // printf(CTYPE_FMT "(" MVTYPE_FMT ")%s", val_to_write, cluster_data.data[d], (d == DIMENSION - 1) ? "]\n" : ", ");
                    ret = fwrite(&val_to_write, sizeof(divftree::CTYPE), 1, args.output_fp);
                    if (ret != 1) {
                        std::cerr << "Error writing centroid_vector for cluster " << cluster_data.id._id << " in level " << (uint32_t)level << " to output file.\n";
                        fclose(args.output_fp);
                        FatalAssert(false, LOG_TAG_BASIC, "Error writing centroid_vector to output file!");
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
    }

    // for (uint32_t i = 0; i < args.num_points_to_use; i++) {
        // std::cout << "\tWriting vector " << i << "/" << args.num_points_to_use << " with id {hash=" << data[i].id.vector_hash
        //           << ", val=" << data[i].id.value
        //           << "}, num_duplicates=" << data[i].num_duplicates
        //           << ", data=[";
        // for (uint16_t d = 0; d < DIMENSION; d++) {
        //     std::cout << (uint32_t)data[i].data[d];
        //     if (d < DIMENSION - 1) {
        //         std::cout << ", ";
        //     }
        // }
        // std::cout << "]\n";
    // }

    ret = fwrite(data, 1, sizeof(VectorData) * (size_t)args.num_points_to_use, args.output_fp);
    if (ret != sizeof(VectorData) * (size_t)args.num_points_to_use) {
        std::cerr << "Error writing vector data to output file.\n";
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error writing vector data to output file!");
        exit(EXIT_FAILURE);
    }
    fclose(args.output_fp);
}

void build_ivf_index(DataSet<VectorData>& data, Args& args) {
    std::cout << "Building IVF index using k-means clustering...\n" << std::flush;
    ClusterManager cluster_manager(1);

    if (args.algorithm == AlgorithmType::KMEANS) {
        kmeans_simple(data, args.num_points_to_use, args.kmeans_args.num_clusters, args.max_iterations,
                      args.num_threads, cluster_manager);
    } else if (args.algorithm == AlgorithmType::KMEANS_SAMPLED) {
        kmeans_sampled(data, args.num_points_to_use, args.kmeans_sampled_args.min_sample_size,
                       args.kmeans_sampled_args.num_point_per_cluster,
                       args.kmeans_sampled_args.num_clusters, args.max_iterations,
                       args.num_threads, cluster_manager);
    } else {
        FatalAssert(false, LOG_TAG_BASIC, "invalid algorithm type");
    }

    write_output(data.Data(), &cluster_manager, 1, args);
}

void build_capped_ivf_index(DataSet<VectorData>& data, Args& args) {
    std::cout << "Building IVF index with using capped k-means clustering...\n" << std::flush;
    ClusterManager cluster_manager(1);
    kmeans_capped(data, args.num_points_to_use, args.kmeans_capped_args.cluster_cap, args.max_iterations,
                  args.num_threads, cluster_manager, args.kmeans_capped_args.conf);
    write_output(data.Data(), &cluster_manager, 1, args);
}

void build_divftree_index(DataSet<VectorData>& data, Args& args) {
    std::cout << "Building DIVF-Tree index using capped k-means clustering...\n" << std::flush;
    std::vector<ClusterManager> cluster_managers;
    uint8_t cluster_level = 1;
    cluster_managers.emplace_back(cluster_level); /* leaf clusters */
    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Building level 1 clusters with leaf cap %u...", args.divftree_kmeans_capped_args.leaf_cap);
    kmeans_capped(data, args.num_points_to_use, args.divftree_kmeans_capped_args.leaf_cap, args.max_iterations,
                  args.num_threads, cluster_managers.back(), args.divftree_kmeans_capped_args.conf,
                  args.divftree_kmeans_capped_args.weighted_kmeans);
    while (cluster_managers.back().clusters.size() > args.divftree_kmeans_capped_args.internal_cap) {
        uint8_t data_level = cluster_managers.size();
        cluster_level = data_level + 1;
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Building level %hhu clusters with internal cap %u...", cluster_level, args.divftree_kmeans_capped_args.internal_cap);
        cluster_managers.emplace_back(cluster_level);
        DataSet<ClusterManager> centroid_dataset(cluster_managers[data_level - 1]);
        kmeans_capped(centroid_dataset, centroid_dataset.Size(), args.divftree_kmeans_capped_args.internal_cap,
                      args.max_iterations, args.num_threads, cluster_managers.back(), args.divftree_kmeans_capped_args.conf,
                      args.divftree_kmeans_capped_args.weighted_kmeans);
    }

    FatalAssert(cluster_managers.size() <= UINT8_MAX, LOG_TAG_BASIC, "Number of levels Exeeds MAX");

    write_output(data.Data(), cluster_managers.data(), (uint8_t)cluster_managers.size(), args);

}

int main(int argc, char* argv[]) {
    divftree::Thread st(100);
    st.InitDIVFThread(true);

    Args args;
    ParseArgs(argc, argv, args);
    args.print();

    std::cout << "Allocating enough memory for vector data...\n" << std::flush;
    VectorData* data = (VectorData*)mmap64(nullptr, (size_t)args.num_points_to_use * sizeof(VectorData),
                                           PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (data == MAP_FAILED) {
        std::cerr << "Error allocating memory for vector data. errno: " << errno << ", msg: " << strerror(errno) << "\n";
        fclose(args.input_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error allocating memory for vector data!");
        exit(EXIT_FAILURE);
    }

    std::cout << "Reading vector data from file...\n" << std::flush;
    size_t ret = fread(data, 1, (size_t)args.num_points_to_use * sizeof(VectorData), args.input_fp);
    fclose(args.input_fp);
    if (ret != (size_t)args.num_points_to_use * sizeof(VectorData)) {
        std::cerr << "Error reading data from file.\n";
        fclose(args.output_fp);
        munmap(data, (size_t)args.num_points_to_use * sizeof(VectorData));
        FatalAssert(false, LOG_TAG_BASIC, "Error reading data from file!");
        exit(EXIT_FAILURE);
    }

    std::cout << "Successfully read " << args.num_points_to_use << " vectors from file.\n" << std::flush;
    std::cout << "Some final checks on the data before clustering...\n" << std::flush;
    for (uint32_t i = 0; i < args.num_points_to_use; i++) {
        args.num_total_points_used += (data[i].num_duplicates + 1);
        // std::cout << "Vector " << i << ": id={hash=" << data[i].id.vector_hash << ", val=" << data[i].id.value << "}, num_duplicates=" << data[i].num_duplicates << "data=[";
        // for (uint16_t d = 0; d < args.dimension; d++) {
        //     std::cout << (uint32_t)data[i].data[d];
        //     if (d < args.dimension - 1) {
        //         std::cout << ", ";
        //     }
        // }
        // std::cout << "]\n";
    }

    FatalAssert(args.num_total_points_used >= args.num_points_to_use, LOG_TAG_BASIC,
                "Total number of points represented by the unique vectors (num_total_points_used) should be greater than or equal to num_points_to_use!");

    DataSet<VectorData> dataset(data, args.num_points_to_use);
    switch (args.algorithm) {
        case AlgorithmType::KMEANS:
        case AlgorithmType::KMEANS_SAMPLED:
            build_ivf_index(dataset, args);
            break;
        case AlgorithmType::KMEANS_CAPPED:
            build_capped_ivf_index(dataset, args);
            break;
        case AlgorithmType::KMEANS_HIERARCHICAL:
            build_divftree_index(dataset, args);
            break;
    }

    std::cout << "Clustering and index building completed successfully. freeing memory...\n" << std::flush;
    munmap(data, (size_t)args.num_points_to_use * sizeof(VectorData));
    st.DestroyDIVFThread();
}