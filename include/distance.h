#ifndef DISTANCE_H_
#define DISTANCE_H_

#include "common.h"
#include "vector_utils.h"

#include <type_traits>
#include <vector>

namespace divftree {

using SimilarityComparator = int (*)(const ANNVectorInfo&, const ANNVectorInfo&);

namespace L2 {

inline constexpr DTYPE Distance(const VTYPE* a, const VTYPE* b, uint16_t dim) {
    CHECK_NOT_NULLPTR(a, LOG_TAG_BASIC);
    CHECK_NOT_NULLPTR(b, LOG_TAG_BASIC);

    DTYPE dist = 0;
    for (size_t i = 0; i < dim; ++i) {
        const DTYPE abs = static_cast<DTYPE>(a[i]) - static_cast<DTYPE>(b[i]);
        dist += abs * abs;
    }
    return static_cast<DTYPE>(dist);
}

/* todo: A better method(compared to passing a pointer) to allow inlining for optimization */
inline constexpr int MoreSimilar(const DTYPE& a, const DTYPE& b) {
    return (a == b ? 0 : (a < b ? 1 : -1));
}

inline constexpr int MoreSimilarCmp(const ANNVectorInfo& a, const ANNVectorInfo& b) {
    return MoreSimilar(a.distance_to_query, b.distance_to_query);
}

inline constexpr int LessSimilarCmp(const ANNVectorInfo& a, const ANNVectorInfo& b) {
    return MoreSimilar(b.distance_to_query, a.distance_to_query);
}

inline bool ComputeCentroid(const VTYPE* vectors1, size_t size1, const VTYPE* vectors2, size_t size2, uint16_t dim,
                            VTYPE* centroid, MVTYPE* temp) {
    FatalAssert(size1 > 0, LOG_TAG_BASIC, "size cannot be 0");
    CHECK_NOT_NULLPTR(vectors1, LOG_TAG_BASIC);
    CHECK_NOT_NULLPTR(centroid, LOG_TAG_BASIC);
    CHECK_NOT_NULLPTR(temp, LOG_TAG_BASIC);
    FatalAssert((vectors2 == nullptr) == (size2 == 0), LOG_TAG_BASIC, "mismatch for second batch of vectors");

    memset(temp, 0, sizeof(MVTYPE) * dim);
    /* todo: use AVX for this operation? */
    for (size_t v = 0; v < size1; ++v) {
        for (uint16_t e = 0; e < dim; ++e) {
            temp[e] += static_cast<MVTYPE>(vectors1[(v * dim) + e]);
        }
    }

    for (size_t v = 0; v < size2; ++v) {
        for (uint16_t e = 0; e < dim; ++e) {
            temp[e] += static_cast<MVTYPE>(vectors2[(v * dim) + e]);
        }
    }

    for (uint16_t e = 0; e < dim; ++e) {
        centroid[e] = static_cast<VTYPE>(temp[e] / (size1 + size2));
    }

    return true;
}

inline bool ComputeCentroid(const Cluster& cluster, uint16_t block_size, uint16_t cluster_cap, bool is_leaf,
                            const VTYPE* vectors, size_t size, uint16_t dim, VTYPE* centroid, MVTYPE* temp) {
    uint16_t cluster_size = cluster.header.visible_size.load(std::memory_order_acquire);
    FatalAssert(cluster_size > 0, LOG_TAG_BASIC, "cluster_size cannot be 0");
    CHECK_NOT_NULLPTR(centroid, LOG_TAG_BASIC);
    CHECK_NOT_NULLPTR(temp, LOG_TAG_BASIC);
    FatalAssert((vectors == nullptr) == (size == 0), LOG_TAG_BASIC, "mismatch for second batch of vectors");
    const uint16_t num_blocks = cluster.NumBlocks(block_size, cluster_cap);
    FatalAssert(num_blocks == 1, LOG_TAG_NOT_IMPLEMENTED, "currently not possible with more than 1 block!");
    UNUSED_VARIABLE(num_blocks);
    uint16_t real_size = 0;
    const VTYPE* data = cluster.Data(0, is_leaf, block_size, cluster_cap, dim);
    const void* meta = cluster.MetaData(0, is_leaf, block_size, cluster_cap, dim);
    memset(temp, 0, sizeof(MVTYPE) * dim);
    /* todo: use AVX for this operation? */
    for (uint16_t offset = 0; offset < cluster_size; ++offset) {
        if (is_leaf) {
            const VectorMetaData* vmt = static_cast<const VectorMetaData*>(meta);
            if (vmt[offset].state.load(std::memory_order_acquire) != VECTOR_STATE_VALID) {
                continue;
            }
        } else {
            const CentroidMetaData* vmt = static_cast<const CentroidMetaData*>(meta);
            if (vmt[offset].state.load(std::memory_order_acquire) != VECTOR_STATE_VALID) {
                continue;
            }
        }

        for (uint16_t e = 0; e < dim; ++e) {
            temp[e] += static_cast<MVTYPE>(data[(offset * dim) + e]);
        }

        ++real_size;
    }

    for (size_t v = 0; v < size; ++v) {
        for (uint16_t e = 0; e < dim; ++e) {
            temp[e] += static_cast<MVTYPE>(vectors[(v * dim) + e]);
        }
    }

    real_size += size;
    if (real_size == 0) {
        return false;
    } else {
        for (uint16_t e = 0; e < dim; ++e) {
            centroid[e] = static_cast<VTYPE>(temp[e] / static_cast<MVTYPE>(real_size));
        }
    }
    return true;
}

inline void ComputeCentroids(const VTYPE* cluster_vectors, const VTYPE* batch_vectors,
                             uint16_t* cluster_offsets, uint16_t* batch_offsets,
                             uint16_t cluster_size, uint16_t batch_size,
                             uint16_t cluster_total_size,
                             uint16_t* cluster_valid_offsets, VTYPE* centroids,
                             uint16_t target_centroid_idx, uint16_t new_sibling_idx,
                             uint16_t* assignments, uint16_t* new_cluster_size,
                             uint16_t dim, MVTYPE* temp) {
    constexpr size_t NUM_CENTROIDS = 2;
    memset(temp, 0, sizeof(MVTYPE) * dim * NUM_CENTROIDS);
    new_cluster_size[target_centroid_idx] = 0;
    new_cluster_size[new_sibling_idx] = 0;
    // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BASIC,
    //          "Computing centroids for target centroid %hu and new sibling %hu",
    //          target_centroid_idx, new_sibling_idx);

    /* todo: use AVX for this operation? */
    for (uint16_t i = 0; i < cluster_size; ++i) {
        uint16_t c = assignments[cluster_offsets[i]];
        // String currentVectorLog = String("computing centroid %hu: vector[%u] from cluster -> offset = %u = [", c,
        //                                  cluster_offsets[i],
        //                                  cluster_valid_offsets[cluster_offsets[i]]);
        // String currentTmpLog = String("Current sum for centroid %hu: [", c);
        FatalAssert(c == target_centroid_idx || c == new_sibling_idx, LOG_TAG_BASIC,
                    "invalid centroid index assigned!");
        uint16_t c_idx = (c == target_centroid_idx ? 0 : 1);
        for (uint16_t e = 0; e < dim; ++e) {
            // currentVectorLog += String(VTYPE_FMT ", ",
                                    //    cluster_vectors[(cluster_valid_offsets[cluster_offsets[i]] * dim) + e]);
            temp[(c_idx * dim) + e] +=
                static_cast<MVTYPE>(cluster_vectors[(cluster_valid_offsets[cluster_offsets[i]] * dim) + e]);
            // currentTmpLog += String(MVTYPE_FMT ", ", temp[(c_idx * dim) + e]);
        }
        // currentVectorLog += String("]");
        // currentTmpLog += String("]");
        // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BASIC, "%s", currentVectorLog.ToCStr());
        // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BASIC, "%s", currentTmpLog.ToCStr());
        ++(new_cluster_size[c]);
    }

    for (size_t v = 0; v < batch_size; ++v) {
        uint16_t c = assignments[batch_offsets[v] + cluster_total_size];
        FatalAssert(c == target_centroid_idx || c == new_sibling_idx, LOG_TAG_BASIC,
                    "invalid centroid index assigned!");
        // String currentVectorLog = String("computing centroid %hu: vector[%u] from batch -> offset = %u = [", c,
        //                                  batch_offsets[v] + cluster_total_size,
        //                                  batch_offsets[v]);
        // String currentTmpLog = String("Current sum for centroid %hu: [", c);
        uint16_t c_idx = (c == target_centroid_idx ? 0 : 1);
        for (uint16_t e = 0; e < dim; ++e) {
            // currentVectorLog += String(VTYPE_FMT ", ",
            //                            batch_vectors[(batch_offsets[v] * dim) + e]);
            temp[(c_idx * dim) + e] += static_cast<MVTYPE>(batch_vectors[(batch_offsets[v] * dim) + e]);
            // currentTmpLog += String(MVTYPE_FMT ", ", temp[(c_idx * dim) + e]);
        }
        ++(new_cluster_size[c]);
        // currentVectorLog += String("]");
        // currentTmpLog += String("]");
        // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BASIC, "%s", currentVectorLog.ToCStr());
        // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BASIC, "%s", currentTmpLog.ToCStr());
    }

    // FatalAssert(new_cluster_size[target_centroid_idx] > 0, LOG_TAG_BASIC, "centroid size cannot be 0");
    // FatalAssert(new_cluster_size[new_sibling_idx] > 0, LOG_TAG_BASIC, "centroid size cannot be 0");
    // String centroidLog = String("centroid %hu updated to [", target_centroid_idx);
    // String siblingLog = String("centroid %hu updated to [", new_sibling_idx);
    for (uint16_t e = 0; e < dim; ++e) {
        centroids[(target_centroid_idx * dim) + e] =
            static_cast<VTYPE>(temp[e] / static_cast<MVTYPE>(new_cluster_size[target_centroid_idx]));
        centroids[(new_sibling_idx * dim) + e] =
            static_cast<VTYPE>(temp[dim + e] / static_cast<MVTYPE>(new_cluster_size[new_sibling_idx]));
        // centroidLog += String(VTYPE_FMT ", ", centroids[(target_centroid_idx * dim) + e]);
        // siblingLog += String(VTYPE_FMT ", ", centroids[(new_sibling_idx * dim) + e]);
    }
    // centroidLog += String("]");
    // siblingLog += String("]");
    // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BASIC, "%s", centroidLog.ToCStr());
    // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BASIC, "%s", siblingLog.ToCStr());
}

};

inline bool ComputeCentroid(const VTYPE* vectors1, size_t size1, const VTYPE* vectors2, size_t size2,
                              uint16_t dim, DistanceType distanceAlg, VTYPE* centroid, MVTYPE* temp) {
    switch (distanceAlg) {
    case DistanceType::L2:
        return L2::ComputeCentroid(vectors1, size1, vectors2, size2, dim, centroid, temp);
    default:
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC,
             "ComputeCentroid: Invalid distance type: %s", DISTANCE_TYPE_NAME[(int8_t)distanceAlg]);
    }
    return false; // Return an empty vector if the distance type is invalid
}

/* cluster should be locked in S or X mode! This will only caount Valid vectors and it will use visible size not reserved size!*/
inline bool ComputeCentroid(const Cluster& cluster, uint16_t block_size, uint16_t cluster_cap, bool is_leaf,
                            const VTYPE* vectors, size_t size, uint16_t dim,
                            DistanceType distanceAlg, VTYPE* centroid, MVTYPE* temp) {
    switch (distanceAlg) {
    case DistanceType::L2:
        return L2::ComputeCentroid(cluster, block_size, cluster_cap, is_leaf, vectors, size, dim, centroid, temp);
    default:
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC,
             "ComputeCentroid: Invalid distance type: %s", DISTANCE_TYPE_NAME[(int8_t)distanceAlg]);
    }
    return false; // Return an empty vector if the distance type is invalid
}

inline void ComputeCentroids(const VTYPE* cluster_vectors, const VTYPE* batch_vectors,
                             uint16_t* cluster_offsets, uint16_t* batch_offsets,
                             uint16_t cluster_size, uint16_t batch_size,
                             uint16_t cluster_total_size,
                             uint16_t* cluster_valid_offsets, VTYPE* centroids,
                             uint16_t target_centroid_idx, uint16_t new_sibling_idx,
                             uint16_t* assignments, uint16_t* new_cluster_size,
                             uint16_t dim, MVTYPE* temp, DistanceType distanceAlg) {
    switch (distanceAlg) {
    case DistanceType::L2:
        L2::ComputeCentroids(cluster_vectors, batch_vectors,
                             cluster_offsets, batch_offsets,
                             cluster_size, batch_size,
                             cluster_total_size,
                             cluster_valid_offsets, centroids,
                             target_centroid_idx, new_sibling_idx,
                             assignments, new_cluster_size,
                             dim, temp);
        break;
    default:
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC,
             "ComputeCentroid: Invalid distance type: %s", DISTANCE_TYPE_NAME[(int8_t)distanceAlg]);
    }
}


inline constexpr DTYPE Distance(const VTYPE* a, const VTYPE* b, uint16_t dim, DistanceType distanceAlg) {
    switch (distanceAlg) {
    case DistanceType::L2:
        return L2::Distance(a, b, dim);
    default:
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC,
             "Distance: Invalid distance type: %s", DISTANCE_TYPE_NAME[(int8_t)distanceAlg]);
    }
    return 0; // Return 0 if the distance type is invalid
}

inline constexpr int MoreSimilar(const DTYPE& a, const DTYPE& b, DistanceType distanceAlg) {
    switch (distanceAlg) {
    case DistanceType::L2:
        return L2::MoreSimilar(a, b);
    default:
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC,
             "MoreSimilar: Invalid distance type: %s", DISTANCE_TYPE_NAME[(int8_t)distanceAlg]);
    }
    return false; // Return false if the distance type is invalid
}

inline constexpr SimilarityComparator GetDistancePairSimilarityComparator(DistanceType distanceAlg, bool reverse) {
    switch (distanceAlg) {
    case DistanceType::L2:
        return (reverse ? L2::MoreSimilarCmp : L2::LessSimilarCmp);
    default:
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC,
                "MoreSimilarVPair: Invalid distance type: %s", DISTANCE_TYPE_NAME[(int8_t)distanceAlg]);
    }
    return nullptr; // Return nullptr if the distance type is invalid
}

};

#endif