#ifndef DIVFTREE_INTERFACE_H_
#define DIVFTREE_INTERFACE_H_

#include "common.h"
#include "vector_utils.h"
#include "distance.h"

#include "utils/thread.h"
#include "utils/sorted_list.h"

#include <unordered_set>

namespace divftree {

/* in functions such as migrate we may lock all clusters in a parent cluster + some other clusters */
constexpr uint16_t MAX_CLUSTER_SIZE = UINT16_MAX - 8;
constexpr uint16_t MAX_VECTOR_DIMENSION = 512;

class DIVFTreeInterface;

struct DIVFTreeVertexAttributes {
    VectorID centroid_id;
    Version version;
    uint16_t min_size; // todo: do we need this at compute nodes?
    uint16_t cap; // todo: do we need this at compute nodes?
    uint16_t block_size;

    DIVFTreeInterface *index;

    DIVFTreeVertexAttributes() : centroid_id{INVALID_VECTOR_ID}, version{0}, min_size{0}, cap{0}, block_size{0},
                                 index{nullptr} {} /* todo: to be remvoed */
    DIVFTreeVertexAttributes(VectorID id, Version ver, uint16_t min,
                             uint16_t max, uint16_t blck_size, DIVFTreeInterface* idx) :
                                centroid_id{id}, version{ver}, min_size{min}, cap{max}, block_size{blck_size},
                                index{idx} {}
    String ToString() const {
        return String("{"
            "centroid_id:" VECTORID_LOG_FMT ", "
            "version:%u, "
            "size:[%hu, %hu], "
            "block-size:%hu, "
            "index:%p"
        "}", VECTORID_LOG(centroid_id), version, min_size, cap, block_size, index);
    }
};

struct DIVFTreeAttributes {
    ClusteringType clusteringAlg;
    DistanceType distanceAlg;
    SimilarityComparator similarityComparator;
    SimilarityComparator reverseSimilarityComparator;

    uint16_t leaf_min_size;
    uint16_t leaf_max_size;
    uint16_t internal_min_size;
    uint16_t internal_max_size;

    bool use_block_bytes;
    size_t leaf_blck_bytes;
    uint16_t leaf_blck_size;
    size_t internal_blck_bytes;
    uint16_t internal_blck_size;

    uint16_t split_internal;
    uint16_t split_leaf;

    uint16_t dimension;

    size_t num_searchers;
    size_t num_migrators;
    size_t num_mergers;
    size_t num_compactors;

    uint32_t migration_check_triger_rate;
    uint32_t migration_check_triger_single_rate;

    uint32_t random_base_perc;

    size_t memory_pool_size_gb;

    bool collect_stats;

    String ToString() const {
        return String("{"
            "clusteringAlg:%s, "
            "distanceAlg:%s, "
            "leaf_size:[%hu, %hu], "
            "internal_size:[%hu, %hu], "
            "use_block_bytes:%s, "
            "leaf_blck_bytes:%lu"
            "leaf_blck_size:%hu, "
            "internal_blck_bytes:%lu "
            "internal_blck_size:%hu, "
            "split_internal:%hu, "
            "split_leaf:%hu, "
            "dimension:%hu, "
            "num_searchers:%lu, "
            "num_migrators:%lu, "
            "num_mergers:%lu, "
            "num_compactors:%lu, "
            "migration_check_triger_rate:%u, "
            "migration_check_triger_single_rate:%u, "
            "random_base_perc:%u, "
            "memory_pool_size:%luGB, "
            "collect_stats:%s"
        "}", CLUSTERING_TYPE_NAME[(int8_t)clusteringAlg], DISTANCE_TYPE_NAME[(int8_t)distanceAlg],
        leaf_min_size, leaf_max_size, internal_min_size, internal_max_size, (use_block_bytes ? "T" : "F"),
        leaf_blck_bytes, leaf_blck_size, internal_blck_bytes, internal_blck_size, split_internal, split_leaf,
        dimension, num_searchers, num_migrators, num_mergers, num_compactors, migration_check_triger_rate,
        migration_check_triger_single_rate, random_base_perc, memory_pool_size_gb, (collect_stats ? "T" : "F"));
    }
};

enum class DuplicateState : uint8_t {
    UNIQUE = 0,
    DUPLICATE_NOT_INSERTED = 1,
    DUPLICATE_INSERTED = 2
};

class DIVFTreeVertexInterface {
public:
    DIVFTreeVertexInterface() = default;
    DIVFTreeVertexInterface(DIVFTreeVertexInterface&) = delete;
    DIVFTreeVertexInterface(const DIVFTreeVertexInterface&) = delete;
    DIVFTreeVertexInterface(DIVFTreeVertexInterface&&) = delete;
    virtual ~DIVFTreeVertexInterface() = default;

    virtual void Unpin() = 0;
    virtual void MarkForRecycle(uint64_t pinCount) = 0;

    virtual RetStatus BatchInsert(const ConstVectorBatch& batch, DuplicateState* dup_state,
                                  uint16_t num_duplicates_found, uint16_t checked_offset,
                                  uint16_t marked_for_update = INVALID_OFFSET) = 0;
    virtual RetStatus ChangeVectorState(VectorID target, uint16_t targetOffset,
                                        VectorState& expectedState, VectorState finalState,
                                        Version* version = nullptr) = 0;
    virtual RetStatus ChangeVectorState(Address meta, VectorState& expectedState, VectorState finalState,
                                        Version* version = nullptr) = 0;
    virtual void Search(const VTYPE* query, size_t k,
                        SortedList<ANNVectorInfo, SimilarityComparator>* neighbours,
                        std::unordered_set<std::pair<VectorID, Version>, VectorIDVersionPairHash>& seen) = 0;
    virtual void Exists(const ConstVectorBatch batch, uint16_t expected_size, DuplicateState* dup_state,
                        uint16_t& num_found, DuplicateState set_if_found,
                        uint16_t checked_for_duplicates = UINT16_MAX) = 0;

    virtual VectorID CentroidID() const = 0;
    virtual Version VertexVersion() const = 0;
    virtual Cluster& GetCluster() = 0;
    virtual const DIVFTreeVertexAttributes& GetAttributes() const = 0;

    virtual String ToString(bool detailed = false) const = 0;
};

class DIVFTreeInterface {
public:
    DIVFTreeInterface() = default;
    virtual ~DIVFTreeInterface() = default;

    virtual RetStatus Insert(const VTYPE* vec, VectorID& vec_id, uint8_t search_span,
                             bool create_completion_notification = false) = 0;
    virtual RetStatus Delete(VectorID vec_id, bool create_completion_notification = false) = 0;

    virtual RetStatus ApproximateKNearestNeighbours(const VTYPE* query, size_t k,
                                                    uint8_t internal_node_search_span, uint8_t leaf_node_search_span,
                                                    SortType sort_type, std::vector<ANNVectorInfo>& neighbours) = 0;

    virtual size_t Size() const = 0;
    virtual const DIVFTreeAttributes& GetAttributes() const = 0;

    virtual void EndBGThreads() = 0;

    virtual String GetStatistics(std::string title_extention = "", bool clear_stats = false) = 0;
    virtual void StartStatsCollection() = 0;
    virtual void StopStatsCollection() = 0;
    virtual void ClearStats() = 0;

    // virtual String ToString() = 0;
};

};

#endif