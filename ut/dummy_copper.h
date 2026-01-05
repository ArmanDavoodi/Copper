#ifndef DIVFTREE_H_
#define DIVFTREE_H_

#include "common.h"
#include "vector_utils.h"

#include "interface/divftree.h"

/* Needs distance.h and buffer.h */

namespace divftree {

class DIVFTreeVertex : public DIVFTreeVertexInterface {
public:
    DIVFTreeVertex(VectorID id, DIVFTreeVertexAttributes attr) : _centroid_id(id), _parent_id(INVALID_VECTOR_ID),
        _clusteringAlg(attr.core.clusteringAlg), _distanceAlg(attr.core.distanceAlg), _min_size(attr.min_size),
        _similarityComparator(attr.similarityComparator),
        _reverseSimilarityComparator(attr.reverseSimilarityComparator),
        _cluster(attr.core.dimension, attr.max_size) {}

    ~DIVFTreeVertex() = default;
    RetStatus AssignParent(VectorID parent_id) override {
        _parent_id = parent_id;
        return RetStatus::Success();
    }

    Address Insert(const Vector& vec, VectorID vec_id) override {
        return _cluster.Insert(vec, vec_id);
    }

    VectorUpdate MigrateLastVectorTo(DIVFTreeVertexInterface* _dest) override {
        UNUSED_VARIABLE(_dest);
        return VectorUpdate{INVALID_VECTOR_ID, INVALID_ADDRESS}; // Placeholder for actual implementation
    }

    RetStatus Search(const Vector& query, size_t k,
                     std::vector<std::pair<VectorID, DTYPE>>& neighbours) override {
        UNUSED_VARIABLE(query);
        UNUSED_VARIABLE(k);
        UNUSED_VARIABLE(neighbours);
        return RetStatus::Success(); // Placeholder for actual implementation
    }

    VectorID CentroidID() const override {
        return _centroid_id;
    }

    VectorID ParentID() const override {
        return _parent_id;
    }

    uint16_t Size() const override {
        return _cluster.Size();
    }

    bool IsFull() const override {
        return _cluster.Size() >= _cluster.Capacity();
    }

    bool IsAlmostEmpty() const override {
        return _cluster.Size() <= _min_size;
    }

    bool Contains(VectorID id) const override {
        return _cluster.Contains(id);
    }

    bool IsLeaf() const override {
        return _centroid_id.IsLeaf();
    }

    uint8_t Level() const override {
        return _centroid_id.Level();
    }

    Vector ComputeCurrentCentroid() const override {
        return Vector(); // Placeholder for actual centroid computation
    }

    uint16_t MinSize() const override {
        return _min_size;
    }

    uint16_t MaxSize() const override {
        return _cluster.Capacity();
    }

    uint16_t VectorDimension() const override {
        return _cluster.Dimension();
    }

    /* todo: A better method(compared to function pointer) to allow inlining for optimization */
    inline SimilarityComparator GetSimilarityComparator(bool reverese) const override {
        CHECK_VERTEX_SELF_IS_VALID(LOG_TAG_DIVFTREE_VERTEX, false);
        return (reverese ? _reverseSimilarityComparator : _similarityComparator);
    }

    DTYPE Distance(const Vector& a, const Vector& b) const override {
        UNUSED_VARIABLE(a);
        UNUSED_VARIABLE(b);
        return 0.0; // Placeholder for actual distance computation
    }

    String BucketToString() const override {
        return _cluster.ToString(); // Return a string representation of the bucket
    }

    static size_t Bytes(uint16_t dim, uint16_t capacity) {
        return sizeof(DIVFTreeVertex) + Cluster::DataBytes(dim, capacity);
    }

protected:
    const VectorID _centroid_id;
    VectorID _parent_id;
    const ClusteringType _clusteringAlg;
    const DistanceType _distanceAlg;
    const uint16_t _min_size;
    const SimilarityComparator _similarityComparator;
    const SimilarityComparator _reverseSimilarityComparator;
    Cluster _cluster;

TESTABLE;
};

class DIVFTree : public DIVFTreeInterface {
public:

    DIVFTree(DIVFTreeAttributes attr) : core_attr(attr.core), leaf_min_size(attr.leaf_min_size),
                                         leaf_max_size(attr.leaf_max_size),
                                         internal_min_size(attr.internal_min_size),
                                         internal_max_size(attr.internal_max_size),
                                         split_internal(attr.split_internal), split_leaf(attr.split_leaf),
                                         _similarityComparator(
                                            GetDistancePairSimilarityComparator(attr.core.distanceAlg, false)),
                                         _reverseSimilarityComparator(
                                            GetDistancePairSimilarityComparator(attr.core.distanceAlg, true)),
                                         _size(0), _root(INVALID_VECTOR_ID), _levels(2) {
        RetStatus rs = RetStatus::Success();
        rs = _bufmgr.Init();
        FatalAssert(rs.IsOK(), LOG_TAG_DIVFTREE, "Failed to init buffer manager.");
        _root = _bufmgr.RecordRoot();
        FatalAssert(_root.IsValid(), LOG_TAG_DIVFTREE, "Invalid root ID: " VECTORID_LOG_FMT, VECTORID_LOG(_root));
        FatalAssert(_root.IsCentroid(), LOG_TAG_DIVFTREE, "root should be a centroid: "
                    VECTORID_LOG_FMT, VECTORID_LOG(_root));
        FatalAssert(_root.IsLeaf(), LOG_TAG_DIVFTREE, "first root should be a leaf: "
                    VECTORID_LOG_FMT, VECTORID_LOG(_root));

        rs = _bufmgr.UpdateClusterAddress(_root, CreateNewVertex(_root));
        FatalAssert(rs.IsOK(), LOG_TAG_DIVFTREE, "Failed to update cluster address for root: "
                    VECTORID_LOG_FMT, VECTORID_LOG(_root));
        _levels = 2;
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_DIVFTREE,
             "Init DIVFTree Index End: rootID= " VECTORID_LOG_FMT ", _levels = %hhu, attr=%s",
             VECTORID_LOG(_root), _levels, attr.ToString().ToCStr());
    }

    ~DIVFTree() override {
        RetStatus rs = RetStatus::Success();
        rs = _bufmgr.Shutdown();
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_DIVFTREE, "Shutdown DIVFTree Index End: rs=%s", rs.Msg());
    }

    RetStatus Insert(const Vector& vec, VectorID& vec_id, uint16_t vertex_per_layer) override {
        UNUSED_VARIABLE(vec);
        UNUSED_VARIABLE(vec_id);
        UNUSED_VARIABLE(vertex_per_layer);
        return RetStatus::Success(); // Placeholder for actual insertion logic
    }

    RetStatus Delete(VectorID vec_id) override {
        UNUSED_VARIABLE(vec_id);
        return RetStatus::Success(); // Placeholder for actual deletion logic
    }

    RetStatus ApproximateKNearestNeighbours(const Vector& query, size_t k,
                                                    uint16_t _internal_k, uint16_t _leaf_k,
                                                    std::vector<std::pair<VectorID, DTYPE>>& neighbours,
                                                    bool sort = true,
                                                    bool sort_from_more_similar_to_less = true) override {
        UNUSED_VARIABLE(query);
        UNUSED_VARIABLE(k);
        UNUSED_VARIABLE(_internal_k);
        UNUSED_VARIABLE(_leaf_k);
        UNUSED_VARIABLE(neighbours);
        UNUSED_VARIABLE(sort);
        UNUSED_VARIABLE(sort_from_more_similar_to_less);
        return RetStatus::Success(); // Placeholder for actual search logic
    }

    size_t Size() const override {
        return _size; // Return the size of the index
    }

    DTYPE Distance(const Vector& a, const Vector& b) const override {
        UNUSED_VARIABLE(a);
        UNUSED_VARIABLE(b);
        return 0.0;
    }

    size_t Bytes(bool is_internal_vertex) const override {
        return DIVFTreeVertex::Bytes(core_attr.dimension, is_internal_vertex ? internal_max_size : leaf_max_size);
    }

    inline String ToString() override {
        return String();
    }
protected:
    const DIVFTreeCoreAttributes core_attr;
    const uint16_t leaf_min_size;
    const uint16_t leaf_max_size;
    const uint16_t internal_min_size;
    const uint16_t internal_max_size;
    const uint16_t split_internal;
    const uint16_t split_leaf;
    const SimilarityComparator _similarityComparator;
    const SimilarityComparator _reverseSimilarityComparator;
    size_t _size;
    BufferManager _bufmgr;
    VectorID _root;
    uint64_t _levels;

    RetStatus SearchVertexs(const Vector& query,
                          const std::vector<std::pair<VectorID, DTYPE>>& upper_layer,
                          std::vector<std::pair<VectorID, DTYPE>>& lower_layer, size_t n) override {
        UNUSED_VARIABLE(query);
        UNUSED_VARIABLE(upper_layer);
        UNUSED_VARIABLE(lower_layer);
        UNUSED_VARIABLE(n);
        return RetStatus::Success(); // Placeholder for actual search logic
    }

    VectorID RecordInto(const Vector& vec, DIVFTreeVertexInterface* container_vertex,
                        DIVFTreeVertexInterface* vertex = nullptr) override {
        UNUSED_VARIABLE(vec);
        UNUSED_VARIABLE(container_vertex);
        UNUSED_VARIABLE(vertex);
        return INVALID_VECTOR_ID; // Placeholder for actual record logic
    }

    RetStatus ExpandTree(DIVFTreeVertexInterface* root, const Vector& centroid) override {
        UNUSED_VARIABLE(root);
        UNUSED_VARIABLE(centroid);
        return RetStatus::Success(); // Placeholder for actual tree expansion logic
    }

    size_t FindClosestCluster(const std::vector<DIVFTreeVertexInterface*>& candidates,
                              const Vector& vec) override {
        UNUSED_VARIABLE(candidates);
        UNUSED_VARIABLE(vec);
        return 0; // Placeholder for actual cluster finding logic
    }

    RetStatus Split(std::vector<DIVFTreeVertexInterface*>& candidates, size_t vertex_idx) override {
        UNUSED_VARIABLE(candidates);
        UNUSED_VARIABLE(vertex_idx);
        return RetStatus::Success(); // Placeholder for actual split logic
    }

    RetStatus Split(DIVFTreeVertexInterface* leaf) override {
        UNUSED_VARIABLE(leaf);
        return RetStatus::Success(); // Placeholder for actual split logic
    }

    DIVFTreeVertexInterface* CreateNewVertex(VectorID id) override {
        const bool is_internal_vertex = id.IsInternalVertex();
        DIVFTreeVertex* new_vertex = static_cast<DIVFTreeVertex*>(malloc(Bytes(is_internal_vertex)));
        CHECK_NOT_NULLPTR(new_vertex, LOG_TAG_DIVFTREE_VERTEX);

        DIVFTreeVertexAttributes attr;
        attr.core = core_attr;
        attr.similarityComparator = _similarityComparator;
        attr.reverseSimilarityComparator = _reverseSimilarityComparator;
        if (is_internal_vertex) {
            attr.max_size = internal_max_size;
            attr.min_size = internal_min_size;
        }
        else {
            attr.max_size = leaf_max_size;
            attr.min_size = leaf_min_size;
        }

        new (new_vertex) DIVFTreeVertex(id, attr);
        CHECK_VERTEX_IS_VALID(new_vertex, LOG_TAG_DIVFTREE_VERTEX, false);
        return new_vertex;
    }

    RetStatus Cluster(std::vector<DIVFTreeVertexInterface*>& vertices, size_t target_vertex_index,
                      std::vector<Vector>& centroids, uint16_t split_into) override {
        UNUSED_VARIABLE(vertices);
        UNUSED_VARIABLE(target_vertex_index);
        UNUSED_VARIABLE(centroids);
        UNUSED_VARIABLE(split_into);
        return RetStatus::Success(); // Placeholder for actual clustering logic
    }

TESTABLE;
};

};

#endif