#ifndef COPPER_H_
#define COPPER_H_

#include "common.h"
#include "vector_utils.h"
#include "buffer.h"

#include "queue"

namespace copper {

template <typename T, uint16_t min_size, uint16_t max_size>
class Copper_Node {
static_assert(min_size > 0);
static_assert(max_size > min_size);
public:

    inline RetStatus Insert(const Vector& vec, VectorID vec_id) {
        assert(_bucket._size < max_size);
        _bucket.Insert<T>(vec, vec_id, _dimention);
        return RetStatus::Success();
    }

    inline RetStatus Delete(VectorID vec_id, VectorID& swapped_vec_id, Vector& swapped_vec) {
        assert(_bucket._size > min_size);
        _bucket.Delete<T>(vec_id, _dimention, swapped_vec_id, swapped_vec);
        return RetStatus::Success();
    }

    inline RetStatus KNearestNeighbours(const Vector& query, size_t k, 
            std::priority_queue<std::pair<double, VectorID>, 
            std::vector<std::pair<double, VectorID>>, 
            Similarity<T>>& neighbours) const {

        assert(k > 0);
        assert(_bucket._size >= min_size);
        assert(_bucket._size <= max_size);
        assert(neighbours.size() < k);

        for (uint16_t i = 0; i < _bucket._size; ++i) {
            double distance = (*_dist)(query, _bucket.Get_Vector<T>(i, _dimention));
            neighbours.push({distance, _bucket.Get_VectorID(i)});
            if (neighbours.size() == k) {
                break;
            }
        }

        return RetStatus::Success();
    }
 
    inline uint16_t Size() const {
        return _bucket._size;
    }

    inline bool Is_Leaf() {
        return _centroid_id.level == 1;
    }

    inline bool Is_Full() {
        return _bucket._size == max_size;
    }

    inline bool Is_Almost_Empty() {
        return _bucket._size == min_size;
    }

    inline VectorID Find_Nearest(const Vector& query) const {
        double best_dist = (*_dist)(query, _bucket.beg, _dimention);
        VectorID best_vect = _bucket._ids[0];
        
        for (uint16_t i = 1; i < _bucket._size; ++i) {
            Vector next_vec = (T*)(_bucket.beg) + (i * _dimention);
            double new_dist = (*_dist)(query, next_vec, _dimention);
            if ((*_dist)(new_dist, best_dist)) {
                best_dist = new_dist;
                best_vect = _bucket._ids[i];
            }
        }

        return best_vect;
    }

    inline void* Get_Parent_Node() {
        return parent;
    }

protected:

    Distance<T>* _dist;
    uint16_t _dimention;
    VectorID _centroid_id;
    Vector _centroid;
    VectorSet _bucket;
    void* parent;

// friend class VectorIndex;
};

template <typename T, uint16_t ki_min, uint16_t ki_max, uint16_t kl_min, uint16_t kl_max>
class VectorIndex {
static_assert(ki_min > 0);
static_assert(ki_max > ki_min);
static_assert(kl_min > 0);
static_assert(kl_max > kl_min);

protected:
    typedef Copper_Node<T, ki_min, ki_max> Internal_Node;
    typedef Copper_Node<T, kl_min, kl_max> Leaf_Node;

    Distance<T>* _dist;
    uint16_t _dimention;
    size_t _size;
    Buffer_Manager<T, ki_min, ki_max, kl_min, kl_max> _bufmgr;
    VectorID _root;
    std::vector<VectorID> _last_id_per_level;
    uint16_t _internal_search;
    uint16_t _leaf_search;
    uint16_t _vector_search;
    uint8_t _levels;

    inline Leaf_Node* Find_Leaf(const Vector& query) const {
        AssertFatal(_root != INVALID_VECTOR_ID, LOG_TAG_DEFAULT, "Invalid root ID.");
        AssertFatal(_root.Is_Centroid(), LOG_TAG_DEFAULT, "Invalid root ID -> root should be a centroid.");
        AssertFatal(_levels > 1, LOG_TAG_DEFAULT, "Height of the tree should be at least two but is %hhu.", _levels);

        if (_root.Is_Leaf()) {
            return _bufmgr.Get_Leaf(_root);
        }
        VectorID next = _root;

        while (!next.Is_Leaf()) {
            Internal_Node* node = _bufmgr.Get_Node(next);
            AssertFatal(node != nullptr, LOG_TAG_DEFAULT, "Invisible Node with id %lu.", next._id);
            next = node->Find_Nearest(query);
        }

        return _bufmgr.Get_Leaf(next);
    }

    inline VectorID Next_ID(uint8_t level) {
        AssertFatal(_levels > 1, LOG_TAG_DEFAULT, "Height of the tree should be at least two but is %hhu.", _levels);
        AssertFatal(level < _levels, LOG_TAG_DEFAULT, "Input level(%hhu) should be less than height of the tree(%hhu)."
                , level, _levels);

        _last_id_per_level[level] = _last_id_per_level[level].Get_Next_ID();
        return _last_id_per_level[level];
    }
public:

    inline RetStatus Insert(const Vector& vec, VectorID& vec_id) {
        vec_id = Next_ID(0);
        Leaf_Node* leaf = Find_Leaf(vec);
        RetStatus rc = RetStatus::Success();
        AssertFatal(leaf != nullptr, LOG_TAG_DEFAULT, "Leaf not found.");

        if (leaf.Is_Full()) {
            // todo split
            CLOG(LOG_LEVEL_PANIC, LOG_TAG_NOT_IMPLEMENTED, "Split is not implemented.");
        }

        rc = leaf->Insert(vec, vec_id);
        if (rc.Is_OK()) {
            ++_size;
        }

        return rc;
    }

    inline RetStatus Delete(VectorID vec_id) {
        AssertFatal(_root != INVALID_VECTOR_ID, LOG_TAG_DEFAULT, "Invalid root ID.");
        AssertFatal(_root.Is_Centroid(), LOG_TAG_DEFAULT, "Invalid root ID -> root should be a centroid.");
        AssertFatal(vec_id != INVALID_VECTOR_ID, LOG_TAG_DEFAULT, "Invalid input vector id.");
        AssertFatal(!vec_id.Is_Centroid(), LOG_TAG_DEFAULT, "Invalid input vector id -> vector should not be a centroid.");
        AssertFatal(_levels > 1, LOG_TAG_DEFAULT, "Height of the tree should be at least two but is %hhu.", _levels);

        RetStatus rc = RetStatus::Success();
        Leaf_Node* leaf = _bufmgr->Get_Container_Leaf(vec_id);
        AssertFatal(leaf != nullptr, LOG_TAG_DEFAULT, "Container leaf not found.");

        if (leaf->Is_Almost_Empty()) {
            // todo handle merge
            CLOG(LOG_LEVEL_PANIC, LOG_TAG_NOT_IMPLEMENTED, "Merge is not implemented.");
        }

        Vector swapped_vector = INVALID_VECTOR;
        VectorID swapped_vector_id = INVALID_VECTOR_ID;

        rc = leaf->Delete(vec_id, swapped_vector_id, swapped_vector);
        if (!rc.Is_OK()) {
            CLOG(LOG_LEVEL_PANIC, LOG_TAG_NOT_IMPLEMENTED, "Recovery in case of delete failure not implemented.");
        }

        // todo handle the swapped vector
        CLOG(LOG_LEVEL_ERROR, LOG_TAG_NOT_IMPLEMENTED, "Handling swapped vectors is not implemented.");
    }

    inline RetStatus KNearestNeighbours(const Vector& query, size_t k, std::vector<VectorID>& neighbours) const {
        AssertFatal(_root != INVALID_VECTOR_ID, LOG_TAG_DEFAULT, "Invalid root ID.");
        AssertFatal(_root.Is_Centroid(), LOG_TAG_DEFAULT, "Invalid root ID -> root should be a centroid.");
        AssertFatal(query != INVALID_VECTOR, LOG_TAG_DEFAULT, "Invalid query vector.");
        AssertFatal(k > 0, LOG_TAG_DEFAULT, "Number of neighbours cannot be 0.");
        AssertFatal(_levels > 1, LOG_TAG_DEFAULT, "Height of the tree should be at least two but is %hhu.", _levels);

        std::priority_queue<std::pair<double, VectorID>
            , std::vector<std::pair<double, VectorID>>, Similarity<T>> 
                heap_stack{_dist}, heap_store{_dist};
        RetStatus rc = RetStatus::Success();

        if (_root.Is_Leaf()) {
            Leaf_Node* root = _bufmgr.Get_Leaf(_root);
            AssertFatal(root != nullptr, LOG_TAG_DEFAULT, "Root vector is null.");

            rc = root->KNearestNeighbours(query, _vector_search, heap_store);
            AssertFatal(rc.Is_OK(), LOG_TAG_DEFAULT, "ANN search failed at leaf root with err(%s).", rc.Msg());
            goto Neighbour_Extraction;
        }

        VectorID node_id = _root;
        Internal_Node* node = _bufmgr.Get_Node(node_id);
        AssertFatal(node != nullptr, LOG_TAG_DEFAULT, "Failed to get node with id %lu.", node_id._id);
        heap_stack.push({(*_dist)(query, node, _dimention), node_id});
        uint8_t level = node_id._level;

        while (level > 1) {
            uint16_t search_threshold = (level > 2 ? _internal_search : _leaf_search);

            while (!heap_stack.empty() && heap_store.size() < search_threshold) {
                std::pair<double, VectorID> dist_id_pair = heap_stack.top();
                heap_stack.pop();
                AssertFatal(level == dist_id_pair.second._level, LOG_TAG_DEFAULT, 
                    "Level mismatch detected: level(%hhu) != dist_id_pair.level(%hhu).",
                    level, dist_id_pair.second._level);

                AssertFatal(dist_id_pair.second != INVALID_VECTOR_ID, LOG_TAG_DEFAULT,
                            "Invalid vector id at top of the stack.");

                node = _bufmgr.Get_Node(dist_id_pair.second);
                AssertFatal(node != nullptr, LOG_TAG_DEFAULT, "Failed to get node with id %lu.",
                            dist_id_pair.second);
                rc = node->KNearestNeighbours(query, search_threshold, heap_store);
                AssertFatal(rc.Is_OK(), LOG_TAG_DEFAULT, "ANN search failed at node(%lu) with err(%s).", 
                            dist_id_pair.second, rc.Msg());
            }

            std::swap(heap_stack, heap_store);
            --level;
        }
        
        AssertFatal(!heap_stack.empty(), LOG_TAG_DEFAULT, "Heap stack should not be empty.");
        
        AssertFatal(heap_stack.top().second._level == 1, LOG_TAG_DEFAULT, 
                    "node level of the current stack top(%hhu) is not 1", heap_stack.top().second._level);
        AssertFatal(level == 1, LOG_TAG_DEFAULT, 
            "level(%hhu) is not 1", level);

        while (!heap_stack.empty() && heap_store.size() < _vector_search) {
            std::pair<double, VectorID> dist_id_pair = heap_stack.top();
            heap_stack.pop();
            AssertFatal(level == dist_id_pair.second._level, LOG_TAG_DEFAULT, 
                    "Level mismatch detected: level(%hhu) != dist_id_pair.level(%hhu).",
                    level, dist_id_pair.second._level);

            AssertFatal(dist_id_pair.second != INVALID_VECTOR_ID, LOG_TAG_DEFAULT,
                "Invalid vector id at top of the stack.");

            node = _bufmgr.Get_Leaf(dist_id_pair.second);
            AssertFatal(node != nullptr, LOG_TAG_DEFAULT, "Failed to get node with id %lu.",
                            dist_id_pair.second);
            rc = node->KNearestNeighbours(query, _vector_search, heap_store);
            AssertFatal(rc.Is_OK(), LOG_TAG_DEFAULT, "ANN search failed at node(%lu) with err(%s).", 
                            dist_id_pair.second, rc.Msg());
        }
        
        Neighbour_Extraction:
        
        neighbours.clear();
        size_t num_neigbours = std::min(k, heap_store.size());
        neighbours.resize(num_neigbours);

        while (num_neigbours > 0) {
            AssertFatal(!heap_store.empty(), LOG_TAG_DEFAULT, "Heap store should not be empty.");
            --num_neigbours;
            neighbours[num_neigbours] = heap_store.top().second;

            AssertFatal(neighbours[num_neigbours] != INVALID_VECTOR_ID, LOG_TAG_DEFAULT,
                "Invalid vector id in the heap store.");

            AssertFatal(neighbours[num_neigbours].Is_Vector(), LOG_TAG_DEFAULT, "Centroid vector in the heap_store");
            
            heap_store.pop();
        }
        return rc;
    }

    inline size_t Size() const {
        return _size;
    }
    
};

};

#endif