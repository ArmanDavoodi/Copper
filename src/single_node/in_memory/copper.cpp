#include "vector_utils.h"
#include "single_node/in_memory/copper.h"

#include <queue>

namespace Copper {

    ReturnStatus SNIMCopperVertex::Insert(Vector vec, VectorID& ID) {
        assert(_size < _max_size);
        assert(vec != Invalid_Vector);

        ReturnStatus ret;
        Vector new_loc = Invalid_Vector;
        ret = _bufmgr->Insert(vec, _selfID, new_loc, ID);

        if (ret.Is_OK()) { // likely
            assert(ID.Is_Valid());
            assert(new_loc != Invalid_Vector);
            // assert(ID.level == _selfID.level)
            // assert(ID.creator_node_id == ??)
            auto map_ins = _dir.insert({ID, new_loc});
            if (!map_ins.second) {
                assert(false); // todo handle this case later
            }
            ++_size;
        }
        else {
            assert(!ID.Is_Valid());
            // assert(new_loc == Invalid_Vector);
        }

        assert(_dir.size() == _size);
        return ret;
    }

    ReturnStatus SNIMCopperVertex::Delete(VectorID ID) {
        assert(_size > _min_size);
        assert(ID.Is_Valid());

        ReturnStatus ret;
        auto ID_loc = _dir.find(ID);
        if (ID_loc == _dir.end()) { // unlikely
            // ret = could not find
            // todo
            return ret;
        }

        assert(ID_loc->first == ID && ID_loc->second != Invalid_Vector);
        // todo can we do other bound checking to make sure it is not out of bounds?
        Vector swapped_vec = Invalid_Vector;
        VectorID swapped_vec_ID = Invalid_ID;

        ret = _bufmgr->Delete(ID_loc->second, swapped_vec_ID, swapped_vec);

        if (ret.Is_OK()) { // likely
            assert(swapped_vec_ID.Is_Valid());
            assert(swapped_vec != Invalid_Vector)

            auto swapped = _dir.find(swapped_vec_ID);
            assert(swapped != _dir.end());
            assert(swapped->first == swapped_vec_ID && swapped->second == swapped_vec);

            swapped->second = ID_loc->second;
            _dir.erase(ID_loc);
            // todo assert to check succesful?

            --_size;
        }

        assert(_dir.size() == _size);
        return ret;
    }

    ReturnStatus SNIMCopperVertex::KNearestNeighbours(const Vector& query, uint16_t k, std::vector<VectorID>& NNs) const {
        assert(k > 0);
        assert(query != Invalid_Vector);

        NNs.clear();
        NNs.reserve(k);
        NNs.resize(0);

        ReturnStatus ret;
        
        std::priority_queue<double, VectorID> maxHeap;
        for (auto e : _dir) {
            double distance = (*_dist)(query, e.second, _vtrtype, _dim);
            minHeap.push({distance, e.first});
            if (maxHeap.size() > k) {
                maxHeap.pop();
            }
        }

        assert(maxHeap.size() <= k);

        for (const auto& e : maxHeap) {
            NNs.push_back(e.second);
        }

        return ret;
    }

    uint64_t SNIMCopperVertex::Size() const {
        return _size;
    }

};