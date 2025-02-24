#ifndef COPPER_VECTOR_UTILS_H
#define COPPER_VECTOR_UTILS_H

#include "common.h"
#include "queue"

namespace copper {

typedef void* Vector;

constexpr Vector INVALID_VECTOR = nullptr;

template<typename T>
std::string to_string(Vector vec, uint16_t dim) {
    std::string str = "[";
    for (uint16_t i = 0; i < dim; ++i) {
        str += std::to_string(((T*)vec)[i]);
        if (i < dim - 1)
            str += ", ";
    }
    str += "]";
    return str;
}

struct VectorSet {
    void* beg;
    VectorID* _ids;
    uint16_t _size;

    template<typename T>
    inline void Insert(const Vector& vec, VectorID id, uint16_t dim) {
        memcpy(beg + (_size * dim * sizeof(T)), vec, dim * sizeof(T));
        _ids[_size] = id;
        ++_size;
    }

    inline VectorID Get_VectorID(uint16_t idx) const {
        AssertFatal(idx < _size, LOG_TAG_ANY, "idx(%hu) >= _size(%hu)", idx, _size);
        return _ids[idx]; 
    }

    inline uint16_t Get_Index(VectorID id) const {
        AssertFatal(_size > 0, LOG_TAG_ANY, "Bucket is Empty");

        uint16_t index = 0;
        for (; index < _size; ++index) {
            if (_ids[index] == id) {
                break;
            }
        }

        AssertFatal(_ids[index] == id, LOG_TAG_ANY, "_ids[index(%hu)](%lu) != id(%lu)", index, _ids[index]._id, id._id);
        return index;
    }

    template<typename T>
    inline Vector Get_Vector(uint16_t idx, uint16_t dim) {
        AssertFatal(idx < _size, LOG_TAG_ANY, "idx(%hu) >= _size(%hu)", idx, _size);
        
        return ((T*)beg) + (idx * dim);
    }

    template<typename T>
    inline Vector Get_Vector_By_ID(VectorID id, uint16_t dim) {
        return Get_Vector<T>(Get_Index(id), dim); 
    }

    template<typename T>
    inline void Delete(VectorID id, uint16_t dim, VectorID& swapped_vec_id, Vector& swapped_vec) {
        uint16_t idx = Get_Index(id);
        
        if (idx != _size - 1) {
            swapped_vec = Get_Vector<T>(_size - 1, dim);
            swapped_vec_id = Get_VectorID(_size - 1);

            memcpy(beg + (idx * dim * sizeof(T)), swapped_vec, dim * sizeof(T));
            _ids[idx] = swapped_vec_id;
        }
        else {
            swapped_vec_id = INVALID_VECTOR_ID;
            swapped_vec = INVALID_VECTOR;
        }

        --_size;
    }

    template<typename T>
    std::string to_string(uint16_t dim) {
        std::string str = "<Vectors: [";
        for (uint16_t i = 0; i < _size; ++i) {
            str += copper::to_string<T>(Get_Vector<T>(i, dim), dim);
            if (i != _size - 1)
                str += ", ";
        }
        str += "], IDs: [";
        for (uint16_t i = 0; i < _size; ++i) {
            str += std::to_string(_ids[i]._id);
            if (i != _size - 1)
                str += ", ";
        }
        str += "]>";
        return str;
    }
};

template<typename T>
class Distance {
public:
    virtual double operator()(const Vector& a, const Vector& b, uint16_t _dim) = 0;
    virtual bool operator()(const double& a, const double& b) = 0;
};

template<typename T>
class L2_Distance : public Distance<T> {
public:
    double operator()(const Vector& a, const Vector& b, uint16_t _dim) {
        AssertFatal(a != INVALID_VECTOR, LOG_TAG_ANY, "a is invalid");
        AssertFatal(b != INVALID_VECTOR, LOG_TAG_ANY, "b is invalid");

        double dist = 0;
        for (size_t i = 0; i < _dim; ++i) {
            dist += (double)(((T*)a)[i] - ((T*)b)[i]) * (double)(((T*)a)[i] - ((T*)b)[i]);
        }
        return dist;
    }

    bool operator()(const double& a, const double& b) {
        return a < b;
    }
};

template<typename T>
struct Similarity {
public:
    Distance<T>* _dist;

    Similarity(Distance<T>* dist) : _dist(dist) {}

    bool operator()(const std::pair<double, VectorID>& a, const std::pair<double, VectorID>& b) {
        return (*_dist)(a.first, b.first);
    }
};

};

#endif