#ifndef COPPER_VECTOR_UTILS_H
#define COPPER_VECTOR_UTILS_H

#include "common.h"
#include "queue"

namespace copper {

typedef void* Vector;

constexpr Vector INVALID_VECTOR = nullptr;

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

    inline VectorID Get_VectorID(uint16_t idx) {
        assert(idx < _size);

        return _ids[idx]; 
    }

    inline uint16_t Get_Index(VectorID id) {
        assert(_size > 0);

        uint16_t index = 0;
        for (; index < _size; ++index) {
            if (_ids[index] == id) {
                break;
            }
        }

        assert(_ids[index] == id);
        return index;
    }

    template<typename T>
    inline Vector Get_Vector(uint16_t idx, uint16_t dim) {
        assert(idx < _size);
        
        return ((T*)beg) + (idx * dim);
    }

    template<typename T>
    inline Vector Get_Vector_By_ID(VectorID id, uint16_t dim) {
        return Get_Vector(Get_Index(id), dim); 
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
        assert(a != INVALID_VECTOR);
        assert(b != INVALID_VECTOR);

        double dist = 0;
        for (size_t i = 0; i < _dim; ++i) {
            dist += (((T*)a)[i] - ((T*)b)[i]) * (((T*)a)[i] - ((T*)b)[i]);
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