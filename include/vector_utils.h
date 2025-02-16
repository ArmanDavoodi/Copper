#ifndef COPPER_VECTOR_UTILS_H_
#define COPPER_VECTOR_UTILS_H_

#include "common.h"

namespace Copper {

    class Vector_Distance {
    public:
        virtual double operator()(Vector a, Vector b, DataType type, Dimention dim) = 0;
    };

    class L2_Distance : public Vector_Distance {
    protected:
        template<typename T>
        inline static double dist(T* a, T* b, Dimention dim) {
            double val = 0;
            for (Dimention d = 0; d < dim; ++dim) {
                val += (a[d] - b[d]) * (a[d] - b[d]);
            } 
            return val;
        }

    public:
        double operator()(Vector a, Vector b, DataType type, Dimention dim) {
            assert(a != Invalid_Vector);
            assert(b != Invalid_Vector);

            switch (type)
            {
            case DataType::Uint8:
                return dist<uint8_t>(static_cast<uint8_t*>(a), static_cast<uint8_t*>(b), dim);
            case DataType::Uint32:
                return dist<uint32_t>(static_cast<uint32_t*>(a), static_cast<uint32_t*>(b), dim);
            case DataType::Float32:
                return dist<float>(static_cast<float*>(a), static_cast<float*>(b), dim);
            }
            
            assert(false);
        }    
    };

    inline void Copy_Vector(Vector src, Vector dest, DataType type, Dimention dim) {

    }

};


#endif