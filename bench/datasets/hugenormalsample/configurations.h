#ifndef SMALLNORMALSAMPLE_CONFIGURATIONS_H_
#define SMALLNORMALSAMPLE_CONFIGURATIONS_H_

#include <cstdint>

#ifdef VECTOR_TYPE
#undef VECTOR_TYPE
#endif
#ifdef VTYPE_FMT
#undef VTYPE_FMT
#endif
#ifdef VECTOR_TYPE
#undef VECTOR_TYPE
#endif
#ifdef DTYPE_FMT
#undef DTYPE_FMT
#endif

#define UINT8 1
#define UINT16 2
#define UINT32 3
#define FLOAT 4

#define VECTOR_TYPE FLOAT

#define DIMENSION ((uint16_t)10)
#define DISTANCE_ALG (divftree::DistanceType::L2)

// #define EXCESS_LOGING
// #define MEMORY_DEBUG
#define DATASET_NAME "HUGENORMAL"
#define DATA_PATH "bench/datasets/hugenormalsample/raw_data/hugenormalsample1G.fbin"
#define QUERY_PATH "bench/datasets/hugenormalsample/raw_data/hugenormalsampleQ1K.fbin"

#endif