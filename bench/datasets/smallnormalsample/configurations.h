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

#define VECTOR_TYPE UINT8

#define DIMENSION ((uint16_t)3)
#define DISTANCE_ALG (divftree::DistanceType::L2)

// #define EXCESS_LOGING
// #define MEMORY_DEBUG
#define DATASET_NAME "SMALLNORMAL"
#define DATA_PATH "bench/datasets/smallnormalsample/raw_data/smallnormalsample128.u8bin"
#define QUERY_PATH "bench/datasets/smallnormalsample/raw_data/smallnormalQ32.u8bin"

#endif