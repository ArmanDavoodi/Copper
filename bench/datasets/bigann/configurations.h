#ifndef BIGANN_CONFIGURATIONS_H_
#define BIGANN_CONFIGURATIONS_H_

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

#define DIMENSION ((uint16_t)128)
#define DISTANCE_ALG (divftree::DistanceType::L2)

#if DATASET == BIGANN100M
#define DATA_PATH "bench/datasets/bigann/raw_data/BIGANN100M.u8bin"
#define DATASET_NAME "BIGANN100M"
#elif DATASET == BIGANN1B
#define DATASET_NAME "BIGANN1B"
#define DATA_PATH "bench/datasets/bigann/raw_data/BIGANN1B.u8bin"
#endif

#define QUERY_PATH "bench/datasets/bigann/raw_data/BIGANNQ10K.u8bin"

#endif