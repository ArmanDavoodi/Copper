#ifndef FLTNORMAL100M_CONFIGURATIONS_H_
#define FLTNORMAL100M_CONFIGURATIONS_H_

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

#define DIMENSION ((uint16_t)92)
#define DISTANCE_ALG (divftree::DistanceType::L2)

// #define EXCESS_LOGING
// #define MEMORY_DEBUG
#define DATASET_NAME "FLTNORMAL100M"
/*
    mean: 0, stddev: 1 for both data and query path.
    num_total_vectors for dataset is 111848106
 */
#define DATA_PATH "bench/datasets/100MFloatNormal_SkewedQ/raw_data/FLTNORMAL100M.fbin"

/* 
num query vectors is 40000, with 4 skewed directions and alpha of 4 for the skewed query generation.
    skew directions had ratios of 40%, 30%, 20%, and 10% respectively.
 */
// #define QUERY_PATH "bench/datasets/100MFloatNormal_SkewedQ/raw_data/FLTSKEWQ40K_alpha4_numd4.fbin"

/* 
num query vectors is 40000, with 8 skewed directions and alpha of 2.5 for the skewed query generation.
    skew directions had ratios of 20%, 10%, 40%, 10%, 5%, 5%, 5%, and 5% respectively.
 */
// #define QUERY_PATH "bench/datasets/100MFloatNormal_SkewedQ/raw_data/FLTNORMALQ40K_alpha2.5_numd8.fbin"

/*
num query vectors is 40000, with normal distribution (mean: 0, stddev: 1) and no skewed direction.
 */
#define QUERY_PATH "bench/datasets/100MFloatNormal_SkewedQ/raw_data/FLTNORMALQ40K.fbin"

#endif