#ifndef CONFIGURATIONS_H_
#define CONFIGURATIONS_H_

#define BIGANN100M 1
#define BIGANN1B 2
#define SMALLNORMAL 3
#define HUGENORMAL 4
#define FLTNORMAL100M 5

// Hang Detector.+Message: Detected hang in thread id [^0][0-9]*
#define HANG_DETECTION
// todo: enable hugeTLB for release after builds and resolve the issue for them
// #define USE_HUGETLB

// #define DATASET BIGANN100M

#if defined(DATASET)
    #if ((DATASET == BIGANN100M) || (DATASET == BIGANN1B))
    #include "bench/datasets/bigann/configurations.h"
    #elif (DATASET == SMALLNORMAL)
    #include "bench/datasets/smallnormalsample/configurations.h"
    #elif (DATASET == HUGENORMAL)
    #include "bench/datasets/hugenormalsample/configurations.h"
    #elif (DATASET == FLTNORMAL100M)
    #include "bench/datasets/100MFloatNormal_SkewedQ/configurations.h"
    #else
    #error UNDEFINED DATASET!
    #endif
#else
    #error DATASET IS NOT DEFINED!
#endif

namespace divftree {
#if defined(VECTOR_TYPE)
    #if VECTOR_TYPE == UINT8
        using VTYPE = uint8_t;
        #define VTYPE_FMT "%hhu"
        #pragma message("DTYPE = UINT8")
    #elif VECTOR_TYPE == UINT16
        using VTYPE = uint16_t;
        #define VTYPE_FMT "%hu"
        #pragma message("DTYPE = UINT16")
    #elif VECTOR_TYPE == FLOAT
        using VTYPE = float;
        #define VTYPE_FMT "%0.2f"
        #pragma message("DTYPE = FLOAT")
    #else
        #error UNDEFINED VECTOR_TYPE!
    #endif
#else
using VTYPE = uint8_t;
#define VTYPE_FMT "%hhu"
#error VECTOR_TYPE not found!
#endif

#if defined(CENTROID_TYPE)
    #if CENTROID_TYPE == UINT8
        using CTYPE = uint8_t;
        #define CTYPE_FMT "%hhu"
        using MVTYPE = double;
        #define MVTYPE_FMT "%0.4f"
        #pragma message("CTYPE = UINT8")
    #elif CENTROID_TYPE == UINT16
        using CTYPE = uint16_t;
        #define CTYPE_FMT "%hu"
        using MVTYPE = double;
        #define MVTYPE_FMT "%0.4f"
        #pragma message("CTYPE = UINT16")
    #elif CENTROID_TYPE == FLOAT
        using CTYPE = float;
        #define CTYPE_FMT "%0.2f"
        using MVTYPE = double;
        #define MVTYPE_FMT "%0.4f"
        #pragma message("CTYPE = FLOAT")
    #else
        #error UNDEFINED CENTROID_TYPE!
    #endif
#else
using CTYPE = uint8_t;
#define CTYPE_FMT "%hhu"
using MVTYPE = double;
#define MVTYPE_FMT "%0.4f"
#error CENTROID_TYPE not found!
#endif

#if defined(DISTANCE_TYPE)
    #if DISTANCE_TYPE == UINT8
        using DTYPE = uint8_t;
        #define DTYPE_FMT "%hhu"
        #pragma message("DTYPE = UINT8")
    #elif DISTANCE_TYPE == UINT16
        using DTYPE = uint16_t;
        #define DTYPE_FMT "%hu"
        #pragma message("DTYPE = UINT16")
    #elif DISTANCE_TYPE == UINT32
        using DTYPE = uint32_t;
        #define DTYPE_FMT "%u"
        #pragma message("DTYPE = UINT32")
    #elif DISTANCE_TYPE == UINT64
        using DTYPE = uint64_t;
        #define DTYPE_FMT "%lu"
        #pragma message("DTYPE = UINT64")
    #elif DISTANCE_TYPE == FLOAT
        using DTYPE = float;
        #define DTYPE_FMT "%0.2f"
        #pragma message("DTYPE = FLOAT")
    #elif DISTANCE_TYPE == DOUBLE
        using DTYPE = double;
        #define DTYPE_FMT "%0.4f"
        #pragma message("DTYPE = DOUBLE")
    #else
        #error UNDEFINED DISTANCE_TYPE!
    #endif
#else
using DTYPE = uint8_t;
#define DTYPE_FMT "%hhu"
#error DISTANCE_TYPE not found!
#endif
}

#endif