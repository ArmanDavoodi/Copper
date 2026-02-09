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
        using DTYPE = uint32_t;
        using MVTYPE = uint64_t;
        #define VTYPE_FMT "%hhu"
        #define DTYPE_FMT "%u"
        #define MVTYPE_FMT "%lu"
        #pragma message("TYPE = UINT8")
    #elif VECTOR_TYPE == UINT16
        using VTYPE = uint16_t;
        using DTYPE = uint64_t;
        using MVTYPE = uint64_t;
        #define VTYPE_FMT "%hu"
        #define DTYPE_FMT "%lu"
        #define MVTYPE_FMT "%lu"
        #pragma message("TYPE = UINT16")
    // #elif VECTOR_TYPE == UINT32
    //     using VTYPE = uint32_t;
    //     using DTYPE = uint64_t;
    //     using MVTYPE = uint64_t;
    //     #define VTYPE_FMT "%u"
    //     #define DTYPE_FMT "%lu"
    //     #define MVTYPE_FMT "%lu"
    //     #pragma message("TYPE = UINT32")
    #elif VECTOR_TYPE == FLOAT
        using VTYPE = float;
        using DTYPE = double;
        using MVTYPE = double;
        #define VTYPE_FMT "%0.2f"
        #define DTYPE_FMT "%0.4f"
        #define MVTYPE_FMT "%0.4f"
        #pragma message("TYPE = FLOAT")
    #else
        #error UNDEFINED VECTOR_TYPE!
    #endif
#else
using VTYPE = uint8_t;
using DTYPE = uint16_t;
using MVTYPE = uint64_t;
#define VTYPE_FMT "%hhu"
#define DTYPE_FMT "%hu"
#define MVTYPE_FMT "%lu"
#error VECTOR_TYPE not found!
#endif
}

#endif