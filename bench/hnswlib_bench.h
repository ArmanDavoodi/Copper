#ifndef HNSWLIB_BENCHMARK_H_
#define HNSWLIB_BENCHMARK_H_

#include "bench/configurations.h"
#include "third_party/index/hnswlib/hnswlib/hnswlib.h"
#include "common.h"

#include <cstdint>
#include <atomic>

struct HNSWLibAttributes {
    uint32_t dim;
    uint32_t max_elements;
    uint32_t M;

    uint32_t ef_construction;
    uint32_t ef_search;
};

inline HNSWLibAttributes index_attr;
inline uint8_t default_k;
inline size_t num_threads;
inline uint32_t write_ratio; /* write ratio will cause the index to be locked in exclusive mode in-order to support concurrent read/write*/
inline uint32_t delete_ratio;
inline uint32_t bench_batch_size;
inline uint32_t build_size;
inline uint32_t warmup_time;
inline uint32_t run_time;
inline uint32_t throughput_report_time;
inline bool show_runtime_report_for_build_and_warmup;

inline bool collect_avg_distances;

static double DIVFBenchL2Sqrt(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    divftree::VTYPE *pVect1 = (divftree::VTYPE *) pVect1v;
    divftree::VTYPE *pVect2 = (divftree::VTYPE *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    divftree::DTYPE res = 0;
    for (size_t i = 0; i < qty; i++) {
        divftree::DTYPE t = (divftree::DTYPE)(*pVect1) - (divftree::DTYPE)(*pVect2);
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (double)(res);
};

class DIVFBenchL2Space : public hnswlib::SpaceInterface<double> {
protected:
    size_t data_size_;
    size_t dim_;
public:
    DIVFBenchL2Space(size_t dim) : data_size_(dim * sizeof(divftree::VTYPE)), dim_(dim) {}

    inline size_t get_data_size() override {
        return data_size_;
    }

    hnswlib::DISTFUNC<double> get_dist_func() override {
        return DIVFBenchL2Sqrt;
    }

    void *get_dist_func_param() override {
        return &dim_;
    }

    ~DIVFBenchL2Space() {}
};


#endif