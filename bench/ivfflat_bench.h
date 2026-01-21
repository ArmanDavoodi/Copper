#ifndef IVFFLAT_BENCH_H_
#define IVFFLAT_BENCH_H_

#include "configurations.h"
#include "IVF/ivf.h"

#include <cstdint>
#include <atomic>

inline size_t num_clusters;
inline size_t n_probes;
inline uint8_t default_k;
inline size_t num_threads;
inline size_t bench_batch_size;
inline size_t max_iters;
inline bool insert_duplicates;

inline uint32_t build_size;
inline uint32_t warmup_time;
inline uint32_t run_time;
inline uint32_t throughput_report_time;
inline bool show_runtime_report_for_build_and_warmup;

inline bool collect_avg_distances;

inline divftree::VTYPE* data_set = nullptr;

#endif