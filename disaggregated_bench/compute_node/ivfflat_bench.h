#ifndef DIVFFLAT_CN_BENCH_H_
#define DIVFFLAT_CN_BENCH_H_

#include "bench/configurations.h"
#include "DIVF/compute_node/compute_node_divf.h"

#include <cstdint>
#include <atomic>

inline size_t n_probes;
inline uint8_t default_k;
inline size_t num_threads;
inline size_t page_size;
inline size_t pool_size;
inline size_t bench_batch_size; /* unused */

inline uint32_t warmup_time;
inline uint32_t run_time;
inline uint32_t throughput_report_time;
inline bool show_runtime_report_for_build_and_warmup;

inline bool collect_avg_distances;

#endif