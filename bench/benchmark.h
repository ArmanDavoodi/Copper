#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include "bench/configurations.h"
#include "divftree.h"

#include <cstdint>
#include <atomic>

inline divftree::DIVFTreeAttributes index_attr;
inline uint8_t default_leaf_search_span;
inline uint8_t default_internal_search_span;
inline uint8_t default_k;
inline size_t num_threads;
inline uint32_t write_ratio;
inline uint32_t delete_ratio;
/* todo: for every 10 writes, use the next vector for search ->
   not the best benchmark as it does not take the search skew in account */
inline uint32_t bench_batch_size;
inline uint32_t build_size;
inline uint32_t warmup_time;
inline uint32_t run_time;
inline uint32_t throughput_report_time;
inline bool show_runtime_report_for_build_and_warmup;

inline bool collect_build_stats;
inline bool collect_warmup_stats;
inline bool collect_run_stats;

inline bool collect_avg_distances;

#endif