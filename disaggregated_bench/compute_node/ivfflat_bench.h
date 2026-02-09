#ifndef DIVFFLAT_CN_BENCH_H_
#define DIVFFLAT_CN_BENCH_H_

#include "bench/configurations.h"
#include "DIVF/compute_node/compute_node_divf.h"

#include "utils/string.h"

#include <cstdint>
#include <atomic>
#include <mutex>

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

inline char stat_file[256] = "";

#ifdef ENABLE_STAT_COLLECTION
inline size_t _total_num_polls;
inline size_t _total_unsuccsessful_polls;
inline size_t _total_empty_polls;
inline size_t _total_num_remote_reads_polled;
inline size_t _total_remote_accesses;
inline size_t _total_tries_to_get_memory;
inline size_t _total_single_try;
inline size_t _total_got_memory_from_cool_once;
inline size_t _total_got_memory_from_pool_once;
inline size_t _total_got_memory_from_cool_multiple;
inline size_t _total_got_memory_from_pool_multiple;
inline std::mutex stat_file_lock;
inline divftree::String stat_file_buffer;
#endif

#endif