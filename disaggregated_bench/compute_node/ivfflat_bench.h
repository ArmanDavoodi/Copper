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
inline size_t _total_num_queries = 0;
inline size_t _total_num_tasks_created = 0;
inline size_t _total_num_search_queue_polls = 0;
inline size_t _total_num_polls = 0;
inline size_t _total_num_triggered_polls = 0;
inline size_t _total_num_empty_queue_induced_polls = 0;
inline size_t _total_unsuccsessful_polls = 0;
inline size_t _total_empty_polls = 0;
inline size_t _total_num_remote_reads_polled = 0;
inline size_t _total_remote_accesses = 0;
inline size_t _total_tries_to_get_memory = 0;
inline size_t _total_single_try = 0;
inline size_t _total_got_memory_from_cool_once = 0;
inline size_t _total_got_memory_from_pool_once = 0;
inline size_t _total_got_memory_from_cool_multiple = 0;
inline size_t _total_got_memory_from_pool_multiple = 0;
inline std::mutex stat_file_lock;
inline divftree::String stat_file_buffer;
#endif

#endif