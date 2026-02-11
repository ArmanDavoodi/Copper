#ifndef DIVFFLAT_MN_BENCH_H_
#define DIVFFLAT_MN_BENCH_H_

#include "bench/configurations.h"
#include "DIVF/memory_node/memory_node_divf.h"

#include <cstdint>
#include <atomic>

inline size_t num_clusters;
inline size_t num_threads;
inline size_t bench_batch_size;
inline size_t max_iters;
inline bool insert_duplicates;
inline size_t page_size;

inline uint32_t build_size;

inline divftree::VTYPE* data_set = nullptr;

#endif