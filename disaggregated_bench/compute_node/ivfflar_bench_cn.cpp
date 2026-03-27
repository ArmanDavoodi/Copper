#include "disaggregated_bench/compute_node/ivfflat_bench.h"
#include "disaggregated_bench/compute_node/ivfflat_config_reader.h"

#include "bench/dataset.h"

#include <vector>
#include <unordered_map>
#include <cmath>

// #define RECALL_BENCH

inline divftree::DIVFIndex* vector_index = nullptr;

inline std::atomic<size_t> search_queries = 0;
inline std::atomic<size_t> search_errors = 0;

inline divftree::SXSpinLock distance_lock;
inline double sum_search_distance = 0;
inline double avg_search_distance = 0;
inline uint64_t num_returned_neighbours = 0;
inline uint64_t num_total_returned_neighbours = 0;

#ifdef RECALL_BENCH
inline std::vector<uint64_t> num_true_positives_per_query;
inline std::atomic<uint64_t> next_query_batch_to_fetch = 0;
#endif

inline std::atomic<uint32_t> warmup_ready = false;
inline std::atomic<bool> warmup_start = false;
inline std::atomic<bool> warmup_finished = false;

inline std::atomic<uint32_t> run_ready = 0;
inline std::atomic<bool> run_start = false;
inline std::atomic<bool> run_finished = false;

inline std::atomic<uint32_t> run_done = 0;

inline thread_local uint64_t worker_idx = UINT64_MAX;
inline std::vector<std::vector<size_t>> query_latency_lists;

divftree::RetStatus Search(std::vector<std::pair<divftree::DTYPE, divftree::IVFVectorID>>& neighbours, size_t idx) {
    divftree::RetStatus rs;
    neighbours.clear();
    if (divftree::threadSelf->UniformRange32(0, 1000) == 0) {
        divftree::String query_str = divftree::String("search query vector: idx=%zu, data=[", idx);
        for (size_t i = 0; i < DIMENSION; ++i) {
            query_str += divftree::String(VTYPE_FMT "%s", search_query_vectors[idx * DIMENSION + i], (i == DIMENSION - 1) ? "]" : ", ");
        }
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "%s", query_str.ToCStr());
    }

    if (sample_rate_for_latency != 0 &&
        divftree::threadSelf->UniformRange64(0, sample_base_for_latency - 1) < sample_rate_for_latency) {
        auto start_time = std::chrono::high_resolution_clock::now();
        rs = vector_index->ANNSearch(&search_query_vectors[idx * DIMENSION], default_k, internal_n_probes, leaf_n_probes,
                                     neighbours);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        query_latency_lists[worker_idx].push_back(
            std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
    } else {
        rs = vector_index->ANNSearch(&search_query_vectors[idx * DIMENSION], default_k, internal_n_probes, leaf_n_probes,
                                    neighbours);
    }

    if (!rs.IsOK()) {
        DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_TEST, "Error during search: %s", rs.Msg());
        return rs;
    } else if (neighbours.empty()) {
        DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_TEST, "No neighbours found during search!");
        return divftree::RetStatus::Fail(nullptr);
    }

#ifdef RECALL_BENCH
    if (run_start.load(std::memory_order_acquire)) {
        num_true_positives_per_query[idx] = 0;
        size_t i = 0;
        for (size_t exact_idx = 0; ((exact_idx < default_k) && (i < neighbours.size())); ++exact_idx) {
            FatalAssert(i <= exact_idx, LOG_TAG_TEST, "since i iterates approximate answer it can only be worse");
            FatalAssert(i == 0 || divftree::L2::MoreSimilar(neighbours[i-1].first, neighbours[i].first) >= 0,
                        LOG_TAG_TEST, "Neighbours are not sorted by distance!");
            FatalAssert(exact_idx == 0 || divftree::L2::MoreSimilar(exact_knn_results[idx][exact_idx-1].first,
                                                                    exact_knn_results[idx][exact_idx].first) >= 0,
                        LOG_TAG_TEST, "Neighbours are not sorted by distance!");
            FatalAssert(exact_knn_results[idx][i].first <= neighbours[i].first, LOG_TAG_TEST,
                        "Exact KNN result is worse than the returned value");
            if (exact_knn_results[idx][exact_idx].second == neighbours[i].second) {
                FatalAssert(exact_knn_results[idx][exact_idx].first == neighbours[i].first, LOG_TAG_TEST,
                            "Exact KNN result with same ID has different distance than the returned value");
                ++num_true_positives_per_query[idx];
                ++i;
            } else {
                FatalAssert(divftree::L2::MoreSimilar(exact_knn_results[idx][exact_idx].first, neighbours[i].first) >= 0,
                            LOG_TAG_TEST,
                            "Exact KNN result should be better than the returned!");
            }
        }
        FatalAssert(num_true_positives_per_query[idx] != 0, LOG_TAG_TEST, "recall of 0!");
    }
#endif
    double total_distance = 0;
    divftree::DTYPE last_dist = 0;
    if (collect_avg_distances) {
        for (auto& neighbour : neighbours) {
            FatalAssert(divftree::L2::MoreSimilar(last_dist, neighbour.first) >= 0, LOG_TAG_TEST,
                        "Neighbours are not sorted by distance!");
            last_dist = neighbour.first;
            total_distance += std::sqrt(neighbour.first);
        }
        distance_lock.Lock(divftree::SX_EXCLUSIVE);
        num_returned_neighbours += neighbours.size();
        sum_search_distance += total_distance;
        distance_lock.Unlock();

    }

    return rs;
}

divftree::RetStatus Search(std::vector<std::pair<divftree::DTYPE, divftree::IVFVectorID>>& neighbours) {
    size_t idx = divftree::threadSelf->UniformRange64(0, total_num_queries - 1);
    return Search(neighbours, idx);
}

void FlushIncrement(uint64_t& local_cnt, std::atomic<uint64_t>& shared_cnt) {
    constexpr uint64_t local_cnt_thresh = 16;
    ++local_cnt;
    if (local_cnt >= local_cnt_thresh) {
        shared_cnt.fetch_add(local_cnt);
        local_cnt = 0;
    }
}

/* todo: instead of this get a batch per thread and make reading the file atomic? */
void worker(divftree::Thread* self, uint64_t thread_idx) {
    self->InitDIVFThread(DIMENSION);
    worker_idx = thread_idx;
    std::vector<std::pair<divftree::DTYPE, divftree::IVFVectorID>> neighbours;
    divftree::RetStatus rs;
    uint32_t num_ready = warmup_ready.fetch_add(1);
    if (num_ready == index_attr.num_user_threads - 1) {
        warmup_ready.notify_all();
    }
    bool ready = warmup_start.load(std::memory_order_acquire);
    while(!ready) {
        warmup_start.wait(false);
        ready = warmup_start.load(std::memory_order_acquire);
    }

    uint64_t current_search = 0;
    uint64_t current_search_err = 0;
    while(!warmup_finished.load(std::memory_order_acquire)) {
        self->LoopIncrement();
        rs = Search(neighbours);
        if (show_runtime_report_for_build_and_warmup) {
            if (rs.IsOK()) {
                FlushIncrement(current_search, search_queries);
            } else {
                FlushIncrement(current_search_err, search_errors);
            }
        }
    }

    if (show_runtime_report_for_build_and_warmup) {
        if (current_search != 0) {
            search_queries.fetch_add(current_search);
            current_search = 0;
        }
        if (current_search_err != 0) {
            search_errors.fetch_add(current_search_err);
            current_search_err = 0;
        }
    }
#ifdef ENABLE_STAT_COLLECTION
    stat_file_lock.lock();
    self->AppendStats(
        stat_file_buffer,
        _total_num_queries,
        _total_num_tasks_created,
        _total_num_search_queue_polls,
        _total_num_polls,
        _total_num_triggered_polls,
        _total_num_empty_queue_induced_polls,
        _total_unsuccsessful_polls,
        _total_empty_polls,
        _total_num_remote_reads_polled,
        _total_remote_accesses,
        _total_tries_to_get_memory,
        _total_single_try,
        _total_got_memory_from_cool_once,
        _total_got_memory_from_pool_once,
        _total_got_memory_from_cool_multiple,
        _total_got_memory_from_pool_multiple, true /* reset_stats */
    );
    stat_file_lock.unlock();
#endif

    num_ready = run_ready.fetch_add(1);
    if (num_ready == index_attr.num_user_threads - 1) {
        run_ready.notify_all();
    }
    ready = run_start.load(std::memory_order_acquire);
    while(!ready) {
        run_start.wait(false);
        ready = run_start.load(std::memory_order_acquire);
    }

    current_search = 0;
    current_search_err = 0;

#ifdef RECALL_BENCH
    size_t query_per_thread = ((size_t)total_num_queries + index_attr.num_user_threads - 1) / index_attr.num_user_threads;
    size_t idx = next_query_batch_to_fetch.fetch_add(query_per_thread);
    size_t end_idx = std::min(idx + query_per_thread, (size_t)total_num_queries);

    for (; idx < end_idx; ++idx) {
        self->LoopIncrement();
        rs = Search(neighbours, idx);
        if (rs.IsOK()) {
            FlushIncrement(current_search, search_queries);
        } else {
            FlushIncrement(current_search_err, search_errors);
        }
    }
#else
    while(!run_finished.load(std::memory_order_acquire)) {
        self->LoopIncrement();
        rs = Search(neighbours);
        if (rs.IsOK()) {
            FlushIncrement(current_search, search_queries);
        } else {
            FlushIncrement(current_search_err, search_errors);
        }
    }
#endif

    if (current_search != 0) {
        search_queries.fetch_add(current_search);
        current_search = 0;
    }
    if (current_search_err != 0) {
        search_errors.fetch_add(current_search_err);
        current_search_err = 0;
    }

    num_ready = run_done.fetch_add(1);
    if (num_ready == index_attr.num_user_threads - 1) {
        run_done.notify_all();
    }

#ifdef ENABLE_STAT_COLLECTION
    stat_file_lock.lock();
    self->AppendStats(
        stat_file_buffer,
        _total_num_queries,
        _total_num_tasks_created,
        _total_num_search_queue_polls,
        _total_num_polls,
        _total_num_triggered_polls,
        _total_num_empty_queue_induced_polls,
        _total_unsuccsessful_polls,
        _total_empty_polls,
        _total_num_remote_reads_polled,
        _total_remote_accesses,
        _total_tries_to_get_memory,
        _total_single_try,
        _total_got_memory_from_cool_once,
        _total_got_memory_from_pool_once,
        _total_got_memory_from_cool_multiple,
        _total_got_memory_from_pool_multiple
    );
    stat_file_lock.unlock();
#endif

    self->DestroyDIVFThread();
}

#define BenchLog(msg, ...) \
    do { \
        printf(msg __VA_OPT__(,) __VA_ARGS__); \
        printf("\n"); \
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_TEST, msg __VA_OPT__(,) __VA_ARGS__); \
    } while(0)

#define ExclusiveBenchLog(msg, ...) \
    do { \
        printf(msg __VA_OPT__(,) __VA_ARGS__); \
        printf("\n"); \
    } while(0)

#define ExclusiveBenchNewLine() \
    do { \
        printf("\n"); \
    } while(0)

inline void FlushStats(bool clear) {
#ifdef ENABLE_STAT_COLLECTION
    FILE* f = fopen(stat_file, "a");
    if (f == nullptr) {
        DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_TEST, "Cannot open stat file %s for writing!", stat_file);
    } else {
        stat_file_buffer = divftree::String("total_num_queries: %zu, total_num_tasks_created: %zu, total_num_search_queue_polls: %zu, "
                            "avg_num_tasks_created_per_query: %.2f, "
                            "avg_num_search_queue_polls_per_query: %.2f, "
                            "avg_num_search_queue_polls_per_created: %.2f, "
                            "total_num_polls: %zu(triggered: %.2f(%zu), empty_q_end: %.2f(%zu) | "
                            "total_num_fail: %.2f%%(%zu), total_num_empty %.2f%%(%zu), total_num_valid: %.2f%%(%zu)), "
                            "total_num_valid_to_successful_ratio: %.2f%%, "
                            "total_num_remote_reads_polled: %zu, avg_num_polled_per_valid_polls: %zu, "
                            "total_num_remote_rdma_reads: %zu(single_alloc: %.2f%%(%zu), multi_alloc: %.2f%%(%zu)), "
                            "total_num_alloc_tries: %zu, avg_num_alloc_tries_per_req: %zu, "
                            "avg_num_alloc_multi_tries_per_req: %zu, "
                            "total_num_pages_got: %zu(from_pool: %.2f%%(%zu), from_cooling_list: %.2f%%(%zu))\n",
                            _total_num_queries, _total_num_tasks_created, _total_num_search_queue_polls,
                            (_total_num_queries > 0 ? ((double)_total_num_tasks_created / (double)_total_num_queries) : 0),
                            (_total_num_queries > 0 ? ((double)_total_num_search_queue_polls / (double)_total_num_queries) : 0),
                            (_total_num_tasks_created > 0 ? ((double)_total_num_search_queue_polls / (double)_total_num_tasks_created) : 0),
                            _total_num_polls,
                            (_total_num_polls > 0 ? (100.0 * _total_num_triggered_polls / _total_num_polls) : 0), _total_num_triggered_polls,
                            (_total_num_polls > 0 ? (100.0 * _total_num_empty_queue_induced_polls / _total_num_polls) : 0), _total_num_empty_queue_induced_polls,
                            (_total_num_polls > 0 ? (100.0 * _total_unsuccsessful_polls / _total_num_polls) : 0),
                            _total_unsuccsessful_polls,
                            (_total_num_polls > 0 ? (100.0 * _total_empty_polls / _total_num_polls) : 0), _total_empty_polls,
                            (_total_num_polls > 0 ?
                                (100.0 * (_total_num_polls - _total_unsuccsessful_polls - _total_empty_polls) / _total_num_polls) :
                                0),
                            _total_num_polls - _total_unsuccsessful_polls - _total_empty_polls,
                            (_total_num_polls > 0 ? (100.0 * (_total_num_polls - _total_unsuccsessful_polls - _total_empty_polls) /
                                                  (_total_num_polls - _total_unsuccsessful_polls)) : 0),
                            _total_num_remote_reads_polled,
                            (((_total_num_polls > 0) && (_total_num_polls - _total_unsuccsessful_polls - _total_empty_polls) > 0) ?
                             (_total_num_remote_reads_polled / (_total_num_polls - _total_unsuccsessful_polls - _total_empty_polls))
                             : 0), _total_remote_accesses,
                            (_total_remote_accesses > 0 ? (100.0 * _total_single_try / _total_remote_accesses) : 0), _total_single_try,
                            (_total_remote_accesses > 0 ? (100.0 * (_total_remote_accesses - _total_single_try) / _total_remote_accesses) : 0), (_total_remote_accesses - _total_single_try),
                            _total_tries_to_get_memory,
                            (_total_remote_accesses > 0 ? (_total_tries_to_get_memory / _total_remote_accesses) : 0),
                            (((_total_remote_accesses > 0) && (_total_remote_accesses - _total_single_try > 0)) ?
                                (_total_tries_to_get_memory / (_total_remote_accesses - _total_single_try)) : 0),
                            _total_got_memory_from_cool_once + _total_got_memory_from_cool_multiple + _total_got_memory_from_pool_once + _total_got_memory_from_pool_multiple,
                            (_total_got_memory_from_cool_once + _total_got_memory_from_cool_multiple + _total_got_memory_from_pool_once + _total_got_memory_from_pool_multiple > 0) ?
                            ((_total_got_memory_from_pool_once + _total_got_memory_from_pool_multiple) * 100.0) / (_total_got_memory_from_cool_once + _total_got_memory_from_cool_multiple + _total_got_memory_from_pool_once + _total_got_memory_from_pool_multiple) : 0,
                            _total_got_memory_from_pool_once + _total_got_memory_from_pool_multiple,
                            (_total_got_memory_from_cool_once + _total_got_memory_from_cool_multiple + _total_got_memory_from_pool_once + _total_got_memory_from_pool_multiple > 0) ?
                            ((_total_got_memory_from_cool_once + _total_got_memory_from_cool_multiple) * 100.0) / (_total_got_memory_from_cool_once + _total_got_memory_from_cool_multiple + _total_got_memory_from_pool_once + _total_got_memory_from_pool_multiple) : 0,
                            _total_got_memory_from_cool_once + _total_got_memory_from_cool_multiple
                        ) + stat_file_buffer;
        fprintf(f, "%s", stat_file_buffer.ToCStr());
        stat_file_buffer = "";
        fprintf(f, "\n-------------------------\n");
        divftree::MemoryStatsNode* stats = nullptr;
        divftree::String index_Stats = vector_index->GetStats(stats, clear);
        fprintf(f, "%s", index_Stats.ToCStr());
        index_Stats = "";
        fprintf(f, "\n-------------------------\n");
        while (stats != nullptr) {
            divftree::MemoryStatsNode* next = stats->next;
            fprintf(f, "Memory Stats: Allocated Pages: %zu, Bytes in Use: %zu\n", stats->num_allocated_pages, stats->num_bytes_in_use);
            delete stats;
            stats = next;
        }
        fprintf(f, "\n-------------------------\n");
        fclose(f);

        if (clear) {
            _total_num_queries = 0;
            _total_num_tasks_created = 0;
            _total_num_search_queue_polls = 0;
            _total_num_polls = 0;
            _total_num_triggered_polls = 0;
            _total_num_empty_queue_induced_polls = 0;
            _total_unsuccsessful_polls = 0;
            _total_empty_polls = 0;
            _total_num_remote_reads_polled = 0;
            _total_remote_accesses = 0;
            _total_tries_to_get_memory = 0;
            _total_single_try = 0;
            _total_got_memory_from_cool_once = 0;
            _total_got_memory_from_pool_once = 0;
            _total_got_memory_from_cool_multiple = 0;
            _total_got_memory_from_pool_multiple = 0;
        }
    }
#endif
}

/* log-output-file-dir is not the file name but the directory path */
void ReadArgs(int argc, char** argv) {
    if (argc != 9) {
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Usage: %s <self-node-idx> <index-file-path> <log-output-file-dir> <stat-file-path> <page-size-bytes> <pool-size-bytes> <internal_n_probes> <leaf_n_probes>", argv[0]);
    }

    divftree::network_config::self_idx = static_cast<uint8_t>(std::stoul(argv[1]));
    std::string index_file_path = argv[2];
    if (index_file_path.empty()) {
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Index file path cannot be empty!");
    }

    if (!std::filesystem::exists(index_file_path)) {
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Index file does not exist at path: %s", index_file_path.c_str());
    }

    if (!std::filesystem::is_regular_file(index_file_path)) {
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Index file path is not a regular file: %s", index_file_path.c_str());
    }

    var_configs["log-path"] = argv[3];
    var_configs["stat-file"] = argv[4];
    index_attr.page_size = std::stoul(argv[5]);
    if (index_attr.page_size == 0 || !(divftree::ALIGNED(index_attr.page_size))) {
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Page size must be greater than 0!");
    }
    index_attr.pool_size = std::stoul(argv[6]);
    if (index_attr.pool_size < index_attr.page_size || !(divftree::ALIGNED(index_attr.pool_size))) {
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Pool size must be greater than page-size!");
    }

    internal_n_probes = std::stoul(argv[7]);
    if (internal_n_probes == 0) {
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Number of internal probes must be greater than 0!");
    }
    leaf_n_probes = std::stoul(argv[8]);
    if (leaf_n_probes == 0) {
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Number of leaf probes must be greater than 0!");
    }

    FILE* file = fopen(index_file_path.c_str(), "rb");
    if (file == nullptr) {
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Failed to open index file at path: %s", index_file_path.c_str());
    }

    size_t ret = fread(&index_attr.index_meta.type, sizeof(divftree::IndexType), 1, file);
    if (ret != 1) {
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Failed to read index type from index file!");
    }

    switch (index_attr.index_meta.type) {
    case divftree::IndexType::IVF_CAPPED:
        ret = fread(&index_attr.index_meta.leaf_size_cap, sizeof(uint32_t), 1, file);
        if (ret != 1) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Failed to read cluster capacity for capped k-means index from index file!");
        }
        if (index_attr.index_meta.leaf_size_cap == 0) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Cluster capacity for capped k-means index must be greater than 0!");
        }
        break;
    case divftree::IndexType::IVF_TREE:
        ret = fread(&index_attr.index_meta.leaf_size_cap, sizeof(uint32_t), 1, file);
        if (ret != 1) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Failed to read leaf cluster capacity for hierarchical k-means index from index file!");
        }
        if (index_attr.index_meta.leaf_size_cap == 0) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Leaf cluster capacity for hierarchical k-means index must be greater than 0!");
        }
        ret = fread(&index_attr.index_meta.internal_size_cap, sizeof(uint32_t), 1, file);
        if (ret != 1) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Failed to read internal cluster capacity for hierarchical k-means index from index file!");
        }
        if (index_attr.index_meta.internal_size_cap == 0) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Internal cluster capacity for hierarchical k-means index must be greater than 0!");
        }
        break;
    }

    fclose(file);
}

int main(int argc, char** argv) {
    ReadArgs(argc, argv);
    std::pair<divftree::String, divftree::String> node_strs = divftree::ReadNetworkConfigs();
    divftree::NodeID self_id = divftree::NodeID(false, divftree::network_config::compute_node_ids[divftree::network_config::self_idx]);
    ReadConfigs();
    ParseConfigs(self_id);
    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
            "Completed reading network configuration: %hhu memory nodes:%s, %hhu compute nodes:%s, "
            "RDMA device: %s, port: %hhu, GID index: %d",
            ::divftree::network_config::num_memory_nodes, node_strs.first.ToCStr(),
            ::divftree::network_config::num_compute_nodes, node_strs.second.ToCStr(),
            ::divftree::network_config::rdma_device_name, ::divftree::network_config::rdma_port,
            ::divftree::network_config::gid_index);

    divftree::Thread main_thread(100);
    main_thread.InitDIVFThread((uint16_t)DIMENSION);

    BenchLog("Starting benchmark from node:%s for %s(type:%s, dimension:%hu, distance:%s) "
             "with %lu threads, warmup-time:%u(s), and run-time:%u(s). "
             "default k = %hhu, internal_n_probes = %u, leaf_n_probes = %u",
             self_id.ToString().ToCStr(),
             DATASET_NAME, DIVF_MACRO_TO_STR(VECTOR_TYPE), DIMENSION,
             divftree::DISTANCE_TYPE_NAME[(int8_t)DISTANCE_ALG], index_attr.num_user_threads, warmup_time, run_time,
             default_k, internal_n_probes, leaf_n_probes);

    BenchLog("Start Node...");
    vector_index = new divftree::DIVFIndex(index_attr);

    BenchLog("Load Query Vectors...");
    LoadQueryVectors();
#ifdef RECALL_BENCH
    UNUSED_VARIABLE(run_finished);
    LoadExactKNNResults(exact_neighbours_path, default_k);
    num_true_positives_per_query.resize(total_num_queries, 0);
#endif

    std::vector<divftree::Thread*> threads(index_attr.num_user_threads);
    query_latency_lists.resize(index_attr.num_user_threads);
    std::vector<uint64_t> all_query_latencies;
    for (size_t i = 0; i < index_attr.num_user_threads; ++i) {
        threads[i] = new divftree::Thread(100);
    }

    BenchLog("Starting %lu worker threads...", index_attr.num_user_threads);

    for (size_t i = 0; i < index_attr.num_user_threads; ++i) {
        threads[i]->Start(worker, i);
    }

    BenchLog("Start Warmup...");
#ifdef ENABLE_STAT_COLLECTION
    stat_file_buffer = divftree::String("****************** Warmup Phase Stats ****************** \n\n");
#endif
    warmup_start.store(true, std::memory_order_release);
    warmup_start.notify_all();

    if (show_runtime_report_for_build_and_warmup) {
        uint32_t time_to_wait = throughput_report_time;
        size_t last_rps = 0, last_reps = 0;
        for (uint32_t total_wait_time = 0; total_wait_time < warmup_time; total_wait_time += time_to_wait) {
            divftree::sleep(time_to_wait);
            size_t cur_rps = search_queries.load(std::memory_order_acquire);
            size_t cur_reps = search_errors.load(std::memory_order_acquire);

            if (collect_avg_distances) {
                distance_lock.Lock(divftree::SX_EXCLUSIVE);
                ExclusiveBenchLog("[%u s]: Warmup Phase Report: RPS: %.2f | Avg Distance: " DTYPE_FMT " | REPS: %.2f",
                                total_wait_time + time_to_wait,
                                (double)(cur_rps - last_rps) / (double)time_to_wait,
                                (divftree::DTYPE)(sum_search_distance / (double)num_returned_neighbours),
                                (double)(cur_reps - last_reps) / (double)time_to_wait);
                sum_search_distance = 0;
                num_returned_neighbours = 0;
                distance_lock.Unlock();
            } else {
                ExclusiveBenchLog("[%u s]: Warmup Phase Report: RPS: %.2f | REPS: %.2f",
                              total_wait_time + time_to_wait,
                              (double)(cur_rps - last_rps) / (double)time_to_wait,
                              (double)(cur_reps - last_reps) / (double)time_to_wait);
            }
            last_rps = cur_rps;
            last_reps = cur_reps;
            if ((total_wait_time + time_to_wait > run_time)) {
                time_to_wait = run_time - total_wait_time;
            }
        }
    } else {
        divftree::sleep(warmup_time);
    }

    warmup_finished.store(true, std::memory_order_release);
    size_t num_ready = run_ready.load(std::memory_order_acquire);
    while (num_ready != index_attr.num_user_threads) {
        run_ready.wait(num_ready);
        num_ready = run_ready.load(std::memory_order_acquire);
    }

    if (show_runtime_report_for_build_and_warmup) {
        search_queries.store(0, std::memory_order_release);
        search_errors.store(0, std::memory_order_release);
        if (collect_avg_distances) {
            distance_lock.Lock(divftree::SX_EXCLUSIVE);
            sum_search_distance = 0;
            num_returned_neighbours = 0;
            distance_lock.Unlock();
        }
    }

    double avg_latency = 0;
    for (size_t i = 0; i < index_attr.num_user_threads; ++i) {
        all_query_latencies.reserve(all_query_latencies.size() + query_latency_lists[i].size());
        for (uint64_t l : query_latency_lists[i]) {
            avg_latency += l;
            all_query_latencies.push_back(l);
        }
        query_latency_lists[i].clear();
    }
    std::sort(all_query_latencies.begin(), all_query_latencies.end());
    if (!all_query_latencies.empty()) {
        BenchLog("Latency percentiles for warmup phase:");
        BenchLog("P50: %lu us", all_query_latencies[all_query_latencies.size() / 2]);
        BenchLog("P90: %lu us", all_query_latencies[all_query_latencies.size() * 9 / 10]);
        BenchLog("P95: %lu us", all_query_latencies[all_query_latencies.size() * 95 / 100]);
        BenchLog("P99: %lu us", all_query_latencies[all_query_latencies.size() * 99 / 100]);
        BenchLog("Average latency: %.2f us", avg_latency / (double)all_query_latencies.size());

        all_query_latencies.clear();
    } else {
        BenchLog("No latency samples collected for warmup phase!");
    }

    FlushStats(true);
#ifdef ENABLE_STAT_COLLECTION
    stat_file_buffer = divftree::String("\n\n****************** Run Phase Stats ****************** \n\n");
#endif

    BenchLog("Start Run...");
    auto start_time = std::chrono::high_resolution_clock::now();
    run_start.store(true, std::memory_order_release);
    run_start.notify_all();

    if (throughput_report_time == 0) {
#ifndef RECALL_BENCH
        divftree::sleep(run_time);
#endif
    } else {
        uint32_t time_to_wait = throughput_report_time;
        size_t last_rps = 0, last_reps = 0;
#ifdef RECALL_BENCH
        for (uint32_t total_wait_time = 0;
             run_done.load(std::memory_order_acquire) < index_attr.num_user_threads;
             total_wait_time += time_to_wait) {
#else
        for (uint32_t total_wait_time = 0; total_wait_time < run_time; total_wait_time += time_to_wait) {
#endif
            divftree::sleep(time_to_wait);
            size_t cur_rps = search_queries.load(std::memory_order_acquire);
            size_t cur_reps = search_errors.load(std::memory_order_acquire);

            size_t total_qps = cur_rps- last_rps;
            size_t total_eps = cur_reps - last_reps;

            if (collect_avg_distances) {
                distance_lock.Lock(divftree::SX_EXCLUSIVE);
                avg_search_distance += sum_search_distance;
                num_total_returned_neighbours += num_returned_neighbours;

                ExclusiveBenchLog("[%u s]: QPS: %.2f | " "Avg Distance: " DTYPE_FMT " | EPS: %.2f",
                                  total_wait_time + time_to_wait, (double)total_qps / (double)time_to_wait,
                                  (divftree::DTYPE)(sum_search_distance / (double)num_returned_neighbours),
                                  (double)total_eps / (double)time_to_wait);

                sum_search_distance = 0;
                num_returned_neighbours = 0;
                distance_lock.Unlock();
            } else {
                ExclusiveBenchLog("[%u s]: QPS: %.2f | EPS: %.2f",
                                total_wait_time + time_to_wait, (double)total_qps / (double)time_to_wait,
                                (double)total_eps / (double)time_to_wait);
            }
            last_rps = cur_rps;
            last_reps = cur_reps;
#ifndef RECALL_BENCH
            if ((total_wait_time + time_to_wait > run_time)) {
                time_to_wait = run_time - total_wait_time;
            }
#endif
        }
    }

    run_finished.store(true, std::memory_order_release);
    num_ready = run_done.load(std::memory_order_acquire);
    while (num_ready != index_attr.num_user_threads) {
        run_done.wait(num_ready);
        num_ready = run_done.load(std::memory_order_acquire);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    size_t total_run_time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    BenchLog("Stopping all threads...");
    for (size_t i = 0; i < index_attr.num_user_threads; ++i) {
        threads[i]->Join();
        delete threads[i];
        threads[i] = nullptr;
    }

    FlushStats(false);

    delete vector_index;

    if (collect_avg_distances) {
        avg_search_distance += sum_search_distance;
        num_total_returned_neighbours += num_returned_neighbours;

        if (num_total_returned_neighbours > 0) {
            avg_search_distance /= (double)num_total_returned_neighbours;
        } else if (avg_search_distance > 0) {
            DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_TEST, "Error: no neighbours returned but avg distance is not 0!");
            avg_search_distance = 0;
        }
    }

    size_t total_search = search_queries.load(std::memory_order_acquire);
    size_t total_search_err = search_errors.load(std::memory_order_acquire);

    ExclusiveBenchLog("\n___________________________________________\n");
    ExclusiveBenchLog("Run Stats:\n");

    BenchLog("Final Run Time: %lu(ms)", total_run_time);

    ExclusiveBenchLog("------------------------");

    total_run_time /= 1000;

    BenchLog("Total queries: %lu", total_search);

    ExclusiveBenchNewLine();

    BenchLog("Total errors: %lu", total_search_err);

    ExclusiveBenchLog("------------------------");

    BenchLog("Total QPS: %.2f", ((double)total_search) / (double)total_run_time);

    ExclusiveBenchNewLine();

    BenchLog("Total EPS: %.2f", ((double)total_search_err) / (double)total_run_time);

    if (collect_avg_distances) {
        ExclusiveBenchLog("------------------------");

        BenchLog("Average Search Distance: " DTYPE_FMT, (divftree::DTYPE)avg_search_distance);
    }

#ifdef RECALL_BENCH
    ExclusiveBenchLog("------------------------");

    std::sort(num_true_positives_per_query.begin(), num_true_positives_per_query.end());
    double p01_recall = (double)(num_true_positives_per_query[total_num_queries / 100]) / (double)default_k;
    double p05_recall = (double)(num_true_positives_per_query[total_num_queries / 20]) / (double)default_k;
    double p50_recall = (double)(num_true_positives_per_query[total_num_queries / 2]) / (double)default_k;
    double p95_recall = (double)(num_true_positives_per_query[(size_t)((double)total_num_queries * 0.95)]) / (double)default_k;
    double p99_recall = (double)(num_true_positives_per_query[(size_t)((double)total_num_queries * 0.99)]) / (double)default_k;
    double avg_recall = 0;
    for (uint64_t num_tp : num_true_positives_per_query) {
        avg_recall += (double)num_tp / (double)default_k;
        FatalAssert(num_tp <= default_k, LOG_TAG_TEST, "tp cannot be more than k");
    }
    avg_recall /= (double)total_num_queries;

    BenchLog("Recall Info: (99%% of queries have a recall higher/better than p01) "
             "p01:%.2f, p05:%.2f, p50:%.2f, p95:%.f, p99:%2.f, avg:%.2f",
             p01_recall, p05_recall, p50_recall, p95_recall, p99_recall, avg_recall);

    for (size_t i = 0; i < total_num_queries; ++i) {
        delete[] exact_knn_results[i];
    }
    delete[] exact_knn_results;
#endif

    avg_latency = 0;
    for (size_t i = 0; i < index_attr.num_user_threads; ++i) {
        all_query_latencies.reserve(all_query_latencies.size() + query_latency_lists[i].size());
        for (uint64_t l : query_latency_lists[i]) {
            avg_latency += l;
            all_query_latencies.push_back(l);
        }
        query_latency_lists[i].clear();
    }
    std::sort(all_query_latencies.begin(), all_query_latencies.end());
    if (!all_query_latencies.empty()) {
        ExclusiveBenchLog("------------------------");

        BenchLog("Latency percentiles for run phase:");
        BenchLog("P50: %lu us", all_query_latencies[all_query_latencies.size() / 2]);
        BenchLog("P90: %lu us", all_query_latencies[all_query_latencies.size() * 9 / 10]);
        BenchLog("P95: %lu us", all_query_latencies[all_query_latencies.size() * 95 / 100]);
        BenchLog("P99: %lu us", all_query_latencies[all_query_latencies.size() * 99 / 100]);
        BenchLog("Average latency: %.2f us", avg_latency / (double)all_query_latencies.size());

        all_query_latencies.clear();
    } else {
        BenchLog("No latency samples collected for run phase!");
    }


    ExclusiveBenchLog("\n___________________________________________\n");

    delete[] search_query_vectors;
    search_query_vectors = nullptr;
}