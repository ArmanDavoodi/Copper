#include "bench/benchmark.h"

#include "bench/config_reader.h"
#include "bench/dataset.h"

#include <vector>
#include <unordered_map>
#include <cmath>

inline divftree::DIVFTree* vector_index = nullptr;
inline std::atomic<bool> dataset_finished = false;

inline std::atomic<size_t> search_queries = 0;
inline std::atomic<size_t> insert_queries = 0;
inline std::atomic<size_t> delete_queries = 0;
inline std::atomic<size_t> search_errors = 0;
inline std::atomic<size_t> insert_errors = 0;
inline std::atomic<size_t> delete_errors = 0;

inline divftree::SXSpinLock distance_lock;
inline DISTANCE_TYPE sum_search_distance = 0;
inline DISTANCE_TYPE avg_search_distance = 0;
inline uint64_t num_returned_neighbours = 0;
inline uint64_t num_total_returned_neighbours = 0;

inline std::atomic<uint32_t> build_ready = 0;
inline std::atomic<uint32_t> build_num_inserted = 0;
inline std::atomic<bool> build_start = false;

inline std::atomic<uint32_t> warmup_ready = false;
inline std::atomic<bool> warmup_start = false;
inline std::atomic<bool> warmup_finished = false;

inline std::atomic<uint32_t> run_ready = 0;
inline std::atomic<bool> run_start = false;
inline std::atomic<bool> run_finished = false;

inline std::atomic<uint32_t> run_done = 0;

struct IDSet {
    std::unordered_map<divftree::VectorID, size_t, divftree::VectorIDHash> _map;
    std::vector<divftree::VectorID> _data;

    void insert(divftree::VectorID id) {
        CHECK_VECTORID_IS_VALID(id, LOG_TAG_TEST);
        CHECK_VECTORID_IS_VECTOR(id, LOG_TAG_TEST);
        FatalAssert(_map.find(id) == _map.end(), LOG_TAG_TEST, "id should not exist in the set!");
        _map[id] = _data.size();
        _data.push_back(id);
    }

    void remove(divftree::VectorID id) {
        FatalAssert(_map.find(id) != _map.end(), LOG_TAG_TEST, "id should exist in the set!");
        size_t idx = _map[id];
        FatalAssert(idx < _data.size(), LOG_TAG_TEST, "invalid idx!");
        if (idx != _data.size() - 1) {
            _data[idx] = _data.back();
            _map[_data[idx]] = idx;
        }
        _data.pop_back();
        _map.erase(id);
    }

    bool exists(divftree::VectorID id) const {
        return (_map.find(id) != _map.end());
    }

    bool empty() const {
        return _data.empty();
    }

    size_t size() const {
        return _data.size();
    }

    divftree::VectorID get_random_id() const {
        return _data[divftree::threadSelf->UniformRange64(0, _data.size() - 1)];
    }
};

divftree::RetStatus Insert(FILE*& input_file_ptr, size_t& num_read, size_t& total_read, divftree::VTYPE* buffer,
                           divftree::VectorID& id, IDSet& id_set) {
    if (num_read == total_read) {
        total_read = ReadNextBatch(input_file_ptr, buffer);
        num_read = 0;
    }

    if (total_read == 0) {
        if (!dataset_finished.load()) {
            dataset_finished.store(true, std::memory_order_release);
            DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_TEST, "Dataset finished!");
        }
        return divftree::RetStatus::Fail();
    }

    /* todo: make sure that when we return from this it is safe to reuse buffer! */
    divftree::RetStatus rs = vector_index->Insert(&buffer[num_read * DIMENSION], id, default_internal_search_span);
    if (!rs.IsOK()) {
        DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_TEST, "Error during insert: %s", rs.Msg());
    }
    id_set.insert(id);
    ++num_read;
    return rs;
}

divftree::RetStatus Delete(IDSet& id_set) {
    divftree::RetStatus rs;
    if (id_set.empty()) {
        DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_TEST, "id_set is empty!");
        return divftree::RetStatus::Fail();
    }

    divftree::VectorID target = id_set.get_random_id();
    rs = vector_index->Delete(target);
    if (!rs.IsOK()) {
        DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_TEST, "Error during delete: %s", rs.Msg());
    } else {
        id_set.remove(target);
    }
    return rs;
}

divftree::RetStatus Search(std::vector<divftree::ANNVectorInfo>& neighbours) {
    divftree::RetStatus rs;
    neighbours.clear();
    size_t idx = divftree::threadSelf->UniformRange64(0, total_num_queries - 1);
    rs = vector_index->ApproximateKNearestNeighbours(&search_query_vectors[idx * DIMENSION], default_k,
                                                    default_internal_search_span, default_leaf_search_span,
                                                    divftree::SortType::Unsorted, neighbours);
    if (!rs.IsOK()) {
        DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_TEST, "Error during search: %s", rs.Msg());
    } else if (neighbours.empty()) {
        DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_TEST, "No neighbours found during search!");
        rs = divftree::RetStatus::Fail();
    } else if (collect_avg_distances) {
        DISTANCE_TYPE total_distance = 0;
        for (const auto& neighbour : neighbours) {
            total_distance += std::sqrt(neighbour.distance_to_query);
        }
        distance_lock.Lock(divftree::SX_EXCLUSIVE);
        num_returned_neighbours += neighbours.size();
        sum_search_distance += total_distance;
        distance_lock.Unlock();

    }
    return rs;
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
void worker(divftree::Thread* self) {
    self->InitDIVFThread();
    divftree::VTYPE buffer[bench_batch_size * DIMENSION];
    FILE* input_file_ptr = nullptr;
    OpenDataFile(input_file_ptr, false);
    size_t num_read = 0;
    size_t total_read = 0;
    divftree::VectorID id;
    IDSet id_set;
    std::vector<divftree::ANNVectorInfo> neighbours;
    divftree::RetStatus rs;

    uint32_t num_ready = build_ready.fetch_add(1);
    if (num_ready == num_threads - 1) {
        build_ready.notify_all();
    }

    bool ready = build_start.load(std::memory_order_acquire);
    while(!ready) {
        build_start.wait(false);
        ready = build_start.load(std::memory_order_acquire);
    }

    uint32_t insert_batch_size =
        std::max(1u, build_size / ((uint32_t)num_threads * (index_attr.leaf_max_size / 10 + 1)));
    while(build_num_inserted.load(std::memory_order_acquire) < build_size) {
        self->LoopIncrement();
        insert_batch_size = std::min(insert_batch_size,
                            build_size - build_num_inserted.load(std::memory_order_acquire));
        if (insert_batch_size == 0) {
            break;
        }

        uint32_t old_size = build_num_inserted.fetch_add(insert_batch_size);
        insert_batch_size = std::min(insert_batch_size,
                                     build_size - old_size);
        if (old_size >= build_size || insert_batch_size == 0) {
            break;
        }

        for (uint32_t i = 0; i < insert_batch_size; ++i) {
            (void)Insert(input_file_ptr, num_read, total_read, buffer, id, id_set);
        }
    }

    num_ready = warmup_ready.fetch_add(1);
    if (num_ready == num_threads - 1) {
        warmup_ready.notify_all();
    }
    ready = warmup_start.load(std::memory_order_acquire);
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

    num_ready = run_ready.fetch_add(1);
    if (num_ready == num_threads - 1) {
        run_ready.notify_all();
    }
    ready = run_start.load(std::memory_order_acquire);
    while(!ready) {
        run_start.wait(false);
        ready = run_start.load(std::memory_order_acquire);
    }

    current_search = 0;
    uint64_t current_insert = 0;
    uint64_t current_delete = 0;
    current_search_err = 0;
    uint64_t current_insert_err = 0;
    uint64_t current_delete_err = 0;
    while(!run_finished.load(std::memory_order_acquire)) {
        self->LoopIncrement();
        if(!self->UniformBinary(write_ratio)) {
            rs = Search(neighbours);
            if (rs.IsOK()) {
                FlushIncrement(current_search, search_queries);
            } else {
                FlushIncrement(current_search_err, search_errors);
            }
        } else if (id_set.size() > 2 && self->UniformBinary(delete_ratio)) {
            rs = Delete(id_set);
            if (rs.IsOK()) {
                FlushIncrement(current_delete, delete_queries);

            } else {
                FlushIncrement(current_delete_err, delete_errors);
            }
        } else {
            rs = Insert(input_file_ptr, num_read, total_read, buffer, id, id_set);
            if (rs.IsOK()) {
                FlushIncrement(current_insert, insert_queries);
            } else {
                FlushIncrement(current_insert_err, insert_errors);
            }
        }
    }

    if (current_search != 0) {
        search_queries.fetch_add(current_search);
        current_search = 0;
    }
    if (current_insert != 0) {
        insert_queries.fetch_add(current_insert);
        current_insert = 0;
    }
    if (current_delete != 0) {
        delete_queries.fetch_add(current_delete);
        current_delete = 0;
    }
    if (current_search_err != 0) {
        search_errors.fetch_add(current_search_err);
        current_search_err = 0;
    }
    if (current_insert_err != 0) {
        insert_errors.fetch_add(current_insert_err);
        current_insert_err = 0;
    }
    if (current_delete_err != 0) {
        delete_errors.fetch_add(current_delete_err);
        current_delete_err = 0;
    }

    num_ready = run_done.fetch_add(1);
    if (num_ready == num_threads - 1) {
        run_done.notify_all();
    }

    CloseFile(input_file_ptr);
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

int main() {
    ReadConfigs();
    ParseConfigs();
    FILE* file = nullptr;
    OpenDataFile(file, true);
    CloseFile(file);
    LoadQueryVectors();

    if (collect_build_stats) {
        index_attr.collect_stats = true;
    }

    BenchLog("Starting benchmark for %s(type:%s, dimension:%hu, distance:%s) "
           "with %lu threads for build-size:%u, warmup-time:%u(s), and run-time:%u(s) with %u percent writes "
           "and search span of %hhu and %hhu and default k of %hhu for leaf and internal vertices.",
           DATASET_NAME, DIVF_MACRO_TO_STR(VECTOR_TYPE), DIMENSION,
           divftree::DISTANCE_TYPE_NAME[(int8_t)DISTANCE_ALG], num_threads, build_size, warmup_time, run_time,
           write_ratio, default_leaf_search_span, default_internal_search_span, default_k);

    BenchLog("Initing the index with the attributes: %s", index_attr.ToString().ToCStr());

    vector_index = new divftree::DIVFTree(index_attr);
    divftree::String build_stats, warmup_stats, run_stats;

    std::vector<divftree::Thread*> threads(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        threads[i] = new divftree::Thread(100);
    }

    BenchLog("Starting %lu worker threads...", num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        threads[i]->Start(worker);
    }

    uint32_t num_ready = build_ready.load(std::memory_order_acquire);
    while (num_ready != num_threads) {
        build_ready.wait(num_ready);
        num_ready = build_ready.load(std::memory_order_acquire);
    }

    BenchLog("Start Build...");
    auto start_time = std::chrono::high_resolution_clock::now();
    build_start.store(true, std::memory_order_release);
    build_start.notify_all();

    num_ready = warmup_ready.load(std::memory_order_acquire);
    uint32_t last_report_time = 0;
    uint32_t last_num_inserted = 0;
    while (num_ready != num_threads) {
        if (show_runtime_report_for_build_and_warmup) {
            divftree::sleep(throughput_report_time);
            last_report_time += throughput_report_time;
            uint32_t curr_num_inserted = build_num_inserted.load(std::memory_order_acquire);
            ExclusiveBenchLog("[%u s]: Build Phase Report: Inserted %u, Total: %u/%u -> %.2f%% ", last_report_time,
                              curr_num_inserted - last_num_inserted, curr_num_inserted, build_size,
                              ((double)curr_num_inserted * 100.0) / (double)build_size);
            last_num_inserted = curr_num_inserted;
        } else {
            warmup_ready.wait(num_ready);
        }
        num_ready = warmup_ready.load(std::memory_order_acquire);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    size_t build_time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    if (collect_build_stats) {
        if (!collect_warmup_stats) {
            vector_index->StopStatsCollection();
            sleep(1); // make sure all stats are collected
        }
        build_stats = vector_index->GetStatistics("Build", true);
    } else if (collect_warmup_stats) {
        vector_index->StartStatsCollection();
    }

    BenchLog("Start Warmup...");
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
                                sum_search_distance / (DISTANCE_TYPE)num_returned_neighbours,
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
    num_ready = run_ready.load(std::memory_order_acquire);
    while (num_ready != num_threads) {
        run_ready.wait(num_ready);
        num_ready = run_ready.load(std::memory_order_acquire);
    }

    if (collect_warmup_stats) {
        if (!collect_run_stats) {
            vector_index->StopStatsCollection();
            sleep(1); // make sure all stats are collected
        }
        warmup_stats = vector_index->GetStatistics("Warmup", true);
    } else if (collect_run_stats) {
        vector_index->StartStatsCollection();
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

    BenchLog("Start Run...");
    start_time = std::chrono::high_resolution_clock::now();
    run_start.store(true, std::memory_order_release);
    run_start.notify_all();

    if (throughput_report_time == 0) {
        divftree::sleep(run_time);
    } else {
        uint32_t time_to_wait = throughput_report_time;
        size_t last_rps = 0, last_ips = 0, last_dps = 0, last_reps = 0, last_ieps = 0, last_deps = 0;
        for (uint32_t total_wait_time = 0; total_wait_time < run_time; total_wait_time += time_to_wait) {
            divftree::sleep(time_to_wait);
            size_t cur_rps = search_queries.load(std::memory_order_acquire);
            size_t cur_ips = insert_queries.load(std::memory_order_acquire);
            size_t cur_dps = delete_queries.load(std::memory_order_acquire);
            size_t cur_reps = search_errors.load(std::memory_order_acquire);
            size_t cur_ieps = insert_errors.load(std::memory_order_acquire);
            size_t cur_deps = delete_errors.load(std::memory_order_acquire);

            size_t total_qps = cur_rps + cur_ips + cur_dps - (last_rps + last_ips + last_dps);
            size_t total_eps = cur_reps + cur_ieps + cur_deps - (last_reps + last_ieps + last_deps);

            if (collect_avg_distances) {
                distance_lock.Lock(divftree::SX_EXCLUSIVE);
                avg_search_distance += sum_search_distance;
                num_total_returned_neighbours += num_returned_neighbours;

                ExclusiveBenchLog("[%u s]: QPS: %.2f - RPS: %.2f - IPS: %.2f - DPS: %.2f | "
                                  "Avg Distance: " DTYPE_FMT " | "
                                  "EPS: %.2f - REPS: %.2f - IEPS: %.2f - DEPS: %.2f",
                                  total_wait_time + time_to_wait, (double)total_qps / (double)time_to_wait,
                                  (double)(cur_rps - last_rps) / (double)time_to_wait,
                                  (double)(cur_ips - last_ips) / (double)time_to_wait,
                                  (double)(cur_dps - last_dps) / (double)time_to_wait,
                                  sum_search_distance / (DISTANCE_TYPE)num_returned_neighbours,
                                  (double)total_eps / (double)time_to_wait,
                                  (double)(cur_reps - last_reps) / (double)time_to_wait,
                                  (double)(cur_ieps - last_ieps) / (double)time_to_wait,
                                  (double)(cur_deps - last_deps) / (double)time_to_wait);

                sum_search_distance = 0;
                num_returned_neighbours = 0;
                distance_lock.Unlock();
            } else {
                ExclusiveBenchLog("[%u s]: QPS: %.2f - RPS: %.2f - IPS: %.2f - DPS: %.2f | "
                                "EPS: %.2f - REPS: %.2f - IEPS: %.2f - DEPS: %.2f",
                                total_wait_time + time_to_wait, (double)total_qps / (double)time_to_wait,
                                (double)(cur_rps - last_rps) / (double)time_to_wait,
                                (double)(cur_ips - last_ips) / (double)time_to_wait,
                                (double)(cur_dps - last_dps) / (double)time_to_wait,
                                (double)total_eps / (double)time_to_wait,
                                (double)(cur_reps - last_reps) / (double)time_to_wait,
                                (double)(cur_ieps - last_ieps) / (double)time_to_wait,
                                (double)(cur_deps - last_deps) / (double)time_to_wait);
            }
            last_rps = cur_rps;
            last_ips = cur_ips;
            last_dps = cur_dps;
            last_reps = cur_reps;
            last_ieps = cur_ieps;
            last_deps = cur_deps;
            if ((total_wait_time + time_to_wait > run_time)) {
                time_to_wait = run_time - total_wait_time;
            }
        }
    }

    run_finished.store(true, std::memory_order_release);
    num_ready = run_done.load(std::memory_order_acquire);
    while (num_ready != num_threads) {
        run_done.wait(num_ready);
        num_ready = run_done.load(std::memory_order_acquire);
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    size_t total_run_time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    if (collect_run_stats) {
        vector_index->StopStatsCollection();
        sleep(1); // make sure all stats are collected
        run_stats = vector_index->GetStatistics("Run", true);
    }

    BenchLog("Stopping all threads...");
    for (size_t i = 0; i < num_threads; ++i) {
        threads[i]->Join();
        delete threads[i];
        threads[i] = nullptr;
    }
    delete vector_index;

    if (collect_avg_distances) {
        avg_search_distance += sum_search_distance;
        num_total_returned_neighbours += num_returned_neighbours;

        if (num_total_returned_neighbours > 0) {
            avg_search_distance /= (DISTANCE_TYPE)num_total_returned_neighbours;
        } else if (avg_search_distance > 0) {
            DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_TEST, "Error: no neighbours returned but avg distance is not 0!");
            avg_search_distance = 0;
        }
    }

    size_t total_search = search_queries.load(std::memory_order_acquire);
    size_t total_insert = insert_queries.load(std::memory_order_acquire);
    size_t total_delete = delete_queries.load(std::memory_order_acquire);
    size_t total_search_err = search_errors.load(std::memory_order_acquire);
    size_t total_insert_err = insert_errors.load(std::memory_order_acquire);
    size_t total_delete_err = delete_errors.load(std::memory_order_acquire);

    if (dataset_finished.load(std::memory_order_acquire)) {
        ExclusiveBenchLog("Input dataset was finished during run!");
    }

    if (collect_build_stats) {
        ExclusiveBenchLog("%s", build_stats.ToCStr());
    }

    if (collect_warmup_stats) {
        ExclusiveBenchLog("%s", warmup_stats.ToCStr());
    }

    if (collect_run_stats) {
        ExclusiveBenchLog("%s", run_stats.ToCStr());
    }

    ExclusiveBenchLog("\n___________________________________________\n");
    ExclusiveBenchLog("Run Stats:\n");

    BenchLog("Build Time: %lu(ms)", build_time);
    BenchLog("Final Run Time: %lu(ms)", total_run_time);

    ExclusiveBenchLog("------------------------");

    build_time /= 1000;
    total_run_time /= 1000;

    BenchLog("Total queries: %lu", total_search + total_insert + total_delete);
    BenchLog("Total search queries: %lu", total_search);
    BenchLog("Total insert queries: %lu", total_insert);
    BenchLog("Total delete queries: %lu", total_delete);

    ExclusiveBenchNewLine();

    BenchLog("Total errors: %lu", total_search_err + total_insert_err + total_delete_err);
    BenchLog("Total search errors: %lu", total_search_err);
    BenchLog("Total insert errors: %lu", total_insert_err);
    BenchLog("Total delete errors: %lu", total_delete_err);

    ExclusiveBenchLog("------------------------");

    BenchLog("Total QPS: %.2f",
            ((double)(total_search + total_insert + total_delete)) / (double)total_run_time);
    BenchLog("Search QPS: %.2f", ((double)total_search) / (double)total_run_time);
    BenchLog("Insert QPS: %.2f", ((double)total_insert) / (double)total_run_time);
    BenchLog("Delete QPS: %.2f", ((double)total_delete) / (double)total_run_time);

    ExclusiveBenchNewLine();

    BenchLog("Total EPS: %.2f",
            ((double)(total_search_err + total_insert_err + total_delete_err)) / (double)total_run_time);
    BenchLog("Search EPS: %.2f", ((double)total_search_err) / (double)total_run_time);
    BenchLog("Insert EPS: %.2f", ((double)total_insert_err) / (double)total_run_time);
    BenchLog("Delete EPS: %.2f", ((double)total_delete_err) / (double)total_run_time);

    if (collect_avg_distances) {
        ExclusiveBenchLog("------------------------");

        BenchLog("Average Search Distance: " DTYPE_FMT, avg_search_distance);
    }

    ExclusiveBenchLog("\n___________________________________________\n");

    delete[] search_query_vectors;
    search_query_vectors = nullptr;
}