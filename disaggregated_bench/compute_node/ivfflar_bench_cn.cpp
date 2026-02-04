#include "disaggregated_bench/compute_node/ivfflat_bench.h"
#include "disaggregated_bench/compute_node/ivfflat_config_reader.h"

#include "bench/dataset.h"

#include <vector>
#include <unordered_map>
#include <cmath>

inline divftree::DIVFIndex* vector_index = nullptr;

inline std::atomic<size_t> search_queries = 0;
inline std::atomic<size_t> search_errors = 0;

inline divftree::SXSpinLock distance_lock;
inline double sum_search_distance = 0;
inline double avg_search_distance = 0;
inline uint64_t num_returned_neighbours = 0;
inline uint64_t num_total_returned_neighbours = 0;

inline std::atomic<uint32_t> warmup_ready = false;
inline std::atomic<bool> warmup_start = false;
inline std::atomic<bool> warmup_finished = false;

inline std::atomic<uint32_t> run_ready = 0;
inline std::atomic<bool> run_start = false;
inline std::atomic<bool> run_finished = false;

inline std::atomic<uint32_t> run_done = 0;

divftree::RetStatus Search(std::vector<std::pair<divftree::DTYPE, divftree::IVFVectorID>>& neighbours) {
    divftree::RetStatus rs;
    neighbours.clear();
    size_t idx = divftree::threadSelf->UniformRange64(0, total_num_queries - 1);
    rs = vector_index->ANNSearch(&search_query_vectors[idx * DIMENSION], default_k, n_probes, neighbours);
    if (!rs.IsOK()) {
        DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_TEST, "Error during search: %s", rs.Msg());
    } else if (neighbours.empty()) {
        DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_TEST, "No neighbours found during search!");
        rs = divftree::RetStatus::Fail(nullptr);
    } else if (collect_avg_distances) {
        double total_distance = 0;
        for (auto& neighbour : neighbours) {
            total_distance += std::sqrt(neighbour.first);
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
    self->InitDIVFThread(DIMENSION);
    std::vector<std::pair<divftree::DTYPE, divftree::IVFVectorID>> neighbours;
    divftree::RetStatus rs;
    uint32_t num_ready = warmup_ready.fetch_add(1);
    if (num_ready == num_threads - 1) {
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
    current_search_err = 0;
    while(!run_finished.load(std::memory_order_acquire)) {
        self->LoopIncrement();
        rs = Search(neighbours);
        if (rs.IsOK()) {
            FlushIncrement(current_search, search_queries);
        } else {
            FlushIncrement(current_search_err, search_errors);
        }
    }

    if (current_search != 0) {
        search_queries.fetch_add(current_search);
        current_search = 0;
    }
    if (current_search_err != 0) {
        search_errors.fetch_add(current_search_err);
        current_search_err = 0;
    }

    num_ready = run_done.fetch_add(1);
    if (num_ready == num_threads - 1) {
        run_done.notify_all();
    }

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

int main(int argc, char** argv) {
    FatalAssert(argc == 2, LOG_TAG_TEST,
                "Usage: %s <self-node-idx>", argv[0]);
    divftree::network_config::self_idx = static_cast<uint8_t>(std::stoul(argv[1]));
    divftree::ReadNetworkConfigs();
    divftree::NodeID self_id = divftree::NodeID(false, divftree::network_config::compute_node_ids[divftree::network_config::self_idx]);
    ReadConfigs();
    ParseConfigs(self_id);

    divftree::Thread main_thread(100);
    main_thread.InitDIVFThread(DIMENSION);

    BenchLog("Starting benchmark from node:%s for %s(type:%s, dimension:%hu, distance:%s) "
             "with %lu threads, warmup-time:%u(s), and run-time:%u(s). "
             "default k = %hhu, n_probes = %zu",
             self_id.ToString().ToCStr(),
             DATASET_NAME, DIVF_MACRO_TO_STR(VECTOR_TYPE), DIMENSION,
             divftree::DISTANCE_TYPE_NAME[(int8_t)DISTANCE_ALG], num_threads, warmup_time, run_time,
             default_k, n_probes);

    divftree::DIVFIndexAttr attr{.dimension = DIMENSION,
                                 .num_user_threads = num_threads,
                                 .pool_size = pool_size,
                                 .page_size = page_size};
    BenchLog("Start Node...");
    vector_index = new divftree::DIVFIndex(attr);

    BenchLog("Load Query Vectors...");
    LoadQueryVectors();

    std::vector<divftree::Thread*> threads(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        threads[i] = new divftree::Thread(100);
    }

    BenchLog("Starting %lu worker threads...", num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
        threads[i]->Start(worker);
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
    while (num_ready != num_threads) {
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

    BenchLog("Start Run...");
    auto start_time = std::chrono::high_resolution_clock::now();
    run_start.store(true, std::memory_order_release);
    run_start.notify_all();

    if (throughput_report_time == 0) {
        divftree::sleep(run_time);
    } else {
        uint32_t time_to_wait = throughput_report_time;
        size_t last_rps = 0, last_reps = 0;
        for (uint32_t total_wait_time = 0; total_wait_time < run_time; total_wait_time += time_to_wait) {
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

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    size_t total_run_time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

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

    ExclusiveBenchLog("\n___________________________________________\n");

    delete[] search_query_vectors;
    search_query_vectors = nullptr;
}