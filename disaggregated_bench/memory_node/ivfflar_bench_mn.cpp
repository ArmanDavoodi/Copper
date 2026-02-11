#include "disaggregated_bench/memory_node/ivfflat_bench.h"
#include "disaggregated_bench/memory_node/ivfflat_config_reader.h"

#include "bench/dataset.h"

#include <vector>
#include <unordered_map>
#include <cmath>

inline divftree::MN_DIVFIndex* vector_index = nullptr;


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
    std::pair<divftree::String, divftree::String> node_strs = divftree::ReadNetworkConfigs();
    divftree::NodeID self_id = divftree::NodeID(true, divftree::network_config::memory_node_ids[divftree::network_config::self_idx]);
    ReadConfigs();
    ParseConfigs(self_id);
    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
            "Completed reading network configuration: %hhu memory nodes:%s, %hhu compute nodes:%s, "
            "RDMA device: %s, port: %hhu, GID index: %d",
            ::divftree::network_config::num_memory_nodes, node_strs.first.ToCStr(),
            ::divftree::network_config::num_compute_nodes, node_strs.second.ToCStr(),
            ::divftree::network_config::rdma_device_name, ::divftree::network_config::rdma_port,
            ::divftree::network_config::gid_index);
    FILE* file = nullptr;
    OpenDataFile(file, true);

    divftree::Thread main_thread(100);
    main_thread.InitDIVFThread(DIMENSION);

    BenchLog("Starting benchmark from MN for %s(type:%s, dimension:%hu, distance:%s) "
             "with %lu threads for build-size:%u. "
             "num_clusters = %zu, max_iters = %zu, insert_duplicates = %s",
             DATASET_NAME, DIVF_MACRO_TO_STR(VECTOR_TYPE), DIMENSION,
             divftree::DISTANCE_TYPE_NAME[(int8_t)DISTANCE_ALG], num_threads, build_size,
             num_clusters, max_iters, insert_duplicates ? "true" : "false");

    FatalAssert(build_size <= total_num_vectors, LOG_TAG_TEST,
                "Build size (%u) cannot be larger than total number of vectors (%u)!",
                build_size, total_num_vectors);

    data_set = new divftree::VTYPE[(size_t)build_size * DIMENSION];
    bench_batch_size = build_size;

    size_t read_size = ReadNextBatch(file, data_set);
    FatalAssert(read_size == build_size, LOG_TAG_TEST,
                "Error reading data for build! requested: %zu, read: %zu", build_size, read_size);

    CloseFile(file);

    BenchLog("Start Build...");
    auto start_time = std::chrono::high_resolution_clock::now();
    vector_index = new divftree::MN_DIVFIndex(data_set, build_size, num_clusters, insert_duplicates,
                                              max_iters, DIMENSION, page_size, num_threads);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    size_t build_time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    delete[] data_set;
    data_set = nullptr;

    BenchLog("Build Time: %lu(ms)", build_time);
    BenchLog("Start Listening for queries...");
    vector_index->Start();
    BenchLog("All CNs disconnected. Stopping benchmark...");
    delete vector_index;
}