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

/* log-output-file-dir is not the file name but the directory path */
void ReadArgs(int argc, char** argv) {
    if (argc != 4) {
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BASIC, "Usage: %s <self-node-idx> <index-file-path> <log-output-file-dir>", argv[0]);
    }

    divftree::network_config::self_idx = static_cast<uint8_t>(std::stoul(argv[1]));
    index_file_path = argv[2];
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
}

int main(int argc, char** argv) {
    ReadArgs(argc, argv);
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

    divftree::Thread main_thread(100);
    main_thread.InitDIVFThread((uint16_t)DIMENSION);

    BenchLog("Starting benchmark from MN %s for %s(vtype:%s, ctype:%s, dimension:%hu, distance:%s): index-file:%s",
             self_id.ToString().ToCStr(),
             DATASET_NAME, DIVF_MACRO_TO_STR(VECTOR_TYPE), DIVF_MACRO_TO_STR(CENTROID_TYPE), DIMENSION,
             divftree::DISTANCE_TYPE_NAME[(int8_t)DISTANCE_ALG], index_file_path.c_str());

    BenchLog("Start Load...");
    auto start_time = std::chrono::high_resolution_clock::now();
    vector_index = new divftree::MN_DIVFIndex(index_file_path);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    size_t load_time = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    BenchLog("Load Time: %lu(ms)", load_time);
    BenchLog("Start Listening for queries...");
    vector_index->Start();
    BenchLog("All CNs disconnected. Stopping benchmark...");
    delete vector_index;
}