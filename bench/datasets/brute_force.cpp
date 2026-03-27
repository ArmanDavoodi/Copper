#include <iostream>
#include <cstdint>
#include <string>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <omp.h>
#include <sys/mman.h>
#include <set>
#include <math.h>

#define UINT8 1
#define UINT16 2
#define FLOAT 3
#define UINT32 4
#define UINT64 5
#define DOUBLE 6

namespace divftree {
#if defined(VECTOR_TYPE)
    #if VECTOR_TYPE == UINT8
        using VTYPE = uint8_t;
        #define VTYPE_FMT "%hhu"
        #pragma message("DTYPE = UINT8")
    #elif VECTOR_TYPE == UINT16
        using VTYPE = uint16_t;
        #define VTYPE_FMT "%hu"
        #pragma message("DTYPE = UINT16")
    #elif VECTOR_TYPE == FLOAT
        using VTYPE = float;
        #define VTYPE_FMT "%0.2f"
        #pragma message("DTYPE = FLOAT")
    #else
        #error UNDEFINED VECTOR_TYPE!
    #endif
#else
using VTYPE = uint8_t;
#define VTYPE_FMT "%hhu"
#error VECTOR_TYPE not found!
#endif

#if defined(CENTROID_TYPE)
    #if CENTROID_TYPE == UINT8
        using CTYPE = uint8_t;
        #define CTYPE_FMT "%hhu"
        using MVTYPE = double;
        #define MVTYPE_FMT "%0.4f"
        #pragma message("CTYPE = UINT8")
    #elif CENTROID_TYPE == UINT16
        using CTYPE = uint16_t;
        #define CTYPE_FMT "%hu"
        using MVTYPE = double;
        #define MVTYPE_FMT "%0.4f"
        #pragma message("CTYPE = UINT16")
    #elif CENTROID_TYPE == FLOAT
        using CTYPE = float;
        #define CTYPE_FMT "%0.2f"
        using MVTYPE = double;
        #define MVTYPE_FMT "%0.4f"
        #pragma message("CTYPE = FLOAT")
    #else
        #error UNDEFINED CENTROID_TYPE!
    #endif
#else
using CTYPE = uint8_t;
#define CTYPE_FMT "%hhu"
using MVTYPE = double;
#define MVTYPE_FMT "%lu"
#error CENTROID_TYPE not found!
#endif

#if defined(DISTANCE_TYPE)
    #if DISTANCE_TYPE == UINT8
        using DTYPE = uint8_t;
        #define DTYPE_FMT "%hhu"
        #pragma message("DTYPE = UINT8")
    #elif DISTANCE_TYPE == UINT16
        using DTYPE = uint16_t;
        #define DTYPE_FMT "%hu"
        #pragma message("DTYPE = UINT16")
    #elif DISTANCE_TYPE == UINT32
        using DTYPE = uint32_t;
        #define DTYPE_FMT "%u"
        #pragma message("DTYPE = UINT32")
    #elif DISTANCE_TYPE == UINT64
        using DTYPE = uint64_t;
        #define DTYPE_FMT "%lu"
        #pragma message("DTYPE = UINT64")
    #elif DISTANCE_TYPE == FLOAT
        using DTYPE = float;
        #define DTYPE_FMT "%0.2f"
        #pragma message("DTYPE = FLOAT")
    #elif DISTANCE_TYPE == DOUBLE
        using DTYPE = double;
        #define DTYPE_FMT "%0.4f"
        #pragma message("DTYPE = DOUBLE")
    #else
        #error UNDEFINED DISTANCE_TYPE!
    #endif
#else
using DTYPE = uint8_t;
#define DTYPE_FMT "%hhu"
#error DISTANCE_TYPE not found!
#endif
}

#ifndef DIMENSION
#define DIMENSION 128
#error DIMENSION not defined!
#endif

#include "common.h"

template<typename T1, typename T2>
inline divftree::DTYPE L2Squared(const T1* a, const T2* b) {
    divftree:: DTYPE dist = 0;
    for (uint16_t i = 0; i < DIMENSION; i++) {
        divftree::DTYPE diff = static_cast<divftree::DTYPE>(a[i]) - static_cast<divftree::DTYPE>(b[i]);
        dist += diff * diff;
    }
    return dist;
}

inline bool MoreSimilar(const divftree::DTYPE& dist1, const divftree::DTYPE& dist2) {
    return dist1 < dist2;
}

struct IVFDistIDPairCMP {
    inline bool operator()(const std::pair<divftree::DTYPE, divftree::IVFVectorID>& a,
                           const std::pair<divftree::DTYPE, divftree::IVFVectorID>& b) const {
        return MoreSimilar(a.first, b.first) || (a.first == b.first &&
                                                 (a.second.vector_hash < b.second.vector_hash ||
                                                  (a.second.vector_hash == b.second.vector_hash &&
                                                   a.second.value < b.second.value)));
    }
};

struct Args {
    std::string dataset_path;
    std::string query_path;
    std::string output_path;
    FILE* dataset_fp;
    FILE* query_fp;
    FILE* output_fp;
    uint32_t num_threads;
    uint32_t num_points_to_use;
    uint32_t batch_size;
    uint32_t k;
    uint32_t num_queries;

    void print() const {
        std::cout << "Dataset path: " << dataset_path << "\n"
                  << "Query path: " << query_path << "\n"
                  << "Output path: " << output_path << "\n"
                  << "Num threads: " << num_threads << "\n"
                  << "Num points to use: " << num_points_to_use << "\n"
                  << "Batch size: " << batch_size << "\n"
                  << "k: " << k << "\n" << std::flush;
    }
};

struct __attribute__((packed)) VectorData {
    divftree::IVFVectorID id;
    uint32_t num_duplicates;
    divftree::VTYPE data[DIMENSION];
};

inline void PrintUsage(int argc, char* argv[]) {
    std::cerr << "Usage: " << argv[0] << " <dataset_path> <query_path> <output_file> <num_threads> <num_points> <batch_size> <k>\n";

    std::cerr << "Recived " << argc - 1 << " arguments:\n";
    for (int i = 1; i < argc; i++) {
        std::cerr << "\tArgument " << i << ": " << argv[i] << "\n";
    }
}


void ParseArgs(int argc, char* argv[], Args& args) {
    if (argc != 8) {
        PrintUsage(argc, argv);
        FatalAssert(false, LOG_TAG_BASIC, "Invalid number of arguments!");
        exit(EXIT_FAILURE);
    }

    args.dataset_path = argv[1];
    args.dataset_fp = fopen(args.dataset_path.c_str(), "rb");
    if (args.dataset_fp == nullptr) {
        std::cerr << "Error opening dataset file at path: " << args.dataset_path << "\n";
        FatalAssert(false, LOG_TAG_BASIC, "Error opening dataset file!");
        exit(EXIT_FAILURE);
    }

    args.query_path = argv[2];
    if (strcmp(args.query_path.c_str(), args.dataset_path.c_str()) == 0) {
        std::cerr << "Error: Query file path cannot be the same as dataset file path.\n";
        fclose(args.dataset_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Query file path cannot be the same as dataset file path!");
        exit(EXIT_FAILURE);
    }

    args.query_fp = fopen(args.query_path.c_str(), "rb");
    if (args.query_fp == nullptr) {
        std::cerr << "Error opening query file at path: " << args.query_path << "\n";
        fclose(args.dataset_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error opening query file!");
        exit(EXIT_FAILURE);
    }

    args.output_path = argv[3];
    if (strcmp(args.output_path.c_str(), args.query_path.c_str()) == 0) {
        std::cerr << "Error: Output file path cannot be the same as query file path.\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Output file path cannot be the same as query file path!");
        exit(EXIT_FAILURE);
    }

    if (strcmp(args.output_path.c_str(), args.dataset_path.c_str()) == 0) {
        std::cerr << "Error: Output file path cannot be the same as dataset file path.\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Output file path cannot be the same as dataset file path!");
        exit(EXIT_FAILURE);
    }

    if (args.output_path.empty()) {
        std::cerr << "Error: Output file path cannot be empty.\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Output file path cannot be empty!");
        exit(EXIT_FAILURE);
    }

    if (std::filesystem::exists(args.output_path)) {
        if (!std::filesystem::is_regular_file(args.output_path)) {
            std::cerr << "Error: Output file '" << args.output_path << "' exists but is not a regular file.\n";
            fclose(args.dataset_fp);
            fclose(args.query_fp);
            FatalAssert(false, LOG_TAG_BASIC, "Output file is not a regular file!");
            exit(EXIT_FAILURE);
        }
        std::cerr << "Warning: Output file '" << args.output_path << "' already exists and will be overwritten.\n";
    } else {
        // Extract the directory part
        std::filesystem::path out_path_obj(args.output_path);
        std::filesystem::path out_dir = out_path_obj.parent_path();

        // Create directories if they don't exist
        if (!out_dir.empty() && !std::filesystem::exists(out_dir)) {
            std::cout << "Output directory '" << out_dir.string() << "' does not exist. Creating directories...\n" << std::flush;
            std::filesystem::create_directories(out_dir);
        }
    }

    args.output_fp = fopen(args.output_path.c_str(), "wb");
    if (!args.output_fp) {
        std::cerr << "Error: Failed to open output file '" << args.output_path << "'. errno: " << errno << ", msg: " << strerror(errno) << "\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Failed to open output file!");
        exit(EXIT_FAILURE);
    }

    args.num_threads = std::stoul(argv[4]);
    uint32_t max_num_threads = omp_get_max_threads();
    if (args.num_threads == 0 || args.num_threads > max_num_threads) {
        std::cerr << "Warning, num_threads(" << args.num_threads << ") must be between 1 and " << max_num_threads << ". Using " << max_num_threads << " threads.\n";
        args.num_threads = max_num_threads;
    }

    uint32_t num_total_points_in_file;
    size_t ret = fread(&num_total_points_in_file, sizeof(uint32_t), 1, args.dataset_fp);
    if (ret != 1) {
        std::cerr << "Error reading num_total_points from dataset file.\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error reading num_total_points from dataset file!");
        exit(EXIT_FAILURE);
    }

    uint32_t num_unique_points = 0;
    ret = fread(&num_unique_points, sizeof(uint32_t), 1, args.dataset_fp);
    if (ret != 1) {
        std::cerr << "Error reading num_unique_points from dataset file.\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error reading num_unique_points from dataset file!");
        exit(EXIT_FAILURE);
    }

    uint16_t dimension = 0;
    ret = fread(&dimension, sizeof(uint16_t), 1, args.dataset_fp);
    if (ret != 1) {
        std::cerr << "Error reading dimension from dataset file.\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error reading dimension from dataset file!");
        exit(EXIT_FAILURE);
    }

    if (dimension != DIMENSION) {
        std::cerr << "Error: Dimension in dataset file (" << dimension << ") does not match expected dimension (" << DIMENSION << ").\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Dimension in dataset file does not match expected dimension!");
        exit(EXIT_FAILURE);
    }

    if (num_total_points_in_file < num_unique_points) {
        std::cerr << "Error: num_total_points in dataset file (" << num_total_points_in_file << ") is less than num_unique_points (" << num_unique_points << ").\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "num_total_points in dataset file is less than num_unique_points!");
        exit(EXIT_FAILURE);
    }

    args.num_points_to_use = std::stoul(argv[5]);
    if ((args.num_points_to_use > num_unique_points) || (args.num_points_to_use == 0)) {
        std::cerr << "Warning: num_points_to_use (" << args.num_points_to_use << ") must be between 1 and num_unique_points (" << num_unique_points << "). Using " << num_unique_points << " points.\n";
        args.num_points_to_use = num_unique_points;
    }

    args.batch_size = std::stoul(argv[6]);
    if ((args.batch_size == 0) || (args.batch_size > args.num_points_to_use)) {
        std::cerr << "Warning: batch_size (" << args.batch_size << ") must be between 1 and num_points_to_use (" << args.num_points_to_use << "). Using batch_size = num_points_to_use (" << args.num_points_to_use << ").\n";
        args.batch_size = args.num_points_to_use;
    }

    args.k = std::stoul(argv[7]);
    if ((args.k == 0) || (args.k > args.num_points_to_use)) {
        std::cerr << "Error: k (" << args.k << ") must be between 1 and num_points_to_use (" << args.num_points_to_use << ").\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "k must be between 1 and num_points_to_use!");
        exit(EXIT_FAILURE);
    }

    ret = fread(&args.num_queries, sizeof(uint32_t), 1, args.query_fp);
    if (ret != 1) {
        std::cerr << "Error reading num_queries from query file.\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error reading num_queries from query file!");
        exit(EXIT_FAILURE);
    }

    if (args.num_queries == 0) {
        std::cerr << "Error: num_queries must be greater than 0.\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "num_queries must be greater than 0!");
        exit(EXIT_FAILURE);
    }

    ret = fread(&dimension, sizeof(uint32_t), 1, args.query_fp);
    if (ret != 1) {
        std::cerr << "Error reading dimension from query file.\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error reading dimension from query file!");
        exit(EXIT_FAILURE);
    }

    if (dimension != DIMENSION) {
        std::cerr << "Error: Dimension in query file (" << dimension << ") does not match expected dimension (" << DIMENSION << ").\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Dimension in query file does not match expected dimension!");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char* argv[]) {
    divftree::Thread st(100);
    st.InitDIVFThread(true);

    Args args;
    ParseArgs(argc, argv, args);
    args.print();

    std::cout << "Allocating enough memory for batch and query data...\n" << std::flush;
    uint32_t num_remaining_points = args.num_points_to_use;
    VectorData* data = (VectorData*)mmap64(nullptr, (size_t)args.batch_size * sizeof(VectorData),
                                           PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (data == MAP_FAILED) {
        std::cerr << "Error allocating memory for batch data. errno: " << errno << ", msg: " << strerror(errno) << "\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error allocating memory for batch data!");
        exit(EXIT_FAILURE);
    }

    divftree::VTYPE* query_data = (divftree::VTYPE*)mmap64(nullptr, (size_t)args.num_queries * (size_t)DIMENSION * sizeof(divftree::VTYPE),
                                           PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (query_data == MAP_FAILED) {
        std::cerr << "Error allocating memory for query data. errno: " << errno << ", msg: " << strerror(errno) << "\n";
        fclose(args.dataset_fp);
        fclose(args.query_fp);
        fclose(args.output_fp);
        munmap(data, (size_t)args.batch_size * sizeof(VectorData));
        FatalAssert(false, LOG_TAG_BASIC, "Error allocating memory for query data!");
        exit(EXIT_FAILURE);
    }

    std::cout << "Reading query data from file...\n" << std::flush;
    size_t ret = fread(query_data, 1, (size_t)args.num_queries * (size_t)DIMENSION * sizeof(divftree::VTYPE), args.query_fp);
    fclose(args.query_fp);
    if (ret != (size_t)args.num_queries * (size_t)DIMENSION * sizeof(divftree::VTYPE)) {
        std::cerr << "Error reading query data from file.\n";
        fclose(args.dataset_fp);
        fclose(args.output_fp);
        munmap(data, (size_t)args.batch_size * sizeof(VectorData));
        munmap(query_data, (size_t)args.num_queries * (size_t)DIMENSION * sizeof(divftree::VTYPE));
        FatalAssert(false, LOG_TAG_BASIC, "Error reading query data from file!");
        exit(EXIT_FAILURE);
    }

    std::set<std::pair<divftree::DTYPE, divftree::IVFVectorID>, IVFDistIDPairCMP> knn_results[args.num_queries];

    std::cout << "Starting KNN Computation...\n" << std::flush;
    uint64_t batch_num = 0;
    uint64_t num_total_batches = (args.num_points_to_use + args.batch_size - 1) / args.batch_size;
    while (num_remaining_points > 0) {
        uint32_t current_batch_size = std::min(num_remaining_points, args.batch_size);
        num_remaining_points -= current_batch_size;

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Reading batch %lu/%lu (batch size: %u, remaining points after this batch: %u)...",
                batch_num + 1, num_total_batches, current_batch_size, num_remaining_points);
        ret = fread(data, 1, (size_t)(current_batch_size) * sizeof(VectorData), args.dataset_fp);
        if (ret != (size_t)(current_batch_size) * sizeof(VectorData)) {
            std::cerr << "Error reading batch data from file.\n";
            fclose(args.dataset_fp);
            fclose(args.output_fp);
            munmap(data, (size_t)args.batch_size * sizeof(VectorData));
            munmap(query_data, (size_t)args.num_queries * (size_t)DIMENSION * sizeof(divftree::VTYPE));
            FatalAssert(false, LOG_TAG_BASIC, "Error reading batch data from file!");
            exit(EXIT_FAILURE);
        }

        #pragma omp parallel for num_threads(args.num_threads) schedule(static)
        for (uint32_t q = 0; q < args.num_queries; q++) {
            for (uint32_t i = 0; i < current_batch_size; i++) {
                divftree::DTYPE dist = L2Squared(data[i].data, query_data + (size_t)q * (size_t)DIMENSION);
                divftree::IVFVectorID id = data[i].id;
                knn_results[q].emplace(dist, id);
                if (knn_results[q].size() > args.k) {
                    knn_results[q].erase(std::prev(knn_results[q].end()));
                    // knn_results[q].erase(knn_results[q].begin());
                }
            }
        }

        ++batch_num;
    }

    std::cout << "Finished KNN Computation for all batches.\n" << std::flush;
    fclose(args.dataset_fp);
    munmap(data, (size_t)args.batch_size * sizeof(VectorData));
    munmap(query_data, (size_t)args.num_queries * (size_t)DIMENSION * sizeof(divftree::VTYPE));

    std::cout << "Writing KNN results to output file...\n" << std::flush;
    ret = fwrite(&args.num_queries, sizeof(uint32_t), 1, args.output_fp);
    if (ret != 1) {
        std::cerr << "Error writing num_queries to output file.\n";
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error writing num_queries to output file!");
        exit(EXIT_FAILURE);
    }

    uint16_t dim = DIMENSION;
    ret = fwrite(&dim, sizeof(uint16_t), 1, args.output_fp);
    if (ret != 1) {
        std::cerr << "Error writing dimension to output file.\n";
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error writing dimension to output file!");
        exit(EXIT_FAILURE);
    }

    ret = fwrite(&args.k, sizeof(uint32_t), 1, args.output_fp);
    if (ret != 1) {
        std::cerr << "Error writing k to output file.\n";
        fclose(args.output_fp);
        FatalAssert(false, LOG_TAG_BASIC, "Error writing k to output file!");
        exit(EXIT_FAILURE);
    }

    divftree::MVTYPE avg_dist = 0;
    for (uint32_t q = 0; q < args.num_queries; q++) {
        divftree::DTYPE last_dist = 0;
        for (const auto& [dist, id] : knn_results[q]) {
            avg_dist += (divftree::MVTYPE)dist;
            FatalAssert(last_dist <= dist, LOG_TAG_BASIC, "KNN results are not sorted by distance!");
            last_dist = dist;
            ret = fwrite(&id, sizeof(divftree::IVFVectorID), 1, args.output_fp);
            if (ret != 1) {
                std::cerr << "Error writing vector ID for query " << q << " to output file.\n";
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing vector ID to output file!");
                exit(EXIT_FAILURE);
            }

            ret = fwrite(&dist, sizeof(divftree::DTYPE), 1, args.output_fp);
            if (ret != 1) {
                std::cerr << "Error writing distance for query " << q << " to output file.\n";
                fclose(args.output_fp);
                FatalAssert(false, LOG_TAG_BASIC, "Error writing distance to output file!");
                exit(EXIT_FAILURE);
            }
        }
        FatalAssert(args.k == knn_results[q].size(), LOG_TAG_BASIC, "KNN result size does not match k!");
    }

    avg_dist /= (divftree::MVTYPE)(args.num_queries) * (divftree::MVTYPE)(args.k);
    avg_dist = sqrt(avg_dist);
    divftree::DTYPE avg_dist_uint = static_cast<divftree::DTYPE>(avg_dist);
    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Average distance across all queries: " DTYPE_FMT, avg_dist_uint);

    fclose(args.output_fp);
}