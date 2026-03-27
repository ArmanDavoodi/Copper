#ifndef DATASET_H_
#define DATASET_H_

#include "bench/configurations.h"

inline uint32_t total_num_vectors = 0;
inline std::atomic<uint32_t> next_offset = 0;

inline uint32_t total_num_queries = 0;
inline divftree::VTYPE* search_query_vectors = nullptr;
inline std::pair<divftree::DTYPE, divftree::IVFVectorID>** exact_knn_results = nullptr;

void OpenDataFile(FILE*& input_file_ptr, bool read_header = false) {
    if (input_file_ptr != nullptr) {
        fclose(input_file_ptr);
        input_file_ptr = nullptr;
    }
    input_file_ptr = fopen(DATA_PATH, "rb");
    if (input_file_ptr == nullptr) {
        throw std::runtime_error(
                divftree::String("Could not open the output data file! errno %d, errno msg: %s",
                                    errno, strerror(errno)).ToCStr());
    }

    uint32_t dummy_num_v = 0;
    uint32_t* num_vec = nullptr;
    if (!read_header) {
        num_vec = &dummy_num_v;
    } else {
        num_vec = &total_num_vectors;
    }

    size_t ret_code = fread(num_vec, sizeof(uint32_t), 1, input_file_ptr);
    if (ret_code != 1) {
        if (feof(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected end of file!");
        else if (ferror(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected error!");
    }

    if (*num_vec == 0) {
        throw std::runtime_error("Error: num vectors cannot be 0!");
    }

    if (!read_header && (total_num_vectors != dummy_num_v)) {
        throw std::runtime_error("Error: total num vectors mismatch!");
    }

    uint32_t num_dimensions = 0;
    ret_code = fread(&num_dimensions, sizeof(uint32_t), 1, input_file_ptr);
    if (ret_code != 1) {
        if (feof(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected end of file!");
        else if (ferror(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected error!");
    }

    if (num_dimensions != DIMENSION) {
        throw std::runtime_error("Error: dimension mismatch!");
    }
}

size_t ReadNextBatch(FILE* input_file_ptr, divftree::VTYPE* buffer) {
    if (input_file_ptr == nullptr) {
        throw std::runtime_error("Error: file is not open!");
    }

    if (buffer == nullptr) {
        throw std::runtime_error("Error: buffer is null!");
    }

    uint32_t offset = next_offset.fetch_add(bench_batch_size * DIMENSION * sizeof(divftree::VTYPE));

    if (offset >= total_num_vectors * DIMENSION * sizeof(divftree::VTYPE)) {
        return 0;
    }

    size_t ret_code = fseek(input_file_ptr, sizeof(uint32_t) * 2 + offset, SEEK_SET);
    if (ret_code != 0) {
        throw std::runtime_error("Error seeking data file!");
    }

    ret_code = fread(buffer, sizeof(divftree::VTYPE) * DIMENSION, bench_batch_size, input_file_ptr);
    if (ret_code < bench_batch_size && offset < (total_num_vectors * DIMENSION * sizeof(divftree::VTYPE))) {
        if (feof(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected end of file!");
        else if (ferror(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected error!");
    }

    return ret_code;
}

void CloseFile(FILE* input_file_ptr) {
    if (input_file_ptr != nullptr) {
        fclose(input_file_ptr);
        input_file_ptr = nullptr;
    }
}

void LoadQueryVectors() {
    if (search_query_vectors != nullptr || total_num_queries != 0) {
        throw std::runtime_error("query vectors are already loaded!");
    }

    FILE* input_file_ptr = fopen(QUERY_PATH, "rb");
    if (input_file_ptr == nullptr) {
        throw std::runtime_error(
                divftree::String("Could not open the query data file! errno %d, errno msg: %s",
                                    errno, strerror(errno)).ToCStr());
    }

    size_t ret_code = fread(&total_num_queries, sizeof(uint32_t), 1, input_file_ptr);
    if (ret_code != 1) {
        if (feof(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected end of file!");
        else if (ferror(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected error!");
    }

    if (total_num_queries == 0) {
        throw std::runtime_error("Error: num vectors cannot be 0!");
    }

    uint32_t num_dimensions = 0;
    ret_code = fread(&num_dimensions, sizeof(uint32_t), 1, input_file_ptr);
    if (ret_code != 1) {
        if (feof(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected end of file!");
        else if (ferror(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected error!");
    }

    if (num_dimensions != DIMENSION) {
        throw std::runtime_error("Error: dimension mismatch!");
    }

    search_query_vectors = new divftree::VTYPE[total_num_queries * DIMENSION];
    ret_code = fread(search_query_vectors, sizeof(divftree::VTYPE) * DIMENSION, total_num_queries, input_file_ptr);
    if (ret_code < total_num_queries) {
        if (feof(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected end of file!");
        else if (ferror(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected error!");
    }

    CloseFile(input_file_ptr);
}

void LoadExactKNNResults(std::string& file_path, uint8_t K) {
    if (exact_knn_results != nullptr) {
        throw std::runtime_error("Exact KNN results are already loaded!");
    }

    FILE* input_file_ptr = fopen(file_path.c_str(), "rb");
    if (input_file_ptr == nullptr) {
        throw std::runtime_error(
                divftree::String("Could not open the exact KNN results file! errno %d, errno msg: %s",
                                    errno, strerror(errno)).ToCStr());
    }

    uint32_t num_queries_in_file = 0;
    size_t ret_code = fread(&num_queries_in_file, sizeof(uint32_t), 1, input_file_ptr);
    if (ret_code != 1) {
        if (feof(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected end of file!");
        else if (ferror(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected error!");
    }

    if (num_queries_in_file != total_num_queries) {
        throw std::runtime_error("Error: num queries in file does not match total_num_queries!");
    }

    uint16_t num_dimensions = 0;
    ret_code = fread(&num_dimensions, sizeof(uint16_t), 1, input_file_ptr);
    if (ret_code != 1) {
        if (feof(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected end of file!");
        else if (ferror(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected error!");
    }

    if (num_dimensions != DIMENSION) {
        throw std::runtime_error("Error: dimension in file does not match expected dimension!");
    }

    uint32_t k_in_file = 0;
    ret_code = fread(&k_in_file, sizeof(uint32_t), 1, input_file_ptr);
    if (ret_code != 1) {
        if (feof(input_file_ptr))
            throw std::runtime_error("Error reading data file: unexpected end of file!");
        else if (ferror(input_file_ptr))
        throw std::runtime_error("Error reading data file: unexpected error!");
    }

    if (k_in_file < (uint32_t)K) {
        throw std::runtime_error("Error: K in file is less than target K");
    }

    exact_knn_results = new std::pair<divftree::DTYPE, divftree::IVFVectorID>*[total_num_queries];
    for (uint32_t q = 0; q < total_num_queries; q++) {
        exact_knn_results[q] = new std::pair<divftree::DTYPE, divftree::IVFVectorID>[K];
        for (uint32_t k = 0; k < K; k++) {
            ret_code = fread(&exact_knn_results[q][k].second, sizeof(divftree::IVFVectorID), 1, input_file_ptr);
            if (ret_code != 1) {
                if (feof(input_file_ptr))
                    throw std::runtime_error("Error reading data file: unexpected end of file!");
                else if (ferror(input_file_ptr))
                    throw std::runtime_error("Error reading data file: unexpected error!");
            }
            ret_code = fread(&exact_knn_results[q][k].first, sizeof(divftree::DTYPE), 1, input_file_ptr);
            if (ret_code != 1) {
                if (feof(input_file_ptr))                    throw std::runtime_error("Error reading data file: unexpected end of file!");
                else if (ferror(input_file_ptr))
                    throw std::runtime_error("Error reading data file: unexpected error!");
            }
            if (k > 0 && exact_knn_results[q][k].first < exact_knn_results[q][k - 1].first) {
                throw std::runtime_error("Error: KNN results in file are not sorted by distance!");
            }
        }

        if (K != k_in_file) {
            // skip the remaining neighbors in the file for this query
            ret_code = fseek(input_file_ptr, (k_in_file - K) * (sizeof(divftree::IVFVectorID) + sizeof(divftree::DTYPE)), SEEK_CUR);
            if (ret_code != 0) {
                throw std::runtime_error("Error seeking data file!");
            }
        }
    }

    CloseFile(input_file_ptr);
}

#endif