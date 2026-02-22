#include <iostream>
#include <cstdint>
#include <string>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <omp.h>
#include <sys/mman.h>

#define UINT8 1
#define UINT16 2
#define FLOAT 3

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

#include "utils/vector_directory.h"

int main(int argc, char* argv[]) {
    std::string input_file;
    std::string output_file;
    size_t num_threads;

    uint32_t num_total_points;
    uint16_t dimension;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <num_threads>\n";
        return EXIT_FAILURE;
    }

    input_file = argv[1];
    output_file = argv[2];
    num_threads = std::stoul(argv[3]);

    if (!std::filesystem::exists(input_file)) {
        std::cerr << "Error: Input file does not exist.\n";
        return EXIT_FAILURE;
    }

    if (!std::filesystem::is_regular_file(input_file)) {
        std::cerr << "Error: Input path is not a regular file.\n";
        return EXIT_FAILURE;
    }

    size_t max_num_threads = omp_get_max_threads();
    if (num_threads == 0 || num_threads > max_num_threads) {
        std::cerr << "Warning, num_threads must be between 1 and " << max_num_threads << ". Using " << max_num_threads << " threads.\n";
        num_threads = max_num_threads;
    }

    // Extract the directory part
    std::filesystem::path out_path_obj(output_file);
    std::filesystem::path out_dir = out_path_obj.parent_path();

    // Create directories if they don't exist
    if (!out_dir.empty() && !std::filesystem::exists(out_dir)) {
        std::filesystem::create_directories(out_dir);
    }

    // Now open the file
    FILE* file = fopen(input_file.c_str(), "rb");
    if (!file) {
        perror("fopen");
        return EXIT_FAILURE;
    }

    size_t ret = fread(&num_total_points, sizeof(uint32_t), 1, file);
    if (ret != 1) {
        std::cerr << "Error reading num_total_points from file.\n";
        fclose(file);
        return EXIT_FAILURE;
    }

    if (num_total_points == 0) {
        std::cerr << "Error: num_total_points must be greater than 0.\n";
        fclose(file);
        return EXIT_FAILURE;
    }

    uint32_t dimension_32;
    ret = fread(&dimension_32, sizeof(uint32_t), 1, file);
    if (ret != 1) {
        std::cerr << "Error reading dimension from file.\n";
        fclose(file);
        return EXIT_FAILURE;
    }

    if (dimension_32 == 0 || dimension_32 > UINT16_MAX) {
        std::cerr << "Error: Dimension must be between 1 and " << UINT16_MAX << ".\n";
        fclose(file);
        return EXIT_FAILURE;
    }
    dimension = static_cast<uint16_t>(dimension_32);

    std::cout << "num_total_points: " << num_total_points << ", dimension: " << dimension << "\n";
    std::cout << "Reading vector data from file: " << input_file << "\n";

    divftree::VTYPE* data =
        (divftree::VTYPE*)mmap64(nullptr, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE),
                                 PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (data == MAP_FAILED) {
        std::cerr << "Error allocating memory for vector data. errno: " << errno << ", msg: " << strerror(errno) << "\n";
        fclose(file);
        return EXIT_FAILURE;
    }

    ret = fread(data, 1, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE), file);
    if (ret != (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE)) {
        std::cerr << "Error reading data from file.\n";
        fclose(file);
        munmap(data, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE));
        return EXIT_FAILURE;
    }
    fclose(file);

    divftree::VectorDirectory v_dir(num_total_points, dimension, num_threads * 2);

    std::cout << "Inserting vectors into VectorDirectory...\n";
    std::atomic<bool> error_occurred(false);
    std::atomic<size_t> insert_count(0);
    #pragma omp parallel num_threads(num_threads) shared(v_dir, data, error_occurred, insert_count)
    {
        divftree::Thread st(100);
        st.InitDIVFThread(true);
        #pragma omp for schedule(guided)
        for (size_t i = 0; i < num_total_points; i++) {
            if (error_occurred.load()) {
                continue;
            }

            bool is_duplicate;
            uint64_t hash = v_dir.GetVectorHash(&data[i * dimension]);
            v_dir.LockVector(hash, divftree::LockMode::SX_EXCLUSIVE);
            divftree::IVFVectorID vid = v_dir.Insert(hash, &data[i * dimension], false, &is_duplicate);

            size_t current_count = insert_count.fetch_add(1) + 1;
            if (current_count % (num_total_points / 100) == 0 && num_total_points >= 100) {
                printf("Progress: %zu%% (%zu/%u)\n", (current_count * 100) / num_total_points, current_count, num_total_points);
            }

            if (vid == divftree::INVALID_IVF_VECTOR_ID) {
                fprintf(stderr, "Error inserting vector %zu\n", i);
                error_occurred = true;
                v_dir.UnlockVector(hash);
                continue;
            }

            if (hash != vid.vector_hash) {
                fprintf(stderr, "Error: Hash mismatch for vector %zu. Expected hash: %lu, got: %lu\n", i, hash, vid.vector_hash);
                error_occurred = true;
                v_dir.UnlockVector(hash);
                continue;
            }

            if (is_duplicate) {
                v_dir.UnlockVector(hash);
                continue;
            }

            divftree::IVFVectorInfo* info = v_dir.Find(vid);
            if (info == nullptr) {
                fprintf(stderr, "Error finding vector info for vector %zu\n", i);
                v_dir.UnlockVector(hash);
                error_occurred = true;
                continue;
            }

            info->vector = &data[i * dimension];
            v_dir.UnlockVector(hash);
        }
        st.DestroyDIVFThread();
    }

    if (error_occurred.load()) {
        munmap(data, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE));
        return EXIT_FAILURE;
    }

    if (v_dir.Size(false) != num_total_points) {
        std::cerr << "Error: Expected " << num_total_points << " vectors, but found " << v_dir.Size(false) << ".\n";
    } else {
        std::cout << "Successfully inserted " << num_total_points << " vectors.\n";
    }

    size_t num_unique_points = v_dir.Size(true);
    if (num_unique_points > num_total_points || num_unique_points == 0) {
        std::cerr << "Error: Number of unique vectors is invalid.\n";
    }

    std::cout << "Number of unique vectors: " << num_unique_points << "\n";
    std::cout << "Writing vector information to file: " << output_file << "\n";

    file = fopen(output_file.c_str(), "wb");
    if (!file) {
        perror("fopen");
        munmap(data, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE));
        return EXIT_FAILURE;
    }

    divftree::VectorDirectoryNode* current = v_dir.GetHead();
    if (current == nullptr) {
        std::cerr << "Error: No vectors found in VectorDirectory.\n";
        fclose(file);
        munmap(data, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE));
        return EXIT_FAILURE;
    }

    size_t u_count = 0;
    size_t count = 0;
    uint32_t num_u = static_cast<uint32_t>(num_unique_points);
    ret = fwrite(&num_total_points, sizeof(uint32_t), 1, file);
    if (ret != 1) {
        std::cerr << "Error writing num_total_points to file.\n";
        fclose(file);
        munmap(data, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE));
        return EXIT_FAILURE;
    }
    ret = fwrite(&num_u, sizeof(uint32_t), 1, file);
    if (ret != 1) {
        std::cerr << "Error writing num_unique_points to file.\n";
        fclose(file);
        munmap(data, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE));
        return EXIT_FAILURE;
    }
    ret = fwrite(&dimension, sizeof(uint16_t), 1, file);
    if (ret != 1) {
        std::cerr << "Error writing dimension to file.\n";
        fclose(file);
        munmap(data, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE));
        return EXIT_FAILURE;
    }

    while(current != nullptr) {
        ret = fwrite(&(current->id), sizeof(divftree::IVFVectorID), 1, file);
        if (ret != 1) {
            std::cerr << "Error writing vector ID to file.\n";
            fclose(file);
            munmap(data, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE));
            return EXIT_FAILURE;
        }
        if (current->num_duplicates >= num_total_points) {
            std::cerr << "Error: Invalid number of duplicates for vector ID " << current->id.ToString().ToCStr() << ".\n";
            fclose(file);
            munmap(data, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE));
            return EXIT_FAILURE;
        }
        if (current->info.vector == nullptr) {
            std::cerr << "Error: Vector data is null for vector ID " << current->id.ToString().ToCStr() << ".\n";
            fclose(file);
            munmap(data, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE));
            return EXIT_FAILURE;
        }

        uint32_t num_dup = static_cast<uint32_t>(current->num_duplicates);
        u_count++;
        count += (current->num_duplicates + 1);
        ret = fwrite(&num_dup, sizeof(uint32_t), 1, file);
        if (ret != 1) {
            std::cerr << "Error writing num_duplicates to file.\n";
            fclose(file);
            munmap(data, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE));
            return EXIT_FAILURE;
        }

        ret = fwrite(current->info.vector, sizeof(divftree::VTYPE) * (size_t)dimension, 1, file);
        if (ret != 1) {
            std::cerr << "Error writing vector data to file.\n";
            fclose(file);
            munmap(data, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE));
            return EXIT_FAILURE;
        }
        current = current->next;
    }

    if (u_count != num_unique_points) {
        std::cerr << "Error: Expected " << num_unique_points << " unique vectors, but found " << u_count << ".\n";
    } else {
        std::cout << "Successfully wrote information for " << u_count << " unique vectors.\n";
    }

    if (count != num_total_points) {
        std::cerr << "Error: Expected total count of " << num_total_points << ", but found " << count << ".\n";
    } else {
        std::cout << "Successfully wrote information for all " << count << " vectors.\n";
    }

    fflush(file);
    fclose(file);
    munmap(data, (size_t)num_total_points * (size_t)dimension * sizeof(divftree::VTYPE));
}