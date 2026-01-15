#include <iostream>
#include <cstdint>
#include <string>
#include <cstring>
#include <random>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <filesystem>
#include <algorithm>

using namespace std;

uint32_t num_vectors;
uint32_t dimension; // max should fit in uint32

string filepath;

double mean;
double stddev;
double alpha;

#define UINT8 1
#define UINT16 2
#define UINT32 3
#define FLOAT 4

#if defined(VECTOR_TYPE)
    #if VECTOR_TYPE == UINT8
        using VTYPE = uint8_t;
        using DTYPE = uint32_t;
        using MVTYPE = uint64_t;
        #define VTYPE_FMT "%hhu"
        #define DTYPE_FMT "%u"
        #define MVTYPE_FMT "%lu"
        #pragma message("TYPE = UINT8")
    #elif VECTOR_TYPE == UINT16
        using VTYPE = uint16_t;
        using DTYPE = uint64_t;
        using MVTYPE = uint64_t;
        #define VTYPE_FMT "%hu"
        #define DTYPE_FMT "%lu"
        #define MVTYPE_FMT "%lu"
        #pragma message("TYPE = UINT16")
    // #elif VECTOR_TYPE == UINT32
    //     using VTYPE = uint32_t;
    //     using DTYPE = uint64_t;
    //     using MVTYPE = uint64_t;
    //     #define VTYPE_FMT "%u"
    //     #define DTYPE_FMT "%lu"
    //     #define MVTYPE_FMT "%lu"
    //     #pragma message("TYPE = UINT32")
    #elif VECTOR_TYPE == FLOAT
        using VTYPE = float;
        using DTYPE = double;
        using MVTYPE = double;
        #define VTYPE_FMT "%0.2f"
        #define DTYPE_FMT "%0.4f"
        #define MVTYPE_FMT "%0.4f"
        #pragma message("TYPE = FLOAT")
    #else
        #error UNDEFINED VECTOR_TYPE!
    #endif
#else
using VTYPE = uint8_t;
using DTYPE = uint16_t;
using MVTYPE = uint64_t;
#define VTYPE_FMT "%hhu"
#define DTYPE_FMT "%hu"
#error VECTOR_TYPE not found!
#endif

class SkewNormal {
public:
    SkewNormal()
        : mu(mean), sigma(stddev), alpha(alpha), gen(random_device{}()), dist(mean, stddev) {}

    VTYPE operator()() {
        double z0 = dist(gen);
        double z1 = dist(gen);
        double delta = alpha / sqrt(1 + alpha * alpha);
        double x = mu + sigma * (delta * fabs(z0) + sqrt(1 - delta * delta) * z1);

        if constexpr (is_integral_v<VTYPE>) {
            // clamp to valid range for integer types
            x = round(x);
            x = clamp(x, (double)numeric_limits<VTYPE>::min(), (double)numeric_limits<VTYPE>::max());
        }

        return static_cast<VTYPE>(x);
    }

private:
    double mu, sigma, alpha;
    mt19937 gen;
    normal_distribution<double> dist;
};

bool parse_uint32(const char* str, uint32_t& value) {
    char* end;
    unsigned long val = strtoul(str, &end, 10);
    if (*end != '\0' || val > UINT32_MAX) return false;
    value = static_cast<uint32_t>(val);
    return true;
}

bool parse_double(const char* str, double& value) {
    char* end;
    value = strtod(str, &end);
    return (*end == '\0');
}

int ParseInput(int argc, char* argv[]) {
    if (argc != 7) {
        cerr << "Usage: " << argv[0]
             << " <num_vectors> <dimension> <filepath> <mean> <stddev> <alpha>\n";
        return EXIT_FAILURE;
    }

    filepath = argv[3];

    // Parse and validate numeric inputs
    if (!parse_uint32(argv[1], num_vectors) || num_vectors == 0) {
        cerr << "Error: num_vectors must be a positive 32-bit unsigned integer.\n";
        return EXIT_FAILURE;
    }

    if (!parse_uint32(argv[2], dimension) || dimension < 1 || dimension > 256) {
        cerr << "Error: dimension must be in range [1, 256].\n";
        return EXIT_FAILURE;
    }

    if (filepath.empty()) {
        cerr << "Error: filepath cannot be empty.\n";
        return EXIT_FAILURE;
    }

    if (!parse_double(argv[4], mean)) {
        cerr << "Error: mean must be a valid floating-point number.\n";
        return EXIT_FAILURE;
    }

    if (!parse_double(argv[5], stddev)) {
        cerr << "Error: stddev must be a valid floating-point number.\n";
        return EXIT_FAILURE;
    }

    if (!parse_double(argv[6], alpha) || alpha < -10.0 || alpha > 10.0) {
        cerr << "Error: alpha must be in range [-10, 10].\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
    size_t ret = ParseInput(argc, argv);
    if (ret != EXIT_SUCCESS) {
        return ret;
    }

    // Extract the directory part
    filesystem::path pathObj(filepath);
    filesystem::path dir = pathObj.parent_path();

    // Create directories if they don't exist
    if (!dir.empty() && !filesystem::exists(dir)) {
        filesystem::create_directories(dir);
    }

    // Now open the file
    FILE* file = fopen(filepath.c_str(), "wb"); // "w" creates the file if it doesn't exist
    if (!file) {
        perror("fopen");
        return 1;
    }

    ret = fwrite(&num_vectors, sizeof(uint32_t), 1, file);
    if (ret != 1) {
        perror("fwrite num vectors");
        return 1;
    }

    ret = fwrite(&dimension, sizeof(uint32_t), 1, file);
    if (ret != 1) {
        perror("fwrite dimension");
        return 1;
    }

    SkewNormal generator;
    VTYPE* buffer = new VTYPE[dimension];
    for (uint32_t i = 0; i < num_vectors; ++i) {
        for (uint32_t d = 0; d < dimension; ++d) {
            buffer[d] = generator();
        }
        ret = fwrite(buffer, dimension * sizeof(VTYPE), 1, file);
        if (ret != 1) {
            cerr << "Error: fwrite failed to write " << i << "th vector!\n";
            return 1;
        }
    }
    fflush(file);
    delete[] buffer;
    fclose(file);

    return 0;
}