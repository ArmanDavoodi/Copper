#ifndef CONFIG_READER_H_
#define CONFIG_READER_H_

#include "bench/hnswlib_bench.h"

#include <fstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <sstream>
#include <charconv>
#include <typeinfo>

inline constexpr char config_path[] = "bench/hnswlib_run.conf";

inline std::unordered_map<std::string, std::string> var_configs;
inline std::unordered_map<std::string, std::vector<std::string>> list_configs;

void ReadConfigs() {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open the config file");
    }

    std::string line;
    while (std::getline(file, line)) {
        /* remove all whitespace */
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());

        // Split into name and value
        auto pos = line.find(':');
        if (pos == std::string::npos)
            continue; // invalid line, skip

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        // Parse value type
        if (!value.empty() && value.front() == '[' && value.back() == ']') {
            // Parse list of numbers
            value = value.substr(1, value.size() - 2); // remove brackets
            std::vector<std::string> list;
            std::stringstream ss(value);
            std::string segment;
            while (std::getline(ss, segment, ',')) {
                if (!segment.empty()) {
                    list.push_back(segment);
                }
            }
            list_configs[key] = std::move(list);
        } else if (!value.empty()) {
            var_configs[key] = std::move(value);
        }
    }

    file.close();
}

template <typename T>
bool parseUnsignedInt(const std::string& s, T& value) {
    static_assert(std::is_unsigned_v<T>, "T must be an unsigned integer type");

    // Parse into the widest type (uint64_t) to detect overflow safely
    uint64_t temp = 0;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), temp);

    if (ec != std::errc() || ptr != s.data() + s.size()) {
        // Parsing failed or extra characters were found
        return false;
    }

    if (temp > std::numeric_limits<T>::max()) {
        // Out of range for the target type
        return false;
    }

    value = static_cast<T>(temp);
    return true;
}

bool parseBool(const std::string& s, bool& value) {
    uint8_t val;
    if (!parseUnsignedInt<uint8_t>(s, val)) {
        return false;
    }

    value = (val == 0 ? false : true);
    return true;
}

void ParseConfigs() {
    if (var_configs.empty() && list_configs.empty()) {
        throw std::runtime_error("Configs not available!");
    }

    index_attr.dim = DIMENSION;
    if (index_attr.dim == 0) {
        throw std::runtime_error("Invalid dimension");
    }

    auto vit = var_configs.find("log-path");
    if (vit != var_configs.end()) {
        strncpy(::divftree::debug::output_log_path, vit->second.c_str(), 255);
        char file_name[512] = "";
        strcat(file_name, ::divftree::debug::output_log_path);
        strcat(file_name, (std::to_string(::divftree::debug::next_log_sn) + ".log").c_str());
        divftree::debug::output_log = fopen(file_name, "w");
        printf("Logging to %s\n", file_name);
        if (divftree::debug::output_log == nullptr) {
            throw std::runtime_error(
                    divftree::String("Could not open the output log file! errno %d, errno msg: %s",
                                     errno, strerror(errno)).ToCStr());
        }
    }

    /* graph connectivity */
    vit = var_configs.find("hnsw-M");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, index_attr.M) ||
            index_attr.M < 1) {
            throw std::runtime_error("Invalid hnsw M!");
        }
    } else {
        throw std::runtime_error("hnsw M not provided!");
    }

    /* same as search-span */
    vit = var_configs.find("hnsw-ef-construction");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, index_attr.ef_construction) ||
            index_attr.ef_construction < 1) {
            throw std::runtime_error("Invalid hnsw ef construction!");
        }
    } else {
        throw std::runtime_error("hnsw ef construction not provided!");
    }

    vit = var_configs.find("hnsw-ef-search");;
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, index_attr.ef_search) ||
            index_attr.ef_search < 1) {
            throw std::runtime_error("Invalid hnsw ef search!");
        }
    } else {
        throw std::runtime_error("hnsw ef search not provided!");
    }

    vit = var_configs.find("default-k");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, default_k) ||
            default_k < 1) {
            throw std::runtime_error("Invalid k!");
        }
    } else {
        throw std::runtime_error("k not provided!");
    }

    vit = var_configs.find("num-client-threads");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, num_threads) || (num_threads < 1)) {
            throw std::runtime_error("Invalid number of client threads!");
        }
    } else {
        throw std::runtime_error("Number of client threads not provided!");
    }

    vit = var_configs.find("write-ratio");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, write_ratio) || (write_ratio > 100)) {
            throw std::runtime_error("Invalid write ratio!");
        }
    } else {
        throw std::runtime_error("Write ratio not provided!");
    }

    vit = var_configs.find("delete-ratio");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, delete_ratio) || (delete_ratio > 100)) {
            throw std::runtime_error("Invalid delete ratio!");
        }
    } else {
        throw std::runtime_error("Delete ratio not provided!");
    }

    vit = var_configs.find("bench-batch-size");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, bench_batch_size) || (bench_batch_size == 0)) {
            throw std::runtime_error("Invalid bench batch size!");
        }
    } else {
        throw std::runtime_error("Bench batch size not provided!");
    }

    vit = var_configs.find("build-size");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, build_size)) {
            throw std::runtime_error("Invalid build size!");
        }
    } else {
        throw std::runtime_error("Build size not provided!");
    }

    vit = var_configs.find("warmup-time");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, warmup_time)) {
            throw std::runtime_error("Invalid warmup time!");
        }
    } else {
        throw std::runtime_error("Warmup time not provided!");
    }

    vit = var_configs.find("run-time");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, run_time)) {
            throw std::runtime_error("Invalid run time!");
        }
    } else {
        throw std::runtime_error("Run time not provided!");
    }

    vit = var_configs.find("throughput-report-time");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, throughput_report_time)) {
            throw std::runtime_error("Invalid throughput report time!");
        }
    } else {
        throughput_report_time = 0;
    }

    if (throughput_report_time > 0) {
        vit = var_configs.find("show-runtime-report-for-build-and-warmup");
        if (vit != var_configs.end()) {
            if (!parseBool(vit->second, show_runtime_report_for_build_and_warmup)) {
                throw std::runtime_error("Invalid show runtime report for build and warmup!");
            }
        } else {
            throw std::runtime_error("Show runtime report for build and warmup not provided!");
        }
    }

    vit = var_configs.find("collect-avg-distances");
    if (vit != var_configs.end()) {
        if (!parseBool(vit->second, collect_avg_distances)) {
            throw std::runtime_error("Invalid collect average distances!");
        }
    } else {
        throw std::runtime_error("Collect average distances not provided!");
    }
}

#endif