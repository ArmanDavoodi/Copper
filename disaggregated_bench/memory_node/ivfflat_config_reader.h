#ifndef DIVFFLAT_MN_CONFIG_READER_H_
#define DIVFFLAT_MN_CONFIG_READER_H_

#include "disaggregated_bench/memory_node/ivfflat_bench.h"

#include <fstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <sstream>
#include <charconv>
#include <typeinfo>

inline constexpr char config_path[] = "disaggregated_bench/memory_node/run.conf";

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

void ParseConfigs(divftree::NodeID self_id) {
    if (var_configs.empty()) {
        throw std::runtime_error("Configs not available!");
    }

    auto vit = var_configs.find("log-path");
    if (vit != var_configs.end()) {
        strncpy(::divftree::debug::output_log_path, vit->second.c_str(), 255);
        char file_name[512] = "";
        strcat(file_name, ::divftree::debug::output_log_path);
        strcat(file_name, (self_id.ToString() + ::divftree::String(".log")).ToCStr());
        divftree::debug::output_log = fopen(file_name, "w");
        printf("Logging to %s\n", file_name);
        if (divftree::debug::output_log == nullptr) {
            throw std::runtime_error(
                    divftree::String("Could not open the output log file! errno %d, errno msg: %s",
                                     errno, strerror(errno)).ToCStr());
        }
    }

    vit = var_configs.find("num-clusters");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, num_clusters) ||
            num_clusters < 1) {
            throw std::runtime_error("Invalid number of clusters!");
        }
    } else {
        throw std::runtime_error("Number of clusters not provided!");
    }

    vit = var_configs.find("kmeans-max-iters");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, max_iters)) {
            throw std::runtime_error("Invalid kmeans max iters!");
        }
    } else {
        throw std::runtime_error("kmeans max iters not provided!");
    }

    vit = var_configs.find("insert-duplicates");;
    if (vit != var_configs.end()) {
        if (!parseBool(vit->second, insert_duplicates)) {
            throw std::runtime_error("Invalid insert duplicates!");
        }
    } else {
        throw std::runtime_error("Insert duplicates not provided!");
    }

    vit = var_configs.find("page-size");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, page_size) || (page_size < 1)) {
            throw std::runtime_error("Invalid page size!");
        }
    } else {
        throw std::runtime_error("Page size not provided!");
    }

    vit = var_configs.find("num-build-threads");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, num_threads) || (num_threads < 1)) {
            throw std::runtime_error("Invalid number of build threads!");
        }
    } else {
        throw std::runtime_error("Number of build threads not provided!");
    }

    vit = var_configs.find("build-size");
    if (vit != var_configs.end()) {
        if (!parseUnsignedInt(vit->second, build_size)) {
            throw std::runtime_error("Invalid build size!");
        }
    } else {
        throw std::runtime_error("Build size not provided!");
    }

}

#endif