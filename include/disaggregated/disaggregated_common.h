#ifndef DISAGGREGATED_COMMON_H_
#define DISAGGREGATED_COMMON_H_

#include "common.h"
#include "debug.h"

#include "utils/string.h"

#include <cstdint>
#include <netinet/in.h>
#include <infiniband/verbs.h>
#include <fstream>

#define MEMROY_NODE_ID 0

#define MEMROY_NODE_IDX 0
#define COMPUTE_NODE_IDX 1
#define NUM_NODE_TYPES 2

#ifndef MAX_MNODE_COUNT
#define MAX_MNODE_COUNT 1
#endif
#ifndef MAX_CNODE_COUNT
#define MAX_CNODE_COUNT 1
#endif
#ifndef MAX_CONN_PER_NODE
#define MAX_CONN_PER_NODE 4
#endif

#if !defined(DIVF_SELF_NODE_TYPE) || ((DIVF_SELF_NODE_TYPE != MEMROY_NODE_IDX) && (DIVF_SELF_NODE_TYPE != COMPUTE_NODE_IDX))
#error "DIVF_SELF_NODE_TYPE is not defined or has an invalid value! It should be either MEMROY_NODE_IDX or COMPUTE_NODE_IDX."
#elif DIVF_SELF_NODE_TYPE == MEMROY_NODE_IDX
#define DIVF_IS_MEMORY_NODE
#else
#define DIVF_IS_COMPUTE_NODE
#endif


namespace divftree {

inline constexpr bool IS_MEMORY_NODE() {
#ifdef DIVF_IS_MEMORY_NODE
    return true;
#else
    return false;
#endif
}

inline constexpr bool IS_COMPUTE_NODE() {
#ifdef DIVF_IS_COMPUTE_NODE
    return true;
#else
    return false;
#endif
}

union NodeID {
    struct {
        uint8_t is_mnode : 1;
        uint8_t id : 7;
    };
    uint8_t raw;

    constexpr NodeID() {}

    constexpr NodeID(bool is_memory_node, uint8_t node_id) {
        FatalAssert(node_id < (is_memory_node ? MAX_MNODE_COUNT : MAX_CNODE_COUNT),
                    LOG_TAG_BASIC, "Node ID exceeds the maximum allowed count!");
        is_mnode = is_memory_node ? 1 : 0;
        id = node_id;
    }

    constexpr NodeID(uint8_t raw_id) : raw(raw_id) {
        FatalAssert(id < (IsMemoryNode() ? MAX_MNODE_COUNT : MAX_CNODE_COUNT),
                    LOG_TAG_BASIC, "Node ID exceeds the maximum allowed count!");
    }

    constexpr bool IsComputeNode() const {
        return is_mnode == 0;
    }

    constexpr bool IsMemoryNode() const {
        return is_mnode == 1;
    }

    constexpr uint8_t GetID() const {
        return id;
    }

    constexpr uint8_t ToRaw() const {
        return raw;
    }

    constexpr bool operator==(const NodeID& other) const {
        return raw == other.raw;
    }

    constexpr bool operator!=(const NodeID& other) const {
        return raw != other.raw;
    }

    inline String ToString() const {
        return String("%sNode-%u", IsMemoryNode() ? "M" : "C", id);
    }
};

struct NodeIDHash {
    inline size_t operator()(const NodeID& id) const {
        return splitmix32(static_cast<uint32_t>(id.raw));
    }
};

namespace network_config {

    inline constexpr char network_config_file_path[] = "configs/disaggregated/network_config.conf";
    inline uint8_t num_memory_nodes = 0;
    inline uint8_t num_compute_nodes = 0;

    inline uint8_t memory_node_ids[MAX_MNODE_COUNT];
    inline char* memory_node_ip_lists[MAX_MNODE_COUNT];
    inline char memory_node_ips[MAX_MNODE_COUNT][INET_ADDRSTRLEN];
    inline uint16_t memory_node_ports[MAX_MNODE_COUNT];
    inline uint8_t compute_node_ids[MAX_CNODE_COUNT];
    inline char* compute_node_ip_lists[MAX_CNODE_COUNT];
    inline char compute_node_ips[MAX_CNODE_COUNT][INET_ADDRSTRLEN];
    inline uint16_t compute_node_ports[MAX_CNODE_COUNT];

    inline char rdma_device_name[IBV_SYSFS_NAME_MAX];
    inline uint8_t rdma_port;
    inline int gid_index;
    inline uint8_t self_idx = UINT8_MAX;
};

inline void ReadNetworkConfigs() {
    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
            "Reading network configuration from file: %s",
            network_config::network_config_file_path);
    std::ifstream config_file(network_config::network_config_file_path);
    FatalAssert(config_file.is_open(), LOG_TAG_BASIC,
                "Failed to open network configuration file at path: %s",
                network_config::network_config_file_path);
    int temp;
    config_file >> temp;
    network_config::num_memory_nodes = static_cast<uint8_t>(temp);
    config_file >> temp;
    network_config::num_compute_nodes = static_cast<uint8_t>(temp);

    FatalAssert(network_config::num_memory_nodes <= MAX_MNODE_COUNT, LOG_TAG_BASIC,
                "Number of memory nodes exceeds the maximum allowed count!");
    FatalAssert(network_config::num_compute_nodes <= MAX_CNODE_COUNT, LOG_TAG_BASIC,
                "Number of compute nodes exceeds the maximum allowed count!");
    FatalAssert(network_config::num_compute_nodes > 0, LOG_TAG_BASIC,
                "There should be at least one compute node!");
    FatalAssert(network_config::num_memory_nodes > 0, LOG_TAG_BASIC,
                "There should be at least one memory node!");
    FatalAssert(network_config::num_memory_nodes == 1, LOG_TAG_NOT_IMPLEMENTED,
                "Currently, only one memory node is supported!");

#if DIVF_SELF_NODE_TYPE == MEMROY_NODE_IDX
    FatalAssert(network_config::self_idx < network_config::num_memory_nodes, LOG_TAG_BASIC,
                "Self node index exceeds the number of memory nodes!");
#else
    FatalAssert(network_config::self_idx < network_config::num_compute_nodes, LOG_TAG_BASIC,
                "Self node index exceeds the number of compute nodes!");
#endif
    String memory_node_list = "[";
    String compute_node_list = "[";

    for (uint8_t i = 0; i < network_config::num_memory_nodes; ++i) {
        config_file >> temp;
        network_config::memory_node_ids[i] = static_cast<uint8_t>(temp);
        config_file >> network_config::memory_node_ips[i];
        config_file >> temp;
        network_config::memory_node_ports[i] = static_cast<uint16_t>(temp);
        network_config::memory_node_ip_lists[i] = network_config::memory_node_ips[i];
        memory_node_list +=
            String("%sCNode-%hhu:%s:%u%s",
                   ((IS_MEMORY_NODE() && (network_config::self_idx == i)) ? "*" : ""),
                   network_config::memory_node_ids[i], network_config::memory_node_ips[i],
                   network_config::memory_node_ports[i], i == (network_config::num_memory_nodes - 1) ? "]" : ", ");
    }

    for (uint8_t i = 0; i < network_config::num_compute_nodes; ++i) {
        config_file >> temp;
        network_config::compute_node_ids[i] = static_cast<uint8_t>(temp);
        config_file >> network_config::compute_node_ips[i];
        config_file >> temp;
        network_config::compute_node_ports[i] = static_cast<uint16_t>(temp);
        network_config::compute_node_ip_lists[i] = network_config::compute_node_ips[i];
        compute_node_list +=
            String("%sMNode-%hhu:%s:%u%s",
                   ((IS_COMPUTE_NODE() && (network_config::self_idx == i)) ? "*" : ""),
                   network_config::compute_node_ids[i], network_config::compute_node_ips[i],
                   network_config::compute_node_ports[i], i == (network_config::num_compute_nodes - 1) ? "]" : ", ");
    }

    config_file >> network_config::rdma_device_name;
    config_file >> temp;
    network_config::rdma_port = static_cast<uint8_t>(temp);
    config_file >> temp;
    network_config::gid_index = temp;

    config_file.close();
    DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC,
            "Completed reading network configuration: %hhu memory nodes:%s, %hhu compute nodes:%s, "
            "RDMA device: %s, port: %hhu, GID index: %d",
            network_config::num_memory_nodes, memory_node_list.ToCStr(),
            network_config::num_compute_nodes, compute_node_list.ToCStr(),
            network_config::rdma_device_name, network_config::rdma_port,
            network_config::gid_index);
}

struct ClusterMeta {
    VectorID centroid_id;
    uintptr_t remote_addr;
    size_t remote_size;
};

};

#endif