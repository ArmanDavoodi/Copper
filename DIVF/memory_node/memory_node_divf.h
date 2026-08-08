#ifndef MN_DIVF_H_
#define MN_DIVF_H_

#include "common.h"
#include "vector_utils.h"
#include "distance.h"

#include "utils/thread.h"
#include "utils/sorted_list.h"
#include "utils/concurrent_datastructures.h"
#include "utils/vector_directory.h"
#include "utils/rdma_manager.h"

#include <sys/mman.h>
#include <filesystem>
#include <atomic>

namespace divftree {

inline constexpr size_t MEMORY_POOL_PADDING = 4096;

struct ClusterInfo {
    VectorID id;
    uint32_t offset; // number of points before the data of this cluster
    uint32_t num_points;
    IVFCluster* cluster_ptr;
    size_t bytes;
    CTYPE centroid_vector[DIMENSION];
};

class MN_DIVFIndex {
public:
    MN_DIVFIndex(const std::string& index_path) {
        RetStatus status = LoadIndex(index_path);

        status =
            RDMA_Manager::Initialize(
                network_config::num_memory_nodes,
                network_config::num_compute_nodes,
                network_config::memory_node_ids,
                network_config::memory_node_ip_lists,
                network_config::memory_node_ports,
                network_config::compute_node_ids,
                network_config::compute_node_ip_lists,
                network_config::compute_node_ports,
                network_config::rdma_device_name,
                network_config::rdma_port,
                network_config::gid_index,
                IS_MEMORY_NODE(),
                network_config::self_idx,
                1
            );

        RDMA_Manager* rdma_mgr = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_mgr, LOG_TAG_BASIC);

        status = rdma_mgr->RegisterMemory(memory_pool, pool_size);
        FatalAssert(status.IsOK(), LOG_TAG_BASIC,
                    "Failed to register memory in MN_DIVFIndex constructor: %s",
                    status.Msg());
        status = rdma_mgr->EstablishConnections();
        FatalAssert(status.IsOK(), LOG_TAG_BASIC,
                    "Failed to establish RDMA connections: %s",
                    status.Msg());
    }

    ~MN_DIVFIndex() {
        std::vector<VectorID> tasks;
        RDMA_Manager::DestroyInstance(tasks);
        FatalAssert(tasks.size() == 0, LOG_TAG_BASIC,
                    "There should be no pending tasks when destroying MN_DIVFIndex!");

        for (uint8_t level = 0; level < num_levels; ++level) {
            if (cluster_infos[level] != nullptr) {
                for (uint32_t c = 0; c < num_clusters[level]; ++c) {
                    if (clusters[level][c] != nullptr) {
                        delete clusters[level][c];
                        clusters[level][c] = nullptr;
                    }
                }
                delete[] cluster_infos[level];
                cluster_infos[level] = nullptr;
                delete[] clusters[level];
                clusters[level] = nullptr;
            }
        }
        delete[] num_clusters;

        if (munmap(memory_pool, pool_size) != 0) {
            DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_BASIC, "Failed to free memory for MemoryPool. "
                    "errno %d, errno msg: %s", errno, strerror(errno));
        }
    }

    void Start() {
        std::vector<Thread*> listener_threads;
        std::vector<NodeInfo> compute_nodes, memory_nodes;
        RDMA_Manager* rdma_mgr = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_mgr, LOG_TAG_BASIC);
        rdma_mgr->GetAllNodeInfos(memory_nodes, compute_nodes);
        for (uint8_t cn_idx = 0; cn_idx < compute_nodes.size(); ++cn_idx) {
            Thread* listener_thread = new Thread(100);
            listener_threads.push_back(listener_thread);
            listener_thread->StartMemberFunction(&MN_DIVFIndex::ListenerThread, this, compute_nodes[cn_idx].node_id);
        }

        while (num_connected_cns.load() < compute_nodes.size()) {
            msleep(100);
        }

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_DIVFTREE, "All compute nodes connected. "
                "MN_DIVFIndex is ready to serve requests.");
        
        const bool signal = true;
        for (NodeInfo &cn_info : compute_nodes) {
            rdma_mgr->SendMessage(cn_info.node_id, &signal, sizeof(bool));
        }

        for (auto& thrd : listener_threads) {
            thrd->WaitForThreadToFinish();
            delete thrd;
        }
    }

protected:
    IndexType index_type;

    void* memory_pool = nullptr;
    size_t pool_size = 0;

    uint8_t num_levels = 0;
    ClusterInfo** cluster_infos = nullptr;

    uint32_t leaf_cap = 0;
    uint32_t internal_cap = 0;
    size_t leaf_bytes = 0;
    size_t internal_bytes = 0;

    IVFCluster*** clusters = nullptr;

    uint32_t* num_clusters = nullptr;
    uint32_t num_points = 0;

    std::atomic<size_t> num_connected_cns{0};

    RetStatus LoadIndex(const std::string& index_path) {
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Reading the index from file...");

        if (!std::filesystem::exists(index_path)) {
            FatalAssert(false, LOG_TAG_BASIC, "Input file does not exist: %s", index_path.c_str());
            return RetStatus::Fail("Input file does not exist");
        }

        if (!std::filesystem::is_regular_file(index_path)) {
            FatalAssert(false, LOG_TAG_BASIC, "Input path is not a regular file: %s", index_path.c_str());
            return RetStatus::Fail("Input path is not a regular file");
        }
        // Now open the file
        FILE* file = fopen(index_path.c_str(), "rb");
        if (!file) {
            FatalAssert(false, LOG_TAG_BASIC, "Failed to open input file: %s. errno: %d, msg: %s",
                        index_path.c_str(), errno, strerror(errno));
            return RetStatus::Fail("Failed to open input file");
        }

        size_t ret = fread(&index_type, sizeof(IndexType), 1, file);
        if (ret != 1) {
            fclose(file);
            FatalAssert(false, LOG_TAG_BASIC, "Failed to read index type from index file: %s", index_path.c_str());
            return RetStatus::Fail("Failed to read index type from index file");
        }

        switch (index_type) {
            case IndexType::IVF_FLAT:
                DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Index type: IVF_FLAT");
                break;
            case IndexType::IVF_CAPPED:
                ret = fread(&leaf_cap, sizeof(uint32_t), 1, file);
                if (ret != 1) {
                    fclose(file);
                    FatalAssert(false, LOG_TAG_BASIC, "Failed to read cluster capacity for IVF_CAPPED from index file: %s", index_path.c_str());
                    return RetStatus::Fail("Failed to read cluster capacity for IVF_CAPPED from index file");
                }
                FatalAssert(leaf_cap > 0, LOG_TAG_BASIC, "Cluster capacity for IVF_CAPPED must be greater than 0: %s", index_path.c_str());
                DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Index type: IVF_CAPPED with cluster capacity %u", leaf_cap);
                break;
            case IndexType::IVF_TREE:
                ret = fread(&leaf_cap, sizeof(uint32_t), 1, file);
                if (ret != 1) {
                    fclose(file);
                    FatalAssert(false, LOG_TAG_BASIC, "Failed to read leaf cluster capacity for IVF_TREE from index file: %s", index_path.c_str());
                    return RetStatus::Fail("Failed to read leaf cluster capacity for IVF_TREE from index file");
                }
                FatalAssert(leaf_cap > 0, LOG_TAG_BASIC, "Leaf cluster capacity for IVF_TREE must be greater than 0: %s", index_path.c_str());
                ret = fread(&internal_cap, sizeof(uint32_t), 1, file);
                if (ret != 1) {
                    fclose(file);
                    FatalAssert(false, LOG_TAG_BASIC, "Failed to read internal cluster capacity for IVF_TREE from index file: %s", index_path.c_str());
                    return RetStatus::Fail("Failed to read internal cluster capacity for IVF_TREE from index file");
                }
                FatalAssert(internal_cap > 0, LOG_TAG_BASIC, "Internal cluster capacity for IVF_TREE must be greater than 0: %s", index_path.c_str());
                DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Index type: IVF_TREE with leaf cluster capacity %u and internal cluster capacity %u", leaf_cap, internal_cap);
                break;
            default:
                fclose(file);
                FatalAssert(false, LOG_TAG_BASIC, "Invalid index type in index file: %s", index_path.c_str());
                return RetStatus::Fail("Invalid index type in index file");
        }

        uint32_t tmp;
        ret = fread(&tmp, sizeof(uint32_t), 1, file);
        if (ret != 1) {
            fclose(file);
            FatalAssert(false, LOG_TAG_BASIC, "Failed to read num_points from index file: %s", index_path.c_str());
            return RetStatus::Fail("Failed to read num_points from index file");
        }

        FatalAssert(tmp > 0, LOG_TAG_BASIC, "Number of points in index file must be greater than 0: %s", index_path.c_str());

        ret = fread(&num_points, sizeof(uint32_t), 1, file);
        if (ret != 1) {
            fclose(file);
            FatalAssert(false, LOG_TAG_BASIC, "Failed to read num_unique points from index file: %s", index_path.c_str());
            return RetStatus::Fail("Failed to read num_unique points from index file");
        }

        FatalAssert(num_points > 0, LOG_TAG_BASIC, "Number of unique points in index file must be greater than 0: %s", index_path.c_str());
        FatalAssert(num_points <= tmp, LOG_TAG_BASIC, "Number of unique points cannot be greater than total points: %s", index_path.c_str());

        uint16_t dimension;
        ret = fread(&dimension, sizeof(uint16_t), 1, file);
        if (ret != 1) {
            fclose(file);
            FatalAssert(false, LOG_TAG_BASIC, "Failed to read dimension from index file: %s", index_path.c_str());
            return RetStatus::Fail("Failed to read dimension from index file");
        }
        FatalAssert(dimension == DIMENSION, LOG_TAG_BASIC,
                    "Dimension in index file (%hu) does not match expected dimension (%hu): %s",
                    dimension, DIMENSION, index_path.c_str());

        ret = fread(&num_levels, sizeof(uint8_t), 1, file);
        if (ret != 1) {
            fclose(file);
            FatalAssert(false, LOG_TAG_BASIC, "Failed to read num_levels from index file: %s", index_path.c_str());
            return RetStatus::Fail("Failed to read num_levels from index file");
        }
        FatalAssert(num_levels > 0, LOG_TAG_BASIC, "Number of levels in index file must be greater than 0: %s", index_path.c_str());

        cluster_infos = new ClusterInfo*[num_levels];
        num_clusters = new uint32_t[num_levels];
        clusters = new IVFCluster**[num_levels];
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Reading cluster data...");
        for (uint8_t level = num_levels; level > 0; --level) {
            uint8_t level_idx = level - 1;
            ret = fread(&num_clusters[level_idx], sizeof(uint32_t), 1, file);
            if (ret != 1) {
                fclose(file);
                FatalAssert(false, LOG_TAG_BASIC, "Failed to read num_clusters from index file: %s", index_path.c_str());
                return RetStatus::Fail("Failed to read num_clusters from index file");
            }
            FatalAssert(num_clusters[level_idx] > 0, LOG_TAG_BASIC, "Number of clusters in index file must be greater than 0: %s", index_path.c_str());
            if (level_idx != num_levels - 1) {
                FatalAssert(num_clusters[level_idx] >= num_clusters[level_idx + 1], LOG_TAG_BASIC, "Number of clusters at level %u cannot be greater than number of clusters at level %u: %s",
                            level, level + 1, index_path.c_str());
                if (level_idx == 0) {
                    FatalAssert(num_clusters[level_idx] <= num_points, LOG_TAG_BASIC, "Number of clusters at leaf level cannot be greater than number of unique points: %s", index_path.c_str());
                }
            }

            cluster_infos[level_idx] = new ClusterInfo[num_clusters[level_idx]];
            for (uint32_t cluster_idx = 0; cluster_idx < num_clusters[level_idx]; ++cluster_idx) {
                ret = fread(&cluster_infos[level_idx][cluster_idx].id, sizeof(VectorID), 1, file);
                FatalAssert(ret == 1, LOG_TAG_BASIC, "Failed to read cluster_id for cluster %u in level %u from index file: %s",
                            cluster_idx, level, index_path.c_str());
                FatalAssert(cluster_infos[level_idx][cluster_idx].id._level == level, LOG_TAG_BASIC,
                            "Cluster ID level does not match expected level for cluster %u in level %u in index file: %s",
                            cluster_idx, level, index_path.c_str());
                ret = fread(&cluster_infos[level_idx][cluster_idx].num_points, sizeof(uint32_t), 1, file);
                FatalAssert(ret == 1, LOG_TAG_BASIC, "Failed to read num_points_in_cluster for cluster %u in level %u from index file: %s",
                            cluster_idx, level, index_path.c_str());
                if (level == VectorID::LEAF_LEVEL) {
                    uint32_t num_total_points_in_cluster;
                    ret = fread(&num_total_points_in_cluster, sizeof(uint32_t), 1, file);
                    FatalAssert(ret == 1, LOG_TAG_BASIC, "Failed to read num_total_points_in_cluster for cluster %u in level %u from index file: %s",
                                cluster_idx, level, index_path.c_str());
                    FatalAssert(num_total_points_in_cluster >= cluster_infos[level_idx][cluster_idx].num_points, LOG_TAG_BASIC,
                                "num_total_points_in_cluster should be greater than or equal to num_points in the cluster for cluster %u in level %u in index file: %s",
                                cluster_idx, level, index_path.c_str());
                }
                ret = fread(&cluster_infos[level_idx][cluster_idx].offset, sizeof(uint32_t), 1, file);
                FatalAssert(ret == 1, LOG_TAG_BASIC, "Failed to read cluster_vector_offset for cluster %u in level %u from index file: %s",
                            cluster_idx, level, index_path.c_str());

                ret = fread(cluster_infos[level_idx][cluster_idx].centroid_vector, sizeof(CTYPE), DIMENSION, file);
                FatalAssert(ret == DIMENSION, LOG_TAG_BASIC, "Failed to read centroid_vector for cluster %u in level %u from index file: %s",
                            cluster_idx, level, index_path.c_str());
            }
        }


        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Allocating enough memory for clusters and vectors...");
        if (index_type == IndexType::IVF_FLAT) {
            FatalAssert(num_levels == 1, LOG_TAG_BASIC, "IVFFlat has only a single level");
            pool_size = ALIGNED_SIZE((num_clusters[0] * sizeof(ClusterHeaderData)) +
                                     (num_points * sizeof(VectorData)) +
                                     (num_clusters[0] * CACHE_LINE_SIZE) +
                                     (MEMORY_POOL_PADDING * 2)); // extra padding to ensure we have enough space for alignment and any metadata
        } else if (index_type == IndexType::IVF_CAPPED) {
            FatalAssert(num_levels == 1, LOG_TAG_BASIC, "IVFCapped has only a single level");
            leaf_bytes = ALIGNED_SIZE(sizeof(ClusterHeaderData) + ((size_t)leaf_cap * sizeof(VectorData)));
            pool_size = ALIGNED_SIZE((num_clusters[0] * leaf_bytes) + (MEMORY_POOL_PADDING * 2)); // extra padding to ensure we have enough space for alignment and any metadata
        } else {
            FatalAssert(index_type == IndexType::IVF_TREE, LOG_TAG_BASIC, "Invalid index type specified in index file!");
            leaf_bytes = ALIGNED_SIZE(sizeof(ClusterHeaderData) + ((size_t)leaf_cap * sizeof(VectorData)));
            internal_bytes = ALIGNED_SIZE(sizeof(ClusterHeaderData) + ((size_t)internal_cap * sizeof(CentroidData)));
            size_t num_leaf_clusters = num_clusters[0];
            size_t num_internal_clusters = 0;
            for (uint8_t level = 1; level < num_levels; ++level) {
                num_internal_clusters += num_clusters[level];
            }
            pool_size = ALIGNED_SIZE(ALIGNED_SIZE(num_leaf_clusters * leaf_bytes) +
                                     ALIGNED_SIZE((num_internal_clusters + 1) * internal_bytes) + // one addtional cluster for root
                                     (MEMORY_POOL_PADDING * 2)); // extra padding to ensure we have enough space for alignment and any metadata
        }

#ifdef USE_HUGETLB
        memory_pool = mmap64(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
#else
        memory_pool = mmap64(nullptr, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

        if (memory_pool == MAP_FAILED || memory_pool == nullptr) {
            fclose(file);
            FatalAssert(false, LOG_TAG_BASIC, "Failed to allocate memory for MemoryPool. errno: %d, errno msg: %s",
                        errno, strerror(errno));
            return RetStatus::Fail("Failed to allocate memory for MemoryPool");
        }

        FatalAssert(ALIGNED(memory_pool), LOG_TAG_BASIC,
                    "MemoryPool memory_pool is not properly aligned. Requested alignment: %lu, memory_pool address: %p",
                    CACHE_LINE_SIZE, memory_pool);
        RetStatus rs = RetStatus::Success();
        if (index_type == IndexType::IVF_FLAT) {
            rs = BuildIVFFlat(file);
        } else if (index_type == IndexType::IVF_CAPPED) {
            rs = BuildIVFCapped(file);
        } else {
            FatalAssert(index_type == IndexType::IVF_TREE, LOG_TAG_BASIC, "Invalid index type specified in index file!");
            rs = BuildIVFTree(file);
        }

        FatalAssert(rs.IsOK(), LOG_TAG_BASIC, "failed to build index");

        fclose(file);
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BASIC, "Finished loading the index from file.");
        return rs;
    }

    void ReadVectorDataFromFile(FILE* file, uint64_t file_offset, VectorData* data_arr, uint32_t num_points_to_read) {
        fseek(file, file_offset, SEEK_SET);
        ReadVectorDataFromFile(file, data_arr, num_points_to_read);
    }

    void ReadVectorDataFromFile(FILE* file, VectorData* data_arr, uint32_t num_points_to_read) {
        for (uint32_t p = 0; p < num_points_to_read; ++p) {
            size_t ret = fread(&data_arr[p].id, sizeof(IVFVectorID), 1, file);
            FatalAssert(ret == 1, LOG_TAG_BASIC, "Failed to read vector_id for point");
            uint32_t nd = 0;
            ret = fread(&nd, sizeof(uint32_t), 1, file);
            FatalAssert(ret == 1, LOG_TAG_BASIC, "Failed to read num_duplicate for point");
            ret = fread(data_arr[p].data, sizeof(VTYPE), DIMENSION, file);
            FatalAssert(ret == DIMENSION, LOG_TAG_BASIC, "Failed to read vector data for point");
        }
    }

    RetStatus BuildIVFFlat(FILE* file) {
        clusters[0] = new IVFCluster*[num_clusters[0]];
        uint32_t n_clusters = num_clusters[0];
        IVFCluster** cluster_ptr_arr = clusters[0];
        ClusterInfo* info = cluster_infos[0];

        uint32_t num_seen = 0;
        size_t bytes_offset = MEMORY_POOL_PADDING;
        for (uint32_t c = 0; c < n_clusters; ++c) {
            FatalAssert(info[c].offset == num_seen, LOG_TAG_BASIC, "invalid offset!");
            void* cluster_mem = reinterpret_cast<uint8_t*>(memory_pool) + bytes_offset;
            FatalAssert(ALIGNED(cluster_mem), LOG_TAG_BASIC, "cluster memory should be cache aligned");
            FatalAssert(bytes_offset < pool_size - MEMORY_POOL_PADDING, LOG_TAG_BASIC, "Not enough memory in pool for cluster!");
            cluster_ptr_arr[c] = new (cluster_mem) IVFCluster();
            cluster_ptr_arr[c]->header.id = info[c].id;
            cluster_ptr_arr[c]->header.num_points = info[c].num_points;
            info[c].bytes = ALIGNED_SIZE(sizeof(ClusterHeaderData) + (info[c].num_points * sizeof(VectorData)));
            info[c].cluster_ptr = cluster_ptr_arr[c];
            num_seen += info[c].num_points;
            bytes_offset += info[c].bytes;

            ReadVectorDataFromFile(file, reinterpret_cast<VectorData*>(cluster_ptr_arr[c]->data), info[c].num_points);
        }

        FatalAssert(num_seen == num_points, LOG_TAG_BASIC, "Number of points seen while building IVF_FLAT index does not match expected num_points from index file!");
        return RetStatus::Success();
    }

    RetStatus BuildIVFCapped(FILE* file) {
        clusters[0] = new IVFCluster*[num_clusters[0]];
        uint32_t n_clusters = num_clusters[0];
        IVFCluster** cluster_ptr_arr = clusters[0];
        ClusterInfo* info = cluster_infos[0];

        uint32_t num_seen = 0;
        size_t bytes_offset = MEMORY_POOL_PADDING;
        for (uint32_t c = 0; c < n_clusters; ++c) {
            FatalAssert(info[c].offset == num_seen, LOG_TAG_BASIC, "invalid offset!");
            void* cluster_mem = reinterpret_cast<uint8_t*>(memory_pool) + bytes_offset;
            FatalAssert(ALIGNED(cluster_mem), LOG_TAG_BASIC, "cluster memory should be cache aligned");
            FatalAssert(bytes_offset < pool_size - MEMORY_POOL_PADDING, LOG_TAG_BASIC, "Not enough memory in pool for cluster!");
            cluster_ptr_arr[c] = new (cluster_mem) IVFCluster();
            cluster_ptr_arr[c]->header.id = info[c].id;
            cluster_ptr_arr[c]->header.num_points = info[c].num_points;
            info[c].bytes = ALIGNED_SIZE(sizeof(ClusterHeaderData) + (info[c].num_points * sizeof(VectorData)));
            FatalAssert(leaf_bytes >= info[c].bytes, LOG_TAG_BASIC, "cluster size exceeds capacity!");
            FatalAssert(leaf_cap >= info[c].num_points, LOG_TAG_BASIC, "cluster size exceeds capacity!");
            info[c].cluster_ptr = cluster_ptr_arr[c];
            num_seen += info[c].num_points;
            bytes_offset += leaf_bytes;

            ReadVectorDataFromFile(file, reinterpret_cast<VectorData*>(cluster_ptr_arr[c]->data), info[c].num_points);
        }

        FatalAssert(num_seen == num_points, LOG_TAG_BASIC, "Number of points seen while building IVF_FLAT index does not match expected num_points from index file!");
        return RetStatus::Success();
    }

    RetStatus BuildIVFTree(FILE* file) {
        uint32_t num_seen = 0;
        size_t bytes_offset = MEMORY_POOL_PADDING;
        long file_data_start_offset = 0;
        std::map<uint32_t, uint32_t> old_offsets, new_offsets;
        for (uint8_t level_idx = num_levels - 1; level_idx != UINT8_MAX; --level_idx) {
            uint8_t level = level_idx + 1;
            clusters[level_idx] = new IVFCluster*[num_clusters[level_idx]];
            num_seen = 0;
            if (level == VectorID::LEAF_LEVEL) {
                file_data_start_offset = ftell(file);
            }
            for (uint32_t c = 0; c < num_clusters[level_idx]; ++c) {
                // FatalAssert(cluster_infos[level_idx][c].offset == num_seen, LOG_TAG_BASIC, "invalid offset!");
                void* cluster_mem = reinterpret_cast<uint8_t*>(memory_pool) + bytes_offset;
                FatalAssert(ALIGNED(cluster_mem), LOG_TAG_BASIC, "cluster memory should be cache aligned");
                FatalAssert(bytes_offset < pool_size - MEMORY_POOL_PADDING, LOG_TAG_BASIC, "Not enough memory in pool for cluster!");
                clusters[level_idx][c] = new (cluster_mem) IVFCluster();
                clusters[level_idx][c]->header.id = cluster_infos[level_idx][c].id;
                clusters[level_idx][c]->header.num_points = cluster_infos[level_idx][c].num_points;
                cluster_infos[level_idx][c].cluster_ptr = clusters[level_idx][c];
                FatalAssert(new_offsets.find(cluster_infos[level_idx][c].offset) == new_offsets.end(), LOG_TAG_BASIC, "duplicate offset found for cluster %u in level %u while building IVF_TREE index!", c, level);
                new_offsets[cluster_infos[level_idx][c].offset] = c;
                if (level == VectorID::LEAF_LEVEL) {
                    cluster_infos[level_idx][c].bytes = ALIGNED_SIZE(sizeof(ClusterHeaderData) + (cluster_infos[level_idx][c].num_points * sizeof(VectorData)));
                    FatalAssert(leaf_bytes >= cluster_infos[level_idx][c].bytes, LOG_TAG_BASIC, "cluster size exceeds capacity!");
                    FatalAssert(leaf_cap >= cluster_infos[level_idx][c].num_points, LOG_TAG_BASIC, "cluster size exceeds capacity!");
                    constexpr size_t vector_data_size = sizeof(IVFVectorID) + sizeof(uint32_t) + (sizeof(VTYPE) * DIMENSION);
                    uint64_t file_offset = (uint64_t)file_data_start_offset +
                                           (cluster_infos[level_idx][c].offset * vector_data_size);
                    ReadVectorDataFromFile(file, file_offset,
                                           reinterpret_cast<VectorData*>(clusters[level_idx][c]->data),
                                           cluster_infos[level_idx][c].num_points);
                    bytes_offset += leaf_bytes;
                } else {
                    FatalAssert(level > VectorID::LEAF_LEVEL, LOG_TAG_BASIC, "cannot handle vectors here!");
                    cluster_infos[level_idx][c].bytes = ALIGNED_SIZE(sizeof(ClusterHeaderData) + (cluster_infos[level_idx][c].num_points * sizeof(CentroidData)));
                    FatalAssert(internal_bytes >= cluster_infos[level_idx][c].bytes, LOG_TAG_BASIC, "cluster size exceeds capacity!");
                    FatalAssert(internal_cap >= cluster_infos[level_idx][c].num_points, LOG_TAG_BASIC, "cluster size exceeds capacity!");
                    bytes_offset += internal_bytes;
                }
                num_seen += cluster_infos[level_idx][c].num_points;

                if (level_idx != num_levels - 1) {
                    auto it = old_offsets.upper_bound(c);
                    FatalAssert(it != old_offsets.begin(), LOG_TAG_BASIC, "invalid offset for cluster %u in level %u while building IVF_TREE index!", c, level);
                    uint32_t parent_offset;
                    uint32_t parent_idx;
                    if (it == old_offsets.end()) {
                        parent_offset = old_offsets.rbegin()->first;
                        parent_idx = old_offsets.rbegin()->second;
                    } else {
                        parent_offset = std::prev(it)->first;
                        parent_idx = std::prev(it)->second;
                    }
                    FatalAssert(cluster_infos[level_idx + 1][parent_idx].offset == parent_offset, LOG_TAG_BASIC, "invalid offset for parent cluster while building IVF_TREE index!");
                    FatalAssert(parent_offset <= c, LOG_TAG_BASIC, "invalid parent offset for cluster %u in level %u while building IVF_TREE index!", c, level);
                    FatalAssert(parent_idx < num_clusters[level_idx + 1], LOG_TAG_BASIC, "invalid parent offset for cluster %u in level %u while building IVF_TREE index!", c, level);
                    uint32_t child_idx = c - parent_offset;
                    FatalAssert(child_idx < cluster_infos[level_idx + 1][parent_idx].num_points, LOG_TAG_BASIC, "invalid child index for cluster %u in level %u while building IVF_TREE index!", c, level);
                    CentroidData* centroid_data_arr =
                        reinterpret_cast<CentroidData*>(clusters[level_idx + 1][parent_idx]->data);
                    centroid_data_arr[child_idx].id = cluster_infos[level_idx][c].id;
                    memcpy(centroid_data_arr[child_idx].data, cluster_infos[level_idx][c].centroid_vector, sizeof(CTYPE) * DIMENSION);
                }

            }
            FatalAssert(new_offsets.size() == num_clusters[level_idx], LOG_TAG_BASIC, "new_offsets should not be empty while building IVF_TREE index!");
            old_offsets.swap(new_offsets);
            new_offsets.clear();
            FatalAssert(old_offsets.size() == num_clusters[level_idx], LOG_TAG_BASIC, "new_offsets should not be empty while building IVF_TREE index!");
            FatalAssert(new_offsets.empty(), LOG_TAG_BASIC, "new_offsets should not be empty while building IVF_TREE index!");
        }

        FatalAssert(num_seen == num_points, LOG_TAG_BASIC, "Number of points seen while building IVF_FLAT index does not match expected num_points from index file!");
        return RetStatus::Success();
    }

    inline void SendIndexInfoToNode(NodeID target_cn) {
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_DIVFTREE, "Sending index info to CN %u...", target_cn);

        RDMA_Manager* rdma_mgr = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_mgr, LOG_TAG_BUFFER);
        IndexInfo index_info;
        index_info.type = index_type;
        index_info.num_points = num_points;
        if (index_type == IndexType::IVF_FLAT) {
            index_info.ivf_flat_info.num_clusters = num_clusters[0];
        } else if (index_type == IndexType::IVF_CAPPED) {
            index_info.ivf_capped_info.num_clusters = num_clusters[0];
            index_info.ivf_capped_info.cluster_capacity = leaf_cap;
        } else {
            index_info.ivf_tree_info.num_levels = num_levels;
            index_info.ivf_tree_info.leaf_cluster_capacity = leaf_cap;
            index_info.ivf_tree_info.internal_cluster_capacity = internal_cap;
            index_info.ivf_tree_info.leaf_bytes_cap = leaf_bytes;
            index_info.ivf_tree_info.internal_bytes_cap = internal_bytes;
        }
        rdma_mgr->SendMessage(target_cn, &index_info, sizeof(index_info));

        if (index_type == IndexType::IVF_TREE) {
            rdma_mgr->SendMessage(target_cn, &num_levels, sizeof(num_levels));
            for (uint8_t level_idx = num_levels - 1; level_idx != UINT8_MAX; --level_idx) {
                ClusterMeta* centroids = new ClusterMeta[num_clusters[level_idx]];
                CTYPE* centroid_data = nullptr;
                if (level_idx == num_levels - 1) {
                    centroid_data = new CTYPE[num_clusters[level_idx] * DIMENSION];
                }
                for (size_t c = 0; c < num_clusters[level_idx]; ++c) {
                    centroids[c].centroid_id = cluster_infos[level_idx][c].id;
                    centroids[c].remote_addr = reinterpret_cast<uintptr_t>(cluster_infos[level_idx][c].cluster_ptr);
                    centroids[c].remote_size = cluster_infos[level_idx][c].bytes;
                    centroids[c].num_elements = cluster_infos[level_idx][c].num_points;
                    if (level_idx == num_levels - 1) {
                        DIVF_MEMCOPY(
                            centroid_data + (c * DIMENSION),
                            cluster_infos[level_idx][c].centroid_vector,
                            sizeof(CTYPE) * DIMENSION
                        );
                    }
                }
                rdma_mgr->SendMessage(target_cn, &num_clusters[level_idx], sizeof(num_clusters[level_idx]));
                rdma_mgr->SendMessage(target_cn, centroids, sizeof(ClusterMeta) * num_clusters[level_idx]);
                if (level_idx == num_levels - 1) {
                    rdma_mgr->SendMessage(target_cn, centroid_data, sizeof(CTYPE) * num_clusters[level_idx] * DIMENSION);
                    delete[] centroid_data;
                }
                delete[] centroids;
            }
        } else {
            ClusterMeta* centroids = new ClusterMeta[num_clusters[0]];
            CTYPE* centroid_data = new CTYPE[num_clusters[0] * DIMENSION];
            for (size_t c = 0; c < num_clusters[0]; ++c) {
                centroids[c].centroid_id = cluster_infos[0][c].id;
                centroids[c].remote_addr = reinterpret_cast<uintptr_t>(cluster_infos[0][c].cluster_ptr);
                centroids[c].remote_size = cluster_infos[0][c].bytes;
                centroids[c].num_elements = cluster_infos[0][c].num_points;
                DIVF_MEMCOPY(
                    centroid_data + (c * DIMENSION),
                    cluster_infos[0][c].centroid_vector,
                    sizeof(CTYPE) * DIMENSION
                );
            }
            rdma_mgr->SendMessage(target_cn, centroids, sizeof(ClusterMeta) * num_clusters[0]);
            rdma_mgr->SendMessage(target_cn, centroid_data, sizeof(CTYPE) * num_clusters[0] * DIMENSION);

            delete[] centroids;
            delete[] centroid_data;
        }

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_DIVFTREE, "Index info sent to CN %u successfully.", target_cn);
    }

    inline void ListenerThread(Thread* self, NodeID target_cn) {
        CHECK_NOT_NULLPTR(self, LOG_TAG_DIVFTREE);
        self->InitDIVFThread();
        SendIndexInfoToNode(target_cn);
        num_connected_cns.fetch_add(1);
        bool last_cn_disconnected = RDMA_Manager::ListenForMessages(target_cn);
        UNUSED_VARIABLE(last_cn_disconnected);
        self->DestroyDIVFThread();
    }
};

};

#endif