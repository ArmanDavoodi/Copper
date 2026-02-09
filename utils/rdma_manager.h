#ifndef RDMA_MANAGER_H_
#define RDMA_MANAGER_H_

#include "common.h"
#include "debug.h"

#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <map>

#include "utils/concurrent_datastructures.h"
#include "disaggregated/disaggregated_common.h"

namespace divftree {

inline constexpr size_t MAX_MESSAGE_SIZE = 4096;
inline constexpr uint32_t MAX_NUM_READ_WR = 16;

/* todo: maybe I need to pass this as a runtime arg?! needs tuning */
static constexpr uint32_t MAX_SEND_WR[NUM_NODE_TYPES] = {
    0,                 /* MN_SIDE */
    MAX_NUM_READ_WR,   /* CN_SIDE */
};

static constexpr int MAX_CQE[NUM_NODE_TYPES] = {
    1,                                                              /* MN_SIDE */ /* impossible to set to 0 */
    MAX_SEND_WR[COMPUTE_NODE_IDX] * MAX_CONN_PER_NODE,              /* CN_SIDE */
};

static constexpr uint32_t MAX_SEND_SGE[NUM_NODE_TYPES] = {
    1,   /* MN_SIDE */
    4,   /* CN_SIDE */ /* todo: check that it does not go beyond */ /* max is 30 for the current hardware */
};

/* todo: needs tuning */
static constexpr uint32_t MAX_INLINE_DATA[NUM_NODE_TYPES] = {
    0,                            /* MN_SIDE */
    0,                            /* CN_SIDE */
};


static constexpr uint32_t MAX_RD_ATOMIC[NUM_NODE_TYPES] = {
    0,                 /* MN_SIDE */
    MAX_NUM_READ_WR,   /* CN_SIDE */
};

/* We are not using two-sided verbs */
static constexpr uint32_t MAX_RECV_WR[NUM_NODE_TYPES] = {
    0,  /* MN_SIDE */
    0,  /* CN_SIDE */
};

/* We are not using two-sided verbs */
static constexpr uint32_t MAX_RECV_SGE[NUM_NODE_TYPES] = {
    0,  /* MN_SIDE */
    0,  /* CN_SIDE */
};

static constexpr ibv_mtu DEFAULT_MTU[NUM_NODE_TYPES] = {
    IBV_MTU_4096, /* MN_SIDE */
    IBV_MTU_4096, /* CN_SIDE */
};

static constexpr uint8_t GET_NODE_TYPE_IDX(bool is_memory_node) {
    return is_memory_node ? MEMROY_NODE_IDX : COMPUTE_NODE_IDX;
}

inline constexpr uint64_t MAX_TASK_SEQ_NUM = 0x0000FFFFFFFFFFFF;

union TaskID {
    struct {
        uint64_t task_seq_num : 48;
        uint64_t connection_idx : 8;
        uint64_t node_raw : 8;
    };
    uint64_t raw;

    TaskID() : raw(0) {}
    TaskID (uint64_t task_num, uint8_t conn_idx, NodeID node_id) :
        task_seq_num(task_num & MAX_TASK_SEQ_NUM), connection_idx(conn_idx), node_raw(node_id.raw) {}

    inline bool operator==(const TaskID& other) const {
        return raw == other.raw;
    }

    inline bool operator!=(const TaskID& other) const {
        return raw != other.raw;
    }
};

struct TaskIDHash {
    inline size_t operator()(const TaskID& id) const {
        return splitmix64(id.raw);
    }
};

struct NodeInfo {
    NodeID node_id;
    in_addr_t ip_address;
    uint16_t port;
};

struct ConnectionInfo {
    uint32_t remote_qp_num = 0;
    uint32_t local_psn = 0;
    uint32_t remote_psn = 0;
    struct ibv_qp *qp = nullptr;
    struct ibv_cq *cq = nullptr;
    std::atomic<uint64_t> next_task_id = 0;
    std::atomic<uint16_t> num_pending_requests = 0;
    // size_t num_completed_task_ids = 0;
    // TaskID* completed_task_ids = nullptr;

    ConnectionInfo() = default;
    ConnectionInfo(const ConnectionInfo& other) = delete;
    ConnectionInfo& operator=(const ConnectionInfo& other) = delete;
    ConnectionInfo(ConnectionInfo&& other) noexcept :
        remote_qp_num(other.remote_qp_num),
        local_psn(other.local_psn),
        remote_psn(other.remote_psn),
        qp(other.qp),
        cq(other.cq),
        next_task_id(other.next_task_id.load()),
        num_pending_requests(other.num_pending_requests.load()) {
            other.qp = nullptr;
            other.cq = nullptr;
        }
    inline ConnectionInfo& operator=(ConnectionInfo&& other) noexcept {
        if (this != &other) {
            remote_qp_num = other.remote_qp_num;
            local_psn = other.local_psn;
            remote_psn = other.remote_psn;
            qp = other.qp;
            cq = other.cq;
            next_task_id.store(other.next_task_id.load());
            num_pending_requests.store(other.num_pending_requests.load());
            other.qp = nullptr;
            other.cq = nullptr;
        }
        return *this;
    }
};

struct ConnectionContext {
    NodeInfo remote_node_info;
    int socket;
    union ibv_gid remote_gid;
    uintptr_t remote_region_addr;
    size_t remote_region_size;
    uint32_t remote_region_rkey;
    std::atomic<uint8_t> next_connection_idx;
    ConnectionInfo connections[MAX_CONN_PER_NODE];

    ConnectionContext(NodeID id, in_addr_t ip, uint16_t port) :
        remote_node_info{
            .node_id = id,
            .ip_address = ip,
            .port = port,
        }, socket(-1), remote_region_addr((uintptr_t)nullptr), remote_region_size(0), remote_region_rkey(0),
        next_connection_idx(0) {}

    ConnectionContext(ConnectionContext&& other) noexcept :
        remote_node_info(other.remote_node_info),
        socket(other.socket),
        remote_gid(other.remote_gid),
        remote_region_addr(other.remote_region_addr),
        remote_region_size(other.remote_region_size),
        remote_region_rkey(other.remote_region_rkey),
        next_connection_idx(other.next_connection_idx.load()) {
            for (size_t i = 0; i < MAX_CONN_PER_NODE; i++) {
                connections[i] = std::move(other.connections[i]);
            }
            other.socket = -1;
        }

    inline ConnectionContext& operator=(ConnectionContext&& other) noexcept {
        if (this != &other) {
            remote_node_info = other.remote_node_info;
            socket = other.socket;
            remote_gid = other.remote_gid;
            remote_region_addr = other.remote_region_addr;
            remote_region_size = other.remote_region_size;
            remote_region_rkey = other.remote_region_rkey;
            next_connection_idx.store(other.next_connection_idx.load());
            for (size_t i = 0; i < MAX_CONN_PER_NODE; i++) {
                connections[i] = std::move(other.connections[i]);
            }
            other.socket = -1;
        }
        return *this;
    }
};

struct HandshakeInfo {
    union ibv_gid gid;
    uintptr_t region_addr;
    size_t region_size;
    uint32_t region_rkey;
    uint32_t qp_num[MAX_CONN_PER_NODE];
    uint32_t psn[MAX_CONN_PER_NODE];
};

enum class ConnectionMessage : uint8_t {
    INVALID = 0,
    CN_TO_MN_DISCONNECT = 1,
    MN_TO_CN_DISCONNECT_RESP = 2
};
class RDMA_Manager {
public:
    static RetStatus Initialize(uint8_t num_mnodes, uint8_t num_cnodes, uint8_t* mnode_ids,
                                char** mnode_ips, uint16_t* mnode_ports,
                                uint8_t* cnode_ids, char** cnode_ips, uint16_t* cnode_ports,
                                const char* target_rdma_device_name, uint8_t rdma_port, int gid_index,
                                bool is_memory_node, uint8_t self_idx, size_t num_threads) {
        FatalAssert(instance == nullptr, LOG_TAG_RDMA,
                    "RDMA_Manager is already initialized!");
        instance = new RDMA_Manager(num_mnodes, num_cnodes, mnode_ids,
                                    mnode_ips, mnode_ports,
                                    cnode_ids, cnode_ips, cnode_ports,
                                    target_rdma_device_name, rdma_port, gid_index,
                                    is_memory_node, self_idx, num_threads);
        return RetStatus::Success();
    }

    static RDMA_Manager* GetInstance() {
        FatalAssert(instance != nullptr, LOG_TAG_RDMA,
                    "RDMA_Manager is not initialized!");
        return instance;
    }

    static void DestroyInstance(std::vector<VectorID>& completed_tasks) {
        FatalAssert(instance != nullptr, LOG_TAG_RDMA,
                    "RDMA_Manager is not initialized!");
        instance->ready.store(false, std::memory_order_release);
        if (instance->cq != nullptr) {
            sleep(5); /* wait for pending operations to complete */
            instance->PushCompletedReadsToTaskQueue(completed_tasks, true);
        }

        if (IS_COMPUTE_NODE()) {
            ConnectionMessage disconnect_msg = ConnectionMessage::CN_TO_MN_DISCONNECT;
            ConnectionMessage resp;
            for (auto& [node_id, conn_ctx] : instance->memory_nodes) {
                resp = ConnectionMessage::INVALID;
                instance->SendMessage(node_id, &disconnect_msg, sizeof(disconnect_msg), true);
                instance->ReceiveMessage(node_id, &resp, sizeof(resp), true);
                FatalAssert(resp == ConnectionMessage::MN_TO_CN_DISCONNECT_RESP, LOG_TAG_RDMA,
                            "Failed to receive disconnect response from memory node %s",
                            node_id.ToString().ToCStr());
            }
        }
        delete instance;
        instance = nullptr;
    }

    static bool ListenForMessages(NodeID target_node_id) {
        FatalAssert(instance != nullptr, LOG_TAG_RDMA,
                    "RDMA_Manager is not initialized!");
        FatalAssert(IS_MEMORY_NODE(), LOG_TAG_RDMA,
                    "Only memory nodes can listen for messages.");
        FatalAssert(target_node_id.IsComputeNode(), LOG_TAG_RDMA,
                    "Target node must be a compute node.");
        ConnectionMessage msg = ConnectionMessage::INVALID;
        instance->ReceiveMessage(target_node_id, &msg, sizeof(msg));
        FatalAssert(msg == ConnectionMessage::CN_TO_MN_DISCONNECT, LOG_TAG_RDMA,
                    "Received invalid message from compute node %s",
                    target_node_id.ToString().ToCStr());
        ConnectionMessage resp = ConnectionMessage::MN_TO_CN_DISCONNECT_RESP;
        instance->SendMessage(target_node_id, &resp, sizeof(resp));

        instance->DisconnectFromComputeNode(target_node_id);

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_RDMA,
                "Disconnected from compute node %s successfully.",
                target_node_id.ToString().ToCStr());
        size_t num_connected = instance->num_connected_nodes.fetch_sub(1) - 1;
        return (num_connected == 0);
    }

    static NodeInfo GetSelfNodeInfo() {
        return GetInstance()->selfInfo;
    }

    void GetAllNodeInfos(std::vector<NodeInfo>& mn_infos, std::vector<NodeInfo>& cn_infos) {
        mn_infos.clear();
        cn_infos.clear();
        for (const auto& [node_id, conn_ctx] : memory_nodes) {
            mn_infos.push_back(conn_ctx.remote_node_info);
        }
        for (const auto& [node_id, conn_ctx] : compute_nodes) {
            cn_infos.push_back(conn_ctx.remote_node_info);
        }
    }

    RetStatus RegisterMemory(void* buffer, size_t size) {
        FatalAssert(instance == this, LOG_TAG_RDMA,
                    "RDMA_Manager instance mismatch!");
        FatalAssert(ib_ctx != nullptr, LOG_TAG_RDMA,
                    "RDMA_Manager is not initialized properly!");
        FatalAssert(pd != nullptr, LOG_TAG_RDMA,
                    "Protection Domain is not initialized properly!");
        CHECK_NOT_NULLPTR(buffer, LOG_TAG_RDMA);
        FatalAssert(size > 0, LOG_TAG_RDMA,
                    "Cannot register a memory region of size 0.");
        FatalAssert(mr == nullptr, LOG_TAG_RDMA,
                    "A memory region is already registered.");

        mr = ibv_reg_mr(pd, buffer, size,
                        IBV_ACCESS_LOCAL_WRITE |
                        IBV_ACCESS_REMOTE_READ |
                        IBV_ACCESS_REMOTE_WRITE |
                        IBV_ACCESS_REMOTE_ATOMIC);
        if (mr == nullptr) {
            FatalAssert(false, LOG_TAG_RDMA,
                        "Failed to register memory region for RDMA operations. errno=(%d)%s",
                        errno, strerror(errno));
            return RetStatus::Fail("Failed to register memory region for RDMA operations.");
        }

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_RDMA,
                "Registered memory region for RDMA operations. addr=0x%016lx, length=%zu, lkey=0x%08x, rkey=0x%08x",
                (uintptr_t)(mr->addr), mr->length, mr->lkey, mr->rkey);

        return RetStatus::Success();
    }

    RetStatus EstablishConnections() {
        FatalAssert(instance == this, LOG_TAG_RDMA,
                    "RDMA_Manager instance mismatch!");
        FatalAssert(ib_ctx != nullptr, LOG_TAG_RDMA,
                    "RDMA_Manager is not initialized properly!");
        FatalAssert(mr != nullptr, LOG_TAG_RDMA,
                    "Memory region is not registered for RDMA operations.");

        RetStatus rs = RetStatus::Success();
        String error_msg;
        rs = EstablishTCPConnections();
        if (!rs.IsOK()) {
            error_msg = String("Failed to establish TCP connections. %s", rs.Msg());
            goto EXIT;
        }

        rs = Handshake();
        if (!rs.IsOK()) {
            error_msg = String("Failed to handshake with remote nodes. %s", rs.Msg());
            goto EXIT;
        }

        rs = EstablishRDMAConnections();
        if (!rs.IsOK()) {
            error_msg = String("Failed to establish RDMA connections. %s", rs.Msg());
            goto EXIT;
        }

EXIT:
        FatalAssert(rs.IsOK(), LOG_TAG_RDMA,
                    "Failed to establish RDMA connections. Error: %s", rs.Msg());
        if (rs.IsOK()) {
            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_RDMA,
                    "Successfully established RDMA connections with all remote nodes.");
            ready.store(true);
        }
        return rs;
    }

    RetStatus RDMARead(NodeID target_node, void** local_buffers, uintptr_t* remote_addresses, uint32_t* sizes,
                       size_t num_clusters, std::vector<VectorID>&& cluster_ids) {
        FatalAssert(instance == this, LOG_TAG_RDMA,
                    "RDMA_Manager instance mismatch!");
        FatalAssert(cq != nullptr, LOG_TAG_RDMA,
                    "Completion Queue is not initialized properly.");
        FatalAssert(pd != nullptr, LOG_TAG_RDMA,
                    "Protection Domain is not initialized properly.");
        FatalAssert(mr != nullptr, LOG_TAG_RDMA,
                    "Memory region is not registered for RDMA operations.");
        FatalAssert(selfInfo.node_id.IsComputeNode(), LOG_TAG_RDMA,
                    "Only compute nodes can push completed reads to task queue.");
        FatalAssert(target_node.IsMemoryNode(), LOG_TAG_RDMA,
                    "Target node must be a memory node.");
        if (!ready.load(std::memory_order_acquire)) {
            return RetStatus::Fail("RDMA_Manager is not ready for RDMA operations.");
        }
        RetStatus rs = RetStatus::Success();

        uint8_t num_resources_acquired = 0;
        uint8_t num_remaining = num_clusters;
        auto it = memory_nodes.find(target_node);
        FatalAssert(it != memory_nodes.end(), LOG_TAG_RDMA,
                    "Target memory node not found!");
        ConnectionContext& ctx = it->second;
        while (num_remaining > 0) {
            uint8_t connection_idx = GrabConnection(target_node, num_clusters, num_resources_acquired);
            FatalAssert(num_resources_acquired > 0, LOG_TAG_RDMA,
                        "Failed to acquire any RDMA resources for RDMA read.");
            FatalAssert(num_resources_acquired <= num_remaining, LOG_TAG_RDMA,
                        "Acquired more RDMA resources than remaining clusters to read.");
            TaskID task_id(ctx.connections[connection_idx].next_task_id.fetch_add(1), connection_idx, target_node);
            size_t total_acquired = num_clusters - num_remaining;
            if (num_resources_acquired == num_clusters) {
                pending_tasks.BatchInsert(task_id, std::move(cluster_ids));
            } else {
                pending_tasks.BatchInsert(task_id, cluster_ids.data() + total_acquired, num_resources_acquired);
            }
            rs = RDMAReadInternal(target_node, connection_idx, task_id,
                                  local_buffers + total_acquired, remote_addresses + total_acquired,
                                  sizes + total_acquired, num_resources_acquired);
            FatalAssert(rs.IsOK(), LOG_TAG_RDMA,
                        "RDMAReadInternal() failed. %s", rs.Msg());
            num_remaining -= num_resources_acquired;
        }
        return rs;
    }

    RetStatus RDMASGRead(NodeID target_node, void*** local_buffers, uintptr_t* remote_addresses,
                         uint32_t** sizes, uint32_t* num_sge, size_t num_clusters, std::vector<VectorID>&& cluster_ids) {
        FatalAssert(instance == this, LOG_TAG_RDMA,
                    "RDMA_Manager instance mismatch!");
        FatalAssert(cq != nullptr, LOG_TAG_RDMA,
                    "Completion Queue is not initialized properly.");
        FatalAssert(pd != nullptr, LOG_TAG_RDMA,
                    "Protection Domain is not initialized properly.");
        FatalAssert(mr != nullptr, LOG_TAG_RDMA,
                    "Memory region is not registered for RDMA operations.");
        FatalAssert(selfInfo.node_id.IsComputeNode(), LOG_TAG_RDMA,
                    "Only compute nodes can push completed reads to task queue.");
        FatalAssert(target_node.IsMemoryNode(), LOG_TAG_RDMA,
                    "Target node must be a memory node.");
        if (!ready.load(std::memory_order_acquire)) {
            return RetStatus::Fail("RDMA_Manager is not ready for RDMA operations.");
        }

        RetStatus rs = RetStatus::Success();
        uint8_t num_resources_acquired = 0;
        uint8_t num_remaining = num_clusters;
        auto it = memory_nodes.find(target_node);
        FatalAssert(it != memory_nodes.end(), LOG_TAG_RDMA,
                    "Target memory node not found!");
        ConnectionContext& ctx = it->second;
        while (num_remaining > 0) {
            uint8_t connection_idx = GrabConnection(target_node, num_clusters, num_resources_acquired);
            FatalAssert(num_resources_acquired > 0, LOG_TAG_RDMA,
                        "Failed to acquire any RDMA resources for RDMA scatter-gather read.");
            FatalAssert(num_resources_acquired <= num_remaining, LOG_TAG_RDMA,
                        "Acquired more RDMA resources than remaining clusters to read.");
            TaskID task_id(ctx.connections[connection_idx].next_task_id.fetch_add(1), connection_idx, target_node);
            size_t total_acquired = num_clusters - num_remaining;
            if (num_resources_acquired == num_clusters) {
                pending_tasks.BatchInsert(task_id, std::move(cluster_ids));
            } else {
                pending_tasks.BatchInsert(task_id, cluster_ids.data() + total_acquired, num_resources_acquired);
            }
            rs = RDMASGReadInternal(target_node, connection_idx, task_id,
                                    local_buffers + total_acquired, remote_addresses + total_acquired,
                                    sizes + total_acquired, num_sge + total_acquired,
                                    num_resources_acquired);
            FatalAssert(rs.IsOK(), LOG_TAG_RDMA,
                        "RDMASGReadInternal() failed. %s", rs.Msg());
            num_remaining -= num_resources_acquired;
        }
        return rs;
    }

    RetStatus PushCompletedReadsToTaskQueue(std::vector<VectorID>& completed_tasks, bool destroying = false) {
        FatalAssert(instance == this, LOG_TAG_RDMA,
                    "RDMA_Manager instance mismatch!");
        FatalAssert(cq != nullptr, LOG_TAG_RDMA,
                    "Completion Queue is not initialized properly.");
        FatalAssert(pd != nullptr, LOG_TAG_RDMA,
                    "Protection Domain is not initialized properly.");
        FatalAssert(mr != nullptr, LOG_TAG_RDMA,
                    "Memory region is not registered for RDMA operations.");
        FatalAssert(completed_tasks.empty(), LOG_TAG_RDMA,
                    "Completed tasks vector must be empty on input.");
        // if (destroying) {
        //     FatalAssert(!ready.load(std::memory_order_acquire), LOG_TAG_RDMA,
        //                 "RDMA_Manager must be marked not ready when destroying.");
        //     poll_lock.Lock(SX_EXCLUSIVE);
        // } else if (!poll_lock.TryLock(SX_EXCLUSIVE)) {
        //     return RetStatus::Success();
        // }

        bool done = false;
        while (!done) {
            done = true;
            int num_completions = ibv_poll_cq(cq, MAX_CQE[COMPUTE_NODE_IDX], poll_list);
            if (num_completions < 0) {
                String error_msg = String("Failed to poll Completion Queue for RDMA Read. num_comp=(%d)%s errno=(%d)%s",
                                        num_completions, strerror(-num_completions), errno, strerror(errno));
                FatalAssert(false, LOG_TAG_RDMA, "%s", error_msg.ToCStr());
                // poll_lock.Unlock();
                return RetStatus::Fail(error_msg.ToCStr());
            }

            for (int i = 0; i < num_completions; ++i) {
                ibv_wc& wc = poll_list[i];
                if (wc.status != IBV_WC_SUCCESS) {
                    String error_msg = String("RDMA Read failed. wc_status=%s(%d), wr_id=0x%016lx, byte_len=%u",
                                            ibv_wc_status_str(wc.status), wc.status, wc.wr_id, wc.byte_len);
                    FatalAssert(false, LOG_TAG_RDMA, "%s", error_msg.ToCStr());
                    // poll_lock.Unlock();
                    return RetStatus::Fail(error_msg.ToCStr());
                }

                TaskID task_id;
                task_id.raw = wc.wr_id;
                NodeID target_node(task_id.node_raw);
                auto it = memory_nodes.find(target_node);
                FatalAssert(it != memory_nodes.end(), LOG_TAG_RDMA,
                            "Target memory node not found!");
                ConnectionInfo& conn_info = it->second.connections[task_id.connection_idx];
                size_t old_size = completed_tasks.size();
                bool erased = pending_tasks.Erase(task_id, completed_tasks);
                FatalAssert(erased, LOG_TAG_RDMA,
                            "Completed task ID not found in pending tasks!");
                UNUSED_VARIABLE(erased);
                size_t num_completed = completed_tasks.size() - old_size;
                FatalAssert(num_completed > 0, LOG_TAG_RDMA,
                            "Number of completed tasks should be greater than 0 after erasing from pending tasks!");
                uint16_t num_pending =
                    conn_info.num_pending_requests.fetch_sub(num_completed);
                FatalAssert(num_pending >= num_completed, LOG_TAG_RDMA,
                            "Number of pending requests underflowed!");
                if (destroying && ((num_pending - num_completed) > 0)) {
                    done = false;
                }
            }

            if (!done) {
                FatalAssert(destroying, LOG_TAG_RDMA,
                            "Should be in destroying mode when there are pending requests.");
                // poll_lock.Unlock();
                usleep(10);  /* yield to let other threads run */
                // poll_lock.Lock(SX_EXCLUSIVE);
            }
        }

        // poll_lock.Unlock();
        return RetStatus::Success();
    }

    void SendMessage(NodeID target, const void* msg, size_t size, bool end_message = false) {
        FatalAssert(this == instance, LOG_TAG_RDMA,
                    "RDMA_Manager instance mismatch!");
        FatalAssert(selfInfo.node_id.IsMemoryNode() == target.IsComputeNode(), LOG_TAG_RDMA,
                    "Compute nodes can only send messages to memory nodes and vice versa.");
        FatalAssert(size > 0, LOG_TAG_RDMA,
                    "Cannot send a message of size 0.");
        CHECK_NOT_NULLPTR(msg, LOG_TAG_RDMA);
        if ((!end_message && !ready.load(std::memory_order_acquire)) ||
            (end_message && ready.load(std::memory_order_relaxed))) {
            FatalAssert(false, LOG_TAG_RDMA,
                        "RDMA_Manager is not ready for RDMA operations.");
            return;
        }

        std::unordered_map<NodeID, ConnectionContext, NodeIDHash>* node_map =
            target.IsComputeNode() ? &compute_nodes : &memory_nodes;

        auto it = node_map->find(target);
        FatalAssert(it != node_map->end(), LOG_TAG_RDMA,
                    "Target compute node not found!");
        ConnectionContext& ctx = it->second;
        FatalAssert(ctx.socket != -1, LOG_TAG_RDMA,
                    "Socket to source node is not established.");
        ssize_t ret = send(ctx.socket, &size, sizeof(size), 0);
        FatalAssert(ret == sizeof(size), LOG_TAG_RDMA,
                    "Failed to send message size to target node. sent_bytes=%zd, expected_bytes=%zu, errno=(%d)%s",
                    ret, sizeof(size), errno, strerror(errno));
        ssize_t sent = 0;
        while (sent < static_cast<ssize_t>(size)) {
            ret = send(ctx.socket, static_cast<const uint8_t*>(msg) + sent,
                               std::min(static_cast<ssize_t>(MAX_MESSAGE_SIZE), static_cast<ssize_t>(size) - sent), 0);
            FatalAssert(ret >= 0, LOG_TAG_RDMA,
                        "Failed to send message to target node. sent_bytes=%zd, expected_bytes=%zu, errno=(%d)%s",
                        ret, size - sent, errno, strerror(errno));
            sent += ret;
        }
        FatalAssert(sent == static_cast<ssize_t>(size), LOG_TAG_RDMA,
                    "Failed to send message to target node. sent_bytes=%zd, expected_bytes=%zu, errno=(%d)%s",
                    sent, size, errno, strerror(errno));
    }

    void BroadcastMessage(const void* msg, size_t size) {
        FatalAssert(this == instance, LOG_TAG_RDMA,
                    "RDMA_Manager instance mismatch!");
        FatalAssert(size > 0, LOG_TAG_RDMA,
                    "Cannot send a message of size 0.");
        CHECK_NOT_NULLPTR(msg, LOG_TAG_RDMA);
        if (!ready.load(std::memory_order_acquire)) {
            FatalAssert(false, LOG_TAG_RDMA,
                        "RDMA_Manager is not ready for RDMA operations.");
            return;
        }

        std::unordered_map<NodeID, ConnectionContext, NodeIDHash>* node_map =
            selfInfo.node_id.IsComputeNode() ? &memory_nodes : &compute_nodes;

        for (auto& node_pair : *node_map) {
            NodeID node_id = node_pair.first;
            SendMessage(node_id, msg, size);
        }
    }

    void ReceiveMessage(NodeID source, void* buffer, size_t buffer_size, bool end_message = false) {
        FatalAssert(this == instance, LOG_TAG_RDMA,
                    "RDMA_Manager instance mismatch!");
        FatalAssert(selfInfo.node_id.IsMemoryNode() == source.IsComputeNode(), LOG_TAG_RDMA,
                    "Compute nodes can only send messages to memory nodes and vice versa.");
        FatalAssert(buffer_size > 0, LOG_TAG_RDMA,
                    "Cannot send a message of size 0.");
        CHECK_NOT_NULLPTR(buffer, LOG_TAG_RDMA);
        if ((!end_message && !ready.load(std::memory_order_acquire)) ||
            (end_message && ready.load(std::memory_order_relaxed))) {
            FatalAssert(false, LOG_TAG_RDMA,
                        "RDMA_Manager is not ready for RDMA operations.");
            return;
        }

        std::unordered_map<NodeID, ConnectionContext, NodeIDHash>* node_map =
            source.IsComputeNode() ? &compute_nodes : &memory_nodes;
        auto it = node_map->find(source);
        FatalAssert(it != node_map->end(), LOG_TAG_RDMA,
                    "Source memory node not found!");
        ConnectionContext& ctx = it->second;
        FatalAssert(ctx.socket != -1, LOG_TAG_RDMA,
                    "Socket to source node is not established.");
        size_t msg_size = 0;
        ssize_t ret = recv(ctx.socket, &msg_size, sizeof(msg_size), 0);
        FatalAssert(ret == sizeof(msg_size), LOG_TAG_RDMA,
                    "Failed to receive message size from source node. recv_bytes=%zd, expected_bytes=%zu, errno=(%d)%s",
                    ret, sizeof(msg_size), errno, strerror(errno));
        FatalAssert(msg_size <= buffer_size, LOG_TAG_RDMA,
                    "Received message size exceeds buffer size. msg_size=%zu, buffer_size=%zu",
                    msg_size, buffer_size);

        ssize_t recieved = 0;
        while (recieved < static_cast<ssize_t>(msg_size)) {
            ret = recv(ctx.socket, static_cast<uint8_t*>(buffer) + recieved,
                       std::min(static_cast<ssize_t>(MAX_MESSAGE_SIZE), static_cast<ssize_t>(msg_size) - recieved), 0);
            FatalAssert(ret >= 0, LOG_TAG_RDMA,
                        "Failed to receive message from source node. recv_bytes=%zd, expected_bytes=%zu, errno=(%d)%s",
                        ret, msg_size - recieved, errno, strerror(errno));
            recieved += ret;
        }
        FatalAssert(recieved == static_cast<ssize_t>(msg_size), LOG_TAG_RDMA,
                    "Failed to receive message from source node. recv_bytes=%zd, expected_bytes=%zu, errno=(%d)%s",
                    recieved, msg_size, errno, strerror(errno));
    }

    size_t GetNumMemoryNodes() const {
        return memory_nodes.size();
    }

    size_t GetNumComputeNodes() const {
        return compute_nodes.size();
    }

    NodeID GetMemoryNodeID() const {
        FatalAssert(instance == this, LOG_TAG_RDMA,
                    "RDMA_Manager instance mismatch!");
        FatalAssert(selfInfo.node_id.IsComputeNode(), LOG_TAG_RDMA,
                    "Only compute nodes have memory node IDs.");
        FatalAssert(memory_nodes.size() == 1, LOG_TAG_RDMA,
                    "Currently, only one memory node is supported.");
        return memory_nodes.begin()->first;
    }

protected:
    RDMA_Manager(uint8_t num_mnodes, uint8_t num_cnodes, uint8_t* mnode_ids,
                 char** mnode_ips, uint16_t* mnode_ports,
                 uint8_t* cnode_ids, char** cnode_ips, uint16_t* cnode_ports,
                 const char* target_rdma_device_name, uint8_t rdma_port, int gid_index,
                 bool is_memory_node, uint8_t self_idx, size_t num_threads) :
                    selfInfo{
                            .node_id = NodeID(is_memory_node, self_idx),
                            .ip_address = {0},
                            .port = is_memory_node ? mnode_ports[self_idx] : cnode_ports[self_idx],
                        }, _rdma_port(rdma_port), _gid_index(gid_index), pending_tasks(num_threads * 2),
                        poll_list(is_memory_node ? nullptr : new ibv_wc[MAX_CQE[COMPUTE_NODE_IDX]]) {
        FatalAssert(num_mnodes <= MAX_MNODE_COUNT, LOG_TAG_RDMA,
                    "Number of memory nodes exceeds the maximum allowed count!");
        FatalAssert(num_cnodes <= MAX_CNODE_COUNT, LOG_TAG_RDMA,
                    "Number of compute nodes exceeds the maximum allowed count!");
        FatalAssert(num_cnodes > 0, LOG_TAG_RDMA,
                    "There should be at least one compute node!");
        FatalAssert(num_mnodes > 0, LOG_TAG_RDMA,
                    "There should be at least one memory node!");
        FatalAssert(num_mnodes == 1, LOG_TAG_NOT_IMPLEMENTED,
                    "Currently, only one memory node is supported!");
        FatalAssert(self_idx < (is_memory_node ? num_mnodes : num_cnodes), LOG_TAG_RDMA,
                    "Self node index exceeds the number of nodes of its type!");
        FatalAssert(gid_index >= 0, LOG_TAG_RDMA,
                "gid_index is invalid. gid_index=%d", gid_index);
        FatalAssert(rdma_port > 0, LOG_TAG_RDMA,
                    "rdma_port is invalid. rdma_port=%hhu", rdma_port);
        CHECK_NOT_NULLPTR(mnode_ids, LOG_TAG_RDMA);
        CHECK_NOT_NULLPTR(mnode_ips, LOG_TAG_RDMA);
        CHECK_NOT_NULLPTR(mnode_ports, LOG_TAG_RDMA);
        CHECK_NOT_NULLPTR(cnode_ids, LOG_TAG_RDMA);
        CHECK_NOT_NULLPTR(cnode_ips, LOG_TAG_RDMA);
        CHECK_NOT_NULLPTR(cnode_ports, LOG_TAG_RDMA);
        CHECK_NOT_NULLPTR(target_rdma_device_name, LOG_TAG_RDMA);

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_RDMA,
                "Initializing RDMA_Manager for %s node. self_idx=%hhu, rdma_port=%hhu, gid_index=%d",
                is_memory_node ? "memory" : "compute", self_idx, rdma_port, gid_index);

        int ret = 0;
        String error_msg;
        String dev_list_str;
        int target_device_index;
        uint8_t num_valid_nodes;
        RetStatus rs = RetStatus::Success();
        const uint8_t type = is_memory_node ? MEMROY_NODE_IDX : COMPUTE_NODE_IDX;

        int num_ibv_devices = 0;
        struct ibv_device **dev_list = ibv_get_device_list(&num_ibv_devices);
        if (dev_list == nullptr || num_ibv_devices == 0) {
            error_msg = "Failed to get IB devices list";
            goto ERROR_EXIT;
        }

        dev_list_str = "Available IB devices:[";
        target_device_index = -1;
        for (int i = 0; i < num_ibv_devices; ++i) {
            dev_list_str += String("%s%s", ibv_get_device_name(dev_list[i]),
                                    (i == num_ibv_devices - 1) ? "]" : ", ");
            if (strcmp(ibv_get_device_name(dev_list[i]), target_rdma_device_name) == 0) {
                target_device_index = i;
            }
        }
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_RDMA, "%s", target_rdma_device_name);

        if (target_device_index == -1) {
            ibv_free_device_list(dev_list);
            error_msg = String("Target RDMA device '%s' not found. %s",
                                target_rdma_device_name, dev_list_str.ToCStr());
            goto ERROR_EXIT;
        }
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_RDMA,
                "Using RDMA device '%s'.",
                target_rdma_device_name);

        ib_ctx = ibv_open_device(dev_list[target_device_index]);
        if (ib_ctx == nullptr) {
            ibv_free_device_list(dev_list);
            error_msg = String("Failed to open RDMA device '%s'.", target_rdma_device_name);
            goto ERROR_EXIT;
        }

        ibv_free_device_list(dev_list);
        dev_list = nullptr;

        memset(&dev_attr, 0, sizeof(dev_attr));
        ret = ibv_query_device(ib_ctx, &dev_attr);
        if (ret != 0) {
            error_msg = String("Failed to query RDMA device '%s' attributes. ret=(%d)%s errno=(%d)%s",
                                target_rdma_device_name, ret, strerror(ret), errno, strerror(errno));
            goto ERROR_EXIT;
        }

        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_RDMA,
                "RDMA device '%s' attributes: fw_ver=%s, guid=0x%016lx, sys_img_guid=0x%016lx, max_mr_size=%lu, "
                "page_size_cap=%lu, vendor_id=%u, vendor_part_id=%u, hardware_ver=%u, max_qp=%d, "
                "max_qp_wr=%d, dev_cap_flags=%016x, max_sge=%d, max_sge_rd=%d, max_cq=%d, max_cqe=%d, "
                "max_mr=%d, max_pd=%d, max_qp_rd_atom=%d, max_ee_rd_atom=%d, max_res_rd_atom=%d, max_qp_init_rd_atom=%d, "
                "max_ee_init_rd_atom=%d, atomic_cap=%s, max_ee=%d, max_rdd=%d, max_mw=%d, max_raw_ipv6_qp=%d, "
                "max_raw_ethy_qp=%d, max_mcast_grp=%d, max_mcast_qp_attach=%d, max_total_mcast_qp_attach=%d, "
                "max_ah=%d, max_fmr=%d, max_map_per_fmr=%d, max_srq=%d, max_srq_wr=%d, max_srq_sge=%d, "
                "max_pkeys=%hu, local_ca_ack_delay=%hhu, phys_port_cnt=%hhu",
                target_rdma_device_name,
                dev_attr.fw_ver, dev_attr.node_guid, dev_attr.sys_image_guid, dev_attr.max_mr_size, dev_attr.page_size_cap,
                dev_attr.vendor_id, dev_attr.vendor_part_id, dev_attr.hw_ver, dev_attr.max_qp, dev_attr.max_qp_wr,
                dev_attr.device_cap_flags,
                dev_attr.max_sge, dev_attr.max_sge_rd, dev_attr.max_cq, dev_attr.max_cqe, dev_attr.max_mr, dev_attr.max_pd,
                dev_attr.max_qp_rd_atom, dev_attr.max_ee_rd_atom, dev_attr.max_res_rd_atom, dev_attr.max_qp_init_rd_atom,
                dev_attr.max_ee_init_rd_atom, (dev_attr.atomic_cap == IBV_ATOMIC_NONE) ? "NONE" :
                (dev_attr.atomic_cap == IBV_ATOMIC_HCA) ? "HCA" : "GLOB", dev_attr.max_ee, dev_attr.max_rdd, dev_attr.max_mw,
                dev_attr.max_raw_ipv6_qp, dev_attr.max_raw_ethy_qp, dev_attr.max_mcast_grp, dev_attr.max_mcast_qp_attach,
                dev_attr.max_total_mcast_qp_attach, dev_attr.max_ah, dev_attr.max_fmr, dev_attr.max_map_per_fmr,
                dev_attr.max_srq, dev_attr.max_srq_wr, dev_attr.max_srq_sge, dev_attr.max_pkeys,
                dev_attr.local_ca_ack_delay, dev_attr.phys_port_cnt);

        FatalAssert(_rdma_port <= dev_attr.phys_port_cnt, LOG_TAG_RDMA,
                    "_rdma_port (%hhu) exceeds the number of physical ports (%hhu) on RDMA device '%s'",
                    _rdma_port, dev_attr.phys_port_cnt, target_rdma_device_name);
        memset(&port_attr, 0, sizeof(port_attr));
        ret = ibv_query_port(ib_ctx, _rdma_port, &port_attr);
        if (ret != 0) {
            error_msg = String("Failed to query RDMA device '%s' port %hhu attributes. ret=(%d)%s errno=(%d)%s",
                                target_rdma_device_name, _rdma_port, ret, strerror(ret), errno, strerror(errno));
            goto ERROR_EXIT;
        }

        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_RDMA,
            "RDMA port attributes: state=%u, max_mtu=%u, active_mtu=%u, gid_tbl_len=%u, port_cap_flags=0x%x, "
            "max_msg_sz=%u, max_vl_num=%u, sm_lid=%u, sm_sl=%u, subnet_timeout=%u, init_type_reply=%u, "
            "active_width=%u, active_speed=%u, phys_state=%u, link_layer=%u, pkey_tbl_len=%u, lid=%u, lmc=%u, "
            "qkey_viol_cntr=%u, bad_pkey_cntr=%u",
            (unsigned)port_attr.state,
            (unsigned)port_attr.max_mtu,
            (unsigned)port_attr.active_mtu,
            (unsigned)port_attr.gid_tbl_len,
            (unsigned)port_attr.port_cap_flags,
            (unsigned)port_attr.max_msg_sz,
            (unsigned)port_attr.max_vl_num,
            (unsigned)port_attr.sm_lid,
            (unsigned)port_attr.sm_sl,
            (unsigned)port_attr.subnet_timeout,
            (unsigned)port_attr.init_type_reply,
            (unsigned)port_attr.active_width,
            (unsigned)port_attr.active_speed,
            (unsigned)port_attr.phys_state,
            (unsigned)port_attr.link_layer,
            (unsigned)port_attr.pkey_tbl_len,
            (unsigned)port_attr.lid,
            (unsigned)port_attr.lmc,
            port_attr.qkey_viol_cntr,
            port_attr.bad_pkey_cntr);

        FatalAssert(gid_index < (int)port_attr.gid_tbl_len, LOG_TAG_RDMA,
                    "gid_index (%d) exceeds the GID table length (%u) on RDMA device '%s' port %hhu",
                    gid_index, port_attr.gid_tbl_len, target_rdma_device_name, _rdma_port);

        ret = ibv_query_gid(ib_ctx, _rdma_port, gid_index, &dev_gid);
        if (ret != 0) {
            error_msg = String("Failed to query RDMA device '%s' port %hhu GID at index %d. ret=(%d)%s errno=(%d)%s",
                                target_rdma_device_name, _rdma_port, gid_index, ret, strerror(ret), errno, strerror(errno));
            goto ERROR_EXIT;
        }

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_RDMA,
                "RDMA device '%s' port %hhu GID at index %d: "
                "%02x%02x::%02x%02x::%02x%02x::%02x%02x::%02x%02x::%02x%02x::%02x%02x::%02x%02x",
                target_rdma_device_name, _rdma_port, gid_index,
                dev_gid.raw[0], dev_gid.raw[1], dev_gid.raw[2], dev_gid.raw[3],
                dev_gid.raw[4], dev_gid.raw[5], dev_gid.raw[6], dev_gid.raw[7],
                dev_gid.raw[8], dev_gid.raw[9], dev_gid.raw[10], dev_gid.raw[11],
                dev_gid.raw[12], dev_gid.raw[13], dev_gid.raw[14], dev_gid.raw[15]);
        pd = ibv_alloc_pd(ib_ctx);
        if (pd == nullptr) {
            error_msg = String("Failed to allocate Protection Domain.");
            goto ERROR_EXIT;
        }

        cq = ibv_create_cq(ib_ctx, MAX_CQE[type], nullptr, nullptr, 0);
        if (cq == nullptr) {
            error_msg = String("Failed to create CN Cluster Read Completion Queue. errno=(%d)%s",
                                errno, strerror(errno));
            goto ERROR_EXIT;
        }

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_RDMA, "Created Completion Queues with cqe=%d", cq->cqe);

        std::unordered_map<divftree::NodeID, divftree::ConnectionContext, divftree::NodeIDHash>* target_map;
        uint8_t num_nodes;
        uint16_t* ports;
        char** ips;
        uint8_t* ids;
        if (!is_memory_node) {
            target_map = &memory_nodes;
            num_nodes = num_mnodes;
            ports = mnode_ports;
            ips = mnode_ips;
            ids = mnode_ids;
        } else {
            target_map = &compute_nodes;
            num_nodes = num_cnodes;
            ports = cnode_ports;
            ips = cnode_ips;
            ids = cnode_ids;
        }
        num_valid_nodes = 0;
        for (uint8_t i = 0; i < num_nodes; ++i) {
            NodeID node_id = NodeID(!is_memory_node, ids[i]);
            auto it = target_map->emplace(node_id, ConnectionContext(node_id, inet_addr(ips[i]), ports[i]));
            FatalAssert(it.second, LOG_TAG_RDMA,
                        "Duplicate compute node ID %hhu detected!", ids[i]);
            ++num_valid_nodes;
            rs = InitConnectionCtx(type, it.first->second);
            if (!rs.IsOK()) {
                error_msg = String("Failed to initialize QPs to %s node %hhu. Error: %s",
                                    is_memory_node ? "compute" : "memory", i, rs.Msg());
                goto ERROR_EXIT;
            }
        }

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_RDMA,
                "Created QPs for each of the %hhu/%hhu %s nodes.",
                num_valid_nodes, num_nodes, (is_memory_node ? "compute" : "memory"));

        return;

ERROR_EXIT:
        FatalAssert(false, LOG_TAG_RDMA, "%s", error_msg.ToCStr());
        Cleanup();
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_RDMA, "%s", error_msg.ToCStr());
    }

    ~RDMA_Manager() {

        FatalAssert(instance == this, LOG_TAG_RDMA,
                    "RDMA_Manager instance mismatch!");
        FatalAssert(!ready.load(std::memory_order_acquire), LOG_TAG_RDMA,
                    "RDMA_Manager must be marked not ready before destruction.");

        Cleanup();
    }

    RetStatus Cleanup() {
        FatalAssert(instance == this, LOG_TAG_RDMA,
                    "RDMA_Manager instance mismatch!");
        FatalAssert(pending_tasks.IsEmpty(), LOG_TAG_RDMA,
                    "There are still pending tasks in the pending tasks queue.");
        for (auto& node_pair : compute_nodes) {
            ConnectionContext& ctx = node_pair.second;
            for (uint8_t conn_idx = 0; conn_idx < MAX_CONN_PER_NODE; ++conn_idx) {
                ConnectionInfo& conn_info = ctx.connections[conn_idx];
                FatalAssert(conn_info.num_pending_requests.load() == 0, LOG_TAG_RDMA,
                            "There are still pending requests on connection to compute node %s. pending_requests=%zu",
                            node_pair.first.ToString().ToCStr(),
                            conn_info.num_pending_requests.load());
                conn_info.cq = nullptr;
                if (conn_info.qp != nullptr) {
                    ibv_destroy_qp(conn_info.qp);
                    conn_info.qp = nullptr;
                }
            }
        }

        for (auto& node_pair : memory_nodes) {
            ConnectionContext& ctx = node_pair.second;
            for (uint8_t conn_idx = 0; conn_idx < MAX_CONN_PER_NODE; ++conn_idx) {
                ConnectionInfo& conn_info = ctx.connections[conn_idx];
                FatalAssert(conn_info.num_pending_requests.load() == 0, LOG_TAG_RDMA,
                            "There are still pending requests on connection to memory node %s. pending_requests=%zu",
                            node_pair.first.ToString().ToCStr(),
                            conn_info.num_pending_requests.load());
                conn_info.cq = nullptr;
                if (conn_info.qp != nullptr) {
                    ibv_destroy_qp(conn_info.qp);
                    conn_info.qp = nullptr;
                }
            }
        }

        if (mr != nullptr) {
            ibv_dereg_mr(mr);
            mr = nullptr;
        }

        if (cq != nullptr) {
            ibv_destroy_cq(cq);
            cq = nullptr;
        }

        if (pd != nullptr) {
            ibv_dealloc_pd(pd);
            pd = nullptr;
        }

        if (ib_ctx != nullptr) {
            ibv_close_device(ib_ctx);
            ib_ctx = nullptr;
        }

        if (poll_list != nullptr) {
            delete[] poll_list;
            poll_list = nullptr;
        }

        RetStatus rs = CloseTCPConnections();
        return rs;
    }

    static RetStatus CreateQP(struct ibv_qp** qp, struct ibv_cq* cq, struct ibv_pd* pd, uint8_t type) {
        CHECK_NOT_NULLPTR(qp, LOG_TAG_RDMA);
        CHECK_NOT_NULLPTR(cq, LOG_TAG_RDMA);
        CHECK_NOT_NULLPTR(pd, LOG_TAG_RDMA);
        struct ibv_qp_init_attr qp_init_attr;
        memset(&qp_init_attr, 0, sizeof(qp_init_attr));
        qp_init_attr.send_cq = cq;
        qp_init_attr.recv_cq = cq;
        qp_init_attr.qp_type = IBV_QPT_RC; /* Reliable Connection */
        qp_init_attr.cap.max_send_wr = MAX_SEND_WR[type]; /* max outstanding send requests */
        qp_init_attr.cap.max_recv_wr = MAX_RECV_WR[type]; /* max outstanding recv requests */
        qp_init_attr.cap.max_send_sge = MAX_SEND_SGE[type];   /* max scatter/gather elements in a send request */
        qp_init_attr.cap.max_recv_sge = MAX_RECV_SGE[type];   /* max scatter/gather elements in a recv request */
        qp_init_attr.cap.max_inline_data = MAX_INLINE_DATA[type]; /* max size of inline data */
        qp_init_attr.sq_sig_all = 0; /* we should not create wc for all send requests */

        *qp = ibv_create_qp(pd, &qp_init_attr);
        if (*qp == nullptr) {
            FatalAssert(false, LOG_TAG_RDMA,
                        "Failed to create Queue Pair. errno=(%d)%s",
                        errno, strerror(errno));
            return RetStatus::Fail(String("Failed to create Queue Pair. errno=(%d)%s",
                                            errno, strerror(errno)).ToCStr());
        }
        return RetStatus::Success();
    }

    static RetStatus ModifyQPStateToReset(struct ibv_qp* qp) {
        CHECK_NOT_NULLPTR(qp, LOG_TAG_RDMA);
        struct ibv_qp_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.qp_state = IBV_QPS_RESET;

        int flags = IBV_QP_STATE;
        int ret = ibv_modify_qp(qp, &qp_attr, flags);
        if (ret != 0) {
            FatalAssert(false, LOG_TAG_RDMA,
                        "Failed to modify QP to RESET state. ret=(%d)%s errno=(%d)%s",
                        ret, strerror(ret), errno, strerror(errno));
            return RetStatus::Fail(String("Failed to modify QP to RESET state. ret=(%d)%s errno=(%d)%s",
                                            ret, strerror(ret), errno, strerror(errno)).ToCStr());
        }
        return RetStatus::Success();
    }

    static RetStatus ModifyQPStateToInit(struct ibv_qp* qp, uint8_t port_num) {
        CHECK_NOT_NULLPTR(qp, LOG_TAG_RDMA);
        struct ibv_qp_attr qp_attr;
        int ret = 0;
        memset(&qp_attr, 0, sizeof(qp_attr));
        SANITY_CHECK(
            struct ibv_qp_init_attr init_attr;
            memset(&init_attr, 0, sizeof(init_attr));
            ret = ibv_query_qp(qp, &qp_attr,
                                IBV_QP_STATE,
                                &init_attr);
            FatalAssert(ret == 0, LOG_TAG_RDMA,
                        "Failed to query QP state before modifying to INIT. ret=(%d)%s errno=(%d)%s",
                        ret, strerror(ret), errno, strerror(errno));
            FatalAssert(qp_attr.qp_state == IBV_QPS_RESET, LOG_TAG_RDMA,
                        "QP is not in RESET state before modifying to INIT. current_state=%d",
                        qp_attr.qp_state);
            memset(&qp_attr, 0, sizeof(qp_attr));
        );
        qp_attr.qp_state = IBV_QPS_INIT;
        qp_attr.port_num = port_num;
        qp_attr.pkey_index = 0;
        qp_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
                                    IBV_ACCESS_REMOTE_READ |
                                    IBV_ACCESS_REMOTE_WRITE |
                                    IBV_ACCESS_REMOTE_ATOMIC;

        int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
        ret = ibv_modify_qp(qp, &qp_attr, flags);
        if (ret != 0) {
            FatalAssert(false, LOG_TAG_RDMA,
                        "Failed to modify QP to INIT state. ret=(%d)%s errno=(%d)%s",
                        ret, strerror(ret), errno, strerror(errno));
            return RetStatus::Fail(String("Failed to modify QP to INIT state. ret=(%d)%s errno=(%d)%s",
                                            ret, strerror(ret), errno, strerror(errno)).ToCStr());
        }
        return RetStatus::Success();
    }

    static RetStatus ModifyQPStateToRTR(struct ibv_qp* qp, uint32_t dest_qp_num,
                                        const union ibv_gid& dest_gid,
                                        uint8_t port_num, uint32_t remote_psn, uint8_t type) {
        CHECK_NOT_NULLPTR(qp, LOG_TAG_RDMA);
        struct ibv_qp_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        int ret = 0;
        SANITY_CHECK(
            struct ibv_qp_init_attr init_attr;
            memset(&init_attr, 0, sizeof(init_attr));
            ret = ibv_query_qp(qp, &qp_attr,
                                IBV_QP_STATE,
                                &init_attr);
            FatalAssert(ret == 0, LOG_TAG_RDMA,
                        "Failed to query QP state before modifying to RTR. ret=(%d)%s errno=(%d)%s",
                        ret, strerror(ret), errno, strerror(errno));
            FatalAssert(qp_attr.qp_state == IBV_QPS_INIT, LOG_TAG_RDMA,
                        "QP is not in INIT state before modifying to RTR. current_state=%d",
                        qp_attr.qp_state);
            memset(&qp_attr, 0, sizeof(qp_attr));
        );

        qp_attr.qp_state = IBV_QPS_RTR;
        qp_attr.path_mtu = DEFAULT_MTU[type];
        qp_attr.dest_qp_num = dest_qp_num;
        qp_attr.rq_psn = remote_psn;
        qp_attr.max_dest_rd_atomic = MAX_RD_ATOMIC[(type == MEMROY_NODE_IDX ? COMPUTE_NODE_IDX : MEMROY_NODE_IDX)];
        qp_attr.min_rnr_timer = 1;
        qp_attr.ah_attr.is_global = 1;
        qp_attr.ah_attr.dlid = 0;
        qp_attr.ah_attr.sl = 0;
        qp_attr.ah_attr.src_path_bits = 0;
        qp_attr.ah_attr.port_num = port_num;
        memcpy(&qp_attr.ah_attr.grh.dgid, &dest_gid, sizeof(dest_gid));
        qp_attr.ah_attr.grh.flow_label = 0;
        qp_attr.ah_attr.grh.hop_limit = 1;
        qp_attr.ah_attr.grh.sgid_index = 0;
        qp_attr.ah_attr.grh.traffic_class = 0;

        int flags = IBV_QP_STATE |
                    IBV_QP_AV |
                    IBV_QP_PATH_MTU |
                    IBV_QP_DEST_QPN |
                    IBV_QP_RQ_PSN |
                    IBV_QP_MAX_DEST_RD_ATOMIC |
                    IBV_QP_MIN_RNR_TIMER;
        ret = ibv_modify_qp(qp, &qp_attr, flags);
        if (ret != 0) {
            FatalAssert(false, LOG_TAG_RDMA,
                        "Failed to modify QP to RTR state. ret=(%d)%s errno=(%d)%s",
                        ret, strerror(ret), errno, strerror(errno));
            return RetStatus::Fail(String("Failed to modify QP to RTR state. ret=(%d)%s errno=(%d)%s",
                                            ret, strerror(ret), errno, strerror(errno)).ToCStr());
        }
        return RetStatus::Success();
    }

    static RetStatus ModifyQPStateToRTS(struct ibv_qp* qp, uint32_t local_psn, uint8_t type) {
        CHECK_NOT_NULLPTR(qp, LOG_TAG_RDMA);
        struct ibv_qp_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        int ret = 0;
        SANITY_CHECK(
            struct ibv_qp_init_attr init_attr;
            memset(&init_attr, 0, sizeof(init_attr));
            ret = ibv_query_qp(qp, &qp_attr,
                                IBV_QP_STATE,
                                &init_attr);
            FatalAssert(ret == 0, LOG_TAG_RDMA,
                        "Failed to query QP state before modifying to RTS. ret=(%d)%s errno=(%d)%s",
                        ret, strerror(ret), errno, strerror(errno));
            FatalAssert(qp_attr.qp_state == IBV_QPS_RTR, LOG_TAG_RDMA,
                        "QP is not in RTR state before modifying to RTS. current_state=%d",
                        qp_attr.qp_state);
            memset(&qp_attr, 0, sizeof(qp_attr));
        );

        qp_attr.qp_state = IBV_QPS_RTS;
        qp_attr.sq_psn = local_psn;
        qp_attr.timeout = 14; /* todo: infinite retry -> set to something better later */
        qp_attr.retry_cnt = 7; /* todo: infinite retry -> set to something better later */
        qp_attr.rnr_retry = 7; /* todo: infinite retry -> set to something better later */
        qp_attr.max_rd_atomic = MAX_RD_ATOMIC[type];

        int flags = IBV_QP_STATE |
                    IBV_QP_SQ_PSN |
                    IBV_QP_TIMEOUT |
                    IBV_QP_RETRY_CNT |
                    IBV_QP_RNR_RETRY |
                    IBV_QP_MAX_QP_RD_ATOMIC;
        ret = ibv_modify_qp(qp, &qp_attr, flags);
        if (ret != 0) {
            FatalAssert(false, LOG_TAG_RDMA,
                        "Failed to modify QP to RTS state. ret=(%d)%s errno=(%d)%s",
                        ret, strerror(ret), errno, strerror(errno));
            return RetStatus::Fail(String("Failed to modify QP to RTS state. ret=(%d)%s errno=(%d)%s",
                                            ret, strerror(ret), errno, strerror(errno)).ToCStr());
        }

        SANITY_CHECK(
            memset(&init_attr, 0, sizeof(init_attr));
            struct ibv_qp_attr qp_attr_dummy;
            memset(&qp_attr_dummy, 0, sizeof(qp_attr_dummy));
            ret = ibv_query_qp(qp, &qp_attr_dummy,
                                IBV_QP_STATE,
                                &init_attr);
            FatalAssert(ret == 0, LOG_TAG_RDMA,
                        "Failed to query QP state after modifying to RTS. ret=(%d)%s errno=(%d)%s",
                        ret, strerror(ret), errno, strerror(errno));
            FatalAssert(qp_attr_dummy.qp_state == IBV_QPS_RTS, LOG_TAG_RDMA,
                        "QP is not in RTS state after modifying to RTS. current_state=%d",
                        qp_attr_dummy.qp_state);
        )
        return RetStatus::Success();
    }

    static RetStatus ModifyQPStateToError(struct ibv_qp* qp) {
        CHECK_NOT_NULLPTR(qp, LOG_TAG_RDMA);
        struct ibv_qp_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));

        qp_attr.qp_state = IBV_QPS_ERR;

        int flags = IBV_QP_STATE;
        int ret = ibv_modify_qp(qp, &qp_attr, flags);
        if (ret != 0) {
            FatalAssert(false, LOG_TAG_RDMA,
                        "Failed to modify QP to ERROR state. ret=(%d)%s errno=(%d)%s",
                        ret, strerror(ret), errno, strerror(errno));
            return RetStatus::Fail(String("Failed to modify QP to ERROR state. ret=(%d)%s errno=(%d)%s",
                                            ret, strerror(ret), errno, strerror(errno)).ToCStr());
        }
        return RetStatus::Success();
    }

    RetStatus InitConnectionCtx(uint8_t type, ConnectionContext& ctx) {
        String error_msg;
        RetStatus rs = RetStatus::Success();
        struct ibv_qp* qp = nullptr;
        FatalAssert(cq != nullptr, LOG_TAG_RDMA,
                    "cq cannot be nullptr");
        FatalAssert(pd != nullptr, LOG_TAG_RDMA,
                    "pd cannot be nullptr");

        for (uint8_t conn_id = 0; conn_id < MAX_CONN_PER_NODE; ++conn_id) {
            ctx.connections[conn_id].local_psn = threadSelf->UniformRange32(0, (uint32_t)(1 << 24) - 1);
            ctx.connections[conn_id].cq = cq;

            rs = CreateQP(&qp, ctx.connections[conn_id].cq, pd, type);
            if (!rs.IsOK() || qp == nullptr) {
                error_msg = String("Failed to create QP for connection with id %hhu. Error: %s", conn_id, rs.Msg());
                rs = RetStatus::Fail(error_msg.ToCStr());
                break;
            }

            rs = ModifyQPStateToReset(qp);
            if (!rs.IsOK()) {
                error_msg = String("Failed to modify QP state to RESET for connection %hhu. Error: %s",
                                    conn_id, rs.Msg());
                rs = RetStatus::Fail(error_msg.ToCStr());
                break;
            }

            rs = ModifyQPStateToInit(qp, _rdma_port);
            if (!rs.IsOK()) {
                error_msg = String("Failed to modify QP state to INIT for connection %hhu. Error: %s",
                                    conn_id, rs.Msg());
                rs = RetStatus::Fail(error_msg.ToCStr());
                break;
            }

            ctx.connections[conn_id].qp = qp;
        }

        return rs;
    }

    void FillHandshakeInfo(NodeID node_id, HandshakeInfo& handshake_info) {
        FatalAssert(node_id.IsComputeNode() == selfInfo.node_id.IsMemoryNode(), LOG_TAG_RDMA,
                    "Only MN-CN handshake is supported.");
        CHECK_NOT_NULLPTR(mr, LOG_TAG_RDMA);
        std::unordered_map<divftree::NodeID, divftree::ConnectionContext, divftree::NodeIDHash>* target_map;
        if (node_id.IsComputeNode()) {
            target_map = &compute_nodes;
        } else {
            target_map = &memory_nodes;
        }

        auto it = target_map->find(node_id);
        FatalAssert(it != target_map->end(), LOG_TAG_RDMA,
                    "Connection context for node_id=%s not found.", node_id.ToString().ToCStr());
        ConnectionContext& ctx = it->second;
        memcpy(&handshake_info.gid, &dev_gid, sizeof(dev_gid));
        handshake_info.region_addr = (uintptr_t)(mr->addr);
        handshake_info.region_size = mr->length;
        handshake_info.region_rkey = mr->rkey;
        for (uint8_t conn_id = 0; conn_id < MAX_CONN_PER_NODE; ++conn_id) {
            handshake_info.qp_num[conn_id] = ctx.connections[conn_id].qp->qp_num;
            handshake_info.psn[conn_id] = ctx.connections[conn_id].local_psn;
        }
    }

    void ProcessHandshakeInfo(NodeID node_id, const HandshakeInfo& handshake_info) {
        FatalAssert(node_id.IsComputeNode() == selfInfo.node_id.IsMemoryNode(), LOG_TAG_RDMA,
                    "Only MN-CN handshake is supported.");
        std::unordered_map<divftree::NodeID, divftree::ConnectionContext, divftree::NodeIDHash>* target_map;
        if (node_id.IsComputeNode()) {
            target_map = &compute_nodes;
        } else {
            target_map = &memory_nodes;
        }

        auto it = target_map->find(node_id);
        FatalAssert(it != target_map->end(), LOG_TAG_RDMA,
                    "Connection context for node_id=%s not found.", node_id.ToString().ToCStr());
        ConnectionContext& ctx = it->second;
        memcpy(&ctx.remote_gid, &handshake_info.gid, sizeof(ctx.remote_gid));
        ctx.remote_region_addr = handshake_info.region_addr;
        ctx.remote_region_size = handshake_info.region_size;
        ctx.remote_region_rkey = handshake_info.region_rkey;
        for (uint8_t conn_id = 0; conn_id < MAX_CONN_PER_NODE; ++conn_id) {
            ctx.connections[conn_id].remote_qp_num = handshake_info.qp_num[conn_id];
            ctx.connections[conn_id].remote_psn = handshake_info.psn[conn_id];
        }
    }

    RetStatus EstablishTCPConnections() {
        if (selfInfo.node_id.IsMemoryNode()) {
            return EstablishTCPConnectionsFromMN();
        } else {
            return EstablishTCPConnectionsFromCN();
        }
    }

    RetStatus EstablishTCPConnectionsFromMN() {
        RetStatus rs = RetStatus::Success();
        FatalAssert(!compute_nodes.empty(), LOG_TAG_RDMA, "There should be at least one compute node to connect to.");
        /* creating the server socket */
        FatalAssert(server_socket == -1, LOG_TAG_RDMA,
                    "Server socket is already created.");
        struct sockaddr_in server_addr;
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = INADDR_ANY;
        server_addr.sin_port = htons(selfInfo.port);
        uint8_t num_nodes = static_cast<uint8_t>(compute_nodes.size());
        NodeID remote_node_id;

        server_socket = socket(AF_INET, SOCK_STREAM, 0);
        if (server_socket < 0) {
            rs = RetStatus::Fail(String("Failed to create server socket with errno %d: %s",
                                        errno, strerror(errno)).ToCStr());
            goto EXIT;
        }

        if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            rs = RetStatus::Fail(String("Failed to bind server socket with errno %d: %s",
                                    errno, strerror(errno)).ToCStr());
            goto EXIT;
        }

        if (listen(server_socket, num_nodes) < 0) {
            rs = RetStatus::Fail(String("Failed to listen on server socket with errno %d: %s",
                                    errno, strerror(errno)).ToCStr());
            goto EXIT;
        }

        for (uint8_t i = 0; i < num_nodes; ++i) {
            int socket = accept(server_socket, NULL, NULL);
            if (socket < 0) {
                rs = RetStatus::Fail(String("Failed to accept connection on server socket with errno %d: %s",
                                        errno, strerror(errno)).ToCStr());
                goto EXIT;
            }

            remote_node_id = selfInfo.node_id;
            if (recv(socket, &remote_node_id, sizeof(remote_node_id), 0) != sizeof(remote_node_id)) {
                rs = RetStatus::Fail(String("Failed to receive remote node id with errno %d: %s",
                                        errno, strerror(errno)).ToCStr());
                goto EXIT;
            }

            FatalAssert(remote_node_id.IsComputeNode(), LOG_TAG_RDMA,
                        "Received invalid remote node id from compute node. node_id=%s",
                        remote_node_id.ToString().ToCStr());
            FatalAssert(remote_node_id.id < MAX_CNODE_COUNT, LOG_TAG_RDMA,
                        "Received invalid compute node id from compute node. node_id=%s",
                        remote_node_id.ToString().ToCStr());
            auto it = compute_nodes.find(remote_node_id);
            FatalAssert(it != compute_nodes.end(), LOG_TAG_RDMA,
                        "Received connection from unknown compute node. node_id=%s",
                        remote_node_id.ToString().ToCStr());
            ConnectionContext& ctx = it->second;
            FatalAssert(remote_node_id == ctx.remote_node_info.node_id, LOG_TAG_RDMA,
                        "Remote node id mismatch. expected=%s, received=%s",
                        ctx.remote_node_info.node_id.ToString().ToCStr(),
                        remote_node_id.ToString().ToCStr());
            FatalAssert(ctx.socket == -1, LOG_TAG_RDMA,
                        "TCP connection is already established with compute node. node_id=%s",
                        remote_node_id.ToString().ToCStr());

            ctx.socket = socket;

            remote_node_id = selfInfo.node_id;
            if (send(socket, &remote_node_id, sizeof(remote_node_id), 0) != sizeof(remote_node_id)) {
                rs = RetStatus::Fail(String("Failed to send local node id to compute node with errno %d: %s",
                                        errno, strerror(errno)).ToCStr());
                goto EXIT;
            }

            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_RDMA,
                    "Accepted TCP connection from node (node_id=%s, ip=%u, port=%hu)",
                    ctx.remote_node_info.node_id.ToString().ToCStr(),
                    ctx.remote_node_info.ip_address, ctx.remote_node_info.port);
        }

    EXIT:
        FatalAssert(rs.IsOK(), LOG_TAG_RDMA,
                    "Failed to establish TCP connections from MN. Error: %s", rs.Msg());
        return rs;
    }

    RetStatus EstablishTCPConnectionsFromCN() {
        RetStatus rs = RetStatus::Success();
        FatalAssert(!memory_nodes.empty(), LOG_TAG_RDMA, "There should be at least one memory node to connect to.");
        FatalAssert(memory_nodes.size() == 1, LOG_TAG_NOT_IMPLEMENTED,
                    "Currently, only one memory node is supported.");
        const NodeID MEMORY_NODE_ID = memory_nodes.begin()->first;
        NodeID remote_node_id = selfInfo.node_id;
        ConnectionContext& ctx = memory_nodes.begin()->second;
        FatalAssert(ctx.socket == -1, LOG_TAG_RDMA,
                    "TCP connection is already established with memory node.");
        FatalAssert(ctx.remote_node_info.node_id == MEMORY_NODE_ID, LOG_TAG_RDMA,
                    "Remote node id mismatch. expected=%s, actual=%s",
                    MEMORY_NODE_ID.ToString().ToCStr(),
                    ctx.remote_node_info.node_id.ToString().ToCStr());

        struct sockaddr_in server_addr;
        ctx.socket = socket(AF_INET, SOCK_STREAM, 0);
        if (ctx.socket < 0) {
            rs = RetStatus::Fail(String("Failed to create client socket with errno %d: %s",
                                    errno, strerror(errno)).ToCStr());
            goto EXIT;
        }

        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_addr.s_addr = ctx.remote_node_info.ip_address;
        server_addr.sin_port = htons(ctx.remote_node_info.port);
        if (connect(ctx.socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            rs = RetStatus::Fail(String("Failed to connect to server with errno %d: %s",
                                    errno, strerror(errno)).ToCStr());
            goto EXIT;
        }

        if (send(ctx.socket, &selfInfo.node_id, sizeof(selfInfo.node_id), 0) != sizeof(selfInfo.node_id)) {
            rs = RetStatus::Fail(String("Failed to send self node id with errno %d: %s",
                                        errno, strerror(errno)).ToCStr());
            goto EXIT;
        }

        if (recv(ctx.socket, &remote_node_id, sizeof(remote_node_id), 0) != sizeof(remote_node_id)) {
            rs = RetStatus::Fail(String("Failed to receive remote node id with errno %d: %s",
                                        errno, strerror(errno)).ToCStr());
            goto EXIT;
        }
        FatalAssert(remote_node_id.IsMemoryNode(), LOG_TAG_RDMA,
                    "Received invalid remote node id from memory node. node_id=%s",
                    remote_node_id.ToString().ToCStr());
        FatalAssert(remote_node_id == MEMORY_NODE_ID, LOG_TAG_RDMA,
                    "Remote node id mismatch. expected=%s, received=%s",
                    MEMORY_NODE_ID.ToString().ToCStr(),
                    remote_node_id.ToString().ToCStr());

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_RDMA,
                "Established TCP connection to memory node (node_id=%s, ip=%u, port=%hu)",
                MEMORY_NODE_ID.ToString().ToCStr(), ctx.remote_node_info.ip_address, ctx.remote_node_info.port);

EXIT:
        FatalAssert(rs.IsOK(), LOG_TAG_RDMA,
                    "Failed to establish TCP connection from CN. Error: %s", rs.Msg());
        return rs;
    }

    RetStatus Handshake() {
        if (selfInfo.node_id.IsMemoryNode()) {
            return HandshakeFromMN();
        } else {
            return HandshakeFromCN();
        }
    }

    RetStatus HandshakeFromMN() {
        RetStatus rs = RetStatus::Success();
        FatalAssert(!compute_nodes.empty(), LOG_TAG_RDMA, "There should be at least one compute node to connect to.");

        for (auto& pair : compute_nodes) {
            NodeID node_id = pair.first;
            ConnectionContext& ctx = pair.second;
            HandshakeInfo handshake_info;
            memset(&handshake_info, 0, sizeof(handshake_info));

            FillHandshakeInfo(node_id, handshake_info);

            /* send local handshake info */
            ssize_t bytes_sent = send(ctx.socket, &handshake_info, sizeof(handshake_info), 0);
            if (bytes_sent != sizeof(handshake_info)) {
                rs = RetStatus::Fail(String("Failed to send handshake info to compute node %s. "
                                            "Sent %zd bytes instead of %zu. errno=(%d)%s",
                                            node_id.ToString().ToCStr(),
                                            bytes_sent, sizeof(handshake_info),
                                            errno, strerror(errno)).ToCStr());
                goto EXIT;
            }

            memset(&handshake_info, 0, sizeof(handshake_info));
            /* receive remote handshake info */
            ssize_t bytes_received = recv(ctx.socket, &handshake_info, sizeof(handshake_info), 0);
            if (bytes_received != sizeof(handshake_info)) {
                rs = RetStatus::Fail(String("Failed to receive handshake info from compute node %s. "
                                            "Received %zd bytes instead of %zu. errno=(%d)%s",
                                            node_id.ToString().ToCStr(),
                                            bytes_received, sizeof(handshake_info),
                                            errno, strerror(errno)).ToCStr());
                goto EXIT;
            }

            ProcessHandshakeInfo(node_id, handshake_info);

            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_RDMA,
                    "Completed RDMA handshake with compute node %s.",
                    node_id.ToString().ToCStr());
        }
EXIT:
        FatalAssert(rs.IsOK(), LOG_TAG_RDMA,
                    "Failed to perform RDMA handshake from MN. Error: %s", rs.Msg());
        return rs;
    }

    RetStatus HandshakeFromCN() {
        RetStatus rs = RetStatus::Success();
        FatalAssert(!memory_nodes.empty(), LOG_TAG_RDMA, "There should be at least one memory node to connect to.");
        FatalAssert(memory_nodes.size() == 1, LOG_TAG_NOT_IMPLEMENTED,
                    "Currently, only one memory node is supported.");
        const NodeID MEMORY_NODE_ID = memory_nodes.begin()->first;
        ConnectionContext& ctx = memory_nodes.begin()->second;
        ssize_t bytes = 0;
        FatalAssert(ctx.remote_node_info.node_id == MEMORY_NODE_ID, LOG_TAG_RDMA,
                    "Remote node id mismatch. expected=%s, actual=%s",
                    MEMORY_NODE_ID.ToString().ToCStr(),
                    ctx.remote_node_info.node_id.ToString().ToCStr());
        HandshakeInfo handshake_info;
        memset(&handshake_info, 0, sizeof(handshake_info));

        bytes = recv(ctx.socket, &handshake_info, sizeof(handshake_info), 0);
        if (bytes != sizeof(handshake_info)) {
            rs = RetStatus::Fail(String("Failed to receive handshake info from compute node %s. "
                                        "Received %zd bytes instead of %zu. errno=(%d)%s",
                                        MEMORY_NODE_ID.ToString().ToCStr(),
                                        bytes, sizeof(handshake_info),
                                        errno, strerror(errno)).ToCStr());
            goto EXIT;
        }

        ProcessHandshakeInfo(MEMORY_NODE_ID, handshake_info);

        FillHandshakeInfo(MEMORY_NODE_ID, handshake_info);
        bytes = send(ctx.socket, &handshake_info, sizeof(handshake_info), 0);
        if (bytes != sizeof(handshake_info)) {
            rs = RetStatus::Fail(String("Failed to send handshake info to memory node %s. "
                                        "Sent %zd bytes instead of %zu. errno=(%d)%s",
                                        MEMORY_NODE_ID.ToString().ToCStr(),
                                        bytes, sizeof(handshake_info),
                                        errno, strerror(errno)).ToCStr());
            goto EXIT;
        }
EXIT:
        FatalAssert(rs.IsOK(), LOG_TAG_RDMA,
                    "Failed to perform RDMA handshake from CN. Error: %s", rs.Msg());
        return rs;
    }

    RetStatus EstablishRDMAConnections() {
        RetStatus rs = RetStatus::Success();
        std::unordered_map<divftree::NodeID, divftree::ConnectionContext, divftree::NodeIDHash>* target_map;
        uint8_t type;
        if (selfInfo.node_id.IsMemoryNode()) {
            target_map = &compute_nodes;
            type = MEMROY_NODE_IDX;
        } else {
            target_map = &memory_nodes;
            type = COMPUTE_NODE_IDX;
        }

        for (auto& pair : *target_map) {
            NodeID node_id = pair.first;
            ConnectionContext& ctx = pair.second;

            for (uint8_t conn_id = 0; conn_id < MAX_CONN_PER_NODE; ++conn_id) {
                rs = ModifyQPStateToRTR(ctx.connections[conn_id].qp,
                                        ctx.connections[conn_id].remote_qp_num,
                                        ctx.remote_gid,
                                        _rdma_port,
                                        ctx.connections[conn_id].remote_psn,
                                        type);
                if (!rs.IsOK()) {
                    String error_msg = String("Failed to modify QP to RTR state for connection %hhu with node %s. Error: %s",
                                                conn_id, node_id.ToString().ToCStr(), rs.Msg());
                    rs = RetStatus::Fail(error_msg.ToCStr());
                    goto EXIT;
                }

                rs = ModifyQPStateToRTS(ctx.connections[conn_id].qp,
                                        ctx.connections[conn_id].local_psn,
                                        type);
                if (!rs.IsOK()) {
                    String error_msg = String("Failed to modify QP to RTS state for connection %hhu with node %s. Error: %s",
                                                conn_id, node_id.ToString().ToCStr(), rs.Msg());
                    rs = RetStatus::Fail(error_msg.ToCStr());
                    goto EXIT;
                }

                num_connected_nodes.fetch_add(1);
            }

            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_RDMA,
                    "Established RDMA connections with node %s.",
                    node_id.ToString().ToCStr());
        }
EXIT:
        return rs;
    }

    RetStatus CloseTCPConnections() {
        RetStatus rs = RetStatus::Success();
        std::unordered_map<divftree::NodeID, divftree::ConnectionContext, divftree::NodeIDHash>* target_map;
        if (selfInfo.node_id.IsMemoryNode()) {
            target_map = &compute_nodes;
        } else {
            target_map = &memory_nodes;
        }

        for (auto& pair : *target_map) {
            ConnectionContext& ctx = pair.second;
            if (ctx.socket != -1) {
                close(ctx.socket);
                ctx.socket = -1;
            }
        }

        if (server_socket != -1) {
            close(server_socket);
            server_socket = -1;
        }

        return rs;
    }

    uint8_t GrabConnection(NodeID target, size_t num_clusters, uint8_t& num_resources_acquired) {
        FatalAssert(target.IsMemoryNode(), LOG_TAG_RDMA,
                    "Only connections to memory nodes can be grabbed.");
        FatalAssert(selfInfo.node_id.IsComputeNode(), LOG_TAG_RDMA,
                    "Only compute nodes can grab connections to memory nodes.");
        auto it = memory_nodes.find(target);
        FatalAssert(it != memory_nodes.end(), LOG_TAG_RDMA,
                    "Connection context for target memory node %s not found.",
                    target.ToString().ToCStr());
        ConnectionContext& ctx = it->second;
        uint8_t idx = ctx.next_connection_idx.fetch_add(1) % MAX_CONN_PER_NODE;
        uint64_t num_iterations = 0;
        while (true) {
            if ((num_iterations > 0) && (num_iterations % MAX_CONN_PER_NODE == 0)) {
                DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_RDMA,
                        "High contention detected when grabbing RDMA connection to memory node %s.",
                        target.ToString().ToCStr());
                usleep(1);
            }
            ++num_iterations;

            if ((uint32_t)(ctx.connections[idx].num_pending_requests.load(std::memory_order_acquire)) >=
                MAX_SEND_WR[COMPUTE_NODE_IDX]) {
                idx = (idx + 1) % MAX_CONN_PER_NODE;
                DIVFTREE_YIELD();
                continue;
            }

            uint32_t num_pending = ctx.connections[idx].num_pending_requests.fetch_add(num_clusters);
            if (num_pending >= MAX_SEND_WR[COMPUTE_NODE_IDX]) {
                ctx.connections[idx].num_pending_requests.fetch_sub(num_clusters);
                idx = (idx + 1) % MAX_CONN_PER_NODE;
                DIVFTREE_YIELD();
                continue;
            } else if (num_pending + num_clusters > MAX_SEND_WR[COMPUTE_NODE_IDX]) {
                uint32_t to_free = (num_pending + num_clusters) - MAX_SEND_WR[COMPUTE_NODE_IDX];
                ctx.connections[idx].num_pending_requests.fetch_sub(to_free);
                num_resources_acquired = static_cast<uint8_t>(num_clusters - to_free);
            } else {
                num_resources_acquired = static_cast<uint8_t>(num_clusters);
            }

            break;
        }

        return idx;
    }

    void DisconnectFromComputeNode(NodeID compute_node_id) {
        FatalAssert(selfInfo.node_id.IsMemoryNode(), LOG_TAG_RDMA,
                    "Only memory nodes can disconnect from compute nodes.");
        FatalAssert(compute_node_id.IsComputeNode(), LOG_TAG_RDMA,
                    "compute_node_id must be a compute node.");
        auto it = compute_nodes.find(compute_node_id);
        FatalAssert(it != compute_nodes.end(), LOG_TAG_RDMA,
                    "Connection context for compute node %s not found.",
                    compute_node_id.ToString().ToCStr());
        ConnectionContext& ctx = it->second;
        for (uint8_t conn_id = 0; conn_id < MAX_CONN_PER_NODE; ++conn_id) {
            FatalAssert(ctx.connections[conn_id].qp != nullptr, LOG_TAG_RDMA,
                        "QP for connection %hhu is nullptr.", conn_id);
            ModifyQPStateToError(ctx.connections[conn_id].qp);
            ibv_destroy_qp(ctx.connections[conn_id].qp);
            ctx.connections[conn_id].qp = nullptr;
        }
        FatalAssert(ctx.socket != -1, LOG_TAG_RDMA,
                    "TCP connection to compute node %s is already closed.",
                    compute_node_id.ToString().ToCStr());
        close(ctx.socket);
        ctx.socket = -1;
    }

    RetStatus RDMAReadInternal(NodeID target_node, uint8_t connection_idx, TaskID id,
                               void** local_buffers, uintptr_t* remote_addresses,
                               uint32_t* sizes, size_t num_clusters) {
        FatalAssert(instance == this, LOG_TAG_RDMA,
                "RDMA_Manager instance mismatch in RDMAWrite");
        FatalAssert(local_buffers != nullptr, LOG_TAG_RDMA,
                    "rdma_buffers is null in RDMAWrite");
        FatalAssert(num_clusters > 0, LOG_TAG_RDMA,
                    "num_buffers is zero in RDMAWrite");
        FatalAssert(target_node.IsMemoryNode(), LOG_TAG_RDMA,
                    "RDMARead target_node must be a memory node");
        FatalAssert(selfInfo.node_id.IsComputeNode(), LOG_TAG_RDMA,
                    "RDMARead can only be called from compute nodes");
        FatalAssert(num_clusters <= MAX_SEND_WR[COMPUTE_NODE_IDX], LOG_TAG_RDMA,
                    "num_buffers exceeds MAX_SEND_WR (%u) in RDMARead", MAX_SEND_WR[COMPUTE_NODE_IDX]);
        CHECK_NOT_NULLPTR(remote_addresses, LOG_TAG_RDMA);
        CHECK_NOT_NULLPTR(sizes, LOG_TAG_RDMA);
        CHECK_NOT_NULLPTR(mr, LOG_TAG_RDMA);
        auto it = memory_nodes.find(target_node);
        FatalAssert(it != memory_nodes.end(), LOG_TAG_RDMA,
                    "Connection context for target memory node %s not found.",
                    target_node.ToString().ToCStr());
        ConnectionContext& conn_ctx = it->second;
        FatalAssert(connection_idx < MAX_CONN_PER_NODE, LOG_TAG_RDMA,
                    "Invalid connection_idx %hhu in RDMARead", connection_idx);

        RetStatus rs = RetStatus::Success();

        struct ibv_send_wr* wr_list = new ibv_send_wr[num_clusters];
        struct ibv_sge* sge_list = new ibv_sge[num_clusters];
        struct ibv_send_wr* bad_wr = nullptr;
        SANITY_CHECK(
            std::map<std::pair<void*, uint32_t>, size_t> local_buffers_map;
            std::map<std::pair<void*, uint32_t>, size_t> remote_buffers_map;
        );

        for (size_t i = 0; i < num_clusters; ++i) {
            FatalAssert(local_buffers[i] != nullptr, LOG_TAG_RDMA,
                        "local_buffer[%zu] is null in RDMARead", i);
            FatalAssert(sizes[i] > 0, LOG_TAG_RDMA,
                        "sizes[%zu] is zero in RDMARead", i);
            FatalAssert(local_buffers[i] >= mr->addr &&
                        (reinterpret_cast<uintptr_t>(local_buffers[i]) + sizes[i]) <=
                        (reinterpret_cast<uintptr_t>(mr->addr) + mr->length),
                        LOG_TAG_RDMA,
                        "local_buffer[%zu] is out of registered memory region in RDMARead", i);
            FatalAssert(remote_addresses[i] >= conn_ctx.remote_region_addr &&
                        (remote_addresses[i] + sizes[i]) <=
                        (conn_ctx.remote_region_addr + conn_ctx.remote_region_size),
                        LOG_TAG_RDMA,
                        "remote_address[%zu] is out of remote registered memory region in RDMARead", i);
            SANITY_CHECK(
                auto last = local_buffers_map.upper_bound(std::make_pair(local_buffers[i], sizes[i]));
                for (auto it = local_buffers_map.begin(); it != last; ++it) {
                    FatalAssert((it->first.first < local_buffers[i]) && (it->first.first + it->first.second <= local_buffers[i]),
                                LOG_TAG_MEMORY, "Memory corruption detected before allocated slot");
                }
                for (auto it = last; it != local_buffers_map.end(); ++it) {
                    FatalAssert((it->first.first > local_buffers[i]) && (local_buffers[i] + sizes[i] <= it->first.first),
                                LOG_TAG_MEMORY, "Memory corruption detected after allocated slot");
                }
                local_buffers_map[std::make_pair(local_buffers[i], sizes[i])] = i;

                auto last_r = remote_buffers_map.upper_bound(std::make_pair((void*)remote_addresses[i], sizes[i]));
                for (auto it = remote_buffers_map.begin(); it != last_r; ++it) {
                    FatalAssert((it->first.first < (void*)remote_addresses[i]) && (it->first.first + it->first.second <= (void*)remote_addresses[i]),
                                LOG_TAG_MEMORY, "Memory corruption detected before allocated slot");
                }
                for (auto it = last_r; it != remote_buffers_map.end(); ++it) {
                    FatalAssert((it->first.first > (void*)remote_addresses[i]) && ((void*)remote_addresses[i] + sizes[i] <= it->first.first),
                                LOG_TAG_MEMORY, "Memory corruption detected after allocated slot");
                }
                remote_buffers_map[std::make_pair((void*)remote_addresses[i], sizes[i])] = i;
            );
            sge_list[i].addr = reinterpret_cast<uintptr_t>(local_buffers[i]);
            sge_list[i].length = sizes[i];
            sge_list[i].lkey = mr->lkey;

            memset(&wr_list[i], 0, sizeof(wr_list[i]));
            wr_list[i].wr_id = id.raw;
            wr_list[i].sg_list = sge_list;
            wr_list[i].num_sge = 1;
            wr_list[i].opcode = IBV_WR_RDMA_READ;
            wr_list[i].wr.rdma.remote_addr = remote_addresses[i];
            wr_list[i].wr.rdma.rkey = conn_ctx.remote_region_rkey;
            if (i < num_clusters - 1) {
                wr_list[i].next = &wr_list[i + 1];
                wr_list[i].send_flags = 0;
            } else {
                wr_list[i].next = nullptr;
                wr_list[i].send_flags = IBV_SEND_SIGNALED;
            }
        }

        int ret = ibv_post_send(conn_ctx.connections[connection_idx].qp, &wr_list[0], &bad_wr);
        if (ret != 0) {
            String error_msg = String("Failed to post RDMA Read send work request. ret=(%d)%s errno=(%d)%s",
                                        ret, strerror(ret), errno, strerror(errno));
            FatalAssert(false, LOG_TAG_RDMA, "%s", error_msg.ToCStr());
            rs = RetStatus::Fail(error_msg.ToCStr());
        }

        delete[] wr_list;
        delete[] sge_list;
        return rs;
    }


    RetStatus RDMASGReadInternal(NodeID target_node, uint8_t connection_idx, TaskID id,
                                 void*** local_buffers, uintptr_t* remote_addresses,
                                 uint32_t** sizes, uint32_t* num_sge, uint32_t num_clusters) {
        FatalAssert(instance == this, LOG_TAG_RDMA,
                "RDMA_Manager instance mismatch in RDMAWrite");
        FatalAssert(local_buffers != nullptr, LOG_TAG_RDMA,
                    "rdma_buffers is null in RDMAWrite");
        FatalAssert(num_clusters > 0, LOG_TAG_RDMA,
                    "num_buffers is zero in RDMAWrite");
        FatalAssert(target_node.IsMemoryNode(), LOG_TAG_RDMA,
                    "RDMARead target_node must be a memory node");
        FatalAssert(selfInfo.node_id.IsComputeNode(), LOG_TAG_RDMA,
                    "RDMARead can only be called from compute nodes");
        FatalAssert(num_clusters <= MAX_SEND_WR[COMPUTE_NODE_IDX], LOG_TAG_RDMA,
                    "num_buffers exceeds MAX_SEND_WR (%u) in RDMARead", MAX_SEND_WR[COMPUTE_NODE_IDX]);
        CHECK_NOT_NULLPTR(mr, LOG_TAG_RDMA);
        CHECK_NOT_NULLPTR(num_sge, LOG_TAG_RDMA);
        CHECK_NOT_NULLPTR(remote_addresses, LOG_TAG_RDMA);
        CHECK_NOT_NULLPTR(sizes, LOG_TAG_RDMA);
        auto it = memory_nodes.find(target_node);
        FatalAssert(it != memory_nodes.end(), LOG_TAG_RDMA,
                    "Connection context for target memory node %s not found.",
                    target_node.ToString().ToCStr());
        ConnectionContext& conn_ctx = it->second;
        FatalAssert(connection_idx < MAX_CONN_PER_NODE, LOG_TAG_RDMA,
                    "Invalid connection_idx %hhu in RDMARead", connection_idx);

        RetStatus rs = RetStatus::Success();

        struct ibv_send_wr* wr_list = new ibv_send_wr[num_clusters];
        struct ibv_sge** sge_list = new ibv_sge*[num_clusters];
        struct ibv_send_wr* bad_wr = nullptr;


        for (size_t i = 0; i < num_clusters; ++i) {
            FatalAssert(num_sge[i] > 0, LOG_TAG_RDMA,
                        "num_sge[%zu] is zero in RDMASGRead", i);
            FatalAssert(num_sge[i] <= MAX_SEND_SGE[COMPUTE_NODE_IDX], LOG_TAG_RDMA,
                        "num_sge[%zu] exceeds MAX_SGE_PER_WR (%u) in RDMASGRead", i, MAX_SEND_SGE[COMPUTE_NODE_IDX]);
            CHECK_NOT_NULLPTR(local_buffers[i], LOG_TAG_RDMA);
            CHECK_NOT_NULLPTR(sizes[i], LOG_TAG_RDMA);
            sge_list[i] = new ibv_sge[num_sge[i]];
            for (size_t j = 0; j < num_sge[i]; ++j) {
                FatalAssert(local_buffers[i][j] != nullptr, LOG_TAG_RDMA,
                            "local_buffer[%zu][%zu] is null in RDMASGRead", i, j);
                FatalAssert(sizes[i][j] > 0, LOG_TAG_RDMA,
                            "sizes[%zu][%zu] is zero in RDMASGRead", i, j);
                FatalAssert(local_buffers[i][j] >= mr->addr &&
                            (reinterpret_cast<uintptr_t>(local_buffers[i][j]) + sizes[i][j]) <=
                            (reinterpret_cast<uintptr_t>(mr->addr) + mr->length),
                            LOG_TAG_RDMA,
                            "local_buffer[%zu][%zu] is out of registered memory region in RDMASGRead", i, j);
                FatalAssert(remote_addresses[i] >= conn_ctx.remote_region_addr &&
                            (remote_addresses[i] + sizes[i][j]) <=
                            (conn_ctx.remote_region_addr + conn_ctx.remote_region_size),
                            LOG_TAG_RDMA,
                            "remote_address[%zu] is out of remote registered memory region in RDMASGRead", i);

                sge_list[i][j].addr = reinterpret_cast<uintptr_t>(local_buffers[i][j]);
                sge_list[i][j].length = sizes[i][j];
                sge_list[i][j].lkey = mr->lkey;
            }

            memset(&wr_list[i], 0, sizeof(wr_list[i]));
            wr_list[i].wr_id = id.raw;
            wr_list[i].sg_list = sge_list[i];
            wr_list[i].num_sge = num_sge[i];
            wr_list[i].opcode = IBV_WR_RDMA_READ;
            wr_list[i].wr.rdma.remote_addr = remote_addresses[i];
            wr_list[i].wr.rdma.rkey = conn_ctx.remote_region_rkey;
            if (i < num_clusters - 1) {
                wr_list[i].next = &wr_list[i + 1];
                wr_list[i].send_flags = 0;
            } else {
                wr_list[i].next = nullptr;
                wr_list[i].send_flags = IBV_SEND_SIGNALED;
            }
        }

        int ret = ibv_post_send(conn_ctx.connections[connection_idx].qp, &wr_list[0], &bad_wr);
        if (ret != 0) {
            String error_msg = String("Failed to post RDMA Read send work request. ret=(%d)%s errno=(%d)%s",
                                        ret, strerror(ret), errno, strerror(errno));
            FatalAssert(false, LOG_TAG_RDMA, "%s", error_msg.ToCStr());
            rs = RetStatus::Fail(error_msg.ToCStr());
        }

        delete[] wr_list;
        for (size_t i = 0; i < num_clusters; ++i) {
            delete[] sge_list[i];
        }
        delete[] sge_list;
        return rs;
    }

    inline static RDMA_Manager* instance;
    const NodeInfo selfInfo;
    const uint8_t _rdma_port;
    const int _gid_index;
    std::unordered_map<NodeID, ConnectionContext, NodeIDHash> compute_nodes;
    std::unordered_map<NodeID, ConnectionContext, NodeIDHash> memory_nodes;
    ConcurrentMultiMap<TaskID, VectorID, TaskIDHash> pending_tasks;
    struct ibv_wc* poll_list = nullptr;

    struct ibv_context* ib_ctx = nullptr;
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;
    union ibv_gid dev_gid;
    struct ibv_pd* pd = nullptr;
    struct ibv_cq* cq = nullptr;
    struct ibv_mr* mr = nullptr;
    int server_socket = -1;
    std::atomic<bool> ready = false;
    std::atomic<size_t> num_connected_nodes = 0;
};


};

#endif