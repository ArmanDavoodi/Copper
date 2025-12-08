#ifndef RDMA_MANAGER_H_
#define RDMA_MANAGER_H_

#include "common.h"
#include "debug.h"

#include <cstring>
#include <vector>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

// #define MEMORY_NODE

#define MEMORY_NODE_ID ((uint8_t)0)

namespace divftree {

enum ConnectionType : uint8_t {
    CN_CLUSTER_READ = 0,
    CN_CLUSTER_WRITE,
    CN_COMM,
    MN_COMM,
    MN_URGENT,
    CONNECTION_TYPE_COUNT
};

String ToString(ConnectionType type) {
    switch (type) {
        case CN_CLUSTER_READ:
            return String("CN_CLUSTER_READ");
        case CN_CLUSTER_WRITE:
            return String("CN_CLUSTER_WRITE");
        case CN_COMM:
            return String("CN_COMM");
        case MN_COMM:
            return String("MN_COMM");
        case MN_URGENT:
            return String("MN_URGENT");
        default:
            return String("UNKNOWN_CONNECTION_TYPE");
    }
}

constexpr size_t MAX_NUM_CN = 2;
constexpr size_t MAX_NUM_MN = 1;

constexpr size_t NUM_CONNECTIONS[CONNECTION_TYPE_COUNT] = {
    4,  /* CN_CLUSTER_READ */
    2,  /* CN_CLUSTER_WRITE */
    2,  /* CN_COMM */
    2,  /* MN_COMM */
    2   /* MN_URGENT */
};

constexpr size_t NUM_COMM_BUFFERS_PER_CONNECTION = 2;

constexpr size_t COMM_BUFFER_SIZE = 4096;
constexpr size_t URGENT_BUFFER_SIZE = 4096;
constexpr size_t URGENT_MESSAGE_SIZE = 64; /* todo: needs to be set to max inline data or max size of urgent message */

static_assert(CACHE_LINE_SIZE <= COMM_BUFFER_SIZE, "COMM_BUFFER_SIZE must be at least CACHE_LINE_SIZE");
static_assert(CACHE_LINE_SIZE <= URGENT_BUFFER_SIZE, "URGENT_BUFFER_SIZE must be at least CACHE_LINE_SIZE");
static_assert(CACHE_LINE_SIZE <= URGENT_MESSAGE_SIZE, "URGENT_MESSAGE_SIZE must be at least CACHE_LINE_SIZE");
static_assert(COMM_BUFFER_SIZE % CACHE_LINE_SIZE == 0, "COMM_BUFFER_SIZE must be multiple of CACHE_LINE_SIZE");
static_assert(URGENT_BUFFER_SIZE % CACHE_LINE_SIZE == 0, "URGENT_BUFFER_SIZE must be multiple of CACHE_LINE_SIZE");
static_assert(URGENT_MESSAGE_SIZE % CACHE_LINE_SIZE == 0, "URGENT_MESSAGE_SIZE must be multiple of CACHE_LINE_SIZE");
static_assert(URGENT_MESSAGE_SIZE * 2 <= URGENT_BUFFER_SIZE,
              "URGENT_MESSAGE_SIZE * 2 must be less than or equal to URGENT_BUFFER_SIZE");

union ConnTaskId {
    struct {
        uint64_t task_id : 48;
        uint64_t connection_id : 8;
        uint64_t buffer_id : 8;
    };

    uint64_t _raw;

    ConnTaskId() : task_id{0}, connection_id{0}, buffer_id{0} {}
    ConnTaskId(uint64_t raw) : _raw{raw} {}
    ConnTaskId(uint64_t t_id, uint8_t c_id, uint8_t b_id) :
        task_id{t_id}, connection_id{c_id}, buffer_id{b_id} {}
    ConnTaskId(const ConnTaskId& other) : _raw{other._raw} {}

    ConnTaskId& operator=(uint64_t raw) {
        _raw = raw;
        return *this;
    }

    ConnTaskId& operator=(const ConnTaskId& other) {
        _raw = other._raw;
        return *this;
    }

    inline bool operator==(uint64_t raw) const {
        return _raw == raw;
    }

    inline bool operator!=(uint64_t raw) const {
        return _raw != raw;
    }

    inline bool operator==(const ConnTaskId& other) const {
        return _raw == other._raw;
    }

    inline bool operator!=(const ConnTaskId& other) const {
        return _raw != other._raw;
    }
};

struct ConnTaskIdHash {
    size_t operator()(const ConnTaskId& p) const {
        return splitmix64(p._raw);
    }
};

enum CommBufferState : uint8_t {
    BUFFER_STATE_READY = 0,
    BUFFER_STATE_IN_USE
};

union UrgentMessageMeta {
    struct Detail {
        uint8_t length : 6; /* length in number of bytes */ /* max length is 64 bytes */
        uint8_t seen : 1;
        uint8_t valid : 1;

        Detail() : length{0}, seen{0}, valid{0} {}
        Detail(uint8_t len, uint8_t s, uint8_t v) : length{len}, seen{s}, valid{v} {}
    };

    Detail detail;
    std::atomic<Detail> atomic_detail;

    UrgentMessageMeta() : detail() {}
};

struct UrgentMessageData {
    char data[URGENT_MESSAGE_SIZE - sizeof(UrgentMessageMeta)] = {0};

    UrgentMessageData() = default;
    UrgentMessageData(const UrgentMessageData& other) {
        memcpy(data, other.data, sizeof(data));
    }
};

struct UrgentMessage {
    UrgentMessageData data;
    UrgentMessageMeta meta;

    UrgentMessage() = default;
    ~UrgentMessage() = default;

    void Init() {
        meta.detail.length = 0;
        meta.detail.seen = 0;
        meta.detail.valid = 1;
    }

    void Clear(bool set_zero = true) {
        if (set_zero) {
            memset(data.data, 0, sizeof(data.data));
        }
        meta.detail.length = 0;
    }

    size_t GetLength() const {
        return static_cast<size_t>(meta.detail.length);
    }

    void SetLength(size_t len) {
        FatalAssert(len <= (URGENT_MESSAGE_SIZE - sizeof(UrgentMessageMeta)),
                    LOG_TAG_RDMA,
                    "Setting urgent message length to %zu exceeds maximum size %zu",
                    len, URGENT_MESSAGE_SIZE - sizeof(UrgentMessageMeta));
        meta.detail.length = static_cast<uint8_t>(len);
    }

    // void AppendData(const void* src, size_t len) {
    //     FatalAssert((GetLength() + len) <= (URGENT_MESSAGE_SIZE - sizeof(UrgentMessageMeta)),
    //                 LOG_TAG_RDMA,
    //                 "Appending data of length %zu exceeds urgent message maximum size %zu",
    //                 len, URGENT_MESSAGE_SIZE - sizeof(UrgentMessageMeta));
    //     memcpy(data.data + GetLength(), src, len);
    //     meta.detail.length += static_cast<uint8_t>(len);
    // }

    // void WriteData(size_t offset, const void* src, size_t len) {
    //     FatalAssert((offset + len) <= (URGENT_MESSAGE_SIZE - sizeof(UrgentMessageMeta)),
    //                 LOG_TAG_RDMA,
    //                 "Writing data of length %zu at offset %zu exceeds urgent message maximum size %zu",
    //                 len, offset, URGENT_MESSAGE_SIZE - sizeof(UrgentMessageMeta));
    //     FatalAssert(offset <= GetLength(), LOG_TAG_RDMA,
    //                 "Writing data at offset %zu which is beyond current message length %zu",
    //                 offset, GetLength());
    //     memcpy(data.data + offset, src, len);
    //     meta.detail.length = static_cast<uint8_t>(std::max(static_cast<size_t>(meta.detail.length), offset + len));
    // }

    bool IsUnseenValid() const {
        UrgentMessageMeta::Detail m = meta.atomic_detail.load(std::memory_order_acquire);
        return (m.valid == 1) && (m.seen == 0);
    }

    void SetSeen() {
        UrgentMessageMeta::Detail m = meta.atomic_detail.load(std::memory_order_relaxed);
        FatalAssert(m.valid == 1, LOG_TAG_RDMA,
                    "Cannot set seen flag on an invalid urgent message");
        FatalAssert(m.seen == 0, LOG_TAG_RDMA,
                    "Urgent message is already marked as seen");
        m.seen = 1;
        meta.atomic_detail.store(m, std::memory_order_relaxed);
    }
};

/* In the compute node do I need these two?! I guess not so I only need a buffer for the requests and an index */
struct alignas(CACHE_LINE_SIZE) UrgentBuffer {
    static constexpr size_t MAX_URGENT_MESSAGES = URGENT_BUFFER_SIZE / URGENT_MESSAGE_SIZE;
    UrgentMessage messages[MAX_URGENT_MESSAGES];
};

struct CommBufferMeta {
    std::atomic<uint64_t> num_requests = 0;
    std::atomic<CommBufferState> state = BUFFER_STATE_READY; /* this should be the last byte! */ /* todo alignas(64)? */
};

constexpr size_t COMM_BUFFER_DATA_SIZE = COMM_BUFFER_SIZE - sizeof(CommBufferMeta);
struct alignas(CACHE_LINE_SIZE) CommBuffer {
    /*
     * For each request we first write the type and then the size(if needed) and then a flag indicating
     * whether it is an urgent request or not and then the data.
     *
     * The reciever side will first sort the requests based on urgency, type and cluster id? and then if the sort was
     * not inplace it can use RDMA write to update the viewed_id of the sender side letting it know that it can reuse
     * the buffer.
     *
     * if we use cqes for writes, we can also start using the local buffer but we cannot do an rdma send until the
     * viewed_id is updated.
     */
    char data[COMM_BUFFER_DATA_SIZE] = {0};
    CommBufferMeta meta;
};

struct CommBufferInfo {
    SXLock buffer_lock;
    std::atomic<size_t> num_readers;
    std::atomic<size_t> length;

    uint32_t lkey;
    CommBuffer* buffer = nullptr;

    uint32_t rkey;
    uintptr_t remote_addr; /* if at the reciver side remote addr is the addr of the buffer state */
};

struct ConnectionContext {
    // uint8_t target_node_id; /* 0 is the memory node */
    ConnectionType type;
    uint32_t connection_id; /* todo: do I need this? */
    uint32_t local_psn;
    uint32_t remote_psn;
    uint32_t remote_qp_num;
    std::atomic<uint64_t> last_task_id;
    std::atomic<uint16_t> num_pending_requests;
    struct ibv_qp *qp;
    struct ibv_cq *cq; /* todo: use a single cq per node or total? */

    void Init(ConnectionType t, uint32_t id, struct ibv_qp* q, struct ibv_cq* c) {
        FatalAssert(t < CONNECTION_TYPE_COUNT, LOG_TAG_RDMA,
                    "Invalid ConnectionType %hhu for ConnectionContext", t);
        FatalAssert(q != nullptr, LOG_TAG_RDMA,
                    "qp cannot be nullptr for ConnectionContext");
        FatalAssert(c != nullptr, LOG_TAG_RDMA,
                    "cq cannot be nullptr for ConnectionContext");
        FatalAssert(type == CONNECTION_TYPE_COUNT, LOG_TAG_RDMA,
                    "ConnectionContext is already initialized");
        FatalAssert(connection_id == 0, LOG_TAG_RDMA,
                    "ConnectionContext is already initialized");
        FatalAssert(last_task_id.load(std::memory_order_relaxed) == 0, LOG_TAG_RDMA,
                    "ConnectionContext is already initialized");
        FatalAssert(num_pending_requests.load(std::memory_order_relaxed) == 0, LOG_TAG_RDMA,
                    "ConnectionContext is already initialized");
        FatalAssert(qp == nullptr, LOG_TAG_RDMA,
                    "ConnectionContext is already initialized");
        FatalAssert(cq == nullptr, LOG_TAG_RDMA,
                    "ConnectionContext is already initialized");
        type = t;
        connection_id = id;
        qp = q;
        cq = c;
    }

    ConnectionContext() : type{CONNECTION_TYPE_COUNT}, connection_id{0}, local_psn{0}, remote_psn{0}, remote_qp_num{0},
        last_task_id{0}, num_pending_requests{0}, qp{nullptr}, cq{nullptr} {}
};

struct CommConnectionContext {
    ConnectionContext ctx;
    std::atomic<uint8_t> curr_buffer_idx;
    CommBufferInfo buffer_info[NUM_COMM_BUFFERS_PER_CONNECTION];

    CommConnectionContext() = default;
    void Init(ConnectionType t, uint32_t id, struct ibv_qp* q, struct ibv_cq* c) {
        ctx.Init(t, id, q, c);
        curr_buffer_idx.store(0, std::memory_order_relaxed);
    }

    void SetSendBuffers(CommBuffer* bufs, uint32_t lkey) {
        FatalAssert(bufs != nullptr, LOG_TAG_RDMA,
                    "bufs cannot be nullptr for CommConnectionContext");
        for (size_t i = 0; i < NUM_COMM_BUFFERS_PER_CONNECTION; ++i) {
            FatalAssert(ALIGNED(&bufs[i], CACHE_LINE_SIZE), LOG_TAG_RDMA,
                    "bufs must be aligned to CACHE_LINE_SIZE for CommConnectionContext");
            buffer_info[i].buffer = new (&bufs[i]) CommBuffer();
            buffer_info[i].lkey = lkey;
        }
    }

    void SetReceiveBuffers(uintptr_t bufs, uint32_t rkey, bool self_comm_path) {
        FatalAssert(bufs != 0, LOG_TAG_RDMA,
                    "bufs cannot be nullptr for CommConnectionContext");
        for (size_t i = 0; i < NUM_COMM_BUFFERS_PER_CONNECTION; ++i) {
            FatalAssert(ALIGNED((void*)(bufs + i * COMM_BUFFER_SIZE), CACHE_LINE_SIZE), LOG_TAG_RDMA,
                    "bufs must be aligned to CACHE_LINE_SIZE for CommConnectionContext");
            if (self_comm_path) {
                buffer_info[i].remote_addr = bufs + i * COMM_BUFFER_SIZE;
            } else {
                /* in this case, we are the reciever side so target recive buffer should be the sate so we can
                   signal that we have read the message */
                buffer_info[i].remote_addr = bufs + i * COMM_BUFFER_SIZE + COMM_BUFFER_DATA_SIZE;
            }
            buffer_info[i].rkey = rkey;
        }
    }
};

struct RemoteMemoryRegion {
    uintptr_t addr;
    size_t length;
    uint32_t rkey;

    RemoteMemoryRegion() = default;
    RemoteMemoryRegion(uintptr_t a, size_t len, uint32_t rk) :
        addr{a}, length{len}, rkey{rk} {}
};

struct RDMABuffer {
    void* local_addr;
    uintptr_t remote_addr;
    size_t length;
    uint8_t remote_node_id;
    uint8_t buffer_idx = 0;
};

struct ClusterReadPathInfo {
    std::atomic<uint8_t> curr_conn_idx = 0;
    ConnectionContext connection[NUM_CONNECTIONS[CN_CLUSTER_READ]];
};

struct ClusterWritePathInfo {
    std::atomic<uint8_t> curr_conn_idx = 0;
    ConnectionContext connection[NUM_CONNECTIONS[CN_CLUSTER_WRITE]];
};

struct CNCommPathInfo {
    std::atomic<uint8_t> curr_conn_idx = 0;
    CommConnectionContext connection[NUM_CONNECTIONS[CN_COMM]];
};

struct MNCommPathInfo {
    std::atomic<uint8_t> curr_conn_idx = 0;
    CommConnectionContext connection[NUM_CONNECTIONS[MN_COMM]];
};

struct UrgentPathInfo {
    std::atomic<uint8_t> curr_conn_idx = 0;
    ConnectionContext connection[NUM_CONNECTIONS[MN_URGENT]];
    uintptr_t remote_buffer = 0;
    uint32_t rkey = 0;
#ifdef MEMORY_NODE
    std::atomic<uint64_t>* read_off = nullptr;
    std::atomic<uint64_t> write_off = 0;
#else
    UrgentBuffer* buffer = nullptr;
    size_t read_off = 0;
#endif
};

struct NodeInfo {
    const uint8_t node_id;
    const uint32_t ip_addr;
    const uint16_t port;

    union ibv_gid remote_gid;
    int socket;

    ClusterReadPathInfo cn_cluster_read_connections;
    ClusterWritePathInfo cn_cluster_write_connections;
    CNCommPathInfo cn_comm_connections;
    MNCommPathInfo mn_comm_connections;
    UrgentPathInfo mn_urgent_connections;
    std::map<uintptr_t, RemoteMemoryRegion> remoteMemoryRegions;
    SXSpinLock comm_path_lock;

    NodeInfo(uint8_t id, uint32_t ip, uint16_t p);

    uint32_t GetKey(uintptr_t addr, size_t length) const;
    void AddRegion(uintptr_t addr, size_t length, uint32_t rkey);
};

struct HandshakeConnectionInfo {
    uint32_t qp_num;
    uint32_t psn;
};
struct HandshakeBufferedConnectionInfo {
    HandshakeConnectionInfo conn_info;
    uintptr_t buffer_addr;
};

struct HandshakeInfo {
    union ibv_gid gid;
    uintptr_t buffer_addr;
    uint32_t buffer_rkey;
    size_t buffer_size;
    uintptr_t urgent_buffer_addr;
    HandshakeConnectionInfo cn_cluster_read_conns[NUM_CONNECTIONS[CN_CLUSTER_READ]];
    HandshakeConnectionInfo cn_cluster_write_conns[NUM_CONNECTIONS[CN_CLUSTER_WRITE]];
    HandshakeBufferedConnectionInfo cn_comm_conns[NUM_CONNECTIONS[CN_COMM]];
    HandshakeBufferedConnectionInfo mn_comm_conns[NUM_CONNECTIONS[MN_COMM]];
    HandshakeConnectionInfo urgent_conns[NUM_CONNECTIONS[MN_URGENT]];
};

struct BufferInfo {
    void* buffer;
    size_t max_length;
    union {
        size_t length;
        size_t num_requests;
    };

    uint8_t target_node_id;
    uint8_t conn_id;
    uint8_t buffer_idx;

    void SetLength(size_t len) {
        FatalAssert(len <= max_length, LOG_TAG_RDMA,
                    "Setting buffer length exceeds max_length. set length=%zu, max_length=%zu",
                    len, max_length);
        length = len;
    }

    // void Append(void* data, size_t len) {
    //     FatalAssert(len + length <= max_length, LOG_TAG_RDMA,
    //                 "Appending data exceeds buffer max_length. current length=%zu, append length=%zu, max_length=%zu",
    //                 length, len, max_length);
    //     FatalAssert(buffer != nullptr, LOG_TAG_RDMA,
    //                 "buffer is nullptr in BufferInfo");
    //     FatalAssert(data != nullptr, LOG_TAG_RDMA,
    //                 "data to append is nullptr in BufferInfo");
    //     memcpy(reinterpret_cast<uint8_t*>(buffer) + length, data, len);
    //     length += len;
    // }

    // void Write(void* data, size_t len, size_t offset) {
    //     FatalAssert(offset <= length, LOG_TAG_RDMA,
    //                 "Offset exceeds current length in BufferInfo. current length=%zu, offset=%zu",
    //                 length, offset);
    //     FatalAssert(len + offset <= max_length, LOG_TAG_RDMA,
    //                 "Writing data exceeds buffer max_length. offset=%zu, write length=%zu, max_length=%zu",
    //                 offset, len, max_length);
    //     FatalAssert(buffer != nullptr, LOG_TAG_RDMA,
    //                 "buffer is nullptr in BufferInfo");
    //     FatalAssert(data != nullptr, LOG_TAG_RDMA,
    //                 "data to write is nullptr in BufferInfo");
    //     memcpy(reinterpret_cast<uint8_t*>(buffer) + offset, data, len);
    //     if (offset + len > length) {
    //         length = offset + len;
    //     }
    // }

    // size_t RemainingSpace() const {
    //     return max_length - length;
    // }
};

class RDMA_Manager {
protected:
    inline static RDMA_Manager* rdmaManagerInstance = nullptr;

#ifdef MEMORY_NODE
    /* todo: maybe I need to pass this as a runtime arg?! needs tuning */
    static constexpr uint32_t MAX_SEND_WR[CONNECTION_TYPE_COUNT] = {
        0,      /* CN_CLUSTER_READ */
        0,      /* CN_CLUSTER_WRITE */
        16,     /* CN_COMM */
        16,     /* MN_COMM */
        16      /* MN_URGENT */
    };

    static constexpr int MAX_CQE[CONNECTION_TYPE_COUNT] = {
        1,                                            /* CN_CLUSTER_READ */ /* impossible to set to 0 */
        1,                                            /* CN_CLUSTER_WRITE */ /* impossible to set to 0 */
        /* I just generate a CQE when my send counter reaches MAX_SEND_WR for sending response and wait for its
           completion */
        MAX_NUM_CN * NUM_CONNECTIONS[CN_COMM],        /* CN_COMM */
        MAX_NUM_CN * NUM_CONNECTIONS[MN_COMM],        /* MN_COMM */
        MAX_NUM_CN * NUM_CONNECTIONS[MN_URGENT]       /* MN_URGENT */
    };

    /* todo: we may need to increase this later */
    static constexpr uint32_t MAX_SEND_SGE[CONNECTION_TYPE_COUNT] = {
        1,   /* CN_CLUSTER_READ */ /* todo: need to tune -> maybe should get it as conf -> equal to span */
        1,   /* CN_CLUSTER_WRITE */ /* currently we only use 2 for num clusters during split */
        1,   /* CN_COMM */
        1,   /* MN_COMM */
        1    /* MN_URGENT */
    };

    /* todo: needs tuning */
    static constexpr uint32_t MAX_INLINE_DATA[CONNECTION_TYPE_COUNT] = {
        0,                            /* CN_CLUSTER_READ */
        0,                            /* CN_CLUSTER_WRITE */
        sizeof(CommBufferState),      /* CN_COMM */
        0,                            /* MN_COMM */
        URGENT_MESSAGE_SIZE            /* MN_URGENT */
    };
#else
    /* todo: maybe I need to pass this as a runtime arg?! needs tuning */
    static constexpr uint32_t MAX_SEND_WR[CONNECTION_TYPE_COUNT] = {
        1024 / NUM_CONNECTIONS[CN_CLUSTER_READ],   /* CN_CLUSTER_READ */
        64,                                        /* CN_CLUSTER_WRITE */
        16,                                        /* CN_COMM */
        16,                                        /* MN_COMM */
        16                                         /* MN_URGENT */
    };

    static constexpr int MAX_CQE[CONNECTION_TYPE_COUNT] = {
        /* one per request as I need the cqe for reads */
        MAX_SEND_WR[0] * NUM_CONNECTIONS[CN_CLUSTER_READ],                       /* CN_CLUSTER_READ */
        /* I just generate a CQE when my send counter reaches MAX_SEND_WR for sending response and wait for its
           completion */
        MAX_NUM_MN * NUM_CONNECTIONS[CN_CLUSTER_WRITE],                          /* CN_CLUSTER_WRITE */
        MAX_NUM_MN * NUM_CONNECTIONS[CN_COMM],                                   /* CN_COMM */
        MAX_NUM_MN * NUM_CONNECTIONS[MN_COMM],                                   /* MN_COMM */
        MAX_NUM_MN * NUM_CONNECTIONS[MN_URGENT]                                  /* MN_URGENT */
    };

    /* todo: we may need to increase this later */
    /* Note: sge requires the operation to have a single remote target memory */
    static constexpr uint32_t MAX_SEND_SGE[CONNECTION_TYPE_COUNT] = {
        1,  /* CN_CLUSTER_READ */ /* todo: need to tune -> maybe should get it as conf -> equal to span */
        1,   /* CN_CLUSTER_WRITE */ /* currently we only use 2 for num clusters during split */
        1,   /* CN_COMM */
        1,   /* MN_COMM */
        1    /* MN_URGENT */
    };

    /* todo: needs tuning */
    static constexpr uint32_t MAX_INLINE_DATA[CONNECTION_TYPE_COUNT] = {
        0,                            /* CN_CLUSTER_READ */
        0,                            /* CN_CLUSTER_WRITE */
        0,                            /* CN_COMM */
        sizeof(CommBufferState),      /* MN_COMM */
        sizeof(uint64_t)              /* MN_URGENT */ /* sizeof readoffset */
    };
#endif

    static constexpr uint32_t MAX_RD_ATOMIC[CONNECTION_TYPE_COUNT] = {
        MAX_SEND_WR[0],   /* CN_CLUSTER_READ */
        0,                /* CN_CLUSTER_WRITE */
        0,                /* CN_COMM */
        0,                /* MN_COMM */
        0                 /* MN_URGENT */
    };

    /* We are not using two-sided verbs */
    static constexpr uint32_t MAX_RECV_WR[CONNECTION_TYPE_COUNT] = {
        0,  /* CN_CLUSTER_READ */
        0,  /* CN_CLUSTER_WRITE */
        0,  /* CN_COMM */
        0,  /* MN_COMM */
        0   /* MN_URGENT */
    };

    /* We are not using two-sided verbs */
    static constexpr uint32_t MAX_RECV_SGE[CONNECTION_TYPE_COUNT] = {
        0,  /* CN_CLUSTER_READ */
        0,  /* CN_CLUSTER_WRITE */
        0,  /* CN_COMM */
        0,  /* MN_COMM */
        0   /* MN_URGENT */
    };

    static constexpr ibv_mtu DEFAULT_MTU[CONNECTION_TYPE_COUNT] = {
        IBV_MTU_4096, /* CN_CLUSTER_READ */
        IBV_MTU_4096, /* CN_CLUSTER_WRITE */
        IBV_MTU_4096, /* CN_COMM */
        IBV_MTU_4096, /* MN_COMM */
        IBV_MTU_256   /* MN_URGENT */
    };

    const uint8_t num_nodes;
    const uint8_t self_node_id;

    struct ibv_context *ib_ctx = nullptr;
    struct ibv_device_attr dev_attr;
    struct ibv_port_attr port_attr;
    struct ibv_pd *pd = nullptr;
    struct ibv_cq* cn_cluster_read_cq = nullptr;
    struct ibv_cq* cn_cluster_write_cq = nullptr;
    struct ibv_cq* cn_comm_cq = nullptr;
    struct ibv_cq* mn_comm_cq = nullptr;
    struct ibv_cq* mn_urgent_cq = nullptr;
    uint8_t port_num = 0;
    int gid_index = -1;
    union ibv_gid dev_gid;

    std::vector<NodeInfo> nodes;
    void* buffers = nullptr;
    uint32_t buffer_lkey;
    std::map<uintptr_t, struct ibv_mr*> localMemoryRegions;
#ifndef MEMORY_NODE
    SXSpinLock urgent_msg_lock;
#endif

    static RetStatus CreateQP(struct ibv_qp** qp, struct ibv_cq* cq, struct ibv_pd* pd, ConnectionType type);
    static RetStatus ModifyQPStateToReset(struct ibv_qp* qp);
    static RetStatus ModifyQPStateToInit(struct ibv_qp* qp, uint8_t port_num);
    static RetStatus ModifyQPStateToRTR(struct ibv_qp* qp, uint32_t dest_qp_num,
                                        const union ibv_gid& dest_gid,
                                        uint8_t port_num, uint32_t remote_psn, ConnectionType type);
    static RetStatus ModifyQPStateToRTS(struct ibv_qp* qp, uint32_t local_psn, ConnectionType type);
    static RetStatus ModifyQPStateToError(struct ibv_qp* qp);

    void FillHandshakeInfo(uint8_t node_id, HandshakeInfo& handshake_info);
    void ProcessHandshakeInfo(uint8_t node_id, const HandshakeInfo& handshake_info);

    template<typename PathInfo>
    RetStatus InitConnectionCtx(ConnectionType conn_type, struct ibv_cq* cq, PathInfo& path_info,
                                void*& next_buffer, uint32_t lkey);
    RetStatus EstablishTCPConnections();
    RetStatus Handshake();
    RetStatus EstablishRDMAConnections();

    RetStatus DisconnectTCPConnections();
    RetStatus DisconnectAllNodes();
    RetStatus DeregisterAllMemoryRegions();

    inline void Cleanup();

    /* todo: support multiple memory nodes */
    RDMA_Manager(uint8_t self_id, const std::vector<std::pair<const char*, uint16_t>>& node_addresses,
                 const char* target_rdma_device_name, uint8_t rdma_port, int gid_index);
    ~RDMA_Manager();

    RetStatus FlushCommBuffer(uint8_t target_node_id, uint8_t conn_id, uint8_t buffer_idx);

    uint32_t GetLocalKey(Address addr, size_t length) const;
    RetStatus RDMAWrite(RDMABuffer* rdma_buffers, size_t num_buffers, ConnectionContext& conn_ctx,
                        unsigned int send_flags);
    RetStatus RDMARead(RDMABuffer* rdma_buffers, size_t num_buffers, ConnectionContext& conn_ctx,
                       ConnTaskId& wait_id);

public:
    inline static RDMA_Manager* GetInstance();

    /* --------- start of not thread-safe region --------- */
    static RetStatus Initialize(uint8_t self_id, const std::vector<std::pair<const char*, uint16_t>>& node_addresses,
                                const char* target_rdma_device_name, uint8_t rdma_port, int gid_index);
    static void Destroy();

    inline uint8_t GetSelfNodeId() const {
        return self_node_id;
    }

    inline uint8_t GetNumNodes() const {
        return num_nodes;
    }

    /* Note: this function only registers the memory locally and will not send any notifications to other nodes */
    RetStatus RegisterRemoteMemory(uint8_t target_node_id, uintptr_t remote_addr, size_t length, uint32_t rkey);
    RetStatus RegisterMemory(Address addr, size_t length, uint32_t& lkey, uint32_t& rkey);
    RetStatus EstablishConnections();
    /* todo: this should be done at comm layer */
    /* registers memory locally and notifies the target node. if target node is self, it will be broadcasted to all */
    // RetStatus RegisterMemory(uint8_t target_node_id, Address addr, size_t length);
    /* todo: API for creating new connections and adding new nodes to the system */

    /* --------- end of not thread-safe region --------- */

    /* for urgent buffers we CAS the state for synchronization. for commbuffers we first check the state and if valid
        */
    /* --------------------- Sender Side Operations --------------------- */
    /* Comm Path */
    RetStatus GrabCommBuffer(uint8_t target_node_id, size_t len, BufferInfo& buffer);
    RetStatus ReleaseCommBuffer(BufferInfo buffer, bool flush = false);

#ifdef MEMORY_NODE
    RetStatus BroadCastCommRequest(void* data, size_t len, bool flush);
    /* Urgent Path */
    UrgentMessage* BuildUrgentMessage();
    void ReleaseUrgentMessage(UrgentMessage* msg);
    RetStatus SendUrgentMessage(UrgentMessage* msg, uint8_t target_node_id);
#else
    /* Cluster Write Path */
    RetStatus RDMAWrite(RDMABuffer* rdma_buffers, size_t num_buffers);

    /* Cluster Read Path */
    RetStatus RDMARead(RDMABuffer* rdma_buffers, size_t num_buffers, ConnTaskId& wait_id);
    RetStatus PollCompletion(ConnectionType conn_type, std::vector<ConnTaskId>& completed_task_ids);
#endif

    /* --------------------- Reciever Side Operations --------------------- */
    /* todo: change Reciever Side inputs later */

#ifndef MEMORY_NODE
    /* Recieve UrgentMessage */
    /* Note: this operation is not thread-safe! */
    /* todo: should we make this thread-safe? */
    RetStatus PollUrgentMessages(std::vector<std::pair<UrgentMessageData, uint8_t>>& messages);
#endif

    /* Recieve CommRequest */
    /* Note: these two operations are not thread-safe! */
    /* todo: should we make these two thread-safe? */
    RetStatus PollCommRequests(uint8_t node_id, std::vector<BufferInfo>& request_buffers);
    /* Note: these two operations are not thread-safe! */
    RetStatus ReleaseCommReciveBuffers(uint8_t node_id, std::vector<BufferInfo>& buffers);

TESTABLE;
};


};

#endif