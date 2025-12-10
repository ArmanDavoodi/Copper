#ifndef COMM_LAYER_H_
#define COMM_LAYER_H_

#include "common.h"
#include "debug.h"

#include "utils/rdma_manager.h"

namespace divftree {

constexpr uint8_t BROADCAST_NODE_ID = MEMORY_NODE_ID;

enum class MessageType : uint8_t {
    BASE_MESSAGE_START = 0,
        REGISTER_MEMORY = BASE_MESSAGE_START,
        SHUTDOWN_REQUEST,
        SHUTDOWN_RESPONSE, /* it may be accept or decline */
    BASE_MESSAGE_END = SHUTDOWN_RESPONSE,

    CN_MESSAGE_START,
        CN_TO_MN_REQUEST_ROOT = CN_MESSAGE_START,
        CN_TO_MN_INSERT_REQUEST,
        CN_TO_MN_DELETE_REQUEST,
        CN_TO_MN_MIGRATION_REQUEST,
        CN_TO_MN_MERGE_REQUEST,
        CN_TO_MN_UNPIN_NOTIFICATION,
        CN_TO_MN_CLUSTER_ADDRESS_REQUEST,
    CN_MESSAGE_END = CN_TO_MN_CLUSTER_ADDRESS_REQUEST,

    MN_MESSAGE_START,
        MN_RESPONSE_START = MN_MESSAGE_START,
            MN_TO_CN_RESPONSE_ROOT = MN_RESPONSE_START,
            MN_TO_CN_INSERT_FAILED_RESPONSE,
            MN_TO_CN_DELETE_FAILED_RESPONSE,
            MN_TO_CN_CLUSTER_ADDRESS_RESPONSE,
        MN_RESPONSE_END = MN_TO_CN_CLUSTER_ADDRESS_RESPONSE,

        MN_NOTIF_START,
            MN_TO_CN_VECTOR_INSERT_NOTIFICATION = MN_NOTIF_START,
            MN_TO_CN_CLUSTER_INSERT_NOTIFICATION,
            MN_TO_CN_COMPACTION_NOTIFICATION,
            MN_TO_CN_SPLIT_NOTIFICATION,
            MN_TO_CN_EXPANSION_NOTIFICATION,
            MN_TO_CN_DELETE_NOTIFICATION,
            MN_TO_CN_ROOT_PRUNE_NOTIFICATION,
            MN_TO_CN_MIGRATION_NOTIFICATION,
            MN_TO_CN_MERGE_NOTIFICATION,
        MN_NOTIF_END = MN_TO_CN_MERGE_NOTIFICATION,

        MN_REQUEST_START,
            MN_TO_CN_SPLIT_REQUEST = MN_REQUEST_START,
            MN_TO_CN_MERGE_CHECK_REQUEST,
            // MN_TO_CN_MIGRATION_CHECK_REQUEST,
        MN_REQUEST_END = MN_TO_CN_MERGE_CHECK_REQUEST,

        MN_URGENT_START,
            URGENT_MN_TO_CN_SPLIT_REQUEST = MN_URGENT_START,
        MN_URGENT_END = URGENT_MN_TO_CN_SPLIT_REQUEST,

    MN_MESSAGE_END = MN_URGENT_END,
    MESSAGE_TYPE_COUNT
};

struct  __attribute__((packed)) RegisterMemoryMessage {
    const MessageType type = MessageType::REGISTER_MEMORY;
    uintptr_t addr;
    size_t length;
    uint32_t rkey;

    constexpr static size_t Size() {
        return sizeof(RegisterMemoryMessage);
    }
};

struct  __attribute__((packed)) ShutdownRequestMessage {
    const MessageType type = MessageType::SHUTDOWN_REQUEST;
    /* todo: need more? */

    constexpr static size_t Size() {
        return sizeof(ShutdownRequestMessage);
    }
};

struct  __attribute__((packed)) ShutdownResponseMessage {
    const MessageType type = MessageType::SHUTDOWN_RESPONSE;
    bool accepted;
    /* todo: need more? */
    constexpr static size_t Size() {
        return sizeof(ShutdownResponseMessage);
    }
};

struct __attribute__((packed)) RequestRootMessage {
    const MessageType type = MessageType::CN_TO_MN_REQUEST_ROOT;
    constexpr static size_t Size() {
        return sizeof(RequestRootMessage);
    }
};

struct  __attribute__((packed)) InsertRequestMessage {
    const MessageType type = MessageType::CN_TO_MN_INSERT_REQUEST;
    VectorID target_leaf; /* should be leaf */
    Version target_version;
    VectorID vector; /* cannot be cluster */
    VTYPE data[]; /* todo: does it have paddings? */

    constexpr static size_t Size(uint16_t dim) {
        return sizeof(InsertRequestMessage) + sizeof(VTYPE) * dim;
    }
};

struct  __attribute__((packed)) DeleteRequestMessage {
    const MessageType type = MessageType::CN_TO_MN_DELETE_REQUEST;
    VectorID target_vector; /* cannot be cluster */
    bool need_response;

    constexpr static size_t Size() {
        return sizeof(DeleteRequestMessage);
    }
};

/* todo: currently this is basically a full message as it is more than 3KB. I should add statistics to check the send success rate */
/* todo: if most of the times we need to migrate more than half, we can use a black list instead of white list approach -> should not happen if the clustering algorithm is good though */
/* since this needs to lock a lot of clusters it is considered a general request */
struct  __attribute__((packed)) MigrationRequestMessage {
    const MessageType type = MessageType::CN_TO_MN_MIGRATION_REQUEST;
    VectorID first_cluster;
    VectorID second_cluster;
    Version first_version;
    Version second_version;
    uint64_t num_first_to_second;
    uint64_t num_second_to_first;
    ClusterSizeType offsets[]; /* first num_first_to_second are offsets in first cluster, and the rest are offsets in second cluster */

    constexpr static size_t Size(uint64_t num_first_to_second, uint64_t num_second_to_first) {
        return sizeof(MigrationRequestMessage) +
               sizeof(ClusterSizeType) * (num_first_to_second + num_second_to_first);
    }
};

/* although we can put this in the cluster for the cluster with lower id, it makes things more complicated
because src is locked in X mode and dest in S. As a result if srcId < destId and we lock src and then try to insert into
it too much in a way to cause a major update, we already have it locked in X mode and have to change the main code
to handle this case. todo: maybe do this later but now set it as a general task  */
struct  __attribute__((packed)) MergeRequestMessage {
    const MessageType type = MessageType::CN_TO_MN_MERGE_REQUEST;
    VectorID src_cluster;
    Version src_version;
    VectorID dest_cluster;
    Version dest_version;

    constexpr static size_t Size() {
        return sizeof(MergeRequestMessage);
    }
};

/* a general task as we do not need to lock the whole page we just lock the header I guess */
struct  __attribute__((packed)) UnpinNotificationMessage {
    const MessageType type = MessageType::CN_TO_MN_UNPIN_NOTIFICATION;
    VectorID old_root_id;
    Version old_root_version;

    constexpr static size_t Size() {
        return sizeof(UnpinNotificationMessage);
    }
};

/* a general task as we do not need to lock the whole page we just lock the header I guess */
struct  __attribute__((packed)) ClusterAddressRequestMessage {
    const MessageType type = MessageType::CN_TO_MN_CLUSTER_ADDRESS_REQUEST;
    VectorID cluster_id;
    Version cluster_version;

    constexpr static size_t Size() {
        return sizeof(ClusterAddressRequestMessage);
    }
};

struct __attribute__((packed)) ResponseRootMessage {
    const MessageType type = MessageType::MN_TO_CN_RESPONSE_ROOT;
    VectorID root_id;
    Version root_version;
    uintptr_t root_addr;

    constexpr static size_t Size() {
        return sizeof(ResponseRootMessage);
    }
};

/* a general task as we just need to repeat */
struct __attribute__((packed)) InsertFailedMessage {
    const MessageType type = MessageType::MN_TO_CN_INSERT_FAILED_RESPONSE;
    VectorID target_leaf; /* should be leaf */
    Version target_version;
    VectorID vector; /* cannot be cluster */
    VTYPE data[]; /* todo: does it have paddings? */

    constexpr static size_t Size(uint16_t dim) {
        return sizeof(InsertFailedMessage) + sizeof(VTYPE) * dim;
    }
};

/* a general task */
struct  __attribute__((packed)) DeleteFailedMessage {
    const MessageType type = MessageType::MN_TO_CN_DELETE_FAILED_RESPONSE;
    VectorID target_vector; /* cannot be cluster */

    constexpr static size_t Size() {
        return sizeof(DeleteFailedMessage);
    }
};

/* a general task */
struct  __attribute__((packed)) ClusterAddressResponseMessage {
    const MessageType type = MessageType::MN_TO_CN_CLUSTER_ADDRESS_RESPONSE;
    VectorID cluster_id;
    Version cluster_version;
    uintptr_t cluster_addr;

    constexpr static size_t Size() {
        return sizeof(ClusterAddressResponseMessage);
    }
};

struct __attribute__((packed)) VectorInsertNotificationMessage {
    const MessageType type = MessageType::MN_TO_CN_VECTOR_INSERT_NOTIFICATION;
    VectorID container_id;
    Version container_version;
    ClusterSizeType offset;
    VectorID new_vector_id;
    VTYPE vector_data[];

    constexpr static size_t Size(uint16_t dim) {
        return sizeof(VectorInsertNotificationMessage) + sizeof(VTYPE) * dim;
    }
};

struct __attribute__((packed)) ClusterInsertNotificationMessage {
    const MessageType type = MessageType::MN_TO_CN_CLUSTER_INSERT_NOTIFICATION;
    VectorID container_id;
    Version container_version;
    ClusterSizeType num_inserted;
    ClusterSizeType offset;
    ClusterSizeType outdated_offset;
    char data[]; /* <VectorID, Version, VectorData>[] */

    constexpr static size_t Size(uint16_t dim, ClusterSizeType num_inserted) {
        return sizeof(ClusterInsertNotificationMessage) +
               num_inserted * (sizeof(VectorID) + sizeof(Version) + sizeof(VTYPE) * dim);
    }
};


struct __attribute__((packed)) CompactionNotificationMessage {
    const MessageType type = MessageType::MN_TO_CN_COMPACTION_NOTIFICATION;
    VectorID target_cluster;
    Version new_version;
    uintptr_t cluster_addr;

    constexpr static size_t Size() {
        return sizeof(CompactionNotificationMessage);
    }
};

struct __attribute__((packed)) SplitNotificationMessage {
    const MessageType type = MessageType::MN_TO_CN_SPLIT_NOTIFICATION;
    // VectorID clusters[2]; /* the first one is the target cluster */
    // Version versions[2]; /* the version of the first id is the new version of the cluster */
    // uintptr_t cluster_addrs[2];
    char data[];

    constexpr static size_t Size(ClusterSizeType split_factor) {
        return sizeof(SplitNotificationMessage) +
               split_factor * (sizeof(VectorID) + sizeof(Version) + sizeof(uintptr_t));
    }
};

struct __attribute__((packed)) ExpansionNotificationMessage {
    const MessageType type = MessageType::MN_TO_CN_EXPANSION_NOTIFICATION;
    VectorID new_root_id;
    Version new_root_version;
    uintptr_t new_root_addr;
    // VectorID clusters[2]; /* the first one is the target cluster(which should be old root) */
    // Version versions[2]; /* the version of the first id is the new version of the cluster */
    // uintptr_t cluster_addrs[2];
    char data[];

    constexpr static size_t Size(ClusterSizeType split_factor) {
        return sizeof(ExpansionNotificationMessage) +
               split_factor * (sizeof(VectorID) + sizeof(Version) + sizeof(uintptr_t));
    }
};

struct __attribute__((packed)) DeleteNotificationMessage {
    const MessageType type = MessageType::MN_TO_CN_DELETE_NOTIFICATION;
    VectorID container_id;
    Version container_version;
    ClusterSizeType offset;

    constexpr static size_t Size() {
        return sizeof(DeleteNotificationMessage);
    }
};

struct __attribute__((packed)) RootPruneNotificationMessage {
    const MessageType type = MessageType::MN_TO_CN_ROOT_PRUNE_NOTIFICATION;
    VectorID old_root_id;
    Version old_root_version;
    VectorID new_root_id;
    Version new_root_version;
    uintptr_t new_root_addr;

    constexpr static size_t Size() {
        return sizeof(RootPruneNotificationMessage);
    }
};

/* todo: this message might be too large so we may need to send multiple of them instead of one for one migration */
/* since in this case, we are only the cache, we do not need to lock the children and we can just lock the parents
   but similar to merge, for now we consider this general. todo: later consider it cluster based */
struct __attribute__((packed)) MigrationNotificationMessage {
    const MessageType type = MessageType::MN_TO_CN_MIGRATION_NOTIFICATION;
    VectorID src_cluster;
    VectorID dest_cluster;
    Version src_version;
    Version dest_version;
    ClusterSizeType num_migrated;
    ClusterSizeType offset; /* start offset where they were inserted at dest */
    ClusterSizeType old_offsets[]; /* <old offsets> */

    constexpr static size_t Size(uint64_t num_migrated) {
        return sizeof(MigrationNotificationMessage) +
               num_migrated * (sizeof(ClusterSizeType));
    }
};

struct  __attribute__((packed)) MergeNotificationMessage {
    const MessageType type = MessageType::MN_TO_CN_MERGE_NOTIFICATION;
    VectorID src_cluster;
    Version src_version;
    VectorID dest_cluster;
    Version dest_version;

    constexpr static size_t Size() {
        return sizeof(MergeNotificationMessage);
    }
};

// struct  __attribute__((packed)) MigrationCheckRequestMessage {
//     const MessageType type = MessageType::MN_TO_CN_MIGRATION_CHECK_REQUEST;
//     VectorID first_cluster_id;
//     VectorID second_cluster_id;
//     Version first_cluster_version; /* we do not fail if we have a newer version at CN we just use the newer version */
//     Version second_cluster_version;
//     uintptr_t first_cluster_addr;
//     uintptr_t second_cluster_addr;
// };

struct __attribute__((packed)) MergeCheckRequestMessage {
    const MessageType type = MessageType::MN_TO_CN_MERGE_CHECK_REQUEST;
    VectorID src_cluster_id;
    VectorID parent_cluster_id;
    Version parent_cluster_version; /* we do not fail if we have a newer version at CN we just use the newer version */
    uintptr_t parent_cluster_addr;

    constexpr static size_t Size() {
        return sizeof(MergeCheckRequestMessage);
    }
};

struct __attribute__((packed)) SplitRequestMessage {
    const MessageType type = MessageType::MN_TO_CN_SPLIT_REQUEST;
    VectorID target_vector;
    Version target_version;
    uintptr_t target_vector_addr;
    ClusterSizeType marked_for_update;
    ClusterSizeType num_vectors;
    char data[]; /* uintptr_t[split_factor] new addresses, <VectorID, Version(if applicable?) */
    // uintptr_t new_cluster_addrs[]; /* the CN will use these for the new clusters and puts the VectorData+ID of the cluster itself as the last vector in it */

    constexpr static size_t Size(ClusterSizeType split_factor, ClusterSizeType num_vectors) {
        return sizeof(SplitRequestMessage) +
               sizeof(uintptr_t) * split_factor +
               (sizeof(VectorID) + sizeof(Version)) * num_vectors;
    }
};

struct __attribute__((packed)) UrgentSplitRequestMessage {
    const MessageType type = MessageType::URGENT_MN_TO_CN_SPLIT_REQUEST;
    VectorID target_vector;
    Version target_version;
    uintptr_t target_vector_addr;

    uintptr_t new_cluster_addr[]; /* the CN will use these for the new clusters and puts the VectorData+ID of the cluster itself as the last vector in it */

    constexpr static size_t Size(ClusterSizeType split_factor) {
        return sizeof(UrgentSplitRequestMessage) +
               sizeof(uintptr_t) * split_factor;
    }
};

struct CommLayerMessage {
protected:
    BufferInfo info;
public:
    MessageType GetMessageType() const {
        FatalAssert(info.length >= sizeof(MessageType), LOG_TAG_COMM_LAYER,
                    "Buffer length %zu is smaller than MessageType size %zu",
                    info.length, sizeof(MessageType));
        CHECK_NOT_NULLPTR(info.buffer, LOG_TAG_COMM_LAYER);
        return *reinterpret_cast<MessageType*>(info.buffer);
    }

    void* GetMessageBuffer() const {
        CHECK_NOT_NULLPTR(info.buffer, LOG_TAG_COMM_LAYER);
        return info.buffer;
    }

    size_t GetMessageLength() const {
        return info.length;
    }

friend class CommLayer;
};

struct ClusterRemoteAccessInfo {
    VectorID cluster_id;
    Version cluster_version;
    void* local_addr;
    uintptr_t remote_addr;
};

enum class CommLayerState : uint8_t {
    INITIALIZATION,
    RUNNING,
    SHUTTING_DOWN,
    SHUTDOWN_COMPLETED
};

class CommLayer {
protected:
    const uint16_t dim;
    const ClusterSizeType split_factor;
    const size_t leaf_size_bytes;
    const size_t internal_size_bytes;
    CommLayerState state;
    SXLock state_lock;
#ifndef MEMORY_NODE
    /* todo: replace with concurrent hash table? */
    SXSpinLock pending_reads_lock;
    std::unordered_map<ConnTaskId, std::vector<std::pair<VectorID, Version>>, ConnTaskIdHash> pending_reads;
    SANITY_CHECK(
        std::unordered_map<std::pair<VectorID, Version>, ConnTaskId, VectorIDVersionPairHash> read_requests_map;
    )
#endif
    /* todo: need to implement destructor in such a way that we wait untill everything is done before shutting down  */
    ~CommLayer() {
        state_lock.Lock(SX_EXCLUSIVE);
        state = CommLayerState::SHUTTING_DOWN;
        state_lock.Unlock();
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_COMM_LAYER,
                "Comm Layer is shutting down, waiting for ongoing tasks to finish...");
        sleep(60); /* wait for a minute to let things finish */
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_COMM_LAYER,
                "Comm Layer shutdown completed.");
    }
public:
    inline uint64_t GetMessageSize(MessageType type) const {
        switch (type) {
            case MessageType::REGISTER_MEMORY:
            {
                return RegisterMemoryMessage::Size();
            }
            case MessageType::SHUTDOWN_REQUEST:
            {
                return ShutdownRequestMessage::Size();
            }
            case MessageType::SHUTDOWN_RESPONSE:
            {
                return ShutdownResponseMessage::Size();
            }
            case MessageType::CN_TO_MN_INSERT_REQUEST:
            {
                return InsertRequestMessage::Size(dim);
            }
            case MessageType::CN_TO_MN_DELETE_REQUEST:
            {
                return DeleteRequestMessage::Size();
            }
            case MessageType::CN_TO_MN_MIGRATION_REQUEST:
            {
                const MigrationRequestMessage* req =
                        reinterpret_cast<const MigrationRequestMessage*>(message);
                return MigrationRequestMessage::Size(req->num_first_to_second,
                                                     req->num_second_to_first);
            }
            case MessageType::CN_TO_MN_MERGE_REQUEST:
            {
                return MergeRequestMessage::Size();
            }
            case MessageType::CN_TO_MN_UNPIN_NOTIFICATION:
            {
                return UnpinNotificationMessage::Size();
            }
            case MessageType::CN_TO_MN_CLUSTER_ADDRESS_REQUEST:
            {
                return ClusterAddressRequestMessage::Size();
            }
            case MessageType::MN_TO_CN_INSERT_FAILED_RESPONSE:
            {
                return InsertFailedMessage::Size(dim);
            }
            case MessageType::MN_TO_CN_DELETE_FAILED_RESPONSE:
            {
                return DeleteFailedMessage::Size();
            }
            case MessageType::MN_TO_CN_CLUSTER_ADDRESS_RESPONSE:
            {
                return ClusterAddressResponseMessage::Size();
            }
            case MessageType::MN_TO_CN_VECTOR_INSERT_NOTIFICATION:
            {
                return VectorInsertNotificationMessage::Size(dim);
            }
            case MessageType::MN_TO_CN_CLUSTER_INSERT_NOTIFICATION:
            {
                const ClusterInsertNotificationMessage* notif =
                        reinterpret_cast<const ClusterInsertNotificationMessage*>(message);
                return ClusterInsertNotificationMessage::Size(dim, notif->num_inserted);
            }
            case MessageType::MN_TO_CN_COMPACTION_NOTIFICATION:
            {
                return CompactionNotificationMessage::Size();
            }
            case MessageType::MN_TO_CN_SPLIT_NOTIFICATION:
            {
                return SplitNotificationMessage::Size(split_factor);
            }
            case MessageType::MN_TO_CN_EXPANSION_NOTIFICATION:
            {
                return ExpansionNotificationMessage::Size(split_factor);
            }
            case MessageType::MN_TO_CN_DELETE_NOTIFICATION:
            {
                return DeleteNotificationMessage::Size();
            }
            case MessageType::MN_TO_CN_ROOT_PRUNE_NOTIFICATION:
            {
                return RootPruneNotificationMessage::Size();
            }
            case MessageType::MN_TO_CN_MIGRATION_NOTIFICATION:
            {
                const MigrationNotificationMessage* notif =
                        reinterpret_cast<const MigrationNotificationMessage*>(message);
                return MigrationNotificationMessage::Size(notif->num_migrated);
            }
            case MessageType::MN_TO_CN_MERGE_NOTIFICATION:
            {
                return MergeNotificationMessage::Size();
            }
            case MessageType::MN_TO_CN_MERGE_CHECK_REQUEST:
            {
                return MergeCheckRequestMessage::Size();
            }
            case MessageType::MN_TO_CN_SPLIT_REQUEST:
            {
                const SplitRequestMessage* split_msg =
                        reinterpret_cast<const SplitRequestMessage*>(message);
                return SplitRequestMessage::Size(split_factor, split_msg->num_vectors);
            }
            case MessageType::URGENT_MN_TO_CN_SPLIT_REQUEST:
            {
                const UrgentSplitRequestMessage* urgent_msg =
                        reinterpret_cast<const UrgentSplitRequestMessage*>(message);
                return UrgentSplitRequestMessage::Size(split_factor);
            }
            default:
            {
                FatalAssert(false, LOG_TAG_COMM_LAYER,
                            "Cannot get message size for unknown or variable-size message type %u",
                            static_cast<uint8_t>(type));
                return 0;
            }
        }
    }

    inline uint64_t GetMessageSize(const void* message) const {
        CHECK_NOT_NULLPTR(message, LOG_TAG_COMM_LAYER);
        MessageType type = *reinterpret_cast<const MessageType*>(message);
        return GetMessageSize(type);
    }

    RetStatus BuildRequestMessage(uint8_t target_node_id, MessageType type, CommLayerMessage& message,
                                  size_t message_size, bool need_lock = true) {
#ifdef MEMORY_NODE
        FatalAssert((type >= MessageType::BASE_MESSAGE_START &&
                    type <= MessageType::BASE_MESSAGE_END) ||
                    (type >= MessageType::MN_MESSAGE_START &&
                    type <= MessageType::MN_MESSAGE_END),
                    LOG_TAG_COMM_LAYER,
                    "Memory Node can only send base or MN messages. Invalid message type: %u",
                    static_cast<uint8_t>(type));
#else
        FatalAssert((type >= MessageType::BASE_MESSAGE_START &&
                    type <= MessageType::BASE_MESSAGE_END) ||
                    (type >= MessageType::CN_MESSAGE_START &&
                    type <= MessageType::CN_MESSAGE_END),
                    LOG_TAG_COMM_LAYER,
                    "Compute Node can only send base or CN messages. Invalid message type: %u",
                    static_cast<uint8_t>(type));
#endif
        if (need_lock) {
            state_lock.Lock(SX_SHARED);
        }
        threadSelf->SanityCheckLockHeldByMe(&state_lock);
        if (state != CommLayerState::RUNNING && state != CommLayerState::INITIALIZATION) {
            message.info.buffer = nullptr;
            message.info.length = 0;
            message.info.max_length = 0;
            message.info.target_node_id = target_node_id;
            if (need_lock) {
                state_lock.Unlock();
            }
            return RetStatus{.stat = RetStatus::COMM_LAYER_DOWN,
                                .message = nullptr};
        }

#ifdef MEMORY_NODE
        if (type == MessageType::URGENT_MN_TO_CN_SPLIT_REQUEST) {
            FatalAssert(target_node_id != BROADCAST_NODE_ID,
                        LOG_TAG_COMM_LAYER,
                        "Urgent messages cannot be sent to broadcast node");
            FatalAssert(message_size <= sizeof(UrgentMessageData),
                        LOG_TAG_COMM_LAYER,
                        "Urgent message size %zu exceeds the maximum allowed size %u",
                        message_size,
                        sizeof(UrgentMessageData));
            UrgentMessage* raw_message = reinterpret_cast<UrgentMessage*>(threadSelf->GetUrgentMessageBuffer());
            raw_message->Clear();
            raw_message->meta.detail.length = static_cast<uint8_t>(message_size);
            *(reinterpret_cast<MessageType*>(raw_message->data.data)) = type;
            message.info.max_length = sizeof(UrgentMessageData);
            message.info.length = message_size;
            message.info.buffer = raw_message->data.data;
            message.info.target_node_id = target_node_id;
            message.info.buffer_idx = 0;
            FatalAssert(message.GetMessageType() == type,
                        LOG_TAG_COMM_LAYER,
                        "Failed to set message type %u in urgent message",
                        static_cast<uint8_t>(type));
            return RetStatus::Success();
        } else if (target_node_id == BROADCAST_NODE_ID) {
            message.info.buffer = threadSelf->GetBroadcastBuffer(message.info.max_length);
            FatalAssert(message_size <= message.info.max_length,
                        LOG_TAG_COMM_LAYER,
                        "Broadcast message size %zu exceeds the maximum allowed size %zu",
                        message_size,
                        message.info.max_length);
            message.info.length = message_size;
            message.info.target_node_id = target_node_id;
            *(reinterpret_cast<MessageType*>(message.info.buffer)) = type;
            FatalAssert(message.GetMessageType() == type,
                        LOG_TAG_COMM_LAYER,
                        "Failed to set message type %u in broadcast message",
                        static_cast<uint8_t>(type));
            return RetStatus::Success();
        }
#endif
        RDMA_Manager* rdma_manager = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_manager, LOG_TAG_COMM_LAYER);
        RetStatus rs = rdma_manager->GrabCommBuffer(target_node_id, message_size, message.info);
        if (!rs.IsOK()) {
            FatalAssert(false, LOG_TAG_COMM_LAYER,
                        "Failed to grab communication buffer for target node %u, message size %zu: %s",
                        target_node_id, message_size, rs.Msg());
            state_lock.Unlock();
            return rs;
        }
        CHECK_NOT_NULLPTR(message.info.buffer, LOG_TAG_COMM_LAYER);
        *(reinterpret_cast<MessageType*>(message.info.buffer)) = type;
        FatalAssert(message.GetMessageType() == type,
                    LOG_TAG_COMM_LAYER,
                    "Failed to set message type %u in message",
                    static_cast<uint8_t>(type));
        FatalAssert(rs.IsOK(), LOG_TAG_COMM_LAYER,
                    "Successfully built request message of type %u for target node %u",
                    static_cast<uint8_t>(type), target_node_id);
        message.info.length = message_size;
        return rs;
    }

    RetStatus SendMessage(CommLayerMessage& message, bool flush, bool unlock = true) {
        FatalAssert(message.info.buffer != nullptr,
                    LOG_TAG_COMM_LAYER,
                    "Cannot send message with null buffer");
        MessageType type = message.GetMessageType();
        uint8_t target_node_id = message.info.target_node_id;
#ifdef MEMORY_NODE
        FatalAssert((type >= MessageType::BASE_MESSAGE_START &&
                    type <= MessageType::BASE_MESSAGE_END) ||
                    (type >= MessageType::MN_MESSAGE_START &&
                    type <= MessageType::MN_MESSAGE_END),
                    LOG_TAG_COMM_LAYER,
                    "Memory Node can only send base or MN messages. Invalid message type: %u",
                    static_cast<uint8_t>(type));
#else
        FatalAssert((type >= MessageType::BASE_MESSAGE_START &&
                    type <= MessageType::BASE_MESSAGE_END) ||
                    (type >= MessageType::CN_MESSAGE_START &&
                    type <= MessageType::CN_MESSAGE_END),
                    LOG_TAG_COMM_LAYER,
                    "Compute Node can only send base or CN messages. Invalid message type: %u",
                    static_cast<uint8_t>(type));
#endif
        threadSelf->SanityCheckLockHeldByMe(&state_lock);
        SANITY_CHECK({
            FatalAssert(state == CommLayerState::RUNNING || state == CommLayerState::INITIALIZATION,
                        LOG_TAG_COMM_LAYER,
                        "Cannot send message when communication layer is not running (current state: %u)",
                        static_cast<uint8_t>(state));
        });

        RDMA_Manager* rdma_manager = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_manager, LOG_TAG_COMM_LAYER);
        RetStatus rs;
#ifdef MEMORY_NODE
        if (type == MessageType::URGENT_MN_TO_CN_SPLIT_REQUEST) {
            FatalAssert(target_node_id != BROADCAST_NODE_ID,
                        LOG_TAG_COMM_LAYER,
                        "Urgent messages cannot be sent to broadcast node");
            UrgentMessage* raw_message = reinterpret_cast<UrgentMessage*>(message.info.buffer);
            rs = rdma_manager->SendUrgentMessage(raw_message, target_node_id);
            if (unlock) {
                state_lock.Unlock();
            }
            return rs;
        } else if (target_node_id == BROADCAST_NODE_ID) {
            rs = rdma_manager->BroadCastCommRequest(message.info.buffer, message.info.length, flush);
            if (unlock) {
                state_lock.Unlock();
            }
            return rs;
        }
#endif

        rs = rdma_manager->ReleaseCommBuffer(message.info, flush);
        if (unlock) {
            state_lock.Unlock();
        }
        return rs;
    }

    RetStatus CheckReceivedMessagesFromNode(uint8_t target_node_id, std::vector<CommLayerMessage>& general_messages,
                                            std::unordered_map<VectorID, std::vector<CommLayerMessage>, VectorIDHash>&
                                                cluster_based_messages, bool need_lock = true) {
        RDMA_Manager* rdma_manager = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_manager, LOG_TAG_COMM_LAYER);
        uint8_t self_node_id = rdma_manager->GetSelfNodeId();
        uint8_t num_nodes = rdma_manager->GetNumNodes();
        FatalAssert(self_node_id < num_nodes, LOG_TAG_COMM_LAYER,
                    "Invalid self_node_id %u in CheckReceivedMessagesFromNode", self_node_id);
        FatalAssert(target_node_id < num_nodes, LOG_TAG_COMM_LAYER,
                    "Invalid target_node_id %u in CheckReceivedMessagesFromNode", target_node_id);
        FatalAssert(target_node_id != self_node_id,
                    LOG_TAG_COMM_LAYER,
                    "Cannot check received messages from broadcast node");
#ifdef MEMORY_NODE
            FatalAssert(self_node_id == MEMORY_NODE_ID, LOG_TAG_COMM_LAYER,
                        "Self node id %u is not memory node in CheckReceivedMessagesFromNode",
                        self_node_id);
#else
            FatalAssert(self_node_id != MEMORY_NODE_ID, LOG_TAG_COMM_LAYER,
                        "Self node id %u is memory node in CheckReceivedMessagesFromNode",
                        self_node_id);
#endif
        if (need_lock) {
            state_lock.Lock(SX_SHARED);
        }
        threadSelf->SanityCheckLockHeldByMe(&state_lock);

        if (state != CommLayerState::RUNNING && state != CommLayerState::INITIALIZATION) {
            if (need_lock) {
                state_lock.Unlock();
            }
            return RetStatus{.stat = RetStatus::COMM_LAYER_DOWN,
                                .message = nullptr};
        }
        std::vector<BufferInfo> receive_buffers;
        RetStatus rs = rdma_manager->PollCommRequests(target_node_id, receive_buffers);
        if (!rs.IsOK()) {
            FatalAssert(false, LOG_TAG_COMM_LAYER,
                        "Failed to poll communication requests from node %u: %s",
                        target_node_id, rs.Msg());
            if (need_lock) {
                state_lock.Unlock();
            }
            return rs;
        }

        for (const BufferInfo& buffer_info : receive_buffers) {
            CommLayerMessage message;
            message.info = buffer_info;
            uint64_t num_messages = message.info.num_requests;
            FatalAssert(num_messages > 0, LOG_TAG_COMM_LAYER,
                        "Received buffer with zero messages from node %u",
                        target_node_id);
            void* current_msg_ptr = message.GetMessageBuffer();
            for (uint64_t i = 0; i < num_messages; ++i) {
                FatalAssert(current_msg_ptr != nullptr,
                            LOG_TAG_COMM_LAYER,
                            "Null message pointer for message %zu from node %u",
                            i, target_node_id);
                FatalAssert(current_msg_ptr <= static_cast<char*>(message.GetMessageBuffer()) + message.info.max_length,
                            LOG_TAG_COMM_LAYER,
                            "Message pointer out of bounds for message %zu from node %u",
                            i, target_node_id);
                MessageType type = message.GetMessageType();
                uint64_t message_size = GetMessageSize(current_msg_ptr);
#ifdef MEMORY_NODE
                FatalAssert((type >= MessageType::BASE_MESSAGE_START &&
                            type <= MessageType::BASE_MESSAGE_END) ||
                            (type >= MessageType::CN_MESSAGE_START &&
                            type <= MessageType::CN_MESSAGE_END),
                            LOG_TAG_COMM_LAYER,
                            "Memory Node can only receive base or CN messages. Invalid message type: %u",
                            static_cast<uint8_t>(type));
#else
                FatalAssert((type >= MessageType::BASE_MESSAGE_START &&
                            type <= MessageType::BASE_MESSAGE_END) ||
                            (type >= MessageType::MN_MESSAGE_START &&
                            type <= MessageType::MN_MESSAGE_END),
                            LOG_TAG_COMM_LAYER,
                            "Compute Node can only receive base or MN messages. Invalid message type: %u",
                            static_cast<uint8_t>(type));
#endif
                CommLayerMessage current_msg;
                /* todo: use a single memory allocation for this instead of multiple allocations
                   even better, keep an aditional unregistered buffer for this per thread? or maybe handle the whole
                   thing in the upper layer somehow to avoid additional copy */
                current_msg.info.buffer = new char[message_size];
                memcpy(current_msg.info.buffer, current_msg_ptr, message_size);
                current_msg.info.length = message_size;
                current_msg.info.max_length = message_size;
                current_msg.info.target_node_id = target_node_id;

                VectorID cluster_id = INVALID_VECTOR_ID;
                switch (type) {
                    case MessageType::CN_TO_MN_INSERT_REQUEST:
                    {
                        InsertRequestMessage* insert_msg =
                            reinterpret_cast<InsertRequestMessage*>(current_msg.GetMessageBuffer());
                        CHECK_NOT_NULLPTR(insert_msg, LOG_TAG_COMM_LAYER);
                        FatalAssert(insert_msg->type == type,
                                    LOG_TAG_COMM_LAYER,
                                    "Message type mismatch in InsertRequestMessage: expected %u, got %u",
                                    static_cast<uint8_t>(type),
                                    static_cast<uint8_t>(insert_msg->type));
                        cluster_id = insert_msg->target_leaf;
                        FatalAssert(cluster_id.IsValid() && cluster_id.IsLeaf(),
                                    LOG_TAG_COMM_LAYER,
                                    "Invalid target leaf " VECTORID_LOG_FMT " in InsertRequestMessage",
                                    VECTORID_LOG(cluster_id));
                        break;
                    }
                    case MessageType::MN_TO_CN_VECTOR_INSERT_NOTIFICATION:
                    {
                        VectorInsertNotificationMessage* vector_insert_msg =
                            reinterpret_cast<VectorInsertNotificationMessage*>(current_msg.GetMessageBuffer());
                        CHECK_NOT_NULLPTR(vector_insert_msg, LOG_TAG_COMM_LAYER);
                        FatalAssert(vector_insert_msg->type == type,
                                    LOG_TAG_COMM_LAYER,
                                    "Message type mismatch in VectorInsertNotificationMessage: expected %u, got %u",
                                    static_cast<uint8_t>(type),
                                    static_cast<uint8_t>(vector_insert_msg->type));
                        cluster_id = vector_insert_msg->container_id;
                        FatalAssert(cluster_id.IsValid() && cluster_id.IsLeaf(),
                                    LOG_TAG_COMM_LAYER,
                                    "Invalid container id " VECTORID_LOG_FMT " in VectorInsertNotificationMessage",
                                    VECTORID_LOG(cluster_id));
                        break;
                    }
                    case MessageType::MN_TO_CN_CLUSTER_INSERT_NOTIFICATION:
                    {
                        ClusterInsertNotificationMessage* cluster_insert_msg =
                            reinterpret_cast<ClusterInsertNotificationMessage*>(current_msg.GetMessageBuffer());
                        CHECK_NOT_NULLPTR(cluster_insert_msg, LOG_TAG_COMM_LAYER);
                        FatalAssert(cluster_insert_msg->type == type,
                                    LOG_TAG_COMM_LAYER,
                                    "Message type mismatch in ClusterInsertNotificationMessage: expected %u, got %u",
                                    static_cast<uint8_t>(type),
                                    static_cast<uint8_t>(cluster_insert_msg->type));
                        cluster_id = cluster_insert_msg->container_id;
                        FatalAssert(cluster_id.IsValid() && cluster_id.IsInternalVertex(),
                                    LOG_TAG_COMM_LAYER,
                                    "Invalid container id " VECTORID_LOG_FMT " in ClusterInsertNotificationMessage",
                                    VECTORID_LOG(cluster_id));
                        break;
                    }
                    case MessageType::MN_TO_CN_DELETE_NOTIFICATION:
                    {
                        DeleteNotificationMessage* delete_notif_msg =
                            reinterpret_cast<DeleteNotificationMessage*>(current_msg.GetMessageBuffer());
                        CHECK_NOT_NULLPTR(delete_notif_msg, LOG_TAG_COMM_LAYER);
                        FatalAssert(delete_notif_msg->type == type,
                                    LOG_TAG_COMM_LAYER,
                                    "Message type mismatch in DeleteNotificationMessage: expected %u, got %u",
                                    static_cast<uint8_t>(type),
                                    static_cast<uint8_t>(delete_notif_msg->type));
                        cluster_id = delete_notif_msg->container_id;
                        FatalAssert(cluster_id.IsValid() && cluster_id.IsCentroid(), LOG_TAG_COMM_LAYER,
                                    "Invalid container id " VECTORID_LOG_FMT " in DeleteNotificationMessage",
                                    VECTORID_LOG(cluster_id));
                        break;
                    }
                    case MessageType::MN_TO_CN_ROOT_PRUNE_NOTIFICATION:
                    {
                        RootPruneNotificationMessage* root_prune_msg =
                            reinterpret_cast<RootPruneNotificationMessage*>(current_msg.GetMessageBuffer());
                        CHECK_NOT_NULLPTR(root_prune_msg, LOG_TAG_COMM_LAYER);
                        FatalAssert(root_prune_msg->type == type,
                                    LOG_TAG_COMM_LAYER,
                                    "Message type mismatch in RootPruneNotificationMessage: expected %u, got %u",
                                    static_cast<uint8_t>(type),
                                    static_cast<uint8_t>(root_prune_msg->type));
                        cluster_id = root_prune_msg->old_root_id;
                        FatalAssert(cluster_id.IsValid() && cluster_id.IsInternalVertex(), LOG_TAG_COMM_LAYER,
                                    "Invalid old root id " VECTORID_LOG_FMT " in RootPruneNotificationMessage",
                                    VECTORID_LOG(cluster_id));
                        break;
                    }
                    default:
                    {
                        cluster_id = INVALID_VECTOR_ID;
                        break;
                    }
                }

                if (cluster_id.IsValid()) {
                    auto it = cluster_based_messages.find(cluster_id);
                    if (it == cluster_based_messages.end()) {
                        it = cluster_based_messages.emplace(cluster_id,
                                                            std::vector<CommLayerMessage>()).first;
                    }
                    it->second.push_back(current_msg);
                } else {
                    general_messages.push_back(current_msg);
                }
            }
        }

        rs = rdma_manager->ReleaseCommReciveBuffers(target_node_id, receive_buffers);
        if (!rs.IsOK()) {
            FatalAssert(false, LOG_TAG_COMM_LAYER,
                        "Failed to release communication receive buffers from node %u: %s",
                        target_node_id, rs.Msg());
        }
        if (need_lock) {
            state_lock.Unlock();
        }
        return rs;
    }

    /* general messages may also have vectorID and version but cluster based ones are messages that explicitly
       require locking a cluster. some general messages(e.g. deleteVector) may also need locks so we may need to
       analyze the general messages first in the upper layer to filter such messages*/
    RetStatus CheckReceivedMessages(std::vector<CommLayerMessage>& general_messages,
                                    std::unordered_map<VectorID, std::vector<CommLayerMessage>, VectorIDHash>&
                                        cluster_based_messages) {


        FatalAssert(general_messages.empty(), LOG_TAG_COMM_LAYER,
                    "General messages vector is not empty in CheckReceivedMessages");
        FatalAssert(cluster_based_messages.empty(), LOG_TAG_COMM_LAYER,
                    "Cluster-based messages map is not empty in CheckReceivedMessages");
#ifndef MEMORY_NODE
        return CheckReceivedMessagesFromNode(MEMORY_NODE_ID, general_messages, cluster_based_messages);
#endif
        RDMA_Manager* rdma_manager = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_manager, LOG_TAG_COMM_LAYER);
        uint8_t self_node_id = rdma_manager->GetSelfNodeId();
        uint8_t num_nodes = rdma_manager->GetNumNodes();
        FatalAssert(self_node_id < num_nodes, LOG_TAG_COMM_LAYER,
                    "Invalid self_node_id %u in CheckReceivedMessages", self_node_id);

        state_lock.Lock(SX_SHARED);
        if (state != CommLayerState::RUNNING && state != CommLayerState::INITIALIZATION) {
            state_lock.Unlock();
            return RetStatus{.stat = RetStatus::COMM_LAYER_DOWN,
                                .message = nullptr};
        }

        for (uint8_t node_id = 0; node_id < num_nodes; ++node_id) {
            uint8_t target_node_id = (uint8_t)(((uint64_t)node_id + threadSelf->ID()) % num_nodes);
            if (target_node_id == self_node_id) {
                continue;
            }
            RetStatus rs = CheckReceivedMessagesFromNode(target_node_id, general_messages, cluster_based_messages,
                                                         false);
            if (!rs.IsOK()) {
                FatalAssert(false, LOG_TAG_COMM_LAYER,
                            "Failed to check received messages from node %u: %s",
                            target_node_id, rs.Msg());
                state_lock.Unlock();
                return rs;
            }
        }
        state_lock.Unlock();
        return RetStatus::Success();
    }

    RetStatus FreeMessages(std::vector<CommLayerMessage>& general_messages,
                           std::unordered_map<VectorID, std::vector<CommLayerMessage>, VectorIDHash>&
                                cluster_based_messages) {
        for (CommLayerMessage& message : general_messages) {
            delete[] message.info.buffer;
            message.info.buffer = nullptr;
        }
        general_messages.clear();
        for (auto& pair : cluster_based_messages) {
            for (CommLayerMessage& message : pair.second) {
                delete[] message.info.buffer;
                message.info.buffer = nullptr;
            }
            pair.second.clear();
        }
        cluster_based_messages.clear();
        return RetStatus::Success();
    }

#ifndef MEMORY_NODE
    RetStatus PostClusterReadRequests(const std::vector<ClusterRemoteAccessInfo>& clusters_to_read) {
        FatalAssert(!clusters_to_read.empty(), LOG_TAG_COMM_LAYER,
                    "No clusters to read in PostClusterReadRequests");
        state_lock.Lock(SX_SHARED);
        if (state != CommLayerState::RUNNING && state != CommLayerState::INITIALIZATION) {
            state_lock.Unlock();
            return RetStatus{.stat = RetStatus::COMM_LAYER_DOWN,
                                .message = nullptr};
        }
        RDMA_Manager* rdma_manager = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_manager, LOG_TAG_COMM_LAYER);
        RetStatus rs = RetStatus::Success();
        std::vector<std::pair<VectorID, Version>> new_read_requests;
        new_read_requests.reserve(clusters_to_read.size());
        /* todo: use preallocated buffer */
        RDMABuffer* buffers = new RDMABuffer[clusters_to_read.size()];
        for (size_t i = 0; i < clusters_to_read.size(); ++i) {
            const ClusterRemoteAccessInfo& cluster_info = clusters_to_read[i];
            FatalAssert(cluster_info.cluster_id.IsValid() && cluster_info.cluster_id.IsCentroid() &&
                        cluster_info.remote_addr != 0 &&
                        cluster_info.local_addr != nullptr,
                        LOG_TAG_COMM_LAYER,
                        "Invalid cluster info in PostClusterReadRequests: cluster_id " VECTORID_LOG_FMT
                        ", cluster_version %u, local_addr %p, remote_addr 0x%lx",
                        VECTORID_LOG(cluster_info.cluster_id),
                        cluster_info.cluster_version,
                        cluster_info.local_addr,
                        cluster_info.remote_addr);
            new_read_requests.emplace_back(cluster_info.cluster_id, cluster_info.cluster_version);
            buffers[i].buffer_idx = 0;
            buffers[i].local_addr = cluster_info.local_addr;
            buffers[i].remote_addr = cluster_info.remote_addr;
            buffers[i].length = (cluster_info.cluster_id.IsLeaf()) ?
                                leaf_size_bytes : internal_size_bytes;
            buffers[i].remote_node_id = MEMORY_NODE_ID;
        }
        pending_reads_lock.Lock(SX_EXCLUSIVE);
        ConnTaskId task_id;
        rs = rdma_manager->RDMARead(buffers, clusters_to_read.size(), task_id);
        if (!rs.IsOK()) {
            pending_reads_lock.Unlock();
            FatalAssert(false, LOG_TAG_COMM_LAYER,
                        "Failed to post RDMA read for %zu clusters: %s",
                        clusters_to_read.size(), rs.Msg());
            state_lock.Unlock();
            delete[] buffers;
            return rs;
        }

        SANITY_CHECK(
            for (const auto& req : new_read_requests) {
                FatalAssert(read_requests_map.find(req) == read_requests_map.end(),
                            LOG_TAG_COMM_LAYER,
                            "Duplicate read request for cluster " VECTORID_LOG_FMT " version %u",
                            VECTORID_LOG(req.first),
                            req.second);
                read_requests_map.emplace(req, task_id);
            }
        )
        FatalAssert(pending_reads.find(task_id) == pending_reads.end(),
                    LOG_TAG_COMM_LAYER,
                    "Duplicate task_id 0x%lx in pending_reads map",
                    task_id._raw);
        pending_reads.emplace(task_id, std::move(new_read_requests));
        pending_reads_lock.Unlock();
        state_lock.Unlock();
        delete[] buffers;
        return rs;
    }

    RetStatus PostClusterWriteRequests(const std::vector<ClusterRemoteAccessInfo>& clusters_to_write) {
        FatalAssert(!clusters_to_write.empty(), LOG_TAG_COMM_LAYER,
                    "No clusters to write in PostClusterWriteRequests");
        state_lock.Lock(SX_SHARED);
        if (state != CommLayerState::RUNNING && state != CommLayerState::INITIALIZATION) {
            state_lock.Unlock();
            return RetStatus{.stat = RetStatus::COMM_LAYER_DOWN,
                                .message = nullptr};
        }
        RDMA_Manager* rdma_manager = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_manager, LOG_TAG_COMM_LAYER);
        RetStatus rs = RetStatus::Success();
        /* todo: use preallocated buffer */
        RDMABuffer* buffers = new RDMABuffer[clusters_to_write.size()];
        for (size_t i = 0; i < clusters_to_write.size(); ++i) {
            const ClusterRemoteAccessInfo& cluster_info = clusters_to_write[i];
            FatalAssert(cluster_info.cluster_id.IsValid() && cluster_info.cluster_id.IsCentroid() &&
                        cluster_info.remote_addr != 0 &&
                        cluster_info.local_addr != nullptr,
                        LOG_TAG_COMM_LAYER,
                        "Invalid cluster info in PostClusterReadRequests: cluster_id " VECTORID_LOG_FMT
                        ", cluster_version %u, local_addr %p, remote_addr 0x%lx",
                        VECTORID_LOG(cluster_info.cluster_id),
                        cluster_info.cluster_version,
                        cluster_info.local_addr,
                        cluster_info.remote_addr);
            buffers[i].buffer_idx = 0;
            buffers[i].local_addr = cluster_info.local_addr;
            buffers[i].remote_addr = cluster_info.remote_addr;
            buffers[i].length = (cluster_info.cluster_id.IsLeaf()) ?
                                leaf_size_bytes : internal_size_bytes;
            buffers[i].remote_node_id = MEMORY_NODE_ID;
        }

        rs = rdma_manager->RDMAWrite(buffers, clusters_to_write.size());
        if (!rs.IsOK()) {
            FatalAssert(false, LOG_TAG_COMM_LAYER,
                        "Failed to post RDMA write for %zu clusters: %s",
                        clusters_to_write.size(), rs.Msg());
        }


        state_lock.Unlock();
        delete[] buffers;
        return rs;
    }
    RetStatus CheckClusterReadCompletions(std::vector<std::vector<std::pair<VectorID, Version>>>& completed_reads) {
        RDMA_Manager* rdma_manager = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_manager, LOG_TAG_COMM_LAYER);
        std::vector<ConnTaskId> completed_task_ids;
        completed_reads.clear();
        state_lock.Lock(SX_SHARED);
        if (state != CommLayerState::RUNNING && state != CommLayerState::INITIALIZATION) {
            state_lock.Unlock();
            return RetStatus{.stat = RetStatus::COMM_LAYER_DOWN,
                                .message = nullptr};
        }
        RetStatus rs = rdma_manager->PollCompletion(ConnectionType::CN_CLUSTER_READ, completed_task_ids);
        if (!rs.IsOK()) {
            FatalAssert(false, LOG_TAG_COMM_LAYER,
                        "Failed to poll RDMA read completions from memory node: %s",
                        rs.Msg());
            state_lock.Unlock();
            return rs;
        }

        if (completed_task_ids.empty()) {
            state_lock.Unlock();
            return rs;
        }
        completed_reads.reserve(completed_task_ids.size());

        pending_reads_lock.Lock(SX_EXCLUSIVE);
        for (const ConnTaskId& task_id : completed_task_ids) {
            auto it = pending_reads.find(task_id);
            FatalAssert(it != pending_reads.end(),
                        LOG_TAG_COMM_LAYER,
                        "Completed task_id 0x%lx not found in pending_reads map",
                        task_id._raw);
            SANITY_CHECK(
                for (const auto& req : it->second) {
                    auto map_it = read_requests_map.find(req);
                    FatalAssert(map_it != read_requests_map.end(),
                                LOG_TAG_COMM_LAYER,
                                "Read request for cluster " VECTORID_LOG_FMT " version %u not found in read_requests_map",
                                VECTORID_LOG(req.first),
                                req.second);
                    FatalAssert(map_it->second._raw == task_id._raw,
                                LOG_TAG_COMM_LAYER,
                                "Task ID mismatch for cluster " VECTORID_LOG_FMT " version %u: expected 0x%lx, got 0x%lx",
                                VECTORID_LOG(req.first),
                                req.second,
                                map_it->second._raw,
                                task_id._raw);
                    read_requests_map.erase(map_it);
                }
            )
            completed_reads.push_back(std::move(it->second));
            pending_reads.erase(it);
        }
        pending_reads_lock.Unlock();
        state_lock.Unlock();
        return rs;
    }
    /* todo: since currently the only urgenmessage is split request we do not generalize */
    RetStatus CheckUrgentMessages(std::vector<UrgentSplitRequestMessage>& messages) {
        messages.clear();
        state_lock.Lock(SX_SHARED);
        if (state != CommLayerState::RUNNING && state != CommLayerState::INITIALIZATION) {
            state_lock.Unlock();
            return RetStatus{.stat = RetStatus::COMM_LAYER_DOWN,
                                .message = nullptr};
        }
        RDMA_Manager* rdma_manager = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_manager, LOG_TAG_COMM_LAYER);
        std::vector<std::pair<UrgentMessageData, uint8_t>> raw_messages;
        RetStatus rs = rdma_manager->PollUrgentMessages(raw_messages);
        if (!rs.IsOK()) {
            FatalAssert(false, LOG_TAG_COMM_LAYER,
                        "Failed to poll urgent messages: %s",
                        rs.Msg());
            state_lock.Unlock();
            return rs;
        }

        messages.reserve(raw_messages.size());
        for (const auto& pair : raw_messages) {
            const UrgentMessageData& raw_msg = pair.first;
            uint8_t msg_len = pair.second;
            MessageType type = *(reinterpret_cast<const MessageType*>(raw_msg.data));
            FatalAssert(type == MessageType::URGENT_MN_TO_CN_SPLIT_REQUEST,
                        LOG_TAG_COMM_LAYER,
                        "Invalid urgent message type %u with length %hhu",
                        static_cast<uint8_t>(type),
                        msg_len);
            /* todo: remove additional copy */
            messages.emplace_back();
            memccpy(&messages.back(), &raw_msg, 0, msg_len);
        }

        state_lock.Unlock();
        return rs;
    }
#endif

    CommLayer(uint16_t vector_dimension, ClusterSizeType clustering_split_factor, size_t leaf_bytes,
              size_t internal_bytes, void* memory_pool, size_t pool_size) : dim(vector_dimension),
                                                                            split_factor(clustering_split_factor),
                                                                            leaf_size_bytes(leaf_bytes),
                                                                            internal_size_bytes(internal_bytes),
                                                                            state(CommLayerState::INITIALIZATION) {
        CHECK_NOT_NULLPTR(memory_pool, LOG_TAG_COMM_LAYER);
        FatalAssert(pool_size > 0, LOG_TAG_COMM_LAYER,
                    "Invalid communication layer memory pool size %zu", pool_size);
        FatalAssert(leaf_size_bytes > 0, LOG_TAG_COMM_LAYER,
                    "Invalid leaf size bytes %zu", leaf_size_bytes);
        FatalAssert(internal_size_bytes > 0, LOG_TAG_COMM_LAYER,
                    "Invalid internal size bytes %zu", internal_size_bytes);
        FatalAssert(dim > 0, LOG_TAG_COMM_LAYER,
                    "Invalid vector dimension %u", dim);
        FatalAssert(split_factor >= 2, LOG_TAG_COMM_LAYER,
                    "Invalid clustering split factor %u", split_factor);
        FatalAssert(ALIGNED(memory_pool), LOG_TAG_COMM_LAYER,
                    "Communication layer memory pool %p is not cache aligned",
                    memory_pool);
        state_lock.Lock(SX_EXCLUSIVE);
        RDMA_Manager* rdma_manager = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_manager, LOG_TAG_COMM_LAYER);
        uint32_t lkey;
        uint32_t rkey;
        RetStatus rs = rdma_manager->RegisterMemory(memory_pool, pool_size, lkey, rkey);
        FatalAssert(rs.IsOK(), LOG_TAG_COMM_LAYER,
                    "Failed to register communication layer memory pool of size %zu: %s",
                    pool_size, rs.Msg());
#ifdef MEMORY_NODE
        uint8_t num_nodes = rdma_manager->GetNumNodes();
        uint8_t num_cns = num_nodes - 1; // exclude memory node
        std::vector<bool> registered(num_cns, false);

        std::vector<CommLayerMessage> general_messages;
        std::unordered_map<VectorID, std::vector<CommLayerMessage>, VectorIDHash> cluster_based_messages;

        while (num_cns > 0) {
            for (uint8_t node_id = 0; node_id < num_nodes; ++node_id) {
                if (node_id == MEMORY_NODE_ID || registered[node_id]) {
                    continue;
                }
                rs = CheckReceivedMessagesFromNode(node_id, general_messages, cluster_based_messages, false);
                FatalAssert(rs.IsOK(), LOG_TAG_COMM_LAYER,
                            "Failed to check received messages from node %u: %s",
                            node_id, rs.Msg());
                FatalAssert(cluster_based_messages.empty(), LOG_TAG_COMM_LAYER,
                            "Received unexpected cluster-based messages from node %u during memory registration",
                            node_id);
                FatalAssert(general_messages.size() <= 1, LOG_TAG_COMM_LAYER,
                            "Received multiple general messages from node %u during memory registration",
                            node_id);
                if (general_messages.empty()) {
                    continue;
                }

                for (CommLayerMessage& message : general_messages) {
                    FatalAssert(message.GetMessageType() == MessageType::REGISTER_MEMORY,
                                LOG_TAG_COMM_LAYER,
                                "Unexpected message type %u from node %u during memory registration",
                                static_cast<uint8_t>(message.GetMessageType()),
                                node_id);
                    RegisterMemoryMessage* register_msg =
                        reinterpret_cast<RegisterMemoryMessage*>(message.GetMessageBuffer());
                    CHECK_NOT_NULLPTR(register_msg, LOG_TAG_COMM_LAYER);
                    FatalAssert(register_msg->type == MessageType::REGISTER_MEMORY,
                                LOG_TAG_COMM_LAYER,
                                "Message type mismatch in RegisterMemoryMessage: expected %u, got %u",
                                static_cast<uint8_t>(MessageType::REGISTER_MEMORY),
                                static_cast<uint8_t>(register_msg->type));
                    registered[node_id] = true;
                    --num_cns;
                    rs = rdma_manager->RegisterRemoteMemory(node_id, register_msg->addr,
                                                            register_msg->length,
                                                            register_msg->rkey);
                    FatalAssert(rs.IsOK(), LOG_TAG_COMM_LAYER,
                                "Failed to register remote memory from node %u at addr=0x%lx with length=%zu: %s",
                                node_id, register_msg->addr, register_msg->length, rs.Msg());
                }
                rs = FreeMessages(general_messages, cluster_based_messages);
                FatalAssert(rs.IsOK(), LOG_TAG_COMM_LAYER,
                            "Failed to free messages after memory registration from node %u: %s",
                            node_id, rs.Msg());
            }
        }

        uint8_t target_node_id = BROADCAST_NODE_ID;
        CommLayerMessage message;
        rs = BuildRequestMessage(target_node_id,
                                 MessageType::REGISTER_MEMORY, message,
                                 RegisterMemoryMessage::Size(), false);
        FatalAssert(rs.IsOK(), LOG_TAG_COMM_LAYER,
                    "Failed to build REGISTER_MEMORY message: %s",
                    rs.Msg());
        RegisterMemoryMessage* register_msg =
            reinterpret_cast<RegisterMemoryMessage*>(message.GetMessageBuffer());
        CHECK_NOT_NULLPTR(register_msg, LOG_TAG_COMM_LAYER);
        FatalAssert(register_msg->type == MessageType::REGISTER_MEMORY,
                    LOG_TAG_COMM_LAYER,
                    "Message type mismatch in RegisterMemoryMessage: expected %u, got %u",
                    static_cast<uint8_t>(MessageType::REGISTER_MEMORY),
                    static_cast<uint8_t>(register_msg->type));
        register_msg->addr = reinterpret_cast<uintptr_t>(memory_pool);
        register_msg->length = pool_size;
        register_msg->rkey = rkey;

        rs = SendMessage(message, true, false);
        FatalAssert(rs.IsOK(), LOG_TAG_COMM_LAYER,
                    "Failed to send REGISTER_MEMORY message: %s",
                    rs.Msg());
#else
        uint8_t target_node_id = MEMORY_NODE_ID;
        CommLayerMessage message;
        rs = BuildRequestMessage(target_node_id,
                                 MessageType::REGISTER_MEMORY, message,
                                 RegisterMemoryMessage::Size(), false);
        FatalAssert(rs.IsOK(), LOG_TAG_COMM_LAYER,
                    "Failed to build REGISTER_MEMORY message: %s",
                    rs.Msg());
        RegisterMemoryMessage* register_msg =
            reinterpret_cast<RegisterMemoryMessage*>(message.GetMessageBuffer());
        CHECK_NOT_NULLPTR(register_msg, LOG_TAG_COMM_LAYER);
        FatalAssert(register_msg->type == MessageType::REGISTER_MEMORY,
                    LOG_TAG_COMM_LAYER,
                    "Message type mismatch in RegisterMemoryMessage: expected %u, got %u",
                    static_cast<uint8_t>(MessageType::REGISTER_MEMORY),
                    static_cast<uint8_t>(register_msg->type));
        register_msg->addr = reinterpret_cast<uintptr_t>(memory_pool);
        register_msg->length = pool_size;
        register_msg->rkey = rkey;

        rs = SendMessage(message, true, false);
        FatalAssert(rs.IsOK(), LOG_TAG_COMM_LAYER,
                    "Failed to send REGISTER_MEMORY message: %s",
                    rs.Msg());

        std::vector<CommLayerMessage> general_messages;
        std::unordered_map<VectorID, std::vector<CommLayerMessage>, VectorIDHash> cluster_based_messages;
        bool registered = false;
        while (!registered) {
            rs = CheckReceivedMessagesFromNode(target_node_id, general_messages, cluster_based_messages, false);
            FatalAssert(rs.IsOK(), LOG_TAG_COMM_LAYER,
                        "Failed to check received messages from memory node: %s",
                        rs.Msg());
            FatalAssert(cluster_based_messages.empty(), LOG_TAG_COMM_LAYER,
                        "Received unexpected cluster-based messages from memory node during memory registration");
            FatalAssert(general_messages.size() <= 1, LOG_TAG_COMM_LAYER,
                        "Received multiple general messages from memory node during memory registration");
            if (general_messages.empty()) {
                continue;
            }
            // Process the general message to check if registration is confirmed
            for (const auto& msg : general_messages) {
                FatalAssert(msg.GetMessageType() == MessageType::REGISTER_MEMORY,
                            LOG_TAG_COMM_LAYER,
                            "Unexpected message type %u from memory node during memory registration",
                            static_cast<uint8_t>(msg.GetMessageType()));
                RegisterMemoryMessage* register_msg =
                    reinterpret_cast<RegisterMemoryMessage*>(msg.GetMessageBuffer());
                CHECK_NOT_NULLPTR(register_msg, LOG_TAG_COMM_LAYER);
                FatalAssert(register_msg->type == MessageType::REGISTER_MEMORY,
                            LOG_TAG_COMM_LAYER,
                            "Message type mismatch in RegisterMemoryMessage: expected %u, got %u",
                            static_cast<uint8_t>(MessageType::REGISTER_MEMORY),
                            static_cast<uint8_t>(register_msg->type));
                registered = true;
                rs = rdma_manager->RegisterRemoteMemory(target_node_id, register_msg->addr,
                                                        register_msg->length,
                                                        register_msg->rkey);
                FatalAssert(rs.IsOK(), LOG_TAG_COMM_LAYER,
                            "Failed to register remote memory from memory node at addr=0x%lx with length=%zu: %s",
                            register_msg->addr, register_msg->length, rs.Msg());
            }
            rs = FreeMessages(general_messages, cluster_based_messages);
            FatalAssert(rs.IsOK(), LOG_TAG_COMM_LAYER,
                        "Failed to free messages after memory registration from memory node: %s",
                        rs.Msg());
        }
#endif
        /* todo: have a beeter way to handle this! For now sleep for a while to make sure everyone is ready */
        sleep(30);
        state = CommLayerState::RUNNING;
        state_lock.Unlock();
    }
};

};

#endif