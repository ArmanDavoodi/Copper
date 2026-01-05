#ifndef DIVFTREE_BUFFER_H_
#define DIVFTREE_BUFFER_H_

#include "disaggregated/compute_node/divftree.h"
#include "utils/concurrent_datastructures.h"

#include <memory>
#include <atomic>
#include <unordered_map>

namespace divftree {

struct BufferVertexEntry;
class BufferManager;

enum class ClusterCacheState : uint8_t {
    UNCACHED = 0,
    REMOTE_READ_IN_PROGRESS = 1,
    CACHED_HOT = 2,
    CACHED_COOLING = 3,
    DELETED = 4
};

struct VectorAddress {
    VectorID containerId;
    Version containerVersion;
    ClusterSizeType offset;
};

enum OutOfOrderUpdateType : uint8_t {
    OUT_OF_ORDER_MIGRATION,
    OUT_OF_ORDER_OUTDATED,
    OUT_OF_ORDER_DELETION
};

struct OutOfOrderUpdateInfo {
    OutOfOrderUpdateType type;
    union {
        VectorAddress address; /* for migration */
        ClusterSizeType insertion_batch_offset; /* for outdated */
    };

    OutOfOrderUpdateInfo(VectorID dest_id, Version dest_version, ClusterSizeType dest_insert_offset) :
        type(OUT_OF_ORDER_MIGRATION), address{dest_id, dest_version, dest_insert_offset} {}
    OutOfOrderUpdateInfo(VectorAddress address) : type(OUT_OF_ORDER_MIGRATION), address(address) {}
    OutOfOrderUpdateInfo(ClusterSizeType offset) : type(OUT_OF_ORDER_OUTDATED), insertion_batch_offset(offset) {}
    OutOfOrderUpdateInfo() : type(OUT_OF_ORDER_DELETION) {}
};

enum class CacheUpdateType : uint8_t {
    BATCH_INSERTION,
    ATOMIC_BATCH_INSERTION,
    MIGRATION,
    OUT_OF_ORDER_MIGRATION,
    INSERTION,
    DELETION
};

struct CacheUpdateInfo {
    CacheUpdateType type;
    union {
        struct {
            ClusterSizeType insert_offset;
            ConstVectorBatch batch;
        } batch_insertion;
        struct {
            ClusterSizeType insert_offset;
            ConstVectorBatch batch;
            ClusterSizeType mark_outdated_offset;
        } atomic_batch_insertion;
        struct {
            bool at_src;
            VectorID id;
            Version version;
            ClusterSizeType num_vectors;
            const ClusterSizeType* offsets;
            ClusterSizeType dest_insert_offset;
        } migration;
        struct {
            VectorID src_id;
            Version src_version;
            const std::vector<std::pair<ClusterSizeType, ClusterSizeType>>* offset_pairs;
        } out_of_order_migration;
        struct {
            ConstVectorBatch batch;
            ClusterSizeType* insert_offsets;
        } insertion;
        struct {
            ClusterSizeType num_vectors;
            const ClusterSizeType* offsets;
        } deletion;
    };

    CacheUpdateInfo(ClusterSizeType insert_offset, const ConstVectorBatch& batch) :
        type(CacheUpdateType::BATCH_INSERTION) {
        batch_insertion.insert_offset = insert_offset;
        batch_insertion.batch = batch;
    }

    CacheUpdateInfo(ClusterSizeType insert_offset, const ConstVectorBatch& batch,
                    ClusterSizeType mark_outdated_offset) :
        type(CacheUpdateType::ATOMIC_BATCH_INSERTION) {
        atomic_batch_insertion.insert_offset = insert_offset;
        atomic_batch_insertion.batch = batch;
        atomic_batch_insertion.mark_outdated_offset = mark_outdated_offset;
    }

    CacheUpdateInfo(bool at_src, VectorID id, Version version,
                    ClusterSizeType num_vectors, const ClusterSizeType* offsets,
                    ClusterSizeType dest_insert_offset) :
        type(CacheUpdateType::MIGRATION) {
        migration.at_src = at_src;
        migration.id = id;
        migration.version = version;
        migration.num_vectors = num_vectors;
        migration.offsets = offsets;
        migration.dest_insert_offset = dest_insert_offset;
    }

    CacheUpdateInfo(VectorID src_id, Version src_version,
                    const std::vector<std::pair<ClusterSizeType, ClusterSizeType>>* offset_pairs) :
        type(CacheUpdateType::OUT_OF_ORDER_MIGRATION) {
        out_of_order_migration.src_id = src_id;
        out_of_order_migration.src_version = src_version;
        out_of_order_migration.offset_pairs = offset_pairs;
    }

    CacheUpdateInfo(const ConstVectorBatch& batch, ClusterSizeType* insert_offsets) :
        type(CacheUpdateType::INSERTION) {
        insertion.batch = batch;
        insertion.insert_offsets = insert_offsets;
    }

    CacheUpdateInfo(ClusterSizeType num_vectors, const ClusterSizeType* offsets) :
        type(CacheUpdateType::DELETION) {
        deletion.num_vectors = num_vectors;
        deletion.offsets = offsets;
    }
};

/*
 * This needs to be prtected byt the header lock because we do not have 16Bytes FAA. As a result, if we
 * if we want to do this without locking, we have to use 16Byte CAS which causes a lot of contention on the
 * pin.
 */
struct VertexData {
    std::atomic<ClusterCacheState> cacheState;
    std::atomic<uint64_t> pin;
    SXSpinLock clusterLock;
    uintptr_t remoteAddr;
    DIVFTreeVertex vertex;
    /* a map from offset in the current cluster to new address for vertices
       that are migrating but their insertion has not come yet
       todo: check and if the number of such cases are not that much, use a cache efficent linked list
       instead to save memory
       Note: Should be protected by the cluster lock? */
    /* todo: use a better hash than default? */
    std::unordered_map<ClusterSizeType, OutOfOrderUpdateInfo>* outofOrderUpdates;
    std::vector<CacheUpdateInfo>* pendingUpdates;
    /* todo we also need a pointer to a list of updates if cache state is remote read in progress */

    // VertexData(uint64_t p) : pin{p} {}
    // VertexData() : pin{0} {}
    // ~VertexData() = default;

    // /* has to be protected by a lock! */
    // VertexData(const VertexData& other) {
    //     pin.store(other.pin.load(std::memory_order_relaxed), std::memory_order_relaxed);
    // }

    // /* has to be protected by a lock! */
    // VertexData& operator=(const VertexData& other) {
    //     pin.store(other.pin.load(std::memory_order_relaxed), std::memory_order_relaxed);
    //     return *this;
    // }
};

struct BufferVertexEntry {
    const VectorID selfId;
    SXSpinLock headerLock;
    Version currentVersion;
    // uint64_t currentVersionPin; /* the pin of the cluster it self should be equal or greater than this is a search might pin that version(instead of latest) */
    bool is_root; /* will be set to false when we send unpin -> if set to true it doesn't mean that this is necessarily the current root */

    std::unordered_map<Version, VertexData, VersionHash> liveVersions;
    std::unordered_set<VectorID, VectorIDHash> migrationTasks;

    BufferVertexEntry(VectorID id);
    ~BufferVertexEntry();

    static void* operator new(std::size_t size);
    static void operator delete(void* ptr) noexcept;

    // DIVFTreeVertex& ReadLatestVersion(bool pinCluster = true, bool needsHeaderLock = false);
    VertexData* ReadVersionIfInCache(Version version, CacheUpdateInfo updateInfo, bool pinCluster = false);
    VertexData* ReadVersionIfInCache(Version version, bool pinCluster = false);
    VertexData* ReadAndLockVersion(Version version, bool pinCluster = false);
    DIVFTreeVertex& Read(Version version, bool pinCluster = false);
    void Unpin(Version version);
    // void Unpin(); /* unpins the current version */

    void AddVersion(Version version, uintptr_t remote_addr, ClusterSizeType size, uint64_t initialPin = 0);
    void AddVersion(Version version, uintptr_t remote_addr, ClusterSizeType size, void* local_cpy,
                    uint64_t initialPin = 0);

    String ToString();
};

/* todo: add tostring for buffer entries */
class BufferManager {
// TODO: reuse deleted IDs
public:
    BufferManager(uint64_t internalSize, uint64_t leafSize);
    ~BufferManager();

    /*
     * vertexMetaDataSize should be sizeof(Vertex without the Cluster Header -> maybe we should use a pointer?)
     *
     * will return the rootEntry in the INVALID state and locked in exclusive mode!
     */
    inline static void Init(uint64_t vertexMetaDataSize,
                            ClusterSizeType leaf_blk_size, ClusterSizeType internal_blk_size,
                            ClusterSizeType leaf_cap, ClusterSizeType internal_cap, uint16_t dim);

    /*
     * Note: No thread should be calling any functions from the bufferMgr the moment
     * shutdown is called or some resources may not be cleaned properly
     */
    inline static void Shutdown();

    inline static BufferManager* GetInstance();

    inline CommLayer* GetCommLayer();

    // void UpdateRoot(VectorID newRootId, Version newRootVersion, BufferVertexEntry* oldRootEntry);
    // BufferVertexEntry* CreateNewRootEntry(VectorID expRootId);
    void BatchCreateBufferEntry(ClusterSizeType num_entries, uint8_t level, BufferVertexEntry** entries,
                                VectorID* ids, Version* versions);

    VectorID GetCurrentRootId() const;
    VectorID GetCurrentRootIdAndVersion(Version& version);

    void ReadVectorsFromRemote(VectorID containerId, Version containerVersion,
                               ClusterSizeType offset, ClusterSizeType num_vectors);
    RetStatus PrefetchAndPinVerticesForSearch(SortedList<ANNVectorInfo, SimilarityComparator>* vertices,
                                              SearchTaskGenerator& taskGen);
    /* Note: we should not be able to read the latest version of a random cluster without pinning a root version
        otherwise, that version may be deleted in memory node while we are reading it!! */
    // RetStatus PrefetchAndPinVertex(VectorID vertexId);
    RetStatus UnpinVertex(VectorID vertexId, Version version);
    // RetStatus UnpinVertex(VectorID vertexId);
    RetStatus ReadVertexIfAvailable(VectorID vertexId, Version version, DIVFTreeVertex*& vertex,
                                    bool* outdated = nullptr);
    // RetStatus ReadVertexIfAvailable(VectorID vertexId, DIVFTreeVertex*& vertex);

    BufferVertexEntry* GetBufferEntry(VectorID vertexId);

    /* should always return true for raw vectors */
    bool Exists(VectorID vertexId, Version version);

    uint64_t GetHeight() const;

    String ToString();

    void AddCompactionTaskIfNotExists(VectorID id, BufferVertexEntry* entry = nullptr);
    bool AddMigrationTaskIfNotExists(VectorID first, VectorID second, BufferVertexEntry* firstEntry = nullptr);
    void RemoveMigrationTask(VectorID first, VectorID second, BufferVertexEntry* firstEntry = nullptr);

    /*
     * if both ids are invalid at first, then two random centroids in the cache will be selected and pinned at the same level
     * if one of them is invalid(the first should be valid and second invalid), then the first will be
     *  prefetched and pinned and a different id at the same level will be selected for the second which is also pinned
     * if both of them are valid, then both will be prefetched and pinned -> they should be at the same level
     */
    void GetCentroidsForMigrationCheck(VectorID& firstId, VectorID& secondId);
    void GetCentroidsForMigrationCheck(uint8_t level, VectorID& firstId, VectorID& secondId);

    std::pair<VectorID, VectorID> GetTwoRandomCentroidIdAtLayer(uint8_t level, bool need_lock = true);
    VectorID GetRandomCentroidIdAtNonRootLayer(VectorID exclude = INVALID_VECTOR_ID);
    std::pair<VectorID, VectorID> GetTwoRandomCentroidIdAtNonRootLayer();

protected:
    const uint64_t internalVertexSize;
    const uint64_t leafVertexSize;
    SXSpinLock bufferMgrLock;
    std::vector<BufferVertexEntry*> clusterDirectory[MAX_TREE_HIGHT];
    ConcurrentHashTable<VectorID, BufferVertexEntry*, VectorIDCMP, VectorIDHash> cached_latest_versions[MAX_TREE_HIGHT];
    ConcurrentHashTable<std::pair<VectorID, Version>, bool, VectorIDVersionPairCMP, VectorIDVersionPairHash>
        cached_cluster_set;

    inline static BufferManager *bufferMgrInstance = nullptr;

    void* AllocateMemoryForCluster(bool is_leaf);

TESTABLE;
};

};

#include "disaggregated/compute_node/buffer_impl.h"

#endif