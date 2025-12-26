#ifndef DIVFTREE_BUFFER_H_
#define DIVFTREE_BUFFER_H_

#include "disaggregated/compute_node/divftree.h"
#include "disaggregated/comm_layer.h"
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
    CACHED = 2
};

struct VectorAddress {
    VectorID containerId;
    Version containerVersion;
    ClusterSizeType offset;
};

/*
 * This needs to be prtected byt the header lock because we do not have 16Bytes FAA. As a result, if we
 * if we want to do this without locking, we have to use 16Byte CAS which causes a lot of contention on the
 * pin.
 */
struct VertexData {
    std::atomic<ClusterCacheState> cacheState;
    std::atomic<uint64_t> pin;
    uintptr_t remote_addr;
    DIVFTreeVertex vertex;
    /* a map from offset in the current cluster to new address for vertices
       that are migrating but their insertion has not come yet
       todo: check and if the number of such cases are not that much, use a cache efficent linked list
       instead to save memory
       Note: Should be protected by the cluster lock? */
    /* todo: use a better hash than default? */
    std::unordered_map<ClusterSizeType, VectorAddress>* outofOrderUpdates;
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
    uint64_t currentVersionPin; /* the pin of the cluster it self should be equal or greater than this is a search might pin that version(instead of latest) */
    bool is_root; /* will be set to false when we send unpin -> if set to true it doesn't mean that this is necessarily the current root */

    std::unordered_map<Version, VertexData, VersionHash> liveVersions;
    std::unordered_set<VectorID, VectorIDHash> migrationTasks;

    BufferVertexEntry(VectorID id);
    ~BufferVertexEntry();

    static void* operator new(std::size_t size);
    static void operator delete(void* ptr) noexcept;

    DIVFTreeVertex& ReadLatestVersion(bool pinCluster = true, bool needsHeaderLock = false);
    DIVFTreeVertex& Read(Version version, bool pinCluster = false);
    void Unpin(Version version);
    void Unpin(); /* unpins the current version */

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

    RetStatus PrefetchAndPinVerticesForSearch(SortedList<ANNVectorInfo, SimilarityComparator>* vertices,
                                              SearchTaskGenerator& taskGen);
    RetStatus PrefetchAndPinVertex(VectorID vertexId);
    RetStatus ReadVertexIfAvailable(VectorID vertexId, Version version, BufferVertexEntry*& vertex,
                                    bool* outdated = nullptr);
    RetStatus ReadVertexIfAvailable(VectorID vertexId, BufferVertexEntry*& vertex);

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