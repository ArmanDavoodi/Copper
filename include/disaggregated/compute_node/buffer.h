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
    CACHED_HOT = 2,
    CACHED_COOLING = 3,
    DELETED = 4
};

struct VectorAddress {
    VectorID containerId;
    Version containerVersion;
    ClusterSizeType offset;
};

enum UpdateType : uint8_t {
    UPDATE_TYPE_MIGRATION,
    UPDATE_TYPE_OUTDATED
};

struct UpdateInfo {
    UpdateType type;
    union {
        VectorAddress address; /* for migration */
        ClusterSizeType insertion_batch_offset; /* for outdated */
    };
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
    std::unordered_map<ClusterSizeType, UpdateInfo>* outofOrderUpdates;
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


/*
    Insert AtomicBatch (for split -> cause another vector to become outdated):
        step 1:
        lock(S) header

        step 2:
        if (version not available):
            step 2.1:
            upgrade header to X
            if version is newer than currentVersion:
                change currentVersion to new version
                deduct currentVersionPin from the old currentVersion pin if exists
                add new version with pin = currentVersionPin to live versions
                deduct currentVersionPin from old version pin if exists
            else:
                add new version with pin = 0 to live versions
            set its size to what was given in the message
            if pin is not 0:
                set state to REMOTE_READ_IN_PROGRESS
                issue RDMA read for the whole cluster
            else :
                set state to UNCACHED
            unlock header
            return

        step 3:
        v <- read version(no pin needed)
        increment the pin
        unlock header
        // since these are updates we are not accessing them so we should not make them hot!

        step 4
        set batch state to invalid
        write vector data + id + version + batch meta
        change vector states to valid
        if ( there are out of order updates ):
            for migrations -> save the offsets in a list
            for outdated vectors -> save the offset in another list

        step 5:
        p_off <- go to the vector that needs to be outdated
        change vector state to outdated
        if (vector state is invalid):
            Lock(X) clusterLock
            if outOforder == nullptr -> creat it(do not pin vertex)
            add p_off.offset to outOfOrderUpdates with type outdated
            if (state is outdated_invalid):
                Unlock clusterLock
                unpin vertex
                return
            remove p_off.offset from outOfOrderUpdates
            if outOforder.size == 0:
                delete outOfOrderUpdates
            Unlock clusterLock

        step 6:
        p_batch <- get its batch info
        if the p_batch is invalid:
            Lock(X) clusterLock
            if p_batch.state is still invalid:
                if outOforder == nullptr -> creat it(do not pin vertex)
                add <p_batch.offset, c_batch.offset> to outOfOrderUpdates with type outdated
                Unlock clusterLock
                unpin vertex
                return
            Unlock clusterLock

        set c_batch state to valid
        Lock(X) clusterLock
        if outOforder == nullptr:
            Unlock clusterLock
            unpin vertex
            return
        for each e in migration list:
            if outOfOrder does not contain e.offset:
                continue
            save the target address in a temp variable
        for each e in outdated list:
            if outOfOrder does not contain e.offset:
                continue
            get its batch offset, set it to valid
            check if there are other batches that are chained to it or its elements and set them all to valid
        check if anything depends on the validity of c_batch and if yes:
            validate them and their dependents recursively
        if outOforder is empty:
            delete outOfOrderUpdates
        Unlock clusterLock

        for migrations -> Insert them to their respective clusters if they are cached
        unpin vertex
        return



    Insert Vector (for simple insert to leaves or for migration):
        step 1:
        lock(S) header

        if (version not available):
            step 2.1:
            upgrade header to X
            if version is newer than currentVersion:
                change currentVersion to new version
                deduct currentVersionPin from the old currentVersion pin if exists
                add new version with pin = currentVersionPin to live versions
                deduct currentVersionPin from old version pin if exists
            else:
                add new version with pin = 0 to live versions
            set its size to what was given in the message
            if pin is not 0:
                set state to REMOTE_READ_IN_PROGRESS
                issue RDMA read for the whole cluster
            else :
                set state to UNCACHED
            unlock header
            return

        step 3:
        v <- read version(no pin needed)
        increment the pin
        unlock header
        // since these are updates we are not accessing them so we should not make them hot!

        step 4
        write vector data + id + version + batch meta(should be valid with size 1)
        change vector states to valid
        if ( there are out of order updates ):
            for migrations -> save the offsets in a list
            for outdated vectors -> save the offset in another list

        step 5:
        Lock(X) clusterLock
        if outOforder == nullptr:
            Unlock clusterLock
            unpin vertex
            return

        for each e in migration list:
            if outOfOrder does not contain e.offset:
                continue
            save the target address in a temp variable
        for each e in outdated list:
            if outOfOrder does not contain e.offset:
                continue
            get its batch offset, set it to valid
            check if there are other batches that are chained to it or its elements and set them all to valid
        if outOforder is empty:
            delete outOfOrderUpdates
        Unlock clusterLock

        for migrations -> Insert them to their respective clusters if they are cached
        unpin vertex
        return

    Migrate Vector:
        step 1:
        lock(S) target header

        step 2:
        if (target version not available):
            step 2.1:
            upgrade header to X
            if version is newer than currentVersion:
                change currentVersion to new version
                deduct currentVersionPin from the old currentVersion pin if exists
                add new version with pin = currentVersionPin to live versions
                deduct currentVersionPin from old version pin if exists
            else:
                add new version with pin = 0 to live versions
            set its size to what was given in the message
            if pin is not 0:
                set state to REMOTE_READ_IN_PROGRESS
                issue RDMA read for the whole cluster
            else :
                set state to UNCACHED
            unlock target header
        else:
            tv <- read version(no pin needed)
            increment the pin
            unlock target header
            // since these are updates we are not accessing them so we should not make them hot!

        lock(S) src header
        if (src version not available):
            step 2.2:
            upgrade header to X
            if version is newer than currentVersion:
                change currentVersion to new version
                deduct currentVersionPin from the old currentVersion pin if exists
                add new version with pin = currentVersionPin to live versions
                deduct currentVersionPin from old version pin if exists
            else:
                add new version with pin = 0 to live versions
            set its size to what was given in the message
            if pin is not 0:
                set state to REMOTE_READ_IN_PROGRESS
                issue RDMA read for the whole cluster
            else :
                set state to UNCACHED
            unlock src header
        else:
            sv <- read version(no pin needed)
            increment the pin
            unlock src header
            // since these are updates we are not accessing them so we should not make them hot!

        step 3:
        if (neither sv nor tv is in CACHED_HOT state):
            return

        step 4:
        if sv is cached:
            read offsets to be migrated
            change all their state to migrated

            if tv is cached:
                lock(X) src clusterLock
                if outOforder == nullptr -> creat it(do not pin vertex) only if needed
                for all with invalid state:
                    if state is now valid ignore
                    add their offsets to outOfOrderUpdates with type migration
                unlock src clusterLock

                put all with valid state in a list with their addresses -> during previous steps

        step 5:
        if tv is cached:
            if sv was cached:
                Insert all of the migrated vectors using the list created before
                unpin both sv and tv
                return
            else:
                assert all of them should have invalid states(it can be outdated invalid etc)
                make their batch states valid
                set batch sizes to 1
                use request message data to get migration offset and size and issue RDMA read to read
                    only the vector data and not the metadata!
                we should keep tv pinned and let the RDMA read handler unpin it
                return
        if sv is cached:
            unpin sv
        return

    Delete Vector
        step 1:
        lock(S) header

        step 2:
        if (version not available):
            step 2.1:
            upgrade header to X
            if version is newer than currentVersion:
                change currentVersion to new version
                deduct currentVersionPin from the old currentVersion pin if exists
                add new version with pin = currentVersionPin to live versions
                deduct currentVersionPin from old version pin if exists
            else:
                add new version with pin = 0 to live versions
            if pin is not 0:
                set state to REMOTE_READ_IN_PROGRESS
                issue RDMA read for the whole cluster
            else :
                set state to UNCACHED
            unlock header
            return

        step 3:
        v <- read version(no pin needed)
        increment the pin
        unlock header
        // since these are updates we are not accessing them so we should not make them hot!

        step 4
        p_off <- go to the vector that needs to be deleted
        change vector state to deleted
        if (vector state is invalid):
            Lock(X) clusterLock
            if state is still invalid:
                if outOforder == nullptr -> creat it(do not pin vertex)
                add p_off.offset to outOfOrderUpdates with type deleted
                Unlock clusterLock
                unpin vertex
                return
            Unlock clusterLock

        unpin vertex
        read vector id and version and change the buffer state for that cluster
        return

    ------

    these should only change buffer entry data
    Vertex Split
    Vertex Merge
    Vertex Pruned
    Vertex Compacted

*/
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
    RetStatus UnpinVertex(VectorID vertexId, Version version);
    // RetStatus UnpinVertex(VectorID vertexId);
    RetStatus ReadVertexIfAvailable(VectorID vertexId, Version version, DIVFTreeVertex*& vertex,
                                    bool* outdated = nullptr);
    // RetStatus ReadVertexIfAvailable(VectorID vertexId, DIVFTreeVertex*& vertex);

    BufferVertexEntry* GetBufferEntry(VectorID vertexId, LockMode mode);

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