#ifndef DIVFTREE_BUFFER_H_
#define DIVFTREE_BUFFER_H_

#include "interface/divftree.h"
#include "utils/memory_pool.h"

#include <memory>
#include <atomic>
#include <unordered_map>

namespace divftree {

struct BufferVectorEntry;
struct BufferVertexEntry;
class BufferManager;

struct ReleaseBufferEntryFlags {
    uint8_t notifyAll : 1;
    uint8_t stablize : 1;
    uint8_t unused : 6;

    ReleaseBufferEntryFlags(bool notify_all, bool stablize_state) :
        notifyAll{notify_all ? (uint8_t)1 : (uint8_t)0}, stablize{stablize_state ? (uint8_t)1 : (uint8_t)0},
        unused{(uint8_t)0} {}
};

enum BufferVertexEntryState : uint8_t {
    CLUSTER_CREATED = 0,
    CLUSTER_STABLE = 1,
    CLUSTER_FULL = 2,
    CLUSTER_DELETE_IN_PROGRESS = 3,
    /* this means that currentVersion is deleted but it doesn't mean that we cannot reador unpin older other versions */
    CLUSTER_DELETED = 4
};

/*
 * This needs to be prtected byt the header lock because we do not have 16Bytes FAA. As a result, if we
 * if we want to do this without locking, we have to use 16Byte CAS which causes a lot of contention on the
 * pin.
 */
struct ClusterPtr {
    std::atomic<uint64_t> pin;
    DIVFTreeVertexInterface* clusterPtr;

    ClusterPtr(uint64_t p, DIVFTreeVertexInterface* ptr) : pin{p}, clusterPtr{ptr} {}
    ClusterPtr() : pin{0}, clusterPtr{nullptr} {}
    ~ClusterPtr() = default;

    /* has to be protected by a lock! */
    ClusterPtr(const ClusterPtr& other) {
        pin.store(other.pin.load(std::memory_order_relaxed), std::memory_order_relaxed);
        clusterPtr = other.clusterPtr;
    }

    /* has to be protected by a lock! */
    ClusterPtr& operator=(const ClusterPtr& other) {
        pin.store(other.pin.load(std::memory_order_relaxed), std::memory_order_relaxed);
        clusterPtr = other.clusterPtr;
        return *this;
    }
};

struct VersionedClusterPtr {
    /*
     * 1) If currentVersion of a cluster is this version then it is pinned once
     * 2) If this cluster is the root it is pinned once(since root should only have one
     *    version the previous condition also applies so it is pinned twice in total)
     * 3) If a vertex contains this version of this cluster, this cluster is pinned
     * 4) For every vector in this cluster that their immidiate VectorLocation in the directory is this version
     *    of the cluster, this version is pinned.
     */
    std::atomic<uint64_t> versionPin;
    ClusterPtr clusterPtr;

    VersionedClusterPtr(uint64_t vp, uint64_t p, DIVFTreeVertexInterface* ptr) :
                        versionPin{vp}, clusterPtr(p, ptr) {}
    VersionedClusterPtr() : versionPin{0}, clusterPtr(0, nullptr) {}
    ~VersionedClusterPtr() = default;

    /* has to be protected by a lock! */
    VersionedClusterPtr(const VersionedClusterPtr& other) {
        versionPin.store(other.versionPin.load(std::memory_order_relaxed), std::memory_order_relaxed);
        clusterPtr = other.clusterPtr;
    }

    /* has to be protected by a lock! */
    VersionedClusterPtr& operator=(const VersionedClusterPtr& other) {
        versionPin.store(other.versionPin.load(std::memory_order_relaxed), std::memory_order_relaxed);
        clusterPtr = other.clusterPtr;
        return *this;
    }
};

union alignas(16) VectorLocation {
    struct Detail {
        VectorID containerId;
        Version containerVersion;
        uint16_t entryOffset;

        constexpr Detail() = default;
        constexpr Detail(VectorID cid, Version cv, uint16_t eo) :
            containerId(cid), containerVersion(cv), entryOffset(eo) {}
    } detail;
    __int128_t raw;

    inline constexpr VectorLocation& operator=(const VectorLocation& other) {
        raw = other.raw;
        return *this;
    }

    inline bool operator==(const VectorLocation& other) {
        FatalAssert((raw == other.raw) == (detail.containerId == other.detail.containerId &&
                                        detail.containerVersion == other.detail.containerVersion &&
                                        detail.entryOffset == other.detail.entryOffset),
                    LOG_TAG_BASIC, "VectorLocation comparison operator inconsistency");
        return raw == other.raw;
    }

    inline bool operator!=(const VectorLocation& other) {
        FatalAssert((raw == other.raw) == (detail.containerId == other.detail.containerId &&
                                        detail.containerVersion == other.detail.containerVersion &&
                                        detail.entryOffset == other.detail.entryOffset),
                    LOG_TAG_BASIC, "VectorLocation comparison operator inconsistency");
        return raw != other.raw;
    }

    inline String ToString() const {
        return String("{containerId=" VECTORID_LOG_FMT ", containerVersion=%u, offset=%hu}",
                      VECTORID_LOG(detail.containerId), detail.containerVersion, detail.entryOffset);
    }

    constexpr VectorLocation() : raw(0) {}
    constexpr VectorLocation(__int128_t other) : raw(other) {}
    constexpr VectorLocation(const VectorLocation& other) : raw(other.raw) {}
    constexpr VectorLocation(VectorLocation&& other) : raw(other.raw) {}

    constexpr VectorLocation(VectorID cid, Version cv, uint16_t eo) {
        raw = 0;
        detail.containerId = cid;
        detail.containerVersion = cv;
        detail.entryOffset = eo;
    }
};

constexpr VectorLocation INVALID_VECTOR_LOCATION = VectorLocation(INVALID_VECTOR_ID, 0, UINT16_MAX);

union alignas(16) AtomicVectorLocation {
    struct Detail {
        std::atomic<RawVectorID> containerId;
        std::atomic<RawVersion> containerVersion;
        std::atomic<uint16_t> entryOffset;
    } detail;
    atomic_data128 raw;

    constexpr AtomicVectorLocation() : raw{INVALID_VECTOR_LOCATION.raw} {}

    constexpr AtomicVectorLocation(const VectorLocation& other) : raw{other.raw} {}

    AtomicVectorLocation(const AtomicVectorLocation& other, bool needAtomic = true,
                         bool relaxed = false) : raw{needAtomic ? 0 : other.raw.raw} {
        if (needAtomic) {
            atomic_load128(&other.raw, &raw, relaxed);
        }
    }

    void Store(VectorLocation location, bool needAtomic = true, bool relaxed = false) {
        if (!needAtomic) {
            raw.raw = location.raw;
            return;
        }
        atomic_data128 data(location.raw);
        atomic_store128(&raw, &data, relaxed);
    }

    VectorLocation Load(bool needAtomic = true, bool relaxed = false) {
        if (!needAtomic) {
            return VectorLocation(raw.raw);
        }
        atomic_data128 data;
        atomic_load128(&raw, &data, relaxed);
        return VectorLocation(data.raw);
    }
};

struct alignas(16) BufferVectorEntry {
    const VectorID selfId;
    AtomicVectorLocation location;

    /*
     * The caller has to handle pinVersion and UnpinVersion for the container clusters.
     * Can only use location.Store if:
     *    1) if is vertex: parent is locked in X mode | parent is locked in S mode and this entry is locked in any mode
     *    2) if is vector: parent is locked in X mode |
     *                     parent is locked in S mode and this thread has changed this vector's state to Migrated
     * The atomicity is only meant to synchronize reads with writes and not writes with writes!!
     */
    BufferVectorEntry(VectorID id);
    ~BufferVectorEntry();

    static void* operator new(std::size_t size);
    static void operator delete(void* ptr) noexcept;

    /*
     * If self is a centroid, it is recomended to use the BufferVertex function!
     *  Also, if this is a centroid, it should be locked in X mode.
     */
    BufferVertexEntry* ReadParentEntry(VectorLocation& currentLocation);
    DIVFTreeVertexInterface* ReadAndPinParent(VectorLocation& currentLocation);
};

struct alignas(16) BufferVertexEntry {
    BufferVectorEntry centroidMeta;
    std::atomic<BufferVertexEntryState> state;
    SXSpinLock headerLock;
    SXLock clusterLock;
    CondVar condVar;
    Version currentVersion;
    uint64_t nextVersionPin;
    SANITY_CHECK(
        Version sanity_nextVersion;
    );

    std::atomic<bool> hasCompactionTask;
    std::atomic<bool> hasMergeTask;

    std::unordered_set<VectorID, VectorIDHash> migrationTasks;
    /*
     * liveVersions can only be updated if parent is locked(any mode),
     * self is locked in X mode, and headerLock is locked in X mode
     */
    std::unordered_map<Version, VersionedClusterPtr, VersionHash> liveVersions;

    BufferVertexEntry(DIVFTreeVertexInterface* cluster, VectorID id, uint64_t initialPin = 1);
    ~BufferVertexEntry();

    static void* operator new(std::size_t size);
    static void operator delete(void* ptr) noexcept;

    inline BufferVertexEntry* ReadParentEntry(VectorLocation& currentLocation);
    DIVFTreeVertexInterface* ReadLatestVersion(bool pinCluster = true, bool needsHeaderLock = false);
    // DIVFTreeVertexInterface* MVCCReadAndPinCluster(Version version, bool headerLocked=false);

    RetStatus UpgradeAccessToExclusive(BufferVertexEntryState& targetState, bool unlockOnFail = false);
    void DowngradeAccessToShared();

    /* parentNode should also be locked in shared or exclusive mode and self should be locked in X mode */
    void UpdateClusterPtr(DIVFTreeVertexInterface* newCluster);
    /* parentNode should also be locked in shared or exclusive mode and self should be locked in X mode */
    void UpdateClusterPtr(DIVFTreeVertexInterface* newCluster, Version newVersion, bool isRoot = false);
    void InitiateDelete();

    void UnpinVersion(Version version, uint64_t unpinCnt = 1, bool headerLocked=false);
    void PinVersion(Version version, bool headerLocked=false);

    String ToString();
};

/* todo: add tostring for buffer entries */
class BufferManager {
// TODO: reuse deleted IDs
public:
    BufferManager(uint64_t internalSize, uint64_t leafSize, uint64_t pool_size_bytes);
    ~BufferManager();

    /*
     * vertexMetaDataSize should be sizeof(Vertex without the Cluster Header -> maybe we should use a pointer?)
     *
     * will return the rootEntry in the INVALID state and locked in exclusive mode!
     */
    inline static BufferVertexEntry* Init(uint64_t vertexMetaDataSize,
                                          uint16_t leaf_blk_size, uint16_t internal_blk_size,
                                          uint16_t leaf_cap, uint16_t internal_cap, uint16_t dim,
                                          uint64_t pool_size_gb);

    /*
     * Note: No thread should be calling any functions from the bufferMgr the moment
     * shutdown is called or some resources may not be cleaned properly
     */
    inline static void Shutdown();

    inline static BufferManager* GetInstance();

    DIVFTreeVertexInterface* AllocateMemoryForVertex(uint8_t level);
    void Recycle(DIVFTreeVertexInterface* memory);

    void UpdateRoot(VectorID newRootId, Version newRootVersion, BufferVertexEntry* oldRootEntry);
    BufferVertexEntry* CreateNewRootEntry(VectorID expRootId);
    void BatchCreateBufferEntry(uint16_t num_entries, uint8_t level, BufferVertexEntry** entries,
                                DIVFTreeVertexInterface** clusters = nullptr, VectorID* ids = nullptr,
                                Version* versions = nullptr);
    void BatchCreateVectorEntry(size_t num_entries, BufferVectorEntry** entries, bool create_update_handle = false);

    VectorID GetCurrentRootId() const;
    VectorID GetCurrentRootIdAndVersion(Version& version);

    inline BufferVectorEntry* GetVectorEntry(VectorID id);
    inline void PinVertexVersion(VectorID vertexId, Version version);
    inline void UnpinVertexVersion(VectorID vertexId, Version version);

    inline VectorLocation LoadCurrentVectorLocation(VectorID vectorId, bool return_invalid_if_created = true);

    /*
     * The caller has to handle pinVersion and UnpinVersion for the container clusters.
     * Can only be called if vectorId:
     *    1) if is vertex: parent is locked in X mode | parent is locked in S mode and this entry is locked in any mode
     *    2) if is vector: parent is locked in X mode |
     *                     parent is locked in S mode and this thread has changed this vector's state to Migrated
     * The atomicity is only meant to synchronize reads with writes and not writes with writes!!
     */
    inline void UpdateVectorLocation(VectorID vectorId, VectorLocation newLocation, bool pinNewUnpinOld = true);
    // inline void UpdateVectorLocationOffset(VectorID vectorId, uint16_t newOffset);

    RetStatus ReadAndPinVertex(VectorID vertexId, Version version, DIVFTreeVertexInterface*& vertex,
                               bool* outdated = nullptr);
    DIVFTreeVertexInterface* ReadAndPinRoot();

    /*
     * If we need to lock multiple clusters, then lower level clusters always take priority and
     * are locked first(i.e root is locked the last). If we want to lock to clusters in the same level,
     * the cluster with lower id.val should be locked first.
     */
    BufferVertexEntry* ReadBufferEntry(VectorID vertexId, LockMode mode, bool* blocked = nullptr);
    BufferVertexEntry* TryReadBufferEntry(VectorID vertexId, LockMode mode);
    inline void ReleaseEntriesIfNotNull(BufferVertexEntry** entries, uint16_t num_entries,
                                        ReleaseBufferEntryFlags flags);
    inline void ReleaseBufferEntryIfNotNull(BufferVertexEntry* entry, ReleaseBufferEntryFlags flags);
    void ReleaseBufferEntry(BufferVertexEntry* entry, ReleaseBufferEntryFlags flags);

    uint64_t GetHeight() const;

    String ToString();

    RetStatus CreateUpdateHandle(VectorID target);
    void SignalUpdateHandleIfNeeded(VectorID target);
    RetStatus WaitForUpdateToGoThrough(VectorID target);
    RetStatus CheckIfUpdateHasGoneThrough(VectorID target, bool& updated);

    bool AddMigrationTaskIfNotExists(VectorID first, VectorID second, BufferVertexEntry* firstEntry = nullptr);
    void RemoveMigrationTask(VectorID first, VectorID second, BufferVertexEntry* firstEntry = nullptr);
    bool AddCompactionTaskIfNotExists(VectorID id, BufferVertexEntry* entry = nullptr);
    void RemoveCompactionTask(VectorID id, BufferVertexEntry* entry = nullptr);
    bool AddMergeTaskIfNotExists(VectorID id, BufferVertexEntry* entry = nullptr);
    void RemoveMergeTask(VectorID id, BufferVertexEntry* entry = nullptr);

    VectorID GetRandomCentroidIdAtLayer(uint8_t level, VectorID exclude = INVALID_VECTOR_ID,
                                        bool need_lock = true, uint64_t* num_retries = nullptr);
    std::pair<VectorID, VectorID> GetTwoRandomCentroidIdAtLayer(uint8_t level, bool need_lock = true);
    VectorID GetRandomCentroidIdAtNonRootLayer(VectorID exclude = INVALID_VECTOR_ID);
    std::pair<VectorID, VectorID> GetTwoRandomCentroidIdAtNonRootLayer();

protected:
    const uint64_t internalVertexSize;
    const uint64_t leafVertexSize;
    LocalMemoryPool memoryPool;
    SXSpinLock bufferMgrLock;
    std::atomic<RawVectorID> currentRootId;
    std::vector<BufferVectorEntry*> vectorDirectory;
    std::vector<BufferVertexEntry*> clusterDirectory[MAX_TREE_HIGHT];
    SXSpinLock handleLock;
    std::unordered_map<VectorID, std::atomic<bool>*, VectorIDHash> handles;

    inline static BufferManager *bufferMgrInstance = nullptr;

    inline BufferVertexEntry* GetVertexEntry(VectorID id, bool needLock = true);
    inline BufferVertexEntry* GetRootEntry(bool needLock = true);

TESTABLE;
};

};

#include "buffer_impl.h"

#endif