#ifndef DIVFTREE_BUFFER_IMPL_H_
#define DIVFTREE_BUFFER_IMPL_H_

namespace divftree {

BufferVectorEntry::BufferVectorEntry(VectorID id) : selfId(id), location(INVALID_VECTOR_LOCATION) {
    /* since we have an atomicVectorLocation and it has to be 16byte aligned, the vector entry should be
    16 byte aligned as well */
    FatalAssert(ALIGNED(this, 16), LOG_TAG_BUFFER, "BufferVectorEntry is not alligned!");
#ifdef MEMORY_DEBUG
    DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BUFFER, "%p created buffer vector entry -> ID:" VECTORID_LOG_FMT, this,
            VECTORID_LOG(id));
#endif
}

BufferVectorEntry::~BufferVectorEntry() {
#ifdef MEMORY_DEBUG
    DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BUFFER, "%p destroyed buffer vector entry -> ID:" VECTORID_LOG_FMT, this,
            VECTORID_LOG(selfId));
#endif
}

void* BufferVectorEntry::operator new(std::size_t size) {
    return ::operator new(size, std::align_val_t(16));
}

void BufferVectorEntry::operator delete(void* ptr) noexcept {
    ::operator delete(ptr, std::align_val_t(16));
}

BufferVertexEntry* BufferVectorEntry::ReadParentEntry(VectorLocation& currentLocation) {
    BufferManager* bufferMgr = BufferManager::GetInstance();
    CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_BUFFER);
    currentLocation = INVALID_VECTOR_LOCATION;
    if (selfId == bufferMgr->GetCurrentRootId()) {
        return nullptr;
    }

    BufferVertexEntry* parent = nullptr;
    while (true) {
        currentLocation = INVALID_VECTOR_LOCATION;
        VectorLocation expLocation = location.Load();
        if (expLocation == INVALID_VECTOR_LOCATION) {
            /* this means that we should have become the new root! */
            /* there becomming the new root and change of address are not done atomically so this assertion can fail */
            // FatalAssert(selfId.IsVector() || (selfId == bufferMgr->GetCurrentRootId()), LOG_TAG_BUFFER,
            //             "if location is invalid and we are a centroid we have to be the root!");
            return nullptr;
        }

        parent = bufferMgr->ReadBufferEntry(expLocation.detail.containerId, SX_SHARED);
        currentLocation = location.Load();
        if (currentLocation == INVALID_VECTOR_LOCATION) {
            /* this means that we should have become the new root! */
            if (parent != nullptr) {
                bufferMgr->ReleaseBufferEntry(parent, ReleaseBufferEntryFlags(false, false));
                parent = nullptr;
            }
            // FatalAssert(selfId.IsVector() || (selfId == bufferMgr->GetCurrentRootId()), LOG_TAG_BUFFER,
            //             "if location is invalid and we are a centroid we have to be the root!");
            return nullptr;
        }

        if (currentLocation != expLocation) {
            if ((parent != nullptr) && (currentLocation.detail.containerId == expLocation.detail.containerId) &&
                (currentLocation.detail.containerVersion._split == expLocation.detail.containerVersion._split)) {
                /* This just means that there was a compaction and out offset has changed */
                break;
            }

            if (parent != nullptr) {
                bufferMgr->ReleaseBufferEntry(parent, ReleaseBufferEntryFlags(false, false));
                parent = nullptr;
            }
            continue;
        }

        break;
    }

    CHECK_NOT_NULLPTR(parent, LOG_TAG_BUFFER);
    SANITY_CHECK(
        FatalAssert(parent->centroidMeta.selfId.Level() == selfId.Level() + 1, LOG_TAG_BUFFER,
                    "level mismatch between parent and child!");
        DIVFTreeVertexInterface* parent_vertex = parent->ReadLatestVersion(false);
        CHECK_NOT_NULLPTR(parent_vertex, LOG_TAG_BUFFER);
        FatalAssert(parent_vertex->GetAttributes().version == currentLocation.detail.containerVersion, LOG_TAG_BUFFER,
                    "version mismatch between buffer entry and loaded cluster!");
        bool is_leaf = parent->centroidMeta.selfId.IsLeaf();
        void* meta = parent_vertex->GetCluster().MetaData(currentLocation.detail.entryOffset, false,
                                                            parent_vertex->GetAttributes().block_size,
                                                            parent_vertex->GetAttributes().cap,
                                                            parent_vertex->GetAttributes().index->
                                                            GetAttributes().dimension);
        if (is_leaf) {
            VectorMetaData* vmt = reinterpret_cast<VectorMetaData*>(meta);
            FatalAssert(vmt->id == selfId, LOG_TAG_BUFFER, "Corrupted location data!");
            /* state might change to migrated or invalid if there is a delete or migration task */
            // FatalAssert(vmt->state == VECTOR_STATE_VALID, LOG_TAG_BUFFER, "Corrupted location data!");
        } else {
            CentroidMetaData* vmt = reinterpret_cast<CentroidMetaData*>(meta);
            FatalAssert(vmt->id == selfId, LOG_TAG_BUFFER, "Corrupted location data!");
            /* state might change to migrated or invalid if the child is not locked exclusively */
            // FatalAssert(vmt->state == VECTOR_STATE_VALID, LOG_TAG_BUFFER, "Corrupted location data!");
        }
    )
    return parent;
}

DIVFTreeVertexInterface* BufferVectorEntry::ReadAndPinParent(VectorLocation& currentLocation) {
    BufferManager* bufferMgr = BufferManager::GetInstance();
    CHECK_NOT_NULLPTR(bufferMgr, LOG_TAG_BUFFER);
    currentLocation = INVALID_VECTOR_LOCATION;
    if (selfId == bufferMgr->GetCurrentRootId()) {
        return nullptr;
    }

    DIVFTreeVertexInterface* parent = nullptr;
    while (true) {
        VectorLocation excpectedLocation = location.Load();
        if (excpectedLocation == INVALID_VECTOR_LOCATION) {
            /* this means that we should have become the new root! */
            FatalAssert(selfId.IsVector() || (selfId == bufferMgr->GetCurrentRootId()), LOG_TAG_BUFFER,
                        "if location is invalid and we are a centroid we have to be the root!");
            return nullptr;
        }

        RetStatus rs = bufferMgr->ReadAndPinVertex(excpectedLocation.detail.containerId,
                                                   excpectedLocation.detail.containerVersion, parent);
        UNUSED_VARIABLE(rs);
        if (parent == nullptr) {
            continue;
        }
        currentLocation = location.Load();
        if (currentLocation.detail.containerId != excpectedLocation.detail.containerId ||
            currentLocation.detail.containerVersion != excpectedLocation.detail.containerVersion) {
            parent->Unpin();
            parent = nullptr;
            continue;
        }

        SANITY_CHECK(
            const uint16_t block_size = parent->GetAttributes().block_size;
            const uint16_t cap = parent->GetAttributes().cap;
            const uint16_t dim = parent->GetAttributes().index->GetAttributes().dimension;
            CentroidMetaData* selfMetaData =
                static_cast<CentroidMetaData*>(parent->GetCluster().MetaData(
                    currentLocation.detail.entryOffset, false, block_size, cap, dim));
            FatalAssert(selfMetaData->id == selfId, LOG_TAG_BUFFER, "Corrupted location data!");
        );
        // if (selfMetaData->id != selfId) {
        //     parent->Unpin();
        //     selfMetaData = nullptr;
        //     parent = nullptr;
        //     continue;
        // }

        /* todo: stat collect how many times we have to retry */
        return parent;
    }
}

BufferVertexEntry::BufferVertexEntry(DIVFTreeVertexInterface* cluster, VectorID id, uint64_t initialPin) :
    centroidMeta(id), state(CLUSTER_CREATED), currentVersion(0), nextVersionPin(0),
    SANITY_CHECK(sanity_nextVersion(0), )
    hasCompactionTask(false), hasMergeTask(false) {
    /* since we have an atomicVectorLocation and it has to be 16byte aligned, the vertex entry should be
        16 byte aligned as well */
    FatalAssert(ALIGNED(this, 16), LOG_TAG_BUFFER, "BufferVertexEntry is not alligned!");
    // liveVersions.emplace(initialPin, 0, cluster);
#ifdef MEMORY_DEBUG
    DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BUFFER, "%p created buffer vertex entry -> ID:" VECTORID_LOG_FMT
            ", version pin=%u", this, VECTORID_LOG(id), initialPin);
#endif
    liveVersions[0] = VersionedClusterPtr(initialPin, 0, cluster);
    /*
        * set version pin to 1 as BufferVertexEntry is referencing it with currentVersion and
        * set pin to 0 as no one is using it yet
        */
}

BufferVertexEntry::~BufferVertexEntry() {
#ifdef MEMORY_DEBUG
    DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BUFFER, "%p destroyed buffer vertex entry -> ID:" VECTORID_LOG_FMT, this,
            VECTORID_LOG(centroidMeta.selfId));
#endif
}

void* BufferVertexEntry::operator new(std::size_t size) {
    return ::operator new(size, std::align_val_t(16));
}

void BufferVertexEntry::operator delete(void* ptr) noexcept {
    ::operator delete(ptr, std::align_val_t(16));
}

inline BufferVertexEntry* BufferVertexEntry::ReadParentEntry(VectorLocation& currentLocation) {
    threadSelf->SanityCheckLockHeldByMe(&clusterLock);
    return centroidMeta.ReadParentEntry(currentLocation);
}

DIVFTreeVertexInterface* BufferVertexEntry::ReadLatestVersion(bool pinCluster, bool needsHeaderLock) {
    if (!needsHeaderLock) {
        FatalAssert(threadSelf->LockHeldByMe(&clusterLock) != LockHeld::UNLOCKED ||
                    threadSelf->LockHeldByMe(&headerLock) != LockHeld::UNLOCKED, LOG_TAG_BUFFER,
                    "No lock is held!");
    } else {
        headerLock.Lock(SX_SHARED);
    }

    DIVFTreeVertexInterface* cluster = nullptr;
    auto it = liveVersions.find(currentVersion);
    if (it != liveVersions.end()) {
        FatalAssert(it->second.versionPin.load() != 0, LOG_TAG_BUFFER,
                    "Found current version but its version pin is 0!");
        cluster = it->second.clusterPtr.clusterPtr;
        if (pinCluster) {
            uint64_t pin = it->second.clusterPtr.pin.fetch_add(1);
            UNUSED_VARIABLE(pin);
#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE_VERTEX, "%p cluster pinned -> ID:" VECTORID_LOG_FMT
                ", Version:%u, pinCnt=%lu", cluster, VECTORID_LOG(centroidMeta.selfId), currentVersion, pin+1);
#endif
        }
    }

    if (needsHeaderLock) {
        headerLock.Unlock();
    }

    return cluster;
}

// DIVFTreeVertexInterface* BufferVertexEntry::MVCCReadAndPinCluster(Version version, bool headerLocked) {
//     if (!headerLocked) {
//         headerLock.Lock(SX_SHARED);
//     }

//     FatalAssert(version <= currentVersion, LOG_TAG_BUFFER, "Version is out of bounds: VertexID="
//                 VECTORID_LOG_FMT ", latest version = %u, input version = %u",
//                 VECTORID_LOG(centroidMeta.selfId), currentVersion, version);
//     DIVFTreeVertexInterface* cluster = nullptr;
//     auto it = liveVersions.find(version);
//     if (it != liveVersions.end() && it->second.versionPin.load() != 0) {
//         cluster = it->second.clusterPtr.clusterPtr;
//         it->second.clusterPtr.pin.fetch_add(1);
//     }

//     if (!headerLocked) {
//         headerLock.Unlock();
//     }

//     return cluster;
// }

RetStatus BufferVertexEntry::UpgradeAccessToExclusive(BufferVertexEntryState& targetState, bool unlockOnFail) {
    threadSelf->SanityCheckLockHeldInModeByMe(&clusterLock, SX_SHARED);
    BufferVertexEntryState expected = CLUSTER_STABLE;
    if (state.compare_exchange_strong(expected, targetState)) {
        clusterLock.Unlock();
        clusterLock.Lock(SX_EXCLUSIVE);
        return RetStatus::Success();
    } else {
        condVar.Wait(LockWrapper<SX_SHARED>{clusterLock});
        if (unlockOnFail) {
            clusterLock.Unlock();
        }
        targetState = expected;
        return RetStatus(RetStatus::FAILED_TO_CAS_ENTRY_STATE);
    }
}

void BufferVertexEntry::DowngradeAccessToShared() {
    FatalAssert(state.load() != CLUSTER_STABLE, LOG_TAG_BUFFER, "BufferEntry state is Stable! VertexID="
                VECTORID_LOG_FMT, VECTORID_LOG(centroidMeta.selfId));
    threadSelf->SanityCheckLockHeldInModeByMe(&clusterLock, SX_EXCLUSIVE);
    clusterLock.Unlock();
    clusterLock.Lock(SX_SHARED);
    state.store(CLUSTER_STABLE);
    condVar.NotifyAll();
    state.notify_all();
}

void BufferVertexEntry::UpdateClusterPtr(DIVFTreeVertexInterface* newCluster) {
    DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_BUFFER, "Deprecated function called: UpdateClusterPtr without version!");
    threadSelf->SanityCheckLockHeldInModeByMe(&clusterLock, SX_EXCLUSIVE);
    headerLock.Lock(SX_EXCLUSIVE);
    FatalAssert(liveVersions.find(currentVersion) != liveVersions.end(), LOG_TAG_BUFFER,
                "new version already exists!");
    std::unordered_map<Version, VersionedClusterPtr, VersionHash>::iterator it = liveVersions.find(currentVersion);
    uint64_t pinCount = it->second.clusterPtr.pin.load(std::memory_order_relaxed);
    DIVFTreeVertexInterface* cluster = it->second.clusterPtr.clusterPtr;
    it->second.clusterPtr.pin.store(0, std::memory_order_relaxed);
    it->second.clusterPtr.clusterPtr = newCluster;
    FatalAssert(newCluster != nullptr, LOG_TAG_BUFFER, "Invalid cluster ptr!");
    cluster->MarkForRecycle(pinCount);
    headerLock.Unlock();
}

void BufferVertexEntry::UpdateClusterPtr(DIVFTreeVertexInterface* newCluster, Version newVersion, bool isRoot) {
    threadSelf->SanityCheckLockHeldInModeByMe(&clusterLock, SX_EXCLUSIVE);
    headerLock.Lock(SX_EXCLUSIVE);
    FatalAssert(liveVersions.find(currentVersion) != liveVersions.end(), LOG_TAG_BUFFER,
                "new version already exists!");
    FatalAssert(newVersion == currentVersion.NextSplit() || newVersion == currentVersion.NextCompaction(),
                LOG_TAG_BUFFER, "Version jump detected!");
    Version oldVersion = currentVersion;
    currentVersion = newVersion;
    FatalAssert(currentVersion != oldVersion, LOG_TAG_BUFFER, "Version did not change!");
    FatalAssert(liveVersions.find(currentVersion) == liveVersions.end(), LOG_TAG_BUFFER,
                "new version already exists!");
    FatalAssert(newCluster != nullptr, LOG_TAG_BUFFER, "Invalid cluster ptr!");
    FatalAssert(newCluster->GetAttributes().centroid_id == centroidMeta.selfId, LOG_TAG_BUFFER, "mismatch id!");
    FatalAssert(newCluster->GetAttributes().version == newVersion, LOG_TAG_BUFFER, "mismatch version!");
    FatalAssert(isRoot || (centroidMeta.selfId != BufferManager::GetInstance()->GetCurrentRootId()), LOG_TAG_BUFFER,
                "Cannot call this version of update cluster on root!");
    SANITY_CHECK(
        FatalAssert(((sanity_nextVersion == oldVersion) && (nextVersionPin == 0)) ||
                    ((sanity_nextVersion == newVersion)), LOG_TAG_BUFFER, "Unexcpected version!");
        sanity_nextVersion = newVersion;
    );
    liveVersions[currentVersion] = VersionedClusterPtr((isRoot ? 2 : 1) + nextVersionPin, 0, newCluster);
    nextVersionPin = 0;
    UnpinVersion(oldVersion, (isRoot ? 2 : 1), true);
    headerLock.Unlock();
}

void BufferVertexEntry::InitiateDelete() {
    threadSelf->SanityCheckLockHeldInModeByMe(&clusterLock, SX_EXCLUSIVE);
    headerLock.Lock(SX_EXCLUSIVE);
    FatalAssert(liveVersions.find(currentVersion) != liveVersions.end(), LOG_TAG_BUFFER,
                "new version already exists!");
    FatalAssert(state.load() == CLUSTER_DELETE_IN_PROGRESS, LOG_TAG_BUFFER,
                "cluster should be in pruning process");
    state.store(CLUSTER_DELETED, std::memory_order_release);
    UnpinVersion(currentVersion, 1, true);
    headerLock.Unlock();
}

void BufferVertexEntry::UnpinVersion(Version version, uint64_t unpinCnt, bool headerLocked) {
    FatalAssert(unpinCnt > 0, LOG_TAG_BUFFER, "unpinCnt has to be greater than zero!");
    FatalAssert(unpinCnt <= 2, LOG_TAG_BUFFER, "unpinCnt is too large!");
    if (!headerLocked) {
        headerLock.Lock(SX_SHARED);
    }
    threadSelf->SanityCheckLockHeldByMe(&headerLock);
    FatalAssert(version <= currentVersion, LOG_TAG_BUFFER, "Version is out of bounds: VertexID="
                VECTORID_LOG_FMT ", latest version = %u, input version = %u",
                VECTORID_LOG(centroidMeta.selfId), currentVersion, version);
    auto it = liveVersions.find(version);
    FatalAssert(it != liveVersions.end(), LOG_TAG_BUFFER, "version is not live!");

    /*
    * If versionPins gets to 0, it means that it is an old version so it should no longer be pinned or unpinned
    * therefore, unlocking the lock and acquiring it in exclusive mode should not cause any concurrency issues
    * for this specific version and we only need to get it to handle removing this version from the versionList.
    */
    uint64_t versionPin = it->second.versionPin.fetch_sub(unpinCnt, std::memory_order_relaxed) - unpinCnt;
    FatalAssert(versionPin + unpinCnt > versionPin, LOG_TAG_BUFFER, "Possible unpin underflow!");
#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE_VERTEX, "%p cluster version unpinned -> ID:" VECTORID_LOG_FMT
                ", Version:%u, pinCnt=%lu, currentVersion=%u", it->second.clusterPtr.clusterPtr,
                VECTORID_LOG(centroidMeta.selfId), version, versionPin, currentVersion);
#endif
    bool needed_upgrade = false;
    if (versionPin == 0) {
        if (headerLock.CurrentMode() != SX_EXCLUSIVE) {
            headerLock.Unlock();
            headerLock.Lock(SX_EXCLUSIVE);
            needed_upgrade = true;
        }
        FatalAssert(it->second.versionPin.load() == 0, LOG_TAG_BUFFER,
                    "version pin incremented during access mode upgrade!");
        uint64_t pinCount = it->second.clusterPtr.pin.load(std::memory_order_relaxed);
        DIVFTreeVertexInterface* cluster = it->second.clusterPtr.clusterPtr;
        it->second.clusterPtr.pin.store(0, std::memory_order_relaxed);
        it->second.clusterPtr.clusterPtr = nullptr;
        liveVersions.erase(it);
        cluster->MarkForRecycle(pinCount);
        if (version == currentVersion) {
            FatalAssert(state.load() == CLUSTER_DELETED, LOG_TAG_BUFFER,
                        "cluster should be in deleted state!");
            if (liveVersions.empty()) {
                // todo: delete the buffer entry?
            }
        }
    }

    if (!headerLocked) {
        headerLock.Unlock();
    } else if (needed_upgrade) {
        headerLock.Unlock();
        headerLock.Lock(SX_SHARED);
    }
}

void BufferVertexEntry::PinVersion(Version version, bool headerLocked) {
    if (!headerLocked) {
        headerLock.Lock(SX_SHARED);
    }
    threadSelf->SanityCheckLockHeldByMe(&headerLock);
    FatalAssert(version == currentVersion.NextSplit() || version <= currentVersion.NextCompaction(), LOG_TAG_BUFFER,
                "Version is out of bounds: VertexID="
                VECTORID_LOG_FMT ", latest version = %s, input version = %s",
                VECTORID_LOG(centroidMeta.selfId), currentVersion.ToString().ToCStr(), version.ToString().ToCStr());
    if (version <= currentVersion) {
        auto it = liveVersions.find(version);
        FatalAssert(it != liveVersions.end(), LOG_TAG_BUFFER, "version is not live!");
        uint64_t oldVersionPin = it->second.versionPin.fetch_add(1);
        UNUSED_VARIABLE(oldVersionPin);
        FatalAssert(oldVersionPin != 0, LOG_TAG_BUFFER, "oldVersionPin was zero!");
#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE_VERTEX, "%p cluster version pinned -> ID:" VECTORID_LOG_FMT
                ", Version:%u, pinCnt=%lu, currentVersion=%u", it->second.clusterPtr.clusterPtr,
                VECTORID_LOG(centroidMeta.selfId), version, oldVersionPin + 1, currentVersion);
#endif
    } else {
        SANITY_CHECK(
            if (sanity_nextVersion == currentVersion) {
                sanity_nextVersion = version;
            } else {
                FatalAssert(sanity_nextVersion == version, LOG_TAG_BUFFER, "Unexcpected version!");
            }
        );
        ++nextVersionPin;
#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE_VERTEX, "%p cluster version pinned -> ID:" VECTORID_LOG_FMT
                ", Version:%u, pinCnt=%lu, currentVersion=%u", nullptr, VECTORID_LOG(centroidMeta.selfId),
                version, nextVersionPin, currentVersion);
#endif
    }

    if (!headerLocked) {
        headerLock.Unlock();
    }
}

String BufferVertexEntry::ToString() {
    FatalAssert(false, LOG_TAG_NOT_IMPLEMENTED, "Not Implemented!");
    return String("");
}

BufferManager::BufferManager(uint64_t internalSize, uint64_t leafSize, uint64_t pool_size_bytes) :
    internalVertexSize(internalSize), leafVertexSize(leafSize), memoryPool(internalSize, leafSize, pool_size_bytes),
    currentRootId(INVALID_VECTOR_ID) {
    FatalAssert(bufferMgrInstance == nullptr, LOG_TAG_BUFFER, "Buffer already initialized");
}

BufferManager::~BufferManager() {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BUFFER, "deleting the buffer manager!");

    for (BufferVectorEntry* entry : vectorDirectory) {
        if (entry == nullptr) {
            continue;
        }
        entry->location.Store(INVALID_VECTOR_LOCATION, true);
    }

    for (size_t level = MAX_TREE_HIGHT - 1; level != (size_t)(-1); --level) {
        for (BufferVertexEntry* entry : clusterDirectory[level]) {
            if (entry == nullptr) {
                continue;
            }
            entry->clusterLock.Lock(SX_EXCLUSIVE);
            entry->headerLock.Lock(SX_EXCLUSIVE);
            for (auto& mvccClus : entry->liveVersions) {
                /* todo: need to rethink */
                mvccClus.second.versionPin = 0;
                uint64_t readerPins = mvccClus.second.clusterPtr.pin.load(std::memory_order_relaxed);
                DIVFTreeVertexInterface* clusterPtr = mvccClus.second.clusterPtr.clusterPtr;
                mvccClus.second.clusterPtr.pin.store(0, std::memory_order_relaxed);
                mvccClus.second.clusterPtr.clusterPtr = nullptr;
                clusterPtr->MarkForRecycle(readerPins);
            }
            entry->liveVersions.clear();
            entry->currentVersion = UINT32_MAX;
            entry->state.store(CLUSTER_DELETED);
            entry->state.notify_all();
            entry->centroidMeta.location.Store(INVALID_VECTOR_LOCATION, true);
            entry->headerLock.Unlock();
            entry->clusterLock.Unlock();
        }
    }

    // std::align_val_t al = (std::align_val_t)16;
    for (BufferVectorEntry* entry : vectorDirectory) {
        if (entry != nullptr) {
            delete entry;
            // operator delete(entry, al);
            entry = nullptr;
        }
    }
    vectorDirectory.clear();

    for (size_t level = MAX_TREE_HIGHT - 1; level != (size_t)(-1); --level) {
        for (BufferVertexEntry* entry : clusterDirectory[level]) {
            if (entry != nullptr) {
                delete entry;
                // operator delete(entry, al);
                entry = nullptr;
            }
        }
        clusterDirectory[level].clear();
    }
}

inline BufferVertexEntry* BufferManager::Init(uint64_t vertexMetaDataSize,
                                              uint16_t leaf_blk_size, uint16_t internal_blk_size,
                                              uint16_t leaf_cap, uint16_t internal_cap, uint16_t dim,
                                              uint64_t pool_size_gb) {
    FatalAssert(bufferMgrInstance == nullptr, LOG_TAG_BUFFER, "Buffer already initialized");
    uint64_t internalVertexSize = vertexMetaDataSize + Cluster::TotalBytes(false, internal_blk_size, internal_cap, dim);
    uint64_t leafVertexSize = vertexMetaDataSize + Cluster::TotalBytes(true, leaf_blk_size, leaf_cap, dim);
    bufferMgrInstance = new BufferManager(internalVertexSize, leafVertexSize, pool_size_gb * 1024 * 1024 * 1024);
    /* todo: set creator node id in multi node */
    bufferMgrInstance->nextVectorId[0] = VectorID::CreateVectorID(0, 0);
    for (size_t level = 1; level <= MAX_TREE_HIGHT; ++level) {
        bufferMgrInstance->nextVectorId[level] = VectorID::CreateCentroidID(level, 0);
    }
    BufferVertexEntry* root = bufferMgrInstance->CreateNewRootEntry(INVALID_VECTOR_ID);
    return root;
}

inline void BufferManager::Shutdown() {
    FatalAssert(bufferMgrInstance != nullptr, LOG_TAG_BUFFER, "Buffer not initialized");
    delete bufferMgrInstance;
    bufferMgrInstance = nullptr;
}

inline BufferManager* BufferManager::GetInstance() {
    return bufferMgrInstance;
}

/* todo: add batchallocate and batch free functions! */
DIVFTreeVertexInterface* BufferManager::AllocateMemoryForVertex(uint8_t level) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    FatalAssert(MAX_TREE_HIGHT > (uint64_t)level && (uint64_t)level > VectorID::VECTOR_LEVEL, LOG_TAG_BUFFER,
                "Level is out of bounds.");
    // void* res = std::aligned_alloc(CACHE_LINE_SIZE,
    //                                ((uint64_t)level == VectorID::LEAF_LEVEL ? leafVertexSize : internalVertexSize));
    void* res = memoryPool.Allocate((level == VectorID::LEAF_LEVEL) ? SlotType::Leaf : SlotType::Internal);
    CHECK_NOT_NULLPTR(res, LOG_TAG_BUFFER);
#ifdef MEMORY_DEBUG
    DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BUFFER, "%p allocated", res);
#endif
    return reinterpret_cast<DIVFTreeVertexInterface*>(res);
}

void BufferManager::Recycle(DIVFTreeVertexInterface* memory) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    if (memory == nullptr) {
        return;
    }
#ifdef MEMORY_DEBUG
    DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BUFFER, "%p freed", memory);
#endif
    // std::free(memory);
    memoryPool.Free(reinterpret_cast<void*>(memory));
}

void BufferManager::UpdateRoot(VectorID newRootId, Version newRootVersion, BufferVertexEntry* oldRootEntry) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    CHECK_VECTORID_IS_VALID(newRootId, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_CENTROID(newRootId, LOG_TAG_BUFFER);
    CHECK_NOT_NULLPTR(oldRootEntry, LOG_TAG_BUFFER);
    threadSelf->SanityCheckLockHeldInModeByMe(&oldRootEntry->clusterLock, SX_EXCLUSIVE);
    PinVertexVersion(newRootId, newRootVersion);
    bufferMgrLock.Lock(SX_EXCLUSIVE);
    FatalAssert(currentRootId.load(std::memory_order_relaxed) == oldRootEntry->centroidMeta.selfId,
                LOG_TAG_BUFFER, "entry is not the root!");
    currentRootId.store(newRootId._id, std::memory_order_release);
    currentRootId.notify_all();
    bufferMgrLock.Unlock(); /* unlock buffermanger before unpin to avoid recycling while its lock is held */
    oldRootEntry->UnpinVersion(oldRootEntry->currentVersion);
}

BufferVertexEntry* BufferManager::CreateNewRootEntry(VectorID expRootId) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    VectorID currentId = VectorID::AsID(currentRootId.load(std::memory_order_acquire));
    VectorID newId = INVALID_VECTOR_ID;
    BufferVertexEntry* oldRootEntry = nullptr;
    if (currentId == INVALID_VECTOR_ID) {
        /* this is called during init */
        FatalAssert(expRootId == INVALID_VECTOR_ID, LOG_TAG_BUFFER, "expRootId should be invalid during init!");
        newId = nextVectorId[VectorID::LEAF_LEVEL];
        ++nextVectorId[VectorID::LEAF_LEVEL];

        bufferMgrLock.Lock(SX_EXCLUSIVE);
    } else {
        CHECK_VECTORID_IS_VALID(expRootId, LOG_TAG_BUFFER);
        CHECK_VECTORID_IS_CENTROID(expRootId, LOG_TAG_BUFFER);
        if (currentId != expRootId) {
            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BUFFER, "mismatch root id!");
            currentId = VectorID::AsID(currentRootId.load(std::memory_order_acquire));
            while (currentId != expRootId) {
                currentRootId.wait(currentId._id);
                currentId = VectorID::AsID(currentRootId.load(std::memory_order_acquire));
            }
        }

        bufferMgrLock.Lock(SX_EXCLUSIVE);

        oldRootEntry = GetVertexEntry(currentId, false);
        CHECK_NOT_NULLPTR(oldRootEntry, LOG_TAG_BUFFER);
        threadSelf->SanityCheckLockHeldInModeByMe(&oldRootEntry->clusterLock, SX_EXCLUSIVE);
        FatalAssert(oldRootEntry->centroidMeta.selfId == currentId, LOG_TAG_BUFFER, "id mismatch!");
        FatalAssert(currentId == currentRootId.load(std::memory_order_relaxed), LOG_TAG_BUFFER, "id mismatch!");
        newId = nextVectorId[currentId.Level() + 1];
        ++nextVectorId[currentId.Level() + 1];
    }
    FatalAssert(MAX_TREE_HIGHT > newId.Level() && newId.Level() > VectorID::VECTOR_LEVEL, LOG_TAG_BUFFER,
                "Level is out of bounds.");
    const uint64_t newLevelIdx = newId.Level() - 1;
    FatalAssert(newId.Value() == clusterDirectory[newLevelIdx].size(), LOG_TAG_BUFFER,
                "New ID value does not match the expected size!");
    DIVFTreeVertexInterface* memLoc = AllocateMemoryForVertex(newId.Level());
    CHECK_NOT_NULLPTR(memLoc, LOG_TAG_BUFFER);
    /* because of the currentVersion + currentRootId pin should be 2 */
    // BufferVertexEntry* newRoot = new (std::align_val_t(16)) BufferVertexEntry(memLoc, newId, 2);
    BufferVertexEntry* newRoot = new BufferVertexEntry(memLoc, newId, 2);
    newRoot->clusterLock.Lock(SX_EXCLUSIVE);
    newRoot->headerLock.Lock(SX_EXCLUSIVE);
    clusterDirectory[newLevelIdx].emplace_back(newRoot);
    /* todo: should we pin the version here since we have already pinned it twice in the constructor! */
    // newRoot->PinVersion(newRoot->currentVersion, true);
    currentRootId.store(newId._id, std::memory_order_release);
    currentRootId.notify_all();
    if (oldRootEntry != nullptr) {
        /* Unpin once due to not being root anymore */
        oldRootEntry->UnpinVersion(oldRootEntry->currentVersion);
    }
    bufferMgrLock.Unlock();
    return newRoot;
}

void BufferManager::BatchCreateBufferEntry(uint16_t num_entries, uint8_t level, BufferVertexEntry** entries,
                                           DIVFTreeVertexInterface** clusters, VectorID* ids, Version* versions) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    CHECK_NOT_NULLPTR(entries, LOG_TAG_BUFFER);
    FatalAssert(num_entries > 0, LOG_TAG_BUFFER, "number of entries cannot be 0!");
    FatalAssert(MAX_TREE_HIGHT > (uint64_t)level && (uint64_t)level > VectorID::VECTOR_LEVEL, LOG_TAG_BUFFER,
                "Level is out of bounds.");

    bufferMgrLock.Lock(SX_EXCLUSIVE);
    FatalAssert(level <= VectorID::AsID(currentRootId.load(std::memory_order_relaxed)).Level(), LOG_TAG_BUFFER,
                "Level is out of bounds.");
    const uint64_t levelIdx = level - 1;
    const uint64_t nextVal = clusterDirectory[levelIdx].size();
    UNUSED_VARIABLE(nextVal);
    for (size_t i = 0; i < num_entries; ++i) {
        VectorID id = nextVectorId[level];
        ++nextVectorId[level];
        FatalAssert(id.Value() == nextVal + i, LOG_TAG_BUFFER,
                    "New ID value does not match the expected size!");

        DIVFTreeVertexInterface* cluster = AllocateMemoryForVertex(level);
        if (clusters != nullptr) {
            clusters[i] = cluster;
        }
        if (ids != nullptr) {
            ids[i] = id;
        }
        CHECK_NOT_NULLPTR(cluster, LOG_TAG_BUFFER);

        // entries[i] = new (std::align_val_t(16)) BufferVertexEntry(cluster, id);
        entries[i] = new BufferVertexEntry(cluster, id);
        entries[i]->clusterLock.Lock(SX_EXCLUSIVE);
        if (versions != nullptr) {
            versions[i] = entries[i]->currentVersion;
        }

        clusterDirectory[levelIdx].emplace_back(entries[i]);
    }
    bufferMgrLock.Unlock();
}

void BufferManager::BatchCreateVectorEntry(size_t num_entries, BufferVectorEntry** entries, bool create_update_handle) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    CHECK_NOT_NULLPTR(entries, LOG_TAG_BUFFER);
    FatalAssert(num_entries > 0, LOG_TAG_BUFFER, "number of entries cannot be 0!");

    bufferMgrLock.Lock(SX_EXCLUSIVE);
    const uint64_t nextVal = vectorDirectory.size();
    UNUSED_VARIABLE(nextVal);
    for (size_t i = 0; i < num_entries; ++i) {
        VectorID id = nextVectorId[VectorID::VECTOR_LEVEL];
        ++nextVectorId[VectorID::VECTOR_LEVEL];
        FatalAssert(id.Value() == nextVal + i, LOG_TAG_BUFFER,
                    "New ID value does not match the expected size!");

        // entries[i] = new (std::align_val_t(16)) BufferVectorEntry(id);
        entries[i] = new BufferVectorEntry(id);
        vectorDirectory.emplace_back(entries[i]);

        if (create_update_handle) {
            CreateUpdateHandle(id);
        }
    }
    bufferMgrLock.Unlock();
}

VectorID BufferManager::GetCurrentRootId() const {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    return VectorID::AsID(currentRootId.load());
}

VectorID BufferManager::GetCurrentRootIdAndVersion(Version& version) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    VectorID rootId;
    while (true) {
        BufferVertexEntry* entry = GetRootEntry();
        CHECK_NOT_NULLPTR(entry, LOG_TAG_BUFFER);

        entry->headerLock.Lock(SX_SHARED);
        if (currentRootId.load(std::memory_order_acquire) != entry->centroidMeta.selfId) {
            entry->headerLock.Unlock();
            continue;
        }

        auto it = entry->liveVersions.find(entry->currentVersion);
        if (it == entry->liveVersions.end() || it->second.versionPin.load() == 0) {
            FatalAssert((entry->state.load(std::memory_order_relaxed) == CLUSTER_DELETED) &&
                        currentRootId.load(std::memory_order_acquire) != entry->centroidMeta.selfId,
                        LOG_TAG_BUFFER, "could not find version but it is still root or not deleted!");
            entry->headerLock.Unlock();
            continue;
        }

        version = entry->currentVersion;
        rootId = entry->centroidMeta.selfId;
        entry->headerLock.Unlock();
        return rootId;
    }
}

inline BufferVectorEntry* BufferManager::GetVectorEntry(VectorID id) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    CHECK_VECTORID_IS_VALID(id, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_VECTOR(id, LOG_TAG_BUFFER);
    bufferMgrLock.Lock(SX_SHARED);
    if (vectorDirectory.size() <= id.Value()) {
        bufferMgrLock.Unlock();
        return nullptr;
    }
    BufferVectorEntry* entry = vectorDirectory[id.Value()];
    bufferMgrLock.Unlock();
    return entry;
}

inline void BufferManager::PinVertexVersion(VectorID vertexId, Version version) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    BufferVertexEntry* entry = GetVertexEntry(vertexId, true);
    FatalAssert(entry != nullptr, LOG_TAG_BUFFER, "VertexID not found in buffer. VertexID="
                VECTORID_LOG_FMT, VECTORID_LOG(vertexId));

    FatalAssert(entry->state.load() != CLUSTER_DELETED, LOG_TAG_BUFFER, "BufferEntry state is Deleted! VertexID="
                VECTORID_LOG_FMT, VECTORID_LOG(vertexId));
    FatalAssert(entry->centroidMeta.selfId == vertexId, LOG_TAG_BUFFER, "BufferEntry id mismatch! VertexID="
                VECTORID_LOG_FMT "EntryID = " VECTORID_LOG_FMT, VECTORID_LOG(vertexId),
                VECTORID_LOG(entry->centroidMeta.selfId));
    entry->PinVersion(version);
}

inline void BufferManager::UnpinVertexVersion(VectorID vertexId, Version version) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    BufferVertexEntry* entry = GetVertexEntry(vertexId);
    FatalAssert(entry != nullptr, LOG_TAG_BUFFER, "VertexID not found in buffer. VertexID="
                VECTORID_LOG_FMT, VECTORID_LOG(vertexId));

    /* todo: is it possible to unpin an older version after deletion? */
    // FatalAssert(entry->state.load() != CLUSTER_DELETED, LOG_TAG_BUFFER, "BufferEntry state is Deleted! VertexID="
    //             VECTORID_LOG_FMT, VECTORID_LOG(vertexId));
    FatalAssert(entry->centroidMeta.selfId == vertexId, LOG_TAG_BUFFER, "BufferEntry id mismatch! VertexID="
                VECTORID_LOG_FMT "EntryID = " VECTORID_LOG_FMT, VECTORID_LOG(vertexId),
                VECTORID_LOG(entry->centroidMeta.selfId));
    entry->UnpinVersion(version);
}

inline VectorLocation BufferManager::LoadCurrentVectorLocation(VectorID vectorId) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    CHECK_VECTORID_IS_VALID(vectorId, LOG_TAG_BUFFER);

    if (vectorId.IsCentroid()) {
        BufferVertexEntry* entry = GetVertexEntry(vectorId);
        if (entry == nullptr || entry->state.load() == CLUSTER_DELETED || entry->state.load() == CLUSTER_CREATED) {
            return INVALID_VECTOR_LOCATION;
        }
        FatalAssert(entry->centroidMeta.selfId == vectorId, LOG_TAG_BUFFER, "BufferEntry id mismatch! VertexID="
                    VECTORID_LOG_FMT "EntryID = " VECTORID_LOG_FMT, VECTORID_LOG(vectorId),
                    VECTORID_LOG(entry->centroidMeta.selfId));
        return entry->centroidMeta.location.Load();
    } else {
        BufferVectorEntry* entry = GetVectorEntry(vectorId);
        if (entry == nullptr) {
            return INVALID_VECTOR_LOCATION;
        }
        return entry->location.Load();
    }
}

inline void BufferManager::UpdateVectorLocation(VectorID vectorId, VectorLocation newLocation, bool pinNewUnpinOld) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    CHECK_VECTORID_IS_VALID(vectorId, LOG_TAG_BUFFER);
    VectorLocation oldLocation = INVALID_VECTOR_LOCATION;
    FatalAssert((newLocation != INVALID_VECTOR_LOCATION) || pinNewUnpinOld, LOG_TAG_BUFFER,
                "if newLocation is invalid then pinNewUnpinOld should be true");
    if (pinNewUnpinOld && (newLocation != INVALID_VECTOR_LOCATION)) {
        CHECK_VECTORID_IS_VALID(newLocation.detail.containerId, LOG_TAG_BUFFER);
        CHECK_VECTORID_IS_CENTROID(newLocation.detail.containerId, LOG_TAG_BUFFER);
        FatalAssert(newLocation.detail.entryOffset != INVALID_OFFSET, LOG_TAG_BUFFER,
                    "Invalid entry offset.");
        PinVertexVersion(newLocation.detail.containerId,
                         newLocation.detail.containerVersion);
    }

    if (vectorId.IsCentroid()) {
        BufferVertexEntry* entry = GetVertexEntry(vectorId);
        FatalAssert(entry != nullptr, LOG_TAG_BUFFER, "VertexID not found in buffer. VertexID="
                    VECTORID_LOG_FMT, VECTORID_LOG(vectorId));
        FatalAssert(entry->state.load() != CLUSTER_DELETED, LOG_TAG_BUFFER, "BufferEntry state is Deleted! VertexID="
                    VECTORID_LOG_FMT, VECTORID_LOG(vectorId));
        FatalAssert(entry->centroidMeta.selfId == vectorId, LOG_TAG_BUFFER, "BufferEntry id mismatch! VertexID="
                    VECTORID_LOG_FMT "EntryID = " VECTORID_LOG_FMT, VECTORID_LOG(vectorId),
                    VECTORID_LOG(entry->centroidMeta.selfId));
        oldLocation = entry->centroidMeta.location.Load();
        entry->centroidMeta.location.Store(newLocation);
    } else {
        BufferVectorEntry* entry = GetVectorEntry(vectorId);
        FatalAssert(entry != nullptr, LOG_TAG_BUFFER, "VectorID not found in buffer. VectorID="
                    VECTORID_LOG_FMT, VECTORID_LOG(vectorId));
        oldLocation = entry->location.Load();
        entry->location.Store(newLocation);
    }

    SANITY_CHECK(
        if (oldLocation != INVALID_VECTOR_LOCATION) {
            BufferVertexEntry* old_parent = GetVertexEntry(oldLocation.detail.containerId);
            CHECK_NOT_NULLPTR(old_parent, LOG_TAG_BUFFER);
            threadSelf->SanityCheckLockHeldByMe(&old_parent->clusterLock);
        }

        if (newLocation != INVALID_VECTOR_LOCATION) {
            BufferVertexEntry* new_parent = GetVertexEntry(newLocation.detail.containerId);
            CHECK_NOT_NULLPTR(new_parent, LOG_TAG_BUFFER);
            threadSelf->SanityCheckLockHeldByMe(&new_parent->clusterLock);
        }
    );

    FatalAssert(oldLocation != newLocation, LOG_TAG_BUFFER, "new and old location are the same!");
    FatalAssert((oldLocation != INVALID_VECTOR_LOCATION) || pinNewUnpinOld, LOG_TAG_BUFFER,
                "if oldLocation is invalid then pinNewUnpinOld should be true");
    FatalAssert(!pinNewUnpinOld == ((newLocation.detail.containerId == oldLocation.detail.containerId) &&
                                    (newLocation.detail.containerVersion == oldLocation.detail.containerVersion)),
                LOG_TAG_BUFFER,
                "if pinNewUnpinOld is false, only the offset of the old and new location should be different");
    if (pinNewUnpinOld && (oldLocation != INVALID_VECTOR_LOCATION)) {
        UnpinVertexVersion(oldLocation.detail.containerId,
                           oldLocation.detail.containerVersion);
    }
    // DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_BUFFER, "Updated vector location with id " VECTORID_LOG_FMT
    //         " from {id: " VECTORID_LOG_FMT ", ver:%u, offset:%hu} to {id: "
    //         VECTORID_LOG_FMT ", ver:%u, offset:%hu} with pin=%s",
    //         VECTORID_LOG(vectorId), VECTORID_LOG(oldLocation.detail.containerId), oldLocation.detail.containerVersion,
    //         oldLocation.detail.entryOffset, VECTORID_LOG(newLocation.detail.containerId),
    //         newLocation.detail.containerVersion, newLocation.detail.entryOffset, (pinNewUnpinOld ? "ON" : "OFF"));
}

// inline void BufferManager::UpdateVectorLocationOffset(VectorID vectorId, uint16_t newOffset) {
//     FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
//     CHECK_VECTORID_IS_VALID(vectorId, LOG_TAG_BUFFER);

//     if (vectorId.IsCentroid()) {
//         BufferVertexEntry* entry = GetVertexEntry(vectorId);
//         FatalAssert(entry != nullptr, LOG_TAG_BUFFER, "VertexID not found in buffer. VertexID="
//                     VECTORID_LOG_FMT, VECTORID_LOG(vectorId));
//         FatalAssert(entry->state.load() != CLUSTER_DELETED, LOG_TAG_BUFFER, "BufferEntry state is Deleted! VertexID="
//                     VECTORID_LOG_FMT, VECTORID_LOG(vectorId));
//         FatalAssert(entry->centroidMeta.selfId == vectorId, LOG_TAG_BUFFER, "BufferEntry id mismatch! VertexID="
//                     VECTORID_LOG_FMT "EntryID = " VECTORID_LOG_FMT, VECTORID_LOG(vectorId),
//                     VECTORID_LOG(entry->centroidMeta.selfId));
//         entry->centroidMeta.location.detail.entryOffset.store(newOffset);
//     } else {
//         BufferVectorEntry* entry = GetVectorEntry(vectorId);
//         FatalAssert(entry != nullptr, LOG_TAG_BUFFER, "VectorID not found in buffer. VectorID="
//                     VECTORID_LOG_FMT, VECTORID_LOG(vectorId));
//         entry->location.detail.entryOffset.store(newOffset);
//     }
// }

DIVFTreeVertexInterface* BufferManager::ReadAndPinRoot() {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    while (true) {
        BufferVertexEntry* entry = GetRootEntry();
        CHECK_NOT_NULLPTR(entry, LOG_TAG_BUFFER);

        entry->headerLock.Lock(SX_SHARED);
        if (currentRootId.load(std::memory_order_acquire) != entry->centroidMeta.selfId) {
            entry->headerLock.Unlock();
            continue;
        }
        auto it = entry->liveVersions.find(entry->currentVersion);
        if (it == entry->liveVersions.end() || it->second.versionPin.load() == 0) {
            FatalAssert((entry->state.load(std::memory_order_relaxed) == CLUSTER_DELETED) &&
                        currentRootId.load(std::memory_order_acquire) != entry->centroidMeta.selfId,
                        LOG_TAG_BUFFER, "could not find version but it is still root or not deleted!");
            entry->headerLock.Unlock();
            continue;
        }

        uint64_t oldPin = it->second.clusterPtr.pin.fetch_add(1);
        UNUSED_VARIABLE(oldPin);
        DIVFTreeVertexInterface* vertex = it->second.clusterPtr.clusterPtr;
#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE_VERTEX, "%p cluster pinned(root) -> ID:" VECTORID_LOG_FMT
                ", Version:%u, pinCnt=%lu", vertex, VECTORID_LOG(entry->centroidMeta.selfId),
                entry->currentVersion, oldPin + 1);
#endif
        entry->headerLock.Unlock();
        CHECK_NOT_NULLPTR(vertex, LOG_TAG_BUFFER);
        return vertex;
    }
}

RetStatus BufferManager::ReadAndPinVertex(VectorID vertexId, Version version,
                                          DIVFTreeVertexInterface*& vertex, bool* outdated) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    vertex = nullptr;
    BufferVertexEntry* entry = GetVertexEntry(vertexId);
    if (entry == nullptr) {
        return RetStatus(RetStatus::VERTEX_DELETED);
    }

    FatalAssert(entry->centroidMeta.selfId == vertexId, LOG_TAG_BUFFER, "BufferEntry id mismatch! VertexID="
                VECTORID_LOG_FMT "EntryID = " VECTORID_LOG_FMT, VECTORID_LOG(vertexId),
                VECTORID_LOG(entry->centroidMeta.selfId));
    // if (entry->state.load() == CLUSTER_DELETED) {
    //     return nullptr;
    // }

    entry->headerLock.Lock(SX_SHARED);
    if (version > entry->currentVersion) {
        entry->headerLock.Unlock();
        return RetStatus(RetStatus::VERSION_NOT_APPLIED);
    }

    FatalAssert(version <= entry->currentVersion, LOG_TAG_BUFFER, "Version is out of bounds: VertexID="
                VECTORID_LOG_FMT ", latest version = %u, input version = %u",
                VECTORID_LOG(vertexId), entry->currentVersion, version);
    if (outdated != nullptr) {
        *outdated = (version == entry->currentVersion);
    }
    auto it = entry->liveVersions.find(version);
    if (it == entry->liveVersions.end() || it->second.versionPin.load() == 0) {
        entry->headerLock.Unlock();
        return RetStatus(RetStatus::OUTDATED_VERSION_DELETED);
    }

    uint64_t pin = it->second.clusterPtr.pin.fetch_add(1);
    UNUSED_VARIABLE(pin);
    vertex = it->second.clusterPtr.clusterPtr;
#ifdef MEMORY_DEBUG
        DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_DIVFTREE_VERTEX, "%p cluster pinned -> ID:" VECTORID_LOG_FMT
                ", Version:%u, pinCnt=%lu", vertex, VECTORID_LOG(entry->centroidMeta.selfId),
                entry->currentVersion, pin + 1);
#endif
    entry->headerLock.Unlock();
    CHECK_NOT_NULLPTR(vertex, LOG_TAG_BUFFER);
    // if (vertex == nullptr) {
    //     return vertex;
    // }

    FatalAssert(vertex->CentroidID() == vertexId, LOG_TAG_BUFFER, "Mismatch in ID. BaseID=" VECTORID_LOG_FMT
                ", Found ID=" VECTORID_LOG_FMT, VECTORID_LOG(vertexId), VECTORID_LOG(vertex->CentroidID()));
    return RetStatus::Success();
}

BufferVertexEntry* BufferManager::ReadBufferEntry(VectorID vertexId, LockMode mode, bool* blocked) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    BufferVertexEntry* entry = GetVertexEntry(vertexId);
    if (entry == nullptr) {
        return nullptr;
    }

    FatalAssert(entry->centroidMeta.selfId == vertexId, LOG_TAG_BUFFER, "BufferEntry id mismatch! VertexID="
                VECTORID_LOG_FMT "EntryID = " VECTORID_LOG_FMT, VECTORID_LOG(vertexId),
                VECTORID_LOG(entry->centroidMeta.selfId));
    BufferVertexEntryState state = entry->state.load();
    while (state != CLUSTER_STABLE) {
        if (state == CLUSTER_DELETED) {
            return nullptr;
        }
        entry->state.wait(state);
        state = entry->state.load();
        if (blocked != nullptr) {
            *blocked = true;
        }
    }

    if (blocked == nullptr || (*blocked)) {
        entry->clusterLock.Lock(mode);
    } else {
        if (!entry->clusterLock.TryLock(mode)) {
            *blocked = true;
            entry->clusterLock.Lock(mode);
        } else {
            *blocked = false;
        }
    }

    while ((state = entry->state.load(std::memory_order_acquire)) != CLUSTER_STABLE) {
        if (state == CLUSTER_DELETED) {
            entry->clusterLock.Unlock();
            return nullptr;
        }

        if (blocked != nullptr) {
            *blocked = true;
        }

        if (mode == SX_SHARED) {
            threadSelf->SanityCheckLockHeldInModeByMe(&entry->clusterLock, SX_SHARED);
            entry->condVar.Wait(LockWrapper<SX_SHARED>{entry->clusterLock});
        } else {
            threadSelf->SanityCheckLockHeldInModeByMe(&entry->clusterLock, SX_EXCLUSIVE);
            entry->condVar.Wait(LockWrapper<SX_EXCLUSIVE>{entry->clusterLock});
        }
    }

    FatalAssert(/* (mode != SX_EXCLUSIVE) ||  */(state == CLUSTER_STABLE), LOG_TAG_BUFFER,
                "BufferEntry state is not stable! VertexID=" VECTORID_LOG_FMT, VECTORID_LOG(vertexId));
    return entry;
}

BufferVertexEntry* BufferManager::TryReadBufferEntry(VectorID vertexId, LockMode mode) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    BufferVertexEntry* entry = GetVertexEntry(vertexId);
    if (entry == nullptr) {
        return nullptr;
    }

    FatalAssert(entry->centroidMeta.selfId == vertexId, LOG_TAG_BUFFER, "BufferEntry id mismatch! VertexID="
                VECTORID_LOG_FMT "EntryID = " VECTORID_LOG_FMT, VECTORID_LOG(vertexId),
                VECTORID_LOG(entry->centroidMeta.selfId));
    if (entry->state.load() != CLUSTER_STABLE || !entry->clusterLock.TryLock(mode)) {
        return nullptr;
    }

    if (entry->state.load() != CLUSTER_STABLE) {
        entry->clusterLock.Unlock();
        return nullptr;
    }

    FatalAssert((mode != SX_EXCLUSIVE) || (entry->state.load() == CLUSTER_STABLE), LOG_TAG_BUFFER,
                "BufferEntry state is not stable! VertexID=" VECTORID_LOG_FMT, VECTORID_LOG(vertexId));
    return entry;
}

inline void BufferManager::ReleaseEntriesIfNotNull(BufferVertexEntry** entries, uint16_t num_entries,
                                                   ReleaseBufferEntryFlags flags) {
    CHECK_NOT_NULLPTR(entries, LOG_TAG_BUFFER);
    for (uint16_t i = 0; i < num_entries; ++i) {
        if (entries[i] != nullptr) {
            ReleaseBufferEntry(entries[i], flags);
            entries[i] = nullptr;
        }
    }
}

inline void BufferManager::ReleaseBufferEntryIfNotNull(BufferVertexEntry* entry, ReleaseBufferEntryFlags flags) {
    if (entry != nullptr) {
        ReleaseBufferEntry(entry, flags);
    }
}

void BufferManager::ReleaseBufferEntry(BufferVertexEntry* entry, ReleaseBufferEntryFlags flags) {
    CHECK_NOT_NULLPTR(entry, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_VALID(entry->centroidMeta.selfId, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_CENTROID(entry->centroidMeta.selfId, LOG_TAG_BUFFER);
    if ((bool)(flags.notifyAll)) {
        threadSelf->SanityCheckLockHeldInModeByMe(&entry->clusterLock, SX_EXCLUSIVE);
        FatalAssert(entry->state.load() != CLUSTER_STABLE, LOG_TAG_BUFFER, "BufferEntry state is Stable! VertexID="
                    VECTORID_LOG_FMT, VECTORID_LOG(entry->centroidMeta.selfId));
        if ((bool)(flags.stablize)) {
            entry->state.store(CLUSTER_STABLE);
        } else {
            FatalAssert(entry->state.load() == CLUSTER_DELETED,
                        LOG_TAG_BUFFER, "BufferEntry state is not deleted! "
                        "VertexID=" VECTORID_LOG_FMT, VECTORID_LOG(entry->centroidMeta.selfId));
        }
        entry->condVar.NotifyAll();
        entry->state.notify_all();
    }
    entry->clusterLock.Unlock();
}

uint64_t BufferManager::GetHeight() const {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    return VectorID::AsID(currentRootId.load()).Level() + 1;
}

String BufferManager::ToString() {
    FatalAssert(false, LOG_TAG_NOT_IMPLEMENTED, "Not Implemented!");
    /*
        * Todo: we cannot get any other locks when we get the bufferMgr lock but if we do not get the
        * BufferMgr lock the directory may change! Moreover, we may have multiple versions and printing all versions
        * seem to be too much! Can we do anything about this?
        */
    if (bufferMgrInstance == nullptr) {
        return String("NULL");
    }

    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    bufferMgrLock.Lock(SX_SHARED);
    uint64_t height = GetHeight();
    String str = "<Height: " + std::to_string(height) + ", ";
    str += "Directory:[";
    /* todo: start from higher level to lower */
    for (size_t i = 0; i < height; ++i) {
        str += "Level: " + std::to_string(i+1) + ":[";
        for (size_t j = 0; j < clusterDirectory[i].size(); ++j) {
            BufferVertexEntry *entry = clusterDirectory[i][j];
            if (entry == nullptr) {
                str += "{NULL}";
            } else {
                str += String("{VectorID:" VECTORID_LOG_FMT ", Info:", VECTORID_LOG(entry->centroidMeta.selfId));
                str += entry->ToString();
            }
            if (j != clusterDirectory[i].size() - 1) {
                str += ", ";
            }
        }
        if (i != height - 1) {
            str += "], ";
        }
        else {
            str += "]";
        }
        str += ">";
    }

    /* todo: print vertex entries */
    bufferMgrLock.Unlock();
    return str;
}

RetStatus BufferManager::CreateUpdateHandle(VectorID target) {
    handleLock.Lock(SX_EXCLUSIVE);
    auto it = handles.find(target);
    if (it != handles.end()) {
        handleLock.Unlock();
        return RetStatus(RetStatus::ALREADY_HAS_A_HANDLE,
                         "Cannot have more than one handle per id at a time");
    }

    handles[target] = new std::atomic<bool>{false};
    handleLock.Unlock();
    return RetStatus::Success();
}

void BufferManager::SignalUpdateHandleIfNeeded(VectorID target) {
    handleLock.Lock(SX_SHARED);
    auto it = handles.find(target);
    if (it == handles.end()) {
        handleLock.Unlock();
        return;
    }

    std::atomic<bool>* handle = it->second;
    CHECK_NOT_NULLPTR(handle, LOG_TAG_BUFFER);

    handle->store(true, std::memory_order_release);
    handle->notify_all();
    handleLock.Unlock();

}

RetStatus BufferManager::WaitForUpdateToGoThrough(VectorID target) {
    handleLock.Lock(SX_SHARED);
    auto it = handles.find(target);
    if (it == handles.end()) {
        handleLock.Unlock();
        return RetStatus(RetStatus::NO_HANDLE_FOUND_FOR_TARGET,
                         "no handle found for the target vectorId");
    }
    std::atomic<bool>* handle = it->second;
    CHECK_NOT_NULLPTR(handle, LOG_TAG_BUFFER);
    handleLock.Unlock();

    handle->wait(false, std::memory_order_acquire);
    FatalAssert(*handle, LOG_TAG_BUFFER, "at this point, the operation must have gone through!");
    handleLock.Lock(SX_EXCLUSIVE);
    handles.erase(target);
    delete handle;
    handleLock.Unlock();
    return RetStatus::Success();
}

RetStatus BufferManager::CheckIfUpdateHasGoneThrough(VectorID target, bool& updated) {
    handleLock.Lock(SX_SHARED);
    auto it = handles.find(target);
    if (it == handles.end()) {
        handleLock.Unlock();
        return RetStatus(RetStatus::NO_HANDLE_FOUND_FOR_TARGET,
                         "no handle found for the target vectorId");
    }
    std::atomic<bool>* handle = it->second;
    CHECK_NOT_NULLPTR(handle, LOG_TAG_BUFFER);
    handleLock.Unlock();

    if (handle->load(std::memory_order_acquire)) {
        handleLock.Lock(SX_EXCLUSIVE);
        handles.erase(target);
        delete handle;
        handleLock.Unlock();
        updated = true;
    } else {
        updated = false;
    }
    return RetStatus::Success();
}

bool BufferManager::AddMigrationTaskIfNotExists(VectorID first, VectorID second, BufferVertexEntry* firstEntry) {
    CHECK_VECTORID_IS_VALID(first, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_CENTROID(first, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_VALID(second, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_CENTROID(second, LOG_TAG_BUFFER);
    FatalAssert(first != second, LOG_TAG_BUFFER, "first and second cannot be the same!");

    if (first > second) {
        std::swap(first, second);
    }

    if (firstEntry != nullptr) {
        FatalAssert(firstEntry->centroidMeta.selfId == first, LOG_TAG_BUFFER,
                    "firstEntry does not correspond to first id!");
    } else {
        bufferMgrLock.Lock(SX_SHARED);
        firstEntry = GetVertexEntry(first, false);
        if (firstEntry == nullptr) {
            FatalAssert(false, LOG_TAG_BUFFER, "one of the entries does not exist!");
            bufferMgrLock.Unlock();
            return false;
        }
        bufferMgrLock.Unlock();
    }

    firstEntry->headerLock.Lock(SX_EXCLUSIVE);
    auto it = firstEntry->migrationTasks.find(second);
    if (it != firstEntry->migrationTasks.end()) {
        firstEntry->headerLock.Unlock();
        return false;
    }

    firstEntry->migrationTasks.insert(second);
    firstEntry->headerLock.Unlock();
    return true;
}

void BufferManager::RemoveMigrationTask(VectorID first, VectorID second, BufferVertexEntry* firstEntry) {
    CHECK_VECTORID_IS_VALID(first, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_CENTROID(first, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_VALID(second, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_CENTROID(second, LOG_TAG_BUFFER);
    FatalAssert(first != second, LOG_TAG_BUFFER, "first and second cannot be the same!");

    if (first > second) {
        std::swap(first, second);
    }

    if (firstEntry != nullptr) {
        FatalAssert(firstEntry->centroidMeta.selfId == first, LOG_TAG_BUFFER,
                    "firstEntry does not correspond to first id!");
    } else {
        bufferMgrLock.Lock(SX_SHARED);
        firstEntry = GetVertexEntry(first, false);
        if (firstEntry == nullptr) {
            FatalAssert(false, LOG_TAG_BUFFER, "one of the entries does not exist!");
            bufferMgrLock.Unlock();
            return;
        }
        bufferMgrLock.Unlock();
    }

    firstEntry->headerLock.Lock(SX_EXCLUSIVE);
    auto it = firstEntry->migrationTasks.find(second);
    FatalAssert(it != firstEntry->migrationTasks.end(), LOG_TAG_BUFFER,
                "the migration task to be removed does not exist!");
    firstEntry->migrationTasks.erase(it);
    firstEntry->headerLock.Unlock();
}

bool BufferManager::AddCompactionTaskIfNotExists(VectorID id, BufferVertexEntry* entry) {
    CHECK_VECTORID_IS_VALID(id, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_CENTROID(id, LOG_TAG_BUFFER);

    if (entry != nullptr) {
        FatalAssert(entry->centroidMeta.selfId == id, LOG_TAG_BUFFER,
                    "entry does not correspond to id!");
    } else {
        bufferMgrLock.Lock(SX_SHARED);
        entry = GetVertexEntry(id, false);
        if (entry == nullptr) {
            FatalAssert(false, LOG_TAG_BUFFER, "one of the entries does not exist!");
            bufferMgrLock.Unlock();
            return false;
        }
        bufferMgrLock.Unlock();
    }

    return !(entry->hasCompactionTask.exchange(true));
}

void BufferManager::RemoveCompactionTask(VectorID id, BufferVertexEntry* entry) {
    CHECK_VECTORID_IS_VALID(id, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_CENTROID(id, LOG_TAG_BUFFER);

    if (entry != nullptr) {
        FatalAssert(entry->centroidMeta.selfId == id, LOG_TAG_BUFFER,
                    "entry does not correspond to id!");
    } else {
        bufferMgrLock.Lock(SX_SHARED);
        entry = GetVertexEntry(id, false);
        if (entry == nullptr) {
            FatalAssert(false, LOG_TAG_BUFFER, "one of the entries does not exist!");
            bufferMgrLock.Unlock();
            return;
        }
        bufferMgrLock.Unlock();
    }

    entry->hasCompactionTask.store(false, std::memory_order_release);
}

bool BufferManager::AddMergeTaskIfNotExists(VectorID id, BufferVertexEntry* entry) {
    CHECK_VECTORID_IS_VALID(id, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_CENTROID(id, LOG_TAG_BUFFER);

    if (entry != nullptr) {
        FatalAssert(entry->centroidMeta.selfId == id, LOG_TAG_BUFFER,
                    "entry does not correspond to id!");
    } else {
        bufferMgrLock.Lock(SX_SHARED);
        entry = GetVertexEntry(id, false);
        if (entry == nullptr) {
            FatalAssert(false, LOG_TAG_BUFFER, "one of the entries does not exist!");
            bufferMgrLock.Unlock();
            return false;
        }
        bufferMgrLock.Unlock();
    }

    return !(entry->hasMergeTask.exchange(true));
}

void BufferManager::RemoveMergeTask(VectorID id, BufferVertexEntry* entry) {
    CHECK_VECTORID_IS_VALID(id, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_CENTROID(id, LOG_TAG_BUFFER);

    if (entry != nullptr) {
        FatalAssert(entry->centroidMeta.selfId == id, LOG_TAG_BUFFER,
                    "entry does not correspond to id!");
    } else {
        bufferMgrLock.Lock(SX_SHARED);
        entry = GetVertexEntry(id, false);
        if (entry == nullptr) {
            FatalAssert(false, LOG_TAG_BUFFER, "one of the entries does not exist!");
            bufferMgrLock.Unlock();
            return;
        }
        bufferMgrLock.Unlock();
    }

    entry->hasMergeTask.store(false, std::memory_order_release);
}


VectorID BufferManager::GetRandomCentroidIdAtLayer(uint8_t level, VectorID exclude,
                                                   bool need_lock, uint64_t* num_retries) {
    FatalAssert(MAX_TREE_HIGHT > (uint64_t)level && (uint64_t)level > VectorID::VECTOR_LEVEL, LOG_TAG_BUFFER,
                "Level is out of bounds.");
    const uint64_t levelIdx = level - 1;
    if (need_lock) {
        bufferMgrLock.Lock(SX_SHARED);
    }
    threadSelf->SanityCheckLockHeldByMe(&bufferMgrLock);

    if ((uint64_t)level >= VectorID::AsID(currentRootId.load(std::memory_order_relaxed)).Level()) {
        if (need_lock) {
            bufferMgrLock.Unlock();
        }
        return INVALID_VECTOR_ID;
    }

    VectorID ret = exclude;
    uint64_t num_retry = 0;
    if (num_retries == nullptr) {
        num_retries = &num_retry;
    }
    /* todo: add a stat collection code here to see how many times this fails and rethink if it is a bottelneck */
    while (ret == exclude) {
        ret = VectorID::CreateCentroidID(level, threadSelf->UniformRange64(0, clusterDirectory[levelIdx].size() - 1));
        BufferVertexEntry* vertex = GetVertexEntry(ret, false);
        if (vertex == nullptr || vertex->state.load(std::memory_order_acquire) != CLUSTER_STABLE) {
            ret = exclude;
        }
        ++(*num_retries);
        if ((*num_retries) >= Thread::MAX_RETRY) {
            ret = INVALID_VECTOR_ID;
            break;
        }
    }

    if (need_lock) {
        bufferMgrLock.Unlock();
    }

    return ret;
}

std::pair<VectorID, VectorID> BufferManager::GetTwoRandomCentroidIdAtLayer(uint8_t level, bool need_lock) {
    FatalAssert(MAX_TREE_HIGHT > (uint64_t)level && (uint64_t)level > VectorID::VECTOR_LEVEL, LOG_TAG_BUFFER,
                "Level is out of bounds.");
    const uint64_t levelIdx = level - 1;
    if (need_lock) {
        bufferMgrLock.Lock(SX_SHARED);
    }
    threadSelf->SanityCheckLockHeldByMe(&bufferMgrLock);

    if ((uint64_t)level >= VectorID::AsID(currentRootId.load(std::memory_order_relaxed)).Level()) {
        if (need_lock) {
            bufferMgrLock.Unlock();
        }
        return std::make_pair(INVALID_VECTOR_ID, INVALID_VECTOR_ID);
    }

    uint64_t num_retries = 0;

    VectorID first = INVALID_VECTOR_ID;
    VectorID second = INVALID_VECTOR_ID;

    /* todo: add a stat collection code here to see how many times this fails and rethink if it is a bottelneck */
    while (true) {
        auto index = threadSelf->UniformRangeTwo64(0, clusterDirectory[levelIdx].size() - 1);
        first = VectorID::CreateCentroidID(level, index.first);

        if (index.first == index.second) {
            BufferVertexEntry* vertex = GetVertexEntry(first, false);
            if (vertex == nullptr || vertex->state.load(std::memory_order_acquire) != CLUSTER_STABLE) {
                ++num_retries;
                if (num_retries >= Thread::MAX_RETRY) {
                    first = INVALID_VECTOR_ID;
                    second = INVALID_VECTOR_ID;
                    break;
                }
                continue;
            }

            second = GetRandomCentroidIdAtLayer(level, first, false, &num_retries);
            if (second == INVALID_VECTOR_ID) {
                first = INVALID_VECTOR_ID;
            }
            break;
        }
        second = VectorID::CreateCentroidID(level, index.second);

        BufferVertexEntry* first_vertex = GetVertexEntry(first, false);
        BufferVertexEntry* second_vertex = GetVertexEntry(second, false);
        bool first_unusable =
            (first_vertex == nullptr || first_vertex->state.load(std::memory_order_acquire) != CLUSTER_STABLE);
        bool second_unusable =
            (second_vertex == nullptr || second_vertex->state.load(std::memory_order_acquire) != CLUSTER_STABLE);
        if (first_unusable && second_unusable) {
            ++num_retries;
            if (num_retries >= Thread::MAX_RETRY) {
                first = INVALID_VECTOR_ID;
                second = INVALID_VECTOR_ID;
                break;
            }
            continue;
        }

        if (first_unusable) {
            std::swap(first, second);
            second_unusable = true;
        }

        if (second_unusable) {
            second = GetRandomCentroidIdAtLayer(level, first, false, &num_retries);
        }

        if ((num_retries >= Thread::MAX_RETRY) || (first == INVALID_VECTOR_ID) || (second == INVALID_VECTOR_ID)) {
            first = INVALID_VECTOR_ID;
            second = INVALID_VECTOR_ID;
        }
        break;
    }

    if (need_lock) {
        bufferMgrLock.Unlock();
    }
    return std::make_pair(first, second);
}

VectorID BufferManager::GetRandomCentroidIdAtNonRootLayer(VectorID exclude) {
    bufferMgrLock.Lock(SX_SHARED);
    uint8_t max_level = VectorID::AsID(currentRootId.load(std::memory_order_relaxed)).Level() - 1;
    if (max_level < 2) {
        bufferMgrLock.Unlock();
        return INVALID_VECTOR_ID;
    }
    uint8_t level = threadSelf->UniformRange32(1, max_level);
    VectorID ret = GetRandomCentroidIdAtLayer(level, exclude, false);
    bufferMgrLock.Unlock();
    return ret;
}

std::pair<VectorID, VectorID> BufferManager::GetTwoRandomCentroidIdAtNonRootLayer() {
    bufferMgrLock.Lock(SX_SHARED);
    uint8_t max_level = VectorID::AsID(currentRootId.load(std::memory_order_relaxed)).Level() - 1;
    if (max_level < 2) {
        bufferMgrLock.Unlock();
        return std::make_pair(INVALID_VECTOR_ID, INVALID_VECTOR_ID);
    }
    uint8_t level = threadSelf->UniformRange32(1, max_level);
    std::pair<VectorID, VectorID> ret = GetTwoRandomCentroidIdAtLayer(level, false);
    bufferMgrLock.Unlock();
    return ret;
}

inline BufferVertexEntry* BufferManager::GetVertexEntry(VectorID id, bool needLock) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    CHECK_VECTORID_IS_VALID(id, LOG_TAG_BUFFER);
    CHECK_VECTORID_IS_CENTROID(id, LOG_TAG_BUFFER);
    uint64_t levelIdx = id.Level() - 1;
    FatalAssert(MAX_TREE_HIGHT > levelIdx, LOG_TAG_BUFFER, "Level is out of bounds. VertexID=" VECTORID_LOG_FMT,
                VECTORID_LOG(id));
    if (needLock) {
        bufferMgrLock.Lock(SX_SHARED);
    }
    threadSelf->SanityCheckLockHeldByMe(&bufferMgrLock);
    FatalAssert(clusterDirectory[levelIdx].size() > id.Value(), LOG_TAG_BUFFER, "VertexID val is out of bounds. "
                VECTORID_LOG_FMT ", max_val:%lu", VECTORID_LOG(id), clusterDirectory[levelIdx].size());
    BufferVertexEntry* entry = clusterDirectory[levelIdx][id.Value()];
    if (needLock) {
        bufferMgrLock.Unlock();
    }
    return entry;
}

inline BufferVertexEntry* BufferManager::GetRootEntry(bool needLock) {
    FatalAssert(bufferMgrInstance == this, LOG_TAG_BUFFER, "Buffer not initialized");
    if (needLock) {
        bufferMgrLock.Lock(SX_SHARED);
    }
    threadSelf->SanityCheckLockHeldByMe(&bufferMgrLock);
    BufferVertexEntry* entry = GetVertexEntry(currentRootId.load(std::memory_order_relaxed), false);
    if (needLock) {
        bufferMgrLock.Unlock();
    }
    return entry;
}

};

#endif