#ifndef CN_BUFFER_H_
#define CN_BUFFER_H_

#include "common.h"
#include "debug.h"

#include "utils/single_page_memory_pool.h"
#include "utils/rdma_manager.h"
#include "utils/concurrent_datastructures.h"

#define MAX_NUM_PAGE_PER_CLUSTER 4

namespace divftree {

enum class BufferEntryState : uint8_t {
    BUFFER_ENTRY_CACHED,
    BUFFER_ENTRY_COOLING,
    BUFFER_ENTRY_EVICTED,
    BUFFER_ENTRY_LOADING
};

struct BufferEntry {
    SXSpinLock lock;
    size_t pin = 0;
    BufferEntryState state;
    void* pages[MAX_NUM_PAGE_PER_CLUSTER] = {nullptr};
    size_t page_num_elements[MAX_NUM_PAGE_PER_CLUSTER];
    size_t num_pages = 0;

    uintptr_t remote_addr;
    size_t total_size_bytes;

    size_t cache_list_idx;

    std::vector<IVFSearchTaskFactory*> pending;

    BufferEntry(uintptr_t raddr, size_t size_bytes, size_t page_size, uint16_t dimension, bool is_leaf) :
        state(BufferEntryState::BUFFER_ENTRY_EVICTED),
        remote_addr(raddr), total_size_bytes(size_bytes) {
        FatalAssert(page_size > 0, LOG_TAG_BUFFER,
                    "Page size must be greater than 0 in BufferEntry constructor");
        /* todo: alignment */
        size_t element_size = (is_leaf ? sizeof(IVFVectorID) : sizeof(VectorID)) + sizeof(VTYPE) * dimension;
        size_t num_vectors_per_page = page_size / element_size;
        FatalAssert(total_size_bytes % element_size == 0, LOG_TAG_BUFFER,
                    "Cluster size is not aligned with vector size in BufferEntry constructor");
        size_t total_num_vectors = total_size_bytes / element_size;
        num_pages = (total_num_vectors + num_vectors_per_page - 1) / num_vectors_per_page;
        FatalAssert(num_pages <= MAX_NUM_PAGE_PER_CLUSTER, LOG_TAG_BUFFER,
                    "Cluster requires more than MAX_NUM_PAGE_PER_CLUSTER pages in BufferEntry constructor");
        FatalAssert(total_num_vectors > 0, LOG_TAG_BUFFER,
                    "Cluster must contain at least one vector in BufferEntry constructor");
        for (size_t p = 0; p < num_pages; ++p) {
            if (p == num_pages - 1) {
                page_num_elements[p] = total_num_vectors - (num_vectors_per_page * p);
            } else {
                page_num_elements[p] = num_vectors_per_page;
            }
        }
    }

    BufferEntry(BufferEntry&& other) noexcept {
        FatalAssert(!other.lock.IsLocked(), LOG_TAG_BUFFER,
                    "Cannot move a locked BufferEntry");
        pin = other.pin;
        state = other.state;
        num_pages = other.num_pages;
        remote_addr = other.remote_addr;
        total_size_bytes = other.total_size_bytes;
        cache_list_idx = other.cache_list_idx;
        pending = std::move(other.pending);
        for (size_t p = 0; p < num_pages; ++p) {
            pages[p] = other.pages[p];
            page_num_elements[p] = other.page_num_elements[p];
            other.pages[p] = nullptr;
            other.page_num_elements[p] = 0;
        }
        other.pin = 0;
        other.num_pages = 0;
    }

    inline BufferEntry& operator=(BufferEntry&& other) noexcept {
        FatalAssert(!lock.IsLocked(), LOG_TAG_BUFFER,
                    "Cannot move-assign to a locked BufferEntry");
        FatalAssert(!other.lock.IsLocked(), LOG_TAG_BUFFER,
                    "Cannot move-assign from a locked BufferEntry");
        pin = other.pin;
        state = other.state;
        num_pages = other.num_pages;
        remote_addr = other.remote_addr;
        total_size_bytes = other.total_size_bytes;
        cache_list_idx = other.cache_list_idx;
        pending = std::move(other.pending);
        for (size_t p = 0; p < num_pages; ++p) {
            pages[p] = other.pages[p];
            page_num_elements[p] = other.page_num_elements[p];
            other.pages[p] = nullptr;
            other.page_num_elements[p] = 0;
        }
        other.pin = 0;
        other.num_pages = 0;
        return *this;
    }
};

class CacheMetaContainer {
public:
    CacheMetaContainer(size_t capacity, size_t num_buckets, MemoryPool* pool) :
        _num_buckets(num_buckets), _bucket_cap(std::max((size_t)1, capacity / num_buckets)),
        _page_pool(pool), _hash(PtrHash<BufferEntry>()),
        _num_hot_entries(0) {
        FatalAssert(capacity > 0, LOG_TAG_BUFFER,
                    "CacheMetaContainer capacity must be greater than 0");
        FatalAssert(num_buckets > 0, LOG_TAG_BUFFER,
                    "CacheMetaContainer num_buckets must be greater than 0");
        FatalAssert(capacity >= num_buckets,
                    LOG_TAG_BUFFER,
                    "CacheMetaContainer capacity must be at least num_buckets");
        FatalAssert(pool != nullptr, LOG_TAG_BUFFER,
                    "CacheMetaContainer memory pool cannot be null");

        _hot_entries = new std::vector<BufferEntry*>[num_buckets];
        _cooling_entries = new BufferEntry*[capacity];
        memset(_cooling_entries, 0, sizeof(BufferEntry*) * capacity);
        _cooling_bucket_next_idx = new size_t[num_buckets];
        memset(_cooling_bucket_next_idx, 0, sizeof(size_t) * num_buckets);
        _locks = new SXSpinLock[num_buckets];
    }

    ~CacheMetaContainer() {
        delete[] _hot_entries;
        delete[] _cooling_entries;
        delete[] _cooling_bucket_next_idx;
        delete[] _locks;
    }

    bool TryLockAndPinEntry(BufferEntry* entry) {
        FatalAssert(entry != nullptr, LOG_TAG_BUFFER,
                    "Cannot pin a null entry in CacheMetaContainer");
        size_t hash_value = _hash(entry);
        size_t bucket_idx = hash_value % _num_buckets;
        _locks[bucket_idx].Lock(SX_EXCLUSIVE);
        if (!entry->lock.TryLock(SX_EXCLUSIVE)) {
            _locks[bucket_idx].Unlock();
            return false;
        }

        entry->pin += entry->num_pages;
        FatalAssert(entry->pin >= entry->num_pages, LOG_TAG_BUFFER,
                    "Pin count overflow in CacheMetaContainer::TryLockAndPinEntry()");
        if (entry->pin > entry->num_pages) {
            FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_CACHED ||
                        entry->state == BufferEntryState::BUFFER_ENTRY_LOADING,
                        LOG_TAG_BUFFER,
                        "Pinned entry must be in CACHED state in CacheMetaContainer::TryLockAndPinEntry()");
            _locks[bucket_idx].Unlock();
            return true;
        }

        if (entry->state == BufferEntryState::BUFFER_ENTRY_CACHED) {
            FatalAssert(_hot_entries[bucket_idx].size() > entry->cache_list_idx,
                        LOG_TAG_BUFFER,
                        "Pinned entry's cache list index is out of bounds in CacheMetaContainer::TryLockAndPinEntry()");
            FatalAssert(entry == _hot_entries[bucket_idx][entry->cache_list_idx],
                        LOG_TAG_BUFFER,
                        "Pinned entry must be in hot entries list in CacheMetaContainer::TryLockAndPinEntry()");
            if (entry->cache_list_idx != _hot_entries[bucket_idx].size() - 1) {
                BufferEntry* last_entry = _hot_entries[bucket_idx].back();
                _hot_entries[bucket_idx][entry->cache_list_idx] = last_entry;
                last_entry->cache_list_idx = entry->cache_list_idx;
            }
            _hot_entries[bucket_idx].pop_back();
            _num_hot_entries.fetch_sub(1);
        } else if (entry->state == BufferEntryState::BUFFER_ENTRY_COOLING) {
            size_t idx = entry->cache_list_idx;
            FatalAssert(_cooling_entries[idx] == entry,
                        LOG_TAG_BUFFER,
                        "Pinned entry must be in cooling entries list in CacheMetaContainer::TryLockAndPinEntry()");
            FatalAssert(idx >= bucket_idx * _bucket_cap &&
                        idx < (bucket_idx + 1) * _bucket_cap,
                        LOG_TAG_BUFFER,
                        "Pinned entry's cache list index is out of bounds in CacheMetaContainer::TryLockAndPinEntry()");
            if (idx != _cooling_bucket_next_idx[bucket_idx]) {
                BufferEntry* last_entry = _cooling_entries[_cooling_bucket_next_idx[bucket_idx]];
                _cooling_entries[idx] = last_entry;
                last_entry->cache_list_idx = idx;
            }
            _cooling_entries[idx] = nullptr;
        } else {
            FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_EVICTED,
                        LOG_TAG_BUFFER,
                        "Pinned entry must be in EVICTED state in CacheMetaContainer::TryLockAndPinEntry()");
        }
        _locks[bucket_idx].Unlock();
        return true;
    }

    void UnpinEntry(BufferEntry* entry) {
        CHECK_NOT_NULLPTR(entry, LOG_TAG_BUFFER);
        FatalAssert(entry->pin > 0, LOG_TAG_BUFFER,
                    "Cannot unpin an entry with pin count 0 in CacheMetaContainer::UnpinEntry()");
        FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_CACHED,
                    LOG_TAG_BUFFER,
                    "Unpinned entry must be in CACHED state in CacheMetaContainer::UnpinEntry()");
        size_t hash_value = _hash(entry);
        size_t bucket_idx = hash_value % _num_buckets;
        _locks[bucket_idx].Lock(SX_EXCLUSIVE);
        entry->lock.Lock(SX_EXCLUSIVE);
        --(entry->pin);
        if (entry->pin == 0) {
            _hot_entries[bucket_idx].push_back(entry);
            entry->cache_list_idx = _hot_entries[bucket_idx].size() - 1;
            _num_hot_entries.fetch_add(1);
        }
        entry->lock.Unlock();
        _locks[bucket_idx].Unlock();
    }

    size_t TryMoveToCooling(uint64_t num_pages, void** freed_pages, size_t max_pages_needed) {
        FatalAssert(num_pages > 0, LOG_TAG_BUFFER,
                    "num_pages must be greater than 0 in CacheMetaContainer::TryMoveToCooling()");
        FatalAssert(num_pages <= ((_num_buckets * _bucket_cap) / 10), LOG_TAG_BUFFER,
                    "num_pages exceeds total capacity in CacheMetaContainer::TryMoveToCooling()");
        CHECK_NOT_NULLPTR(freed_pages, LOG_TAG_BUFFER);
        FatalAssert(max_pages_needed > 0, LOG_TAG_BUFFER,
                    "max_pages_needed must be greater than 0 in CacheMetaContainer::TryMoveToCooling()");

        /* todo: check stats */
        size_t num_cooling = 0;
        size_t num_freed = 0;
        while (num_cooling < num_pages) {
            if (_num_hot_entries.load(std::memory_order_acquire) < num_pages) {
                return num_freed;
            }

            size_t bucket_idx = threadSelf->UniformRange64(0, _num_buckets - 1);
            if (_locks[bucket_idx].TryLock(SX_EXCLUSIVE)) {
                continue;
            }

            if (_hot_entries[bucket_idx].empty()) {
                _locks[bucket_idx].Unlock();
                continue;
            }

            size_t hot_idx = threadSelf->UniformRange64(0, _hot_entries[bucket_idx].size() - 1);
            BufferEntry* entry = _hot_entries[bucket_idx][hot_idx];
            if (hot_idx != _hot_entries[bucket_idx].size() - 1) {
                BufferEntry* last_entry = _hot_entries[bucket_idx].back();
                _hot_entries[bucket_idx][hot_idx] = last_entry;
                last_entry->cache_list_idx = hot_idx;
            }
            _hot_entries[bucket_idx].pop_back();
            _num_hot_entries.fetch_sub(1);
            FatalAssert(entry != nullptr, LOG_TAG_BUFFER,
                        "Hot entry is null in CacheMetaContainer::TryMoveToCooling()");
            FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_CACHED,
                        LOG_TAG_BUFFER,
                        "Hot entry is not in CACHED state in CacheMetaContainer::TryMoveToCooling()");
            FatalAssert(entry->pin == 0,
                        LOG_TAG_BUFFER,
                        "Hot entry is pinned in CacheMetaContainer::TryMoveToCooling()");
            entry->lock.Lock(SX_EXCLUSIVE);
            _locks[bucket_idx].Unlock();

            // move to cooling list
            entry->state = BufferEntryState::BUFFER_ENTRY_COOLING;
            size_t idx = _cooling_bucket_next_idx[bucket_idx] + _bucket_cap * bucket_idx;
            if (_cooling_entries[idx] != nullptr) {
                FatalAssert(_cooling_entries[idx]->state == BufferEntryState::BUFFER_ENTRY_COOLING,
                            LOG_TAG_BUFFER,
                            "Cooling entry slot is occupied by a non-cooling entry in CacheMetaContainer::TryMoveToCooling()");
                FatalAssert(_cooling_entries[idx]->pin == 0,
                            LOG_TAG_BUFFER,
                            "Cooling entry slot is occupied by a pinned entry in CacheMetaContainer::TryMoveToCooling()");
                _cooling_entries[idx]->state = BufferEntryState::BUFFER_ENTRY_EVICTED;
                if (num_freed < max_pages_needed) {
                    size_t num_needed = std::min(_cooling_entries[idx]->num_pages, max_pages_needed - num_freed);
                    for (size_t p = 0; p < num_needed; ++p) {
                        freed_pages[num_freed++] = _cooling_entries[idx]->pages[p];
                    }
                    if (num_needed < _cooling_entries[idx]->num_pages) {
                        _page_pool->BatchFree(_cooling_entries[idx]->pages + num_needed,
                                              _cooling_entries[idx]->num_pages - num_needed);
                    }
                } else {
                    _page_pool->BatchFree(_cooling_entries[idx]->pages, _cooling_entries[idx]->num_pages);
                }
            }

            _cooling_entries[idx] = entry;
            entry->cache_list_idx = idx;
            _cooling_bucket_next_idx[bucket_idx] = (_cooling_bucket_next_idx[bucket_idx] + 1) % _bucket_cap;

            num_cooling += entry->num_pages;
            entry->lock.Unlock();
        }

        return num_freed;
    }

    inline size_t NumHotEntries() {
        return _num_hot_entries.load(std::memory_order_acquire);
    }

protected:
    const size_t _num_buckets;
    const size_t _bucket_cap;
    MemoryPool* _page_pool;
    PtrHash<BufferEntry> _hash;

    std::atomic<size_t> _num_hot_entries;
    std::vector<BufferEntry*>* _hot_entries;
    BufferEntry** _cooling_entries;
    size_t* _cooling_bucket_next_idx;
    SXSpinLock* _locks;
};

class BufferMgr {
public:
    /* constructs the memory pool and connection manager + connects to MNs + gets the metadata */
    static RetStatus Init(size_t page_size, size_t pool_size, BlockingQueue<IVFSearchTask*>* search_task_queue,
                          uint16_t dim, size_t num_user_threads,
                          ClusterMeta*& centroids, VTYPE*& centroid_data, size_t& num_centroids) {
        FatalAssert(instance == nullptr, LOG_TAG_BUFFER,
                    "BufferMgr is already initialized!");
        instance = new BufferMgr(page_size, pool_size, search_task_queue, dim,
                                 num_user_threads, centroids, centroid_data, num_centroids);
        return RetStatus::Success();
    }

    static RetStatus Destroy() {
        FatalAssert(instance != nullptr, LOG_TAG_BUFFER,
                    "BufferMgr is not initialized!");
        std::vector<VectorID> completed_tasks;
        RDMA_Manager::DestroyInstance(completed_tasks);
        FatalAssert(completed_tasks.empty(), LOG_TAG_BUFFER,
                    "There are unprocessed completed RDMA tasks during BufferMgr destruction!");
        delete instance;
        instance = nullptr;
        return RetStatus::Success();
    }

    static BufferMgr* GetInstance() {
        FatalAssert(instance != nullptr, LOG_TAG_BUFFER,
                    "BufferMgr is not initialized!");
        return instance;
    }

    void UnpinCluster(VectorID cluster_id) {
        FatalAssert(this == instance, LOG_TAG_BUFFER,
                    "BufferMgr instance mismatch in BufferMgr::UnpinCluster()");
        auto it = _buffer_map.find(cluster_id);
        FatalAssert(it != _buffer_map.end(), LOG_TAG_BUFFER,
                    "Cluster ID not found in BufferMgr::UnpinCluster()");
        BufferEntry& entry = it->second;
        FatalAssert(entry.pin > 0, LOG_TAG_BUFFER,
                    "Cannot unpin a cluster with pin count 0 in BufferMgr::UnpinCluster()");
        _cache_meta_container.UnpinEntry(&entry);
    }

    RetStatus PrefetchClustersForSearch(const VectorID* cluster_ids, size_t num_clusters,
                                        IVFSearchTaskFactory* task_factory) {
        FatalAssert(this == instance, LOG_TAG_BUFFER,
                    "BufferMgr instance mismatch in BufferMgr::PrefetchClusters()");
        CHECK_NOT_NULLPTR(cluster_ids, LOG_TAG_BUFFER);
        FatalAssert(num_clusters > 0, LOG_TAG_BUFFER,
                    "num_clusters must be greater than 0 in BufferMgr::PrefetchClusters()");

        RetStatus status = RetStatus::Success();
        RDMA_Manager* rdma_mgr = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_mgr, LOG_TAG_BUFFER);
        task_factory->num_sibling_tasks = 0;

        std::vector<size_t> current_indices;
        current_indices.reserve(num_clusters);
        for (size_t i = 0; i < num_clusters; ++i) {
            auto it = _buffer_map.find(cluster_ids[i]);
            FatalAssert(it != _buffer_map.end(), LOG_TAG_BUFFER,
                        "Cluster ID not found in BufferMgr::PrefetchClusters()");
            task_factory->num_sibling_tasks += it->second.num_pages;
            current_indices.push_back(i);
        }

        std::vector<IVFSearchTask*> in_cache_tasks;
        in_cache_tasks.reserve(task_factory->num_sibling_tasks);
        std::vector<VectorID> to_load;
        to_load.reserve(num_clusters);
        size_t element_size = (task_factory->is_leaf ? sizeof(IVFVectorID) : sizeof(VectorID)) + sizeof(VTYPE) * _dim;
        std::vector<size_t> remaining;
        remaining.reserve(num_clusters);
        size_t num_pages_to_load = 0;
        while (!current_indices.empty()) {
            for (size_t i = 0; i < num_clusters; ++i) {
                auto it = _buffer_map.find(cluster_ids[i]);
                BufferEntry& entry = it->second;
                if (!_cache_meta_container.TryLockAndPinEntry(&entry)) {
                    remaining.push_back(i);
                    continue;
                }

                if (entry.state == BufferEntryState::BUFFER_ENTRY_LOADING) {
                    FatalAssert(!entry.pending.empty(), LOG_TAG_BUFFER,
                                "BufferEntry in LOADING state must have pending tasks");
                    entry.pending.push_back(task_factory);
                    entry.lock.Unlock();
                } else if (entry.state == BufferEntryState::BUFFER_ENTRY_EVICTED) {
                    FatalAssert(entry.pending.empty(), LOG_TAG_BUFFER,
                                "BufferEntry in EVICTED state must not have pending tasks");
                    entry.state = BufferEntryState::BUFFER_ENTRY_LOADING;
                    entry.pending.push_back(task_factory);
                    entry.lock.Unlock();
                    to_load.push_back(cluster_ids[i]);
                    num_pages_to_load += entry.num_pages;
                } else {
                    FatalAssert(entry.pending.empty(), LOG_TAG_BUFFER,
                                "BufferEntry in EVICTED state must not have pending tasks");
                    FatalAssert(entry.state == BufferEntryState::BUFFER_ENTRY_CACHED,
                                LOG_TAG_BUFFER,
                                "BufferEntry must be in CACHED state in BufferMgr::PrefetchClusters()");
                    for (size_t p = 0; p < entry.num_pages; ++p) {
                        FatalAssert(entry.page_num_elements[p] > 0, LOG_TAG_BUFFER,
                                    "Page must contain at least one element in BufferMgr::PrefetchClusters()");
                        FatalAssert(entry.pages[p] != nullptr, LOG_TAG_BUFFER,
                                    "Page pointer cannot be null in BufferMgr::PrefetchClusters()");
                        FatalAssert(entry.page_num_elements[p] * element_size <= _cache.GetPageSize(),
                                    LOG_TAG_BUFFER,
                                    "Page size is smaller than number of elements in BufferMgr::PrefetchClusters()");
                        IVFSearchTask* task = task_factory->CreateTask(
                            cluster_ids[i],
                            entry.page_num_elements[p],
                            reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(entry.pages[p])));
                        in_cache_tasks.push_back(task);
                    }
                    entry.lock.Unlock();
                }
            }

            if (!in_cache_tasks.empty()) {
                bool res = _search_task_queue->BatchPush(in_cache_tasks.data(), in_cache_tasks.size());
                FatalAssert(res, LOG_TAG_BUFFER,
                            "Failed to push in-cache search tasks to search task queue in BufferMgr::PrefetchClusters()");
                UNUSED_VARIABLE(res);
                in_cache_tasks.clear();
            }

            if (remaining.size() == current_indices.size()) {
                // none of the remaining entries could be locked
                DIVFTREE_YIELD();
            }

            current_indices.swap(remaining);
            remaining.clear();
        }

        if (!to_load.empty()) {
            status = ReadFromRemote(std::move(to_load), element_size, num_pages_to_load);
            FatalAssert(status.IsOK(), LOG_TAG_BUFFER,
                        "Failed to issue RDMA reads in BufferMgr::PrefetchClusters(): %s",
                        status.Msg());
        }

        return RetStatus::Success();
    }

    RetStatus PollRemoteReads() {
        RetStatus status = RetStatus::Success();
        RDMA_Manager* rdma_mgr = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_mgr, LOG_TAG_BUFFER);
        /*  to avoid contention */
        if (!_poll_lock.TryLock(SX_EXCLUSIVE)) {
            return status;
        }
        std::vector<VectorID> completed_tasks;
        status = rdma_mgr->PushCompletedReadsToTaskQueue(completed_tasks);
        FatalAssert(status.IsOK(), LOG_TAG_BUFFER,
                    "Failed to poll completed RDMA reads in BufferMgr::PollRemoteReads(): %s",
                    status.Msg());

        std::vector<IVFSearchTask*> new_tasks;
        for (VectorID cluster_id : completed_tasks) {
            auto it = _buffer_map.find(cluster_id);
            FatalAssert(it != _buffer_map.end(), LOG_TAG_BUFFER,
                        "Cluster ID not found in BufferMgr::PollRemoteReads()");
            BufferEntry& entry = it->second;
            entry.lock.Lock(SX_EXCLUSIVE);
            FatalAssert(entry.state == BufferEntryState::BUFFER_ENTRY_LOADING, LOG_TAG_BUFFER,
                        "BufferEntry must be in LOADING state in BufferMgr::PollRemoteReads()");
            FatalAssert(!entry.pending.empty(), LOG_TAG_BUFFER,
                        "BufferEntry in LOADING state must have pending tasks in BufferMgr::PollRemoteReads()");
            FatalAssert(entry.pin > 0, LOG_TAG_BUFFER,
                        "BufferEntry must be pinned in BufferMgr::PollRemoteReads()");
            entry.state = BufferEntryState::BUFFER_ENTRY_CACHED;

            for (IVFSearchTaskFactory* task_factory : entry.pending) {
                size_t element_size = (task_factory->is_leaf ? sizeof(IVFVectorID) : sizeof(VectorID)) + sizeof(VTYPE) * _dim;
                UNUSED_VARIABLE(element_size);
                for (size_t p = 0; p < entry.num_pages; ++p) {
                    FatalAssert(entry.page_num_elements[p] > 0, LOG_TAG_BUFFER,
                                "Page must contain at least one element in BufferMgr::PollRemoteReads()");
                    FatalAssert(entry.pages[p] != nullptr, LOG_TAG_BUFFER,
                                "Page pointer cannot be null in BufferMgr::PollRemoteReads()");
                    FatalAssert(entry.page_num_elements[p] * element_size <= _cache.GetPageSize(),
                                LOG_TAG_BUFFER,
                                "Page size is smaller than number of elements in BufferMgr::PollRemoteReads()");
                    IVFSearchTask* task = task_factory->CreateTask(
                        cluster_id,
                        entry.page_num_elements[p],
                        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(entry.pages[p])));
                    new_tasks.push_back(task);
                }
            }
            entry.pending.clear();
            entry.lock.Unlock();
        }

        if (!new_tasks.empty()) {
            bool res = _search_task_queue->BatchPush(new_tasks.data(), new_tasks.size());
            FatalAssert(res, LOG_TAG_BUFFER,
                        "Failed to push newly created search tasks to search task queue in BufferMgr::PollRemoteReads()");
            UNUSED_VARIABLE(res);
            new_tasks.clear();
        }

        _poll_lock.Unlock();
        return RetStatus::Success();
    }

protected:
    static constexpr double COOLING_SIZE_RATIO = 0.2;

    BufferMgr(size_t page_size, size_t pool_size, BlockingQueue<IVFSearchTask*>* search_task_queue, uint16_t dim,
              size_t num_user_threads, ClusterMeta*& centroids, VTYPE*& centroid_data,
              size_t& num_centroids) :
        _dim(dim), _cache(page_size, pool_size),
        _cache_meta_container(std::max((size_t)((pool_size / page_size) * COOLING_SIZE_RATIO), 1lu),
                              num_user_threads * 2, &_cache),
        _search_task_queue(search_task_queue) {
        FatalAssert(_search_task_queue != nullptr, LOG_TAG_BUFFER,
                    "search_task_queue cannot be null in BufferMgr constructor");
        FatalAssert(num_user_threads > 0, LOG_TAG_BUFFER,
                    "num_user_threads must be greater than 0 in BufferMgr constructor");
        FatalAssert(IS_COMPUTE_NODE(), LOG_TAG_BUFFER,
                    "BufferMgr can only be initialized on compute nodes");
        FatalAssert(page_size > 0, LOG_TAG_BUFFER,
                    "page_size must be greater than 0 in BufferMgr constructor");
        FatalAssert(pool_size >= page_size, LOG_TAG_BUFFER,
                    "pool_size must be at least equal to page_size in BufferMgr constructor");
        FatalAssert(pool_size % page_size == 0, LOG_TAG_BUFFER,
                    "pool_size must be multiple of page_size in BufferMgr constructor");
        FatalAssert(dim > 0, LOG_TAG_BUFFER,
                    "dim must be greater than 0 in BufferMgr constructor");
        FatalAssert(centroids == nullptr, LOG_TAG_BUFFER,
                    "centroids must be null in BufferMgr constructor");
        FatalAssert(centroid_data == nullptr, LOG_TAG_BUFFER,
                    "centroid_data must be null in BufferMgr constructor");
        RetStatus status = RetStatus::Success();
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
                num_user_threads
            );
        FatalAssert(status.IsOK(), LOG_TAG_BUFFER,
                    "Failed to initialize RDMA_Manager in BufferMgr::Init(): %s",
                    status.Msg());
        RDMA_Manager* rdma_mgr = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_mgr, LOG_TAG_BUFFER);
        status = rdma_mgr->RegisterMemory(_cache.GetBaseAddress(), _cache.GetPoolSize());
        FatalAssert(status.IsOK(), LOG_TAG_BUFFER,
                    "Failed to register memory in BufferMgr::Init(): %s",
                    status.Msg());
        status = rdma_mgr->EstablishConnections();
        FatalAssert(status.IsOK(), LOG_TAG_BUFFER,
                    "Failed to establish RDMA connections in BufferMgr::Init(): %s",
                    status.Msg());

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BUFFER,
                "RDMA_Manager initialized successfully in BufferMgr::Init(). Waiting to receive centroids...");
        NodeID mnode_id = rdma_mgr->GetMemoryNodeID();
        rdma_mgr->ReceiveMessage(mnode_id, &num_centroids, sizeof(num_centroids));
        FatalAssert(num_centroids > 0, LOG_TAG_BUFFER,
                    "Number of centroids must be greater than 0 in BufferMgr::Init()");

        centroids = new ClusterMeta[num_centroids];
        centroid_data = new VTYPE[num_centroids * dim];

        rdma_mgr->ReceiveMessage(mnode_id, centroids, sizeof(ClusterMeta) * num_centroids);
        rdma_mgr->ReceiveMessage(mnode_id, centroid_data, sizeof(VTYPE) * num_centroids * dim);

        for (size_t c = 0; c < num_centroids; ++c) {
            _buffer_map.emplace(
                centroids[c].centroid_id,
                BufferEntry(centroids[c].remote_addr, centroids[c].remote_size, _cache.GetPageSize(), dim, true)
            );
        }

    }

    ~BufferMgr() {}

    RetStatus ReadFromRemote(std::vector<VectorID>&& cluster_ids, size_t element_size, size_t num_pages_to_load) {
        RetStatus status = RetStatus::Success();
        RDMA_Manager* rdma_mgr = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_mgr, LOG_TAG_BUFFER);
        NodeID mnode_id = rdma_mgr->GetMemoryNodeID();
        bool use_sg = (num_pages_to_load != cluster_ids.size());
        void** local_buffers = new void*[num_pages_to_load];
        size_t num_allocated = 0;
        while (num_allocated < num_pages_to_load) {
            size_t num_to_alloc = num_pages_to_load - num_allocated;
            num_allocated += _cache_meta_container.TryMoveToCooling(num_to_alloc, local_buffers + num_allocated,
                                                                    num_to_alloc);
            if (num_allocated < num_pages_to_load) {
                num_to_alloc = num_pages_to_load - num_allocated;
                num_allocated +=
                    _cache.BatchAllocate(local_buffers + num_allocated, num_to_alloc,
                                         AllocationFlags{.clear = 0, .non_blocking = 1, .atomic = 0, .unused = 0});
            }

            if (num_allocated < num_pages_to_load) {
                DIVFLOG(LOG_LEVEL_WARNING, LOG_TAG_BUFFER,
                        "Not enough free pages in BufferMgr::ReadFromRemote(), allocated %zu out of %zu needed. Retrying...",
                        num_allocated, num_pages_to_load);
                usleep(1);
            }
        }

        FatalAssert(num_allocated == num_pages_to_load, LOG_TAG_BUFFER,
                    "Failed to allocate enough pages in BufferMgr::ReadFromRemote()");
        /* todo: more efficnet implementation */
        if (use_sg) {
            size_t num_used = 0;
            void*** local_addrs = new void**[cluster_ids.size()];
            size_t** sizes = new size_t*[cluster_ids.size()];
            uint32_t* num_sge = new uint32_t[cluster_ids.size()];
            uintptr_t* remote_addrs = new uintptr_t[cluster_ids.size()];
            for (size_t i = 0; i < cluster_ids.size(); ++i) {
                auto it = _buffer_map.find(cluster_ids[i]);
                FatalAssert(it != _buffer_map.end(), LOG_TAG_BUFFER,
                            "Cluster ID not found in BufferMgr::ReadFromRemote()");
                BufferEntry& entry = it->second;
                remote_addrs[i] = entry.remote_addr;
                num_sge[i] = entry.num_pages;
                local_addrs[i] = local_buffers + num_used;
                sizes[i] = new size_t[entry.num_pages];
                for (size_t p = 0; p < entry.num_pages; ++p) {
                    entry.pages[p] = local_buffers[num_used + p];
                    sizes[i][p] = entry.page_num_elements[p] * element_size;
                    FatalAssert(sizes[i][p] > 0, LOG_TAG_BUFFER,
                                "Page size must be greater than 0 in BufferMgr::ReadFromRemote()");
                    FatalAssert(sizes[i][p] <= _cache.GetPageSize(), LOG_TAG_BUFFER,
                                "Page size is smaller than number of elements in BufferMgr::ReadFromRemote()");
                }
                num_used += entry.num_pages;
            }
            FatalAssert(num_used == num_pages_to_load, LOG_TAG_BUFFER,
                        "Number of pages to load mismatch in BufferMgr::ReadFromRemote()");
            status = rdma_mgr->RDMASGRead(mnode_id, local_addrs, remote_addrs,
                                          reinterpret_cast<uint32_t**>(sizes), num_sge,
                                          cluster_ids.size(), std::move(cluster_ids));
            FatalAssert(status.IsOK(), LOG_TAG_BUFFER,
                        "Failed to issue scatter-gather RDMA read in BufferMgr::ReadFromRemote(): %s",
                        status.Msg());
            delete[] local_addrs;
            delete[] sizes;
            delete[] num_sge;
            delete[] remote_addrs;
        } else {
            uintptr_t* remote_addrs = new uintptr_t[cluster_ids.size()];
            size_t* sizes = new size_t[cluster_ids.size()];
            for (size_t i = 0; i < cluster_ids.size(); ++i) {
                auto it = _buffer_map.find(cluster_ids[i]);
                FatalAssert(it != _buffer_map.end(), LOG_TAG_BUFFER,
                            "Cluster ID not found in BufferMgr::ReadFromRemote()");
                BufferEntry& entry = it->second;
                remote_addrs[i] = entry.remote_addr;
                sizes[i] = entry.total_size_bytes;
                FatalAssert(sizes[i] <= _cache.GetPageSize() * entry.num_pages, LOG_TAG_BUFFER,
                            "Total cluster size exceeds allocated pages in BufferMgr::ReadFromRemote()");
                FatalAssert(sizes[i] > 0, LOG_TAG_BUFFER,
                            "Cluster size must be greater than 0 in BufferMgr::ReadFromRemote()");
                FatalAssert(entry.num_pages == 1, LOG_TAG_BUFFER,
                            "Cluster with multiple pages must use scatter-gather RDMA read in BufferMgr::ReadFromRemote()");
                FatalAssert(entry.page_num_elements[0] * element_size == sizes[i], LOG_TAG_BUFFER,
                            "Cluster size does not match number of elements in BufferMgr::ReadFromRemote()");
                entry.pages[0] = local_buffers[i];
            }
            status = rdma_mgr->RDMARead(mnode_id, local_buffers, remote_addrs,
                                       reinterpret_cast<uint32_t*>(sizes),
                                       cluster_ids.size(), std::move(cluster_ids));
            FatalAssert(status.IsOK(), LOG_TAG_BUFFER,
                        "Failed to issue RDMA read in BufferMgr::ReadFromRemote(): %s",
                        status.Msg());
            delete[] remote_addrs;
            delete[] sizes;
        }

        delete[] local_buffers;
        return RetStatus::Success();
    }

    inline static BufferMgr* instance;
    const uint16_t _dim;
    MemoryPool _cache;
    std::unordered_map<VectorID, BufferEntry, VectorIDHash> _buffer_map;
    CacheMetaContainer _cache_meta_container;
    SXSpinLock _poll_lock;
    BlockingQueue<IVFSearchTask*>* _search_task_queue;
};

};

#endif