#ifndef CN_BUFFER_H_
#define CN_BUFFER_H_

#include "common.h"
#include "debug.h"

#include "utils/memory_pool.h"
#include "utils/rdma_manager.h"
#include "utils/concurrent_datastructures.h"

namespace divftree {

struct IndexMeta {
    IndexType type;
    uint32_t num_points;
    uint8_t num_levels;
    std::vector<CentroidData> top_centroids;

    uint32_t leaf_size_cap;
    uint32_t internal_size_cap;
};

enum class BufferEntryState : uint8_t {
    BUFFER_ENTRY_CACHED,
    BUFFER_ENTRY_COOLING,
    BUFFER_ENTRY_EVICTED,
    BUFFER_ENTRY_LOADING
};

struct MemoryStatsNode {
    size_t num_allocated_pages_leaf;
    size_t num_bytes_in_use_leaf;
    size_t num_allocated_pages_internal;
    size_t num_bytes_in_use_internal;
    MemoryStatsNode* next;

    MemoryStatsNode(size_t num_allocated_pages_leaf, size_t num_bytes_in_use_leaf, size_t num_allocated_pages_internal, size_t num_bytes_in_use_internal) :
        num_allocated_pages_leaf(num_allocated_pages_leaf),
        num_bytes_in_use_leaf(num_bytes_in_use_leaf),
        num_allocated_pages_internal(num_allocated_pages_internal),
        num_bytes_in_use_internal(num_bytes_in_use_internal),
        next(nullptr) {}
    MemoryStatsNode() : num_allocated_pages_leaf(0), num_bytes_in_use_leaf(0), num_allocated_pages_internal(0), num_bytes_in_use_internal(0), next(nullptr) {}
};

inline size_t ComputeClusterSize(bool is_leaf, uint32_t num_elements) {
    return ALIGNED_SIZE((size_t)(num_elements) * (is_leaf ? sizeof(VectorData) : sizeof(CentroidData))) +
           ALIGNED_SIZE(sizeof(ClusterHeaderData));
}

inline size_t ComputeSubClusterSize(bool is_leaf, bool first_page, bool has_single_page, uint32_t num_elements_in_page) {
    if (has_single_page) {
        FatalAssert(first_page, LOG_TAG_BUFFER, "if the cluster has a single page then this should be the first page!");
        return ComputeClusterSize(is_leaf, num_elements_in_page);
    }

    if (first_page) {
        return ALIGNED_SIZE(sizeof(ClusterHeaderData)) +
               (size_t)(num_elements_in_page) * (is_leaf ? sizeof(VectorData) : sizeof(CentroidData));
    }

    return (size_t)(num_elements_in_page) * (is_leaf ? sizeof(VectorData) : sizeof(CentroidData));
}

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

    VectorID id;
#ifdef ENABLE_STAT_COLLECTION
    size_t num_read_local = 0;
    size_t num_read_remote_in_progress = 0;
    size_t num_read_remote = 0;
    size_t num_read = 0;

    size_t num_moved_to_cool = 0;
    size_t num_removed_from_cool = 0;
    size_t num_evicted = 0;
#endif

    BufferEntry(VectorID cluster_id, uintptr_t raddr, size_t size_bytes, size_t num_elements, size_t page_size) :
        state(BufferEntryState::BUFFER_ENTRY_EVICTED),
        remote_addr(raddr), total_size_bytes(size_bytes), id(cluster_id) {
        FatalAssert(page_size > 0, LOG_TAG_BUFFER,
                    "Page size must be greater than 0 in BufferEntry constructor");
        FatalAssert(id.IsCentroid(), LOG_TAG_BUFFER, "cannot create a buffer entry for raw vectors.");
        const bool is_leaf = id.IsLeaf();
        /* todo: alignment */
        FatalAssert(page_size > ALIGNED_SIZE(sizeof(ClusterHeaderData)), LOG_TAG_BUFFER,
                    "Page size must be greater than cluster header size in BufferEntry constructor");
        size_t raw_data_size = page_size - ALIGNED_SIZE(sizeof(ClusterHeaderData));
        size_t element_size = (is_leaf ? sizeof(VectorData) : sizeof(CentroidData));
        size_t num_vectors_per_page = raw_data_size / element_size;
        FatalAssert(num_vectors_per_page > 0, LOG_TAG_BUFFER,
                    "Page size must be large enough to hold at least one vector in BufferEntry constructor");
        FatalAssert(total_size_bytes == ComputeClusterSize(id.IsLeaf(), num_elements), LOG_TAG_BUFFER,
                    "Cluster size is not aligned with vector size in BufferEntry constructor");
        num_pages = (num_elements + num_vectors_per_page - 1) / num_vectors_per_page;
        FatalAssert(num_pages <= MAX_NUM_PAGE_PER_CLUSTER, LOG_TAG_BUFFER,
                    "Cluster requires more than MAX_NUM_PAGE_PER_CLUSTER pages in BufferEntry constructor");
        FatalAssert(num_elements > 0, LOG_TAG_BUFFER,
                    "Cluster must contain at least one vector in BufferEntry constructor");
        for (size_t p = 0; p < num_pages; ++p) {
            if (p == num_pages - 1) {
                page_num_elements[p] = num_elements - (num_vectors_per_page * p);
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
        id = other.id;
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
        id = other.id;
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

struct CacheEntry {
    BufferEntry* entry;
    size_t next;
    size_t prev;
};

class CacheBucket {
public:
    CacheBucket(size_t _cooling_cap) :
        cooling_cap(_cooling_cap), cooling_list_head(_cooling_cap),
        cooling_list_tail(_cooling_cap), cooling_free_list_head(0),
        hot_list_head(_cooling_cap), hot_list_tail(_cooling_cap), hot_free_list_head(0),
        cooling_list_size(0), hot_list_size(0) {
        hot_list.resize(_cooling_cap);
        cooling_list = new CacheEntry[_cooling_cap];
        for (size_t i = 0; i < _cooling_cap; ++i) {
            cooling_list[i].entry = nullptr;
            cooling_list[i].next = i + 1;
            cooling_list[i].prev = _cooling_cap;
            hot_list[i].entry = nullptr;
            hot_list[i].next = i + 1;
            hot_list[i].prev = _cooling_cap;
        }
    }

    ~CacheBucket() {
        delete[] cooling_list;
    }

    CacheBucket(const CacheBucket&) : cooling_cap(0) {
        /* it is needed to use it as a vector */
        FatalAssert(false, LOG_TAG_BUFFER, "CacheBucket copy constructor should not be called");
    }

    CacheBucket& operator=(const CacheBucket&) {
        /* it is needed to use it as a vector */
        FatalAssert(false, LOG_TAG_BUFFER, "CacheBucket copy assignment operator should not be called");
        return *this;
    }

    bool TryLockBucket() {
        return lock.TryLock(SX_EXCLUSIVE);
    }

    void LockBucket() {
        lock.Lock(SX_EXCLUSIVE);
    }

    void UnlockBucket() {
        lock.Unlock();
    }

    void ResizeHotList() {
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        FatalAssert(hot_free_list_head == hot_list.size(), LOG_TAG_BUFFER,
                    "Hot free list head must point to the end of the hot list when resizing in CacheBucket::ResizeHotList()");
        FatalAssert(hot_list_size == hot_list.size(), LOG_TAG_BUFFER,
                    "Hot list size must be equal to hot list capacity when resizing in CacheBucket::ResizeHotList()");
        FatalAssert(hot_list_size > 0, LOG_TAG_BUFFER,
                    "Hot list size must be greater than 0 when resizing in CacheBucket::ResizeHotList()");
        size_t new_cap = hot_list.size() * 2;
        hot_list.resize(new_cap);
        hot_free_list_head = hot_list_size;
        for (size_t i = hot_list_size; i < new_cap; ++i) {
            hot_list[i].entry = nullptr;
            hot_list[i].next = i + 1;
            hot_list[i].prev = new_cap;
        }
        FatalAssert(hot_list[hot_list_tail].next == hot_list_size, LOG_TAG_BUFFER,
                    "Current tail entry's next index must point to the end of the hot list after resizing in CacheBucket::ResizeHotList()");
        FatalAssert(hot_list[hot_list_head].prev == hot_list_size, LOG_TAG_BUFFER,
                    "Current head entry's prev index must point to the end of the hot list after resizing in CacheBucket::ResizeHotList()");
        hot_list[hot_list_tail].next = new_cap;
        hot_list[hot_list_head].prev = new_cap;
    }

    void AddToHotList(BufferEntry* entry) {
        CHECK_NOT_NULLPTR(entry, LOG_TAG_BUFFER);
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        threadSelf->SanityCheckLockHeldByMe(&entry->lock);
        FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_CACHED, LOG_TAG_BUFFER,
                    "Only entries in CACHED state can be added to hot list in CacheBucket::AddToHotList()");
        FatalAssert(entry->pin == 0, LOG_TAG_BUFFER,
                    "Only unpinned entries can be added to hot list in CacheBucket::AddToHotList()");

        if (hot_free_list_head == hot_list.size()) {
            ResizeHotList();
        }

        size_t idx = hot_free_list_head;
        hot_free_list_head = hot_list[idx].next;

        hot_list[idx].entry = entry;
        hot_list[idx].next = hot_list.size();
        hot_list[idx].prev = hot_list_tail;
        if (hot_list_tail != hot_list.size()) {
            FatalAssert(hot_list[hot_list_tail].entry != nullptr, LOG_TAG_BUFFER,
                        "Current tail entry cannot be null in CacheBucket::AddToHotList()");
            FatalAssert(hot_list[hot_list_tail].next == hot_list.size(), LOG_TAG_BUFFER,
                        "Current tail entry's next index must point to the end of the hot list in CacheBucket::AddToHotList()");
            hot_list[hot_list_tail].next = idx;
        } else {
            FatalAssert(hot_list_head == hot_list.size(), LOG_TAG_BUFFER,
                        "Hot list head must point to the end of the hot list when adding the first entry in CacheBucket::AddToHotList()");
            hot_list_head = idx;
        }
        hot_list_tail = idx;
        entry->cache_list_idx = idx;
        hot_list_size++;
    }

    void RemoveFromHotList(BufferEntry* entry) {
        CHECK_NOT_NULLPTR(entry, LOG_TAG_BUFFER);
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        threadSelf->SanityCheckLockHeldByMe(&entry->lock);
        FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_CACHED, LOG_TAG_BUFFER,
                    "Only entries in CACHED state can be removed from hot list in CacheBucket::RemoveFromHotList()");
        FatalAssert(entry->pin > 0, LOG_TAG_BUFFER,
                    "Only pinned entries can be removed from hot list in CacheBucket::RemoveFromHotList()");
        FatalAssert(entry->cache_list_idx < hot_list.size(), LOG_TAG_BUFFER,
                    "Entry's cache list index is out of bounds in CacheBucket::RemoveFromHotList()");
        FatalAssert(entry == hot_list[entry->cache_list_idx].entry, LOG_TAG_BUFFER,
                    "Entry's cache list index does not point to the entry itself in CacheBucket::RemoveFromHotList()");
        size_t prev_idx = hot_list[entry->cache_list_idx].prev;
        size_t next_idx = hot_list[entry->cache_list_idx].next;
        if (prev_idx != hot_list.size()) {
            FatalAssert(hot_list[prev_idx].entry != nullptr, LOG_TAG_BUFFER,
                        "Previous hot entry cannot be null in CacheBucket::RemoveFromHotList()");
            FatalAssert(hot_list[prev_idx].next == entry->cache_list_idx, LOG_TAG_BUFFER,
                        "Previous hot entry's next index must point to the current entry in CacheBucket::RemoveFromHotList()");
            hot_list[prev_idx].next = next_idx;
        } else {
            FatalAssert(hot_list_head == entry->cache_list_idx, LOG_TAG_BUFFER,
                        "Hot list head must point to the removed entry when there is only one entry in the hot list in CacheBucket::RemoveFromHotList()");
            hot_list_head = next_idx;
        }
        if (next_idx != hot_list.size()) {
            FatalAssert(hot_list[next_idx].entry != nullptr, LOG_TAG_BUFFER,
                        "Next hot entry cannot be null in CacheBucket::RemoveFromHotList()");
            FatalAssert(hot_list[next_idx].prev == entry->cache_list_idx, LOG_TAG_BUFFER,
                        "Next hot entry's prev index must point to the current entry in CacheBucket::RemoveFromHotList()");
            hot_list[next_idx].prev = prev_idx;
        } else {
            FatalAssert(hot_list_tail == entry->cache_list_idx, LOG_TAG_BUFFER,
                        "Hot list tail must point to the removed entry when there is only one entry in the hot list in CacheBucket::RemoveFromHotList()");
            hot_list_tail = prev_idx;
        }

        hot_list[entry->cache_list_idx].entry = nullptr;
        hot_list[entry->cache_list_idx].next = hot_free_list_head;
        hot_list[entry->cache_list_idx].prev = hot_list.size();
        hot_free_list_head = entry->cache_list_idx;
        hot_list_size--;
    }

    BufferEntry* PopFromHotList() {
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        if (hot_list_head == hot_list.size()) {
            FatalAssert(hot_list_tail == hot_list.size(), LOG_TAG_BUFFER,
                        "Hot list tail must point to the end of the hot list when hot list is empty in CacheBucket::PopFromHotList()");
            FatalAssert(hot_list_size == 0, LOG_TAG_BUFFER,
                        "Hot list size must be 0 when hot list is empty in CacheBucket::PopFromHotList()");
            return nullptr;
        }

        size_t idx = hot_list_head;
        FatalAssert(hot_list[idx].entry != nullptr, LOG_TAG_BUFFER,
                    "Popped hot entry's entry pointer cannot be null in CacheBucket::PopFromHotList()");
        FatalAssert(hot_list[idx].prev == hot_list.size(), LOG_TAG_BUFFER,
                    "Popped hot entry's prev index must point to the end of the hot list in CacheBucket::PopFromHotList()");
        hot_list_head = hot_list[idx].next;
        if (hot_list_head != hot_list.size()) {
            FatalAssert(hot_list[hot_list_head].entry != nullptr, LOG_TAG_BUFFER,
                        "Next hot entry's entry pointer cannot be null in CacheBucket::PopFromHotList()");
            FatalAssert(hot_list[hot_list_head].prev == idx, LOG_TAG_BUFFER,
                        "Next hot entry's prev index must point to the current entry in CacheBucket::PopFromHotList()");
            hot_list[hot_list_head].prev = hot_list.size();
        } else {
            FatalAssert(hot_list_tail == idx, LOG_TAG_BUFFER,
                        "Hot list tail must point to the popped entry when there is only one entry in the hot list in CacheBucket::PopFromHotList()");
            hot_list_tail = hot_list.size();
        }

        BufferEntry* entry = hot_list[idx].entry;
        hot_list[idx].entry = nullptr;
        hot_list[idx].next = hot_free_list_head;
        hot_list[idx].prev = hot_list.size();
        hot_free_list_head = idx;
        hot_list_size--;
        return entry;
    }

    size_t GetHotListSize() {
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        return hot_list_size;
    }

    bool CoolingListIsEmpty() {
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        SANITY_CHECK(
            FatalAssert(cooling_list_head <= cooling_cap, LOG_TAG_BUFFER,
                        "Cooling list head index out of bounds in CacheBucket::CoolingListIsEmpty()");
            FatalAssert(cooling_list_tail <= cooling_cap, LOG_TAG_BUFFER,
                        "Cooling list tail index out of bounds in CacheBucket::CoolingListIsEmpty()");
            FatalAssert(cooling_list_size <= cooling_cap, LOG_TAG_BUFFER,
                        "Cooling list size cannot be greater than cooling capacity in CacheBucket::CoolingListIsEmpty()");
            FatalAssert((cooling_free_list_head == cooling_cap) == (cooling_list_size == cooling_cap), LOG_TAG_BUFFER,
                        "Inconsistent cooling list state in CacheBucket::CoolingListIsFull()");
            FatalAssert(((cooling_list_head == cooling_cap) == (cooling_list_tail == cooling_cap)), LOG_TAG_BUFFER,
                        "Inconsistent cooling list state in CacheBucket::CoolingListIsEmpty()");
            FatalAssert(((cooling_list_head == cooling_cap) == (cooling_list_size == 0)), LOG_TAG_BUFFER,
                        "Inconsistent cooling list state in CacheBucket::CoolingListIsEmpty()");
            FatalAssert((cooling_list_head < cooling_cap) || (cooling_free_list_head < cooling_cap), LOG_TAG_BUFFER,
                        "Cooling list head and free list head cannot both point to the end of the cooling list in CacheBucket::CoolingListIsEmpty()");
        );
        return cooling_list_tail == cooling_cap;
    }

    bool CoolingListIsFull() {
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        SANITY_CHECK(
            FatalAssert(cooling_list_head <= cooling_cap, LOG_TAG_BUFFER,
                        "Cooling list head index out of bounds in CacheBucket::CoolingListIsEmpty()");
            FatalAssert(cooling_list_tail <= cooling_cap, LOG_TAG_BUFFER,
                        "Cooling list tail index out of bounds in CacheBucket::CoolingListIsEmpty()");
            FatalAssert(cooling_list_size <= cooling_cap, LOG_TAG_BUFFER,
                        "Cooling list size cannot be greater than cooling capacity in CacheBucket::CoolingListIsEmpty()");
            FatalAssert((cooling_free_list_head == cooling_cap) == (cooling_list_size == cooling_cap), LOG_TAG_BUFFER,
                        "Inconsistent cooling list state in CacheBucket::CoolingListIsFull()");
            FatalAssert(((cooling_list_head == cooling_cap) == (cooling_list_tail == cooling_cap)), LOG_TAG_BUFFER,
                        "Inconsistent cooling list state in CacheBucket::CoolingListIsEmpty()");
            FatalAssert(((cooling_list_head == cooling_cap) == (cooling_list_size == 0)), LOG_TAG_BUFFER,
                        "Inconsistent cooling list state in CacheBucket::CoolingListIsEmpty()");
            FatalAssert((cooling_list_head < cooling_cap) || (cooling_free_list_head < cooling_cap), LOG_TAG_BUFFER,
                        "Cooling list head and free list head cannot both point to the end of the cooling list in CacheBucket::CoolingListIsFull()");
        );
        return cooling_free_list_head == cooling_cap;
    }

    void AddToFreeList(size_t idx) {
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        FatalAssert(idx < cooling_cap, LOG_TAG_BUFFER,
                    "Cooling list node index out of bounds in CacheBucket::AddToFreeList()");
        cooling_list[idx].entry = nullptr;
        cooling_list[idx].next = cooling_free_list_head;
        cooling_list[idx].prev = cooling_cap;
        cooling_free_list_head = idx;
    }

    size_t PopCoolingList() {
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        if (CoolingListIsEmpty()) {
            return cooling_cap;
        }

        size_t idx = cooling_list_head;
        FatalAssert(cooling_list[idx].entry != nullptr, LOG_TAG_BUFFER,
                    "Popped cooling entry's entry pointer cannot be null in CacheBucket::PopCoolingList()");
        FatalAssert(cooling_list[idx].prev == cooling_cap, LOG_TAG_BUFFER,
                    "Popped cooling entry's prev index must point to the end of the cooling list in CacheBucket::PopCoolingList()");
        cooling_list_head = cooling_list[idx].next;
        if (cooling_list_head != cooling_cap) {
            FatalAssert(cooling_list[cooling_list_head].prev == idx, LOG_TAG_BUFFER,
                        "Next cooling entry's prev index must point to the current entry in CacheBucket::PopCoolingList()");
            FatalAssert(cooling_list[cooling_list_head].entry != nullptr, LOG_TAG_BUFFER,
                        "Next cooling entry's entry pointer cannot be null in CacheBucket::PopCoolingList()");
            cooling_list[cooling_list_head].prev = cooling_cap;
        } else {
            FatalAssert(cooling_list_tail == idx, LOG_TAG_BUFFER,
                        "Cooling list tail must point to the popped entry when there is only one entry in the cooling list in CacheBucket::PopCoolingList()");
            cooling_list_tail = cooling_cap;
        }
        cooling_list[idx].next = cooling_cap;
        cooling_list_size--;
        return idx;
    }

    size_t GetCoolingListNode() {
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        if (!CoolingListIsFull()) {
            size_t idx = cooling_free_list_head;
            cooling_free_list_head = cooling_list[idx].next;
            cooling_list[idx].next = cooling_cap;
            FatalAssert(cooling_list[idx].entry == nullptr, LOG_TAG_BUFFER,
                        "New cooling list node must have null entry pointer in CacheBucket::GetCoolingListNode()");
            FatalAssert(cooling_list[idx].prev == cooling_cap, LOG_TAG_BUFFER,
                        "New cooling list node must have prev index pointing to the end of the cooling list in CacheBucket::GetCoolingListNode()");
            return idx;
        }

        return PopCoolingList();
    }

    BufferEntry* AddToCoolingList(BufferEntry* entry) {
        CHECK_NOT_NULLPTR(entry, LOG_TAG_BUFFER);
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        threadSelf->SanityCheckLockHeldByMe(&entry->lock);
        FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_COOLING, LOG_TAG_BUFFER,
                    "Only entries in COOLING state can be added to cooling list in CacheBucket::AddToCoolingList()");
        FatalAssert(entry->pin == 0, LOG_TAG_BUFFER,
                    "Only unpinned entries can be added to cooling list in CacheBucket::AddToCoolingList()");
        size_t idx = GetCoolingListNode();
        BufferEntry* evicted_entry = cooling_list[idx].entry;
        cooling_list[idx].entry = entry;
        cooling_list[idx].prev = cooling_list_tail;
        entry->cache_list_idx = idx;
        if (cooling_list_tail != cooling_cap) {
            FatalAssert(cooling_list[cooling_list_tail].entry != nullptr, LOG_TAG_BUFFER,
                        "Current tail entry cannot be null in CacheBucket::AddToCoolingList()");
            cooling_list[cooling_list_tail].next = idx;
        } else {
            FatalAssert(cooling_list_head == cooling_cap, LOG_TAG_BUFFER,
                        "Cooling list head must point to the end of the cooling list when adding the first entry in CacheBucket::AddToCoolingList()");
            cooling_list_head = idx;
        }
        cooling_list_tail = idx;
        cooling_list_size++;

        FatalAssert((evicted_entry == nullptr) || ((evicted_entry->state == BufferEntryState::BUFFER_ENTRY_COOLING) &&
                                                   (evicted_entry->pin == 0)), LOG_TAG_BUFFER,
                    "Evicted entry must be in EVICTED state in CacheBucket::AddToCoolingList()");
        return evicted_entry;
    }

    BufferEntry* EvictFromCoolingList() {
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        if (CoolingListIsEmpty()) {
            return nullptr;
        }

        size_t idx = PopCoolingList();
        BufferEntry* evicted_entry = cooling_list[idx].entry;
        AddToFreeList(idx);

        FatalAssert(evicted_entry != nullptr, LOG_TAG_BUFFER,
                    "Evicted entry cannot be null in CacheBucket::EvictFromCoolingList()");
        FatalAssert(evicted_entry->state == BufferEntryState::BUFFER_ENTRY_COOLING, LOG_TAG_BUFFER,
                    "Evicted entry must be in COOLING state in CacheBucket::EvictFromCoolingList()");
        FatalAssert(evicted_entry->pin == 0, LOG_TAG_BUFFER,
                    "Evicted entry must be unpinned in CacheBucket::EvictFromCoolingList()");
        return evicted_entry;
    }

    void RemoveFromCoolingList(BufferEntry* entry) {
        CHECK_NOT_NULLPTR(entry, LOG_TAG_BUFFER);
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        threadSelf->SanityCheckLockHeldByMe(&entry->lock);
        FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_COOLING, LOG_TAG_BUFFER,
                    "Only entries in COOLING state can be removed from cooling list in CacheBucket::RemoveFromCoolingList()");
        FatalAssert(entry->pin > 0, LOG_TAG_BUFFER,
                    "Only pinned entries can be removed from cooling list in CacheBucket::RemoveFromCoolingList()");
        size_t idx = entry->cache_list_idx;
        FatalAssert(idx < cooling_cap, LOG_TAG_BUFFER,
                    "Cooling list index out of bounds in CacheBucket::RemoveFromCoolingList()");
        FatalAssert(cooling_list[idx].entry == entry, LOG_TAG_BUFFER,
                    "Entry's cache list index does not point to the entry itself in CacheBucket::RemoveFromCoolingList()");
        size_t prev_idx = cooling_list[idx].prev;
        size_t next_idx = cooling_list[idx].next;
        if (prev_idx != cooling_cap) {
            FatalAssert(cooling_list[prev_idx].entry != nullptr, LOG_TAG_BUFFER,
                        "Previous cooling entry cannot be null in CacheBucket::RemoveFromCoolingList()");
            cooling_list[prev_idx].next = next_idx;
        } else {
            FatalAssert(cooling_list_head == idx, LOG_TAG_BUFFER,
                        "Cooling list head must point to the removed entry when there is only one entry in the cooling list in CacheBucket::RemoveFromCoolingList()");
            cooling_list_head = next_idx;
        }
        if (next_idx != cooling_cap) {
            FatalAssert(cooling_list[next_idx].entry != nullptr, LOG_TAG_BUFFER,
                        "Next cooling entry cannot be null in CacheBucket::RemoveFromCoolingList()");
            cooling_list[next_idx].prev = prev_idx;
        } else {
            FatalAssert(cooling_list_tail == idx, LOG_TAG_BUFFER,
                        "Cooling list tail must point to the removed entry when there is only one entry in the cooling list in CacheBucket::RemoveFromCoolingList()");
            cooling_list_tail = prev_idx;
        }
        AddToFreeList(idx);
        cooling_list_size--;
    }

    void FreeAllEntries(LocalMemoryPool& page_pool) {
        threadSelf->SanityCheckLockHeldInModeByMe(&lock, SX_EXCLUSIVE);
        for (size_t i = 0; i < hot_list.size(); ++i) {
            if (hot_list[i].entry != nullptr) {
                BufferEntry* entry = hot_list[i].entry;
                FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_CACHED, LOG_TAG_BUFFER,
                            "Hot entry must be in CACHED state in CacheBucket::FreeAllEntries()");
                FatalAssert(entry->pin == 0, LOG_TAG_BUFFER,
                            "Hot entry must be unpinned in CacheBucket::FreeAllEntries()");
                for (size_t p = 0; p < entry->num_pages; ++p) {
                    page_pool.Free(entry->pages[p]);
                    entry->pages[p] = nullptr;
                }
                entry->num_pages = 0;
                entry->state = BufferEntryState::BUFFER_ENTRY_EVICTED;
            }
            hot_list[i].entry = nullptr;
        }
        for (size_t i = 0; i < cooling_cap; ++i) {
            if (cooling_list[i].entry != nullptr) {
                BufferEntry* entry = cooling_list[i].entry;
                FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_COOLING, LOG_TAG_BUFFER,
                            "Cooling entry must be in COOLING state in CacheBucket::FreeAllEntries()");
                FatalAssert(entry->pin == 0, LOG_TAG_BUFFER,
                            "Cooling entry must be unpinned in CacheBucket::FreeAllEntries()");
                for (size_t p = 0; p < entry->num_pages; ++p) {
                    page_pool.Free(entry->pages[p]);
                    entry->pages[p] = nullptr;
                }
                entry->num_pages = 0;
                entry->state = BufferEntryState::BUFFER_ENTRY_EVICTED;
            }
            cooling_list[i].entry = nullptr;
        }
        cooling_list_head = cooling_cap;
        cooling_list_tail = cooling_cap;
        cooling_free_list_head = 0;
        hot_list_head = 0;
        hot_list_tail = 0;
        hot_free_list_head = 0;
        cooling_list_size = 0;
        hot_list_size = 0;
    }

protected:
    const size_t cooling_cap;
    std::vector<CacheEntry> hot_list;
    CacheEntry* cooling_list;
    size_t cooling_list_head;
    size_t cooling_list_tail;
    size_t cooling_free_list_head;
    size_t hot_list_head;
    size_t hot_list_tail;
    size_t hot_free_list_head;

    size_t cooling_list_size;
    size_t hot_list_size;
    SXSpinLock lock;
};

class CacheMetaContainerDetail {
public:
    CacheMetaContainerDetail(size_t capacity, size_t num_buckets, LocalMemoryPool& pool) :
        _num_buckets(num_buckets), _bucket_cap(std::max((size_t)2, capacity / num_buckets)),
        _page_pool(pool), _hash(PtrHash<BufferEntry>()),
        _num_hot_entries(0), _num_cooling_entries(0) {
        FatalAssert(capacity > 0, LOG_TAG_BUFFER,
                    "CacheMetaContainerDetail capacity must be greater than 0");
        FatalAssert(num_buckets > 0, LOG_TAG_BUFFER,
                    "CacheMetaContainerDetail num_buckets must be greater than 0");
        FatalAssert(capacity >= num_buckets,
                    LOG_TAG_BUFFER,
                    "CacheMetaContainerDetail capacity must be at least num_buckets");

        buckets.reserve(num_buckets);
        for (size_t i = 0; i < num_buckets; ++i) {
            buckets.emplace_back(_bucket_cap);
        }
    }

    ~CacheMetaContainerDetail() = default;

    bool TryLockAndPinEntry(BufferEntry* entry) {
        FatalAssert(entry != nullptr, LOG_TAG_BUFFER,
                    "Cannot pin a null entry in CacheMetaContainerDetail");
        size_t hash_value = _hash(entry);
        size_t bucket_idx = hash_value % _num_buckets;
        buckets[bucket_idx].LockBucket();
        if (!entry->lock.TryLock(SX_EXCLUSIVE)) {
            buckets[bucket_idx].UnlockBucket();
            return false;
        }

        entry->pin += entry->num_pages;
        FatalAssert(entry->pin >= entry->num_pages, LOG_TAG_BUFFER,
                    "Pin count overflow in CacheMetaContainerDetail::TryLockAndPinEntry()");
        if (entry->pin > entry->num_pages) {
            FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_CACHED ||
                        entry->state == BufferEntryState::BUFFER_ENTRY_LOADING,
                        LOG_TAG_BUFFER,
                        "Pinned entry must be in CACHED state in CacheMetaContainerDetail::TryLockAndPinEntry()");
            buckets[bucket_idx].UnlockBucket();
            return true;
        }

        if (entry->state == BufferEntryState::BUFFER_ENTRY_CACHED) {
            buckets[bucket_idx].RemoveFromHotList(entry);
            _num_hot_entries.fetch_sub(1);
        } else if (entry->state == BufferEntryState::BUFFER_ENTRY_COOLING) {
            buckets[bucket_idx].RemoveFromCoolingList(entry);
            _num_cooling_entries.fetch_sub(1);
            entry->state = BufferEntryState::BUFFER_ENTRY_CACHED;
#ifdef ENABLE_STAT_COLLECTION
            (entry->num_removed_from_cool)++;
#endif
        } else {
            FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_EVICTED,
                        LOG_TAG_BUFFER,
                        "Pinned entry must be in EVICTED state in CacheMetaContainerDetail::TryLockAndPinEntry()");
        }
        buckets[bucket_idx].UnlockBucket();
        return true;
    }

    void UnpinEntry(BufferEntry* entry) {
        CHECK_NOT_NULLPTR(entry, LOG_TAG_BUFFER);
        size_t hash_value = _hash(entry);
        size_t bucket_idx = hash_value % _num_buckets;
        buckets[bucket_idx].LockBucket();
        entry->lock.Lock(SX_EXCLUSIVE);
        FatalAssert(entry->pin > 0, LOG_TAG_BUFFER,
                    "Cannot unpin an entry with pin count 0 in CacheMetaContainerDetail::UnpinEntry()");
        FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_CACHED,
                    LOG_TAG_BUFFER,
                    "Unpinned entry must be in CACHED state in CacheMetaContainerDetail::UnpinEntry()");
        --(entry->pin);
        if (entry->pin == 0) {
            buckets[bucket_idx].AddToHotList(entry);
            _num_hot_entries.fetch_add(1);
        }
        entry->lock.Unlock();
        buckets[bucket_idx].UnlockBucket();
    }

    void PopulateCoolingList(size_t num_pages) {
        FatalAssert(num_pages > 0, LOG_TAG_BUFFER,
                    "num_pages must be greater than 0 in CacheMetaContainerDetail::PopulateCoolingList()");
        num_pages = std::min(num_pages, (_num_buckets * _bucket_cap) / 4);
        FatalAssert(num_pages > 0, LOG_TAG_BUFFER,
                    "num_pages must be greater than 0 in CacheMetaContainerDetail::PopulateCoolingList()");

        size_t num_cooled = 0;
        size_t real_num_pages = num_pages;
        while (num_cooled < real_num_pages) {
            size_t num_hot = _num_hot_entries.load(std::memory_order_acquire);
            if (num_hot == 0) { /* if less than some amount */
                break;
            }

            if (num_hot < real_num_pages) {
                real_num_pages = num_hot;
            } else if (num_hot > real_num_pages && real_num_pages < num_pages) {
                real_num_pages = std::min(num_hot, num_pages);
            }

            size_t bucket_idx = threadSelf->UniformRange64(0, _num_buckets - 1);
            if (!buckets[bucket_idx].TryLockBucket()) {
                DIVFTREE_YIELD();
                continue;
            }

            if (buckets[bucket_idx].CoolingListIsFull()) {
                buckets[bucket_idx].UnlockBucket();
                size_t num_cooling = _num_cooling_entries.load(std::memory_order_acquire);
                if (num_cooling * 10 > (_bucket_cap * _num_buckets) * 9) {
                    break;
                } else {
                    DIVFTREE_YIELD();
                    continue;
                }
            }

            if (buckets[bucket_idx].GetHotListSize() == 0) {
                buckets[bucket_idx].UnlockBucket();
                num_hot = _num_hot_entries.load(std::memory_order_acquire);
                if (num_hot < _num_buckets) {
                    break;
                }
                DIVFTREE_YIELD();
                continue;
            }

            BufferEntry* entry = buckets[bucket_idx].PopFromHotList();
            FatalAssert(entry != nullptr, LOG_TAG_BUFFER,
                        "Popped hot entry cannot be null in CacheMetaContainerDetail::PopulateCoolingList()");
            entry->lock.Lock(SX_EXCLUSIVE);
            FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_CACHED,
                        LOG_TAG_BUFFER,
                        "Popped hot entry must be in CACHED state in CacheMetaContainerDetail::PopulateCoolingList()");
            FatalAssert(entry->pin == 0,
                        LOG_TAG_BUFFER,
                        "Popped hot entry must be unpinned in CacheMetaContainerDetail::PopulateCoolingList()");

            entry->state = BufferEntryState::BUFFER_ENTRY_COOLING;
            BufferEntry* evicted = buckets[bucket_idx].AddToCoolingList(entry);
            entry->lock.Unlock();
            _num_hot_entries.fetch_sub(1);
            _num_cooling_entries.fetch_add(1);
            UNUSED_VARIABLE(evicted);
            FatalAssert(evicted == nullptr, LOG_TAG_BUFFER,
                        "No entry should be evicted when populating cooling list in CacheMetaContainerDetail::PopulateCoolingList()");
#ifdef ENABLE_STAT_COLLECTION
            (entry->num_moved_to_cool)++;
#endif
            num_cooled++;
            buckets[bucket_idx].UnlockBucket();
            num_hot = _num_hot_entries.load(std::memory_order_acquire);
        }
    }

    size_t TriggerEviction(uint64_t num_pages_needed, void** freed_pages, size_t& bytes_freed,
                           size_t& pages_freed, size_t page_size) {
        FatalAssert(num_pages_needed > 0, LOG_TAG_BUFFER,
                    "num_pages_needed must be greater than 0 in CacheMetaContainerDetail::TriggerEviction()");
        CHECK_NOT_NULLPTR(freed_pages, LOG_TAG_BUFFER);


        size_t num_freed = 0;
        while (num_freed < num_pages_needed) {
            size_t bucket_idx = threadSelf->UniformRange64(0, _num_buckets - 1);
            if (!buckets[bucket_idx].TryLockBucket()) {
                DIVFTREE_YIELD();
                continue;
            }

            size_t num_hot = _num_hot_entries.load(std::memory_order_acquire);
            BufferEntry* evicted_entry = nullptr;
            if ((num_hot < _num_buckets) && (buckets[bucket_idx].GetHotListSize() == 0)) {
                if (_num_cooling_entries.load(std::memory_order_acquire) == 0) {
                    DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_BUFFER,
                            "No hot entries and no cooling entries available during eviction in CacheMetaContainerDetail::TriggerEviction()");
                    buckets[bucket_idx].UnlockBucket();
                    break;
                }

                if (buckets[bucket_idx].CoolingListIsEmpty()) {
                    buckets[bucket_idx].UnlockBucket();
                    DIVFTREE_YIELD();
                    continue;
                }

                evicted_entry = buckets[bucket_idx].EvictFromCoolingList();
                FatalAssert(evicted_entry != nullptr, LOG_TAG_BUFFER,
                            "Evicted entry cannot be null in CacheMetaContainerDetail::TriggerEviction()");
                _num_cooling_entries.fetch_sub(1);
            } else {
                if (buckets[bucket_idx].GetHotListSize() == 0) {
                    buckets[bucket_idx].UnlockBucket();
                    DIVFTREE_YIELD();
                    continue;
                }

                BufferEntry* entry = buckets[bucket_idx].PopFromHotList();
                FatalAssert(entry != nullptr, LOG_TAG_BUFFER,
                            "Popped hot entry cannot be null in CacheMetaContainerDetail::TriggerEviction()");
                entry->lock.Lock(SX_EXCLUSIVE);
                FatalAssert(entry->state == BufferEntryState::BUFFER_ENTRY_CACHED,
                            LOG_TAG_BUFFER,
                            "Popped hot entry must be in CACHED state in CacheMetaContainerDetail::TriggerEviction()");
                FatalAssert(entry->pin == 0,
                            LOG_TAG_BUFFER,
                            "Popped hot entry must be unpinned in CacheMetaContainerDetail::TriggerEviction()");

                entry->state = BufferEntryState::BUFFER_ENTRY_COOLING;
#ifdef ENABLE_STAT_COLLECTION
                (entry->num_moved_to_cool)++;
#endif
                evicted_entry = buckets[bucket_idx].AddToCoolingList(entry);
                entry->lock.Unlock();
                _num_hot_entries.fetch_sub(1);
                if (evicted_entry == nullptr) {
                    _num_cooling_entries.fetch_add(1);
                }
            }

            if (evicted_entry != nullptr) {
                evicted_entry->lock.Lock(SX_EXCLUSIVE);
                buckets[bucket_idx].UnlockBucket();
                size_t num_needed = std::min(evicted_entry->num_pages, num_pages_needed - num_freed);
                for (size_t p = 0; p < num_needed; ++p) {
                    freed_pages[num_freed++] = evicted_entry->pages[p];
                }
                if (num_needed < evicted_entry->num_pages) {
                    size_t pf = evicted_entry->num_pages - num_needed;
                    _page_pool.BatchFree(evicted_entry->pages + num_needed, pf);
                    pages_freed += pf;
                    bytes_freed += evicted_entry->total_size_bytes - (num_needed * page_size);
                }
                evicted_entry->state = BufferEntryState::BUFFER_ENTRY_EVICTED;
#ifdef ENABLE_STAT_COLLECTION
                (evicted_entry->num_evicted)++;
#endif
                evicted_entry->lock.Unlock();
            } else {
                buckets[bucket_idx].UnlockBucket();
            }
        }

        return num_freed;
    }

    inline void FreeAll() {
        for (size_t i = 0; i < _num_buckets; ++i) {
            buckets[i].LockBucket();
            buckets[i].FreeAllEntries(_page_pool);
            buckets[i].UnlockBucket();
        }
    }

protected:
    const size_t _num_buckets;
    const size_t _bucket_cap;
    LocalMemoryPool& _page_pool;
    PtrHash<BufferEntry> _hash;

    std::atomic<size_t> _num_hot_entries;
    std::atomic<size_t> _num_cooling_entries;
    std::vector<CacheBucket> buckets;
};

class CacheMetaContainer {
public:
    static constexpr double COOLING_SIZE_RATIO = 0.2;

    CacheMetaContainer(size_t pool_bytes, size_t page_bytes, size_t num_buckets, LocalMemoryPool& pool) :
        _leaf_page_size(page_bytes),
        _internal_page_size(0),
        _leaf_meta_container(new CacheMetaContainerDetail(
            std::max((size_t)((pool_bytes / page_bytes) * COOLING_SIZE_RATIO), 1lu),
            num_buckets, pool)), _internal_meta_container(nullptr) {}

    CacheMetaContainer(size_t pool_bytes, size_t leaf_bytes, size_t internal_bytes, uint32_t leaf_cap,
                       size_t num_buckets, LocalMemoryPool& pool) :
        _leaf_page_size(leaf_bytes), _internal_page_size(internal_bytes),
        _leaf_meta_container(new CacheMetaContainerDetail(
            std::max((size_t)((pool_bytes / leaf_bytes) * COOLING_SIZE_RATIO), 1lu),
            num_buckets, pool)),
        _internal_meta_container(new CacheMetaContainerDetail(
            std::max((size_t)((pool_bytes / (internal_bytes * leaf_cap)) * COOLING_SIZE_RATIO), num_buckets),
            num_buckets, pool)) {}

    ~CacheMetaContainer() {
        CHECK_NOT_NULLPTR(_leaf_meta_container, LOG_TAG_BUFFER);
        delete _leaf_meta_container;
        if (_internal_meta_container != nullptr) {
            delete _internal_meta_container;
        }
    }

    bool TryLockAndPinEntry(BufferEntry* entry) {
        CHECK_NOT_NULLPTR(entry, LOG_TAG_BUFFER);
        CHECK_NOT_NULLPTR(_leaf_meta_container, LOG_TAG_BUFFER);
        if (entry->id.IsLeaf()) {
            return _leaf_meta_container->TryLockAndPinEntry(entry);
        }
        CHECK_NOT_NULLPTR(_internal_meta_container, LOG_TAG_BUFFER);
        return _internal_meta_container->TryLockAndPinEntry(entry);
    }

    void UnpinEntry(BufferEntry* entry) {
        CHECK_NOT_NULLPTR(entry, LOG_TAG_BUFFER);
        CHECK_NOT_NULLPTR(_leaf_meta_container, LOG_TAG_BUFFER);
        if (entry->id.IsLeaf()) {
            return _leaf_meta_container->UnpinEntry(entry);
        }
        CHECK_NOT_NULLPTR(_internal_meta_container, LOG_TAG_BUFFER);
        return _internal_meta_container->UnpinEntry(entry);
    }

    void PopulateCoolingList(size_t num_pages, bool is_leaf) {
        CHECK_NOT_NULLPTR(_leaf_meta_container, LOG_TAG_BUFFER);
        if (is_leaf) {
            _leaf_meta_container->PopulateCoolingList(num_pages);
            return;
        }
        CHECK_NOT_NULLPTR(_internal_meta_container, LOG_TAG_BUFFER);
        _internal_meta_container->PopulateCoolingList(num_pages);
    }

    size_t TriggerEviction(uint64_t num_pages_needed, void** freed_pages, size_t& bytes_freed,
                           size_t& pages_freed, bool is_leaf) {
        CHECK_NOT_NULLPTR(_leaf_meta_container, LOG_TAG_BUFFER);
        if (is_leaf) {
            return _leaf_meta_container->TriggerEviction(num_pages_needed, freed_pages, bytes_freed,
                                                         pages_freed, _leaf_page_size);
        }
        CHECK_NOT_NULLPTR(_internal_meta_container, LOG_TAG_BUFFER);
        return _internal_meta_container->TriggerEviction(num_pages_needed, freed_pages, bytes_freed,
                                                         pages_freed, _internal_page_size);
    }

    inline void FreeAll() {
        CHECK_NOT_NULLPTR(_leaf_meta_container, LOG_TAG_BUFFER);
        _leaf_meta_container->FreeAll();
        if (_internal_meta_container != nullptr) {
            _internal_meta_container->FreeAll();
        }
    }

protected:
    const size_t _leaf_page_size;
    const size_t _internal_page_size;
    CacheMetaContainerDetail* _leaf_meta_container;
    CacheMetaContainerDetail* _internal_meta_container;
};

class BufferMgr {
public:
    /* constructs the memory pool and connection manager + connects to MNs + gets the metadata */
    static RetStatus Init(size_t page_size, size_t pool_size, BlockingQueue<IVFSearchTask*>* search_task_queue,
                          size_t num_user_threads, IndexMeta& index_meta) {
        FatalAssert(instance == nullptr, LOG_TAG_BUFFER,
                    "BufferMgr is already initialized!");
        instance = new BufferMgr(page_size, pool_size, search_task_queue,
                                 num_user_threads, index_meta);
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
        _cache_meta_container->UnpinEntry(&entry);
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
        uint8_t level = cluster_ids[0]._level;
        bool is_leaf = cluster_ids[0].IsLeaf();
        for (size_t i = 0; i < num_clusters; ++i) {
            FatalAssert(cluster_ids[i]._level == level, LOG_TAG_BUFFER,
                        "All cluster IDs must be on the same level in BufferMgr::PrefetchClusters()");
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
        std::vector<size_t> remaining;
        remaining.reserve(num_clusters);
        size_t num_pages_to_load = 0;
        while (!current_indices.empty()) {
            for (size_t i : current_indices) {
                auto it = _buffer_map.find(cluster_ids[i]);
                BufferEntry& entry = it->second;
                FatalAssert(entry.id == cluster_ids[i], LOG_TAG_BUFFER,
                            "BufferEntry ID does not match cluster ID in BufferMgr::PrefetchClusters()");
                if (!_cache_meta_container->TryLockAndPinEntry(&entry)) {
                    remaining.push_back(i);
                    continue;
                }
#ifdef ENABLE_STAT_COLLECTION
                (entry.num_read)++;
                if (entry.state == BufferEntryState::BUFFER_ENTRY_LOADING) {
                    (entry.num_read_remote_in_progress)++;
                } else if (entry.state == BufferEntryState::BUFFER_ENTRY_CACHED) {
                    (entry.num_read_local)++;
                } else {
                    (entry.num_read_remote)++;
                }
#endif
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
                        FatalAssert(ComputeSubClusterSize(is_leaf, p == 0, entry.num_pages == 1,
                                                          entry.page_num_elements[p]) <= _cache->GetPageSize(is_leaf),
                                    LOG_TAG_BUFFER,
                                    "Page size is smaller than number of elements in BufferMgr::PrefetchClusters()");
                        void* data_ptr = nullptr;
                        if (p == 0) {
                            IVFCluster* cluster = reinterpret_cast<IVFCluster*>(entry.pages[p]);
                            FatalAssert(cluster->header.id == entry.id, LOG_TAG_BUFFER,
                                        "Cluster header ID does not match entry ID in BufferMgr::PrefetchClusters()");
                            data_ptr = cluster->data;
                        } else {
                            data_ptr = entry.pages[p];
                        }
                        IVFSearchTask* task = task_factory->CreateTask(
                            cluster_ids[i],
                            entry.page_num_elements[p],
                            data_ptr);
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

            /* todo: stat collection */
            if (remaining.size() == current_indices.size()) {
                // none of the remaining entries could be locked
                DIVFTREE_YIELD();
            }

            current_indices.swap(remaining);
            remaining.clear();
        }

        if (!to_load.empty()) {
            status = ReadFromRemote(std::move(to_load), num_pages_to_load);
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
            threadSelf->UpdatePollStats(false, 0);
            return status;
        }
        std::vector<VectorID> completed_tasks;
        status = rdma_mgr->PushCompletedReadsToTaskQueue(completed_tasks);
        FatalAssert(status.IsOK(), LOG_TAG_BUFFER,
                    "Failed to poll completed RDMA reads in BufferMgr::PollRemoteReads(): %s",
                    status.Msg());
        threadSelf->UpdatePollStats(true, completed_tasks.size());

        std::vector<IVFSearchTask*> new_tasks;
        for (VectorID cluster_id : completed_tasks) {
            auto it = _buffer_map.find(cluster_id);
            FatalAssert(it != _buffer_map.end(), LOG_TAG_BUFFER,
                        "Cluster ID not found in BufferMgr::PollRemoteReads()");
            BufferEntry& entry = it->second;
            FatalAssert(entry.id == cluster_id, LOG_TAG_BUFFER,
                        "BufferEntry ID does not match cluster ID in BufferMgr::PollRemoteReads()");
            entry.lock.Lock(SX_EXCLUSIVE);
            FatalAssert(entry.state == BufferEntryState::BUFFER_ENTRY_LOADING, LOG_TAG_BUFFER,
                        "BufferEntry must be in LOADING state in BufferMgr::PollRemoteReads()");
            FatalAssert(!entry.pending.empty(), LOG_TAG_BUFFER,
                        "BufferEntry in LOADING state must have pending tasks in BufferMgr::PollRemoteReads()");
            FatalAssert(entry.pin > 0, LOG_TAG_BUFFER,
                        "BufferEntry must be pinned in BufferMgr::PollRemoteReads()");
            entry.state = BufferEntryState::BUFFER_ENTRY_CACHED;
            bool is_leaf = cluster_id.IsLeaf();

            for (IVFSearchTaskFactory* task_factory : entry.pending) {
                for (size_t p = 0; p < entry.num_pages; ++p) {
                    FatalAssert(entry.page_num_elements[p] > 0, LOG_TAG_BUFFER,
                                "Page must contain at least one element in BufferMgr::PollRemoteReads()");
                    FatalAssert(entry.pages[p] != nullptr, LOG_TAG_BUFFER,
                                "Page pointer cannot be null in BufferMgr::PollRemoteReads()");
                    FatalAssert(ComputeSubClusterSize(is_leaf, p == 0, entry.num_pages == 1,
                                                      entry.page_num_elements[p]) <= _cache->GetPageSize(is_leaf),
                                LOG_TAG_BUFFER,
                                "Page size is smaller than number of elements in BufferMgr::PollRemoteReads()");
                    void* data_ptr = nullptr;
                    if (p == 0) {
                        IVFCluster* cluster = reinterpret_cast<IVFCluster*>(entry.pages[p]);
                        FatalAssert(cluster->header.id == entry.id, LOG_TAG_BUFFER,
                                    "Cluster header ID does not match entry ID in BufferMgr::PrefetchClusters()");
                        data_ptr = cluster->data;
                    } else {
                        data_ptr = entry.pages[p];
                    }
                    IVFSearchTask* task = task_factory->CreateTask(
                        cluster_id,
                        entry.page_num_elements[p],
                        data_ptr);
                    new_tasks.push_back(task);
                }

                FatalAssert(task_factory->num_sibling_tasks > 0, LOG_TAG_BUFFER,
                            "Sibling task count must be greater than 0 in BufferMgr::PollRemoteReads()");
                FatalAssert(task_factory->num_sibling_tasks >= entry.num_pages, LOG_TAG_BUFFER,
                            "Sibling task count must be at least the number of pages in BufferMgr::PollRemoteReads()");
                FatalAssert(task_factory->num_tasks_completed->load(std::memory_order_acquire) <
                    task_factory->num_sibling_tasks, LOG_TAG_BUFFER,
                            "All tasks should not have been completed in BufferMgr::PollRemoteReads()");
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

    /* not thread-safe */
    inline String GetStats(MemoryStatsNode*& memory_stats_head, bool reset_after_fetch = false) {
#ifdef ENABLE_MEMORY_STAT_COLLECTION
        memory_stats_head = _memory_stats_head;
        if (reset_after_fetch && _memory_stats_head != nullptr) {
            CHECK_NOT_NULLPTR(_memory_stats_tail, LOG_TAG_BUFFER);
            _memory_stats_head = new MemoryStatsNode();
            (*_memory_stats_head) = (*_memory_stats_tail);
            _memory_stats_tail = _memory_stats_head;
        }
#else
        memory_stats_head = nullptr;
#endif
#ifdef ENABLE_STAT_COLLECTION
        String stats(1024*1024); // 1MB buffer
        std::vector<size_t> total_moved_to_cool(256, 0);
        std::vector<size_t> total_removed_from_cool(256, 0);
        std::vector<size_t> total_evicted(256, 0);
        std::vector<size_t> total_read_local(256, 0);
        std::vector<size_t> total_read_remote_in_progress(256, 0);
        std::vector<size_t> total_read_remote(256, 0);
        std::vector<size_t> total_read(256, 0);
        uint8_t max_level = 0;
        for (auto& pair : _buffer_map) {
            uint8_t level = pair.first._level;
            max_level = std::max(max_level, level);
            BufferEntry& entry = pair.second;
            total_moved_to_cool[level] += entry.num_moved_to_cool;
            total_removed_from_cool[level] += entry.num_removed_from_cool;
            total_evicted[level] += entry.num_evicted;
            total_read_local[level] += entry.num_read_local;
            total_read_remote_in_progress[level] += entry.num_read_remote_in_progress;
            total_read_remote[level] += entry.num_read_remote;
            total_read[level] += entry.num_read;

            total_moved_to_cool[0] += entry.num_moved_to_cool;
            total_removed_from_cool[0] += entry.num_removed_from_cool;
            total_evicted[0] += entry.num_evicted;
            total_read_local[0] += entry.num_read_local;
            total_read_remote_in_progress[0] += entry.num_read_remote_in_progress;
            total_read_remote[0] += entry.num_read_remote;
            total_read[0] += entry.num_read;
        }

        stats += String(
                "BufferMgr Stats:\n"
                "Total reads: %zu (local: %.2f%%(%zu), remote in progress: %.2f%%(%zu), remote: %.2f%%(%zu)), "
                "total_moved_to_cool: %zu, total_removed_from_cool: %zu, total_evicted: %zu\n",
                total_read[0],
                (((double)(total_read_local[0]) / total_read[0]) * 100), total_read_local[0],
                (((double)(total_read_remote_in_progress[0]) / total_read[0]) * 100), total_read_remote_in_progress[0],
                (((double)(total_read_remote[0]) / total_read[0]) * 100), total_read_remote[0],
                total_moved_to_cool[0], total_removed_from_cool[0], total_evicted[0]
            );

        for (uint8_t level = max_level; level > 0; --level) {
            stats += String(
                "Level %hhu Stats:\n"
                "Total reads: %zu (local: %.2f%%(%zu), remote in progress: %.2f%%(%zu), remote: %.2f%%(%zu)), "
                "total_moved_to_cool: %zu, total_removed_from_cool: %zu, total_evicted: %zu\n",
                level,
                total_read[level],
                (((double)(total_read_local[level]) / total_read[level]) * 100), total_read_local[level],
                (((double)(total_read_remote_in_progress[level]) / total_read[level]) * 100), total_read_remote_in_progress[level],
                (((double)(total_read_remote[level]) / total_read[level]) * 100), total_read_remote[level],
                total_moved_to_cool[level], total_removed_from_cool[level], total_evicted[level]
            );
        }

        stats += String("\n");

        for (auto& pair : _buffer_map) {
            uint8_t level = pair.first._level;
            max_level = std::max(max_level, level);
            BufferEntry& entry = pair.second;

            stats += String(VECTORID_LOG_FMT ": reads: %zu (local: %.2f%%(%zu), remote in progress: %.2f%%(%zu), remote: %.2f%%(%zu)), moved to cool: %zu, removed from cool: %zu, evicted: %zu\n",
                           VECTORID_LOG(pair.first), entry.num_read,
                           (((double)(entry.num_read_local) / entry.num_read) * 100), entry.num_read_local,
                           (((double)(entry.num_read_remote_in_progress) / entry.num_read) * 100),
                           entry.num_read_remote_in_progress,
                           (((double)(entry.num_read_remote) / entry.num_read) * 100), entry.num_read_remote,
                           entry.num_moved_to_cool, entry.num_removed_from_cool, entry.num_evicted);
            if (reset_after_fetch) {
                entry.num_moved_to_cool = 0;
                entry.num_removed_from_cool = 0;
                entry.num_evicted = 0;
                entry.num_read_local = 0;
                entry.num_read_remote_in_progress = 0;
                entry.num_read_remote = 0;
                entry.num_read = 0;
            }
        }

        stats += String("\n");

        return stats;
#else
        return "BufferMgr Stats: (enable stat collection to see details)";
#endif
    }

protected:

    BufferMgr(size_t page_size, size_t pool_size, BlockingQueue<IVFSearchTask*>* search_task_queue,
              size_t num_user_threads, IndexMeta& index_meta) :
        _cache(nullptr), _cache_meta_container(nullptr),
        _search_task_queue(search_task_queue) {
        FatalAssert(_search_task_queue != nullptr, LOG_TAG_BUFFER,
                    "search_task_queue cannot be null in BufferMgr constructor");
        FatalAssert(num_user_threads > 0, LOG_TAG_BUFFER,
                    "num_user_threads must be greater than 0 in BufferMgr constructor");
        FatalAssert(IS_COMPUTE_NODE(), LOG_TAG_BUFFER,
                    "BufferMgr can only be initialized on compute nodes");
        size_t internal_page_size = 0;
        size_t leaf_page_size = 0;

        if (index_meta.type == IndexType::IVF_FLAT) {
            FatalAssert(page_size > 0, LOG_TAG_BUFFER, "page_size cannot be 0");
            page_size = ALIGNED_SIZE(page_size);
            FatalAssert(pool_size > page_size, LOG_TAG_BUFFER, "pool_size cannot be 0");
            pool_size = ALIGNED_SIZE(pool_size, page_size + CACHE_LINE_SIZE);
            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BUFFER,
                    "Initializing BufferMgr with IVF_FLAT index. page_size: %zu, pool_size: %zu",
                    page_size, pool_size);
            _cache = new LocalMemoryPool(page_size, page_size, pool_size);
            _cache_meta_container = new CacheMetaContainer(pool_size, page_size, 4 * num_user_threads, *_cache);
            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BUFFER,
                    "BufferMgr initialized with IVF_FLAT index. page_size: %zu, pool_size: %zu",
                    page_size, pool_size);
        } else if (index_meta.type == IndexType::IVF_CAPPED) {
            FatalAssert(index_meta.leaf_size_cap > 1, LOG_TAG_BUFFER,
                        "leaf_size_cap must be greater than 1 for IVF_CAPPED index in BufferMgr constructor");
            leaf_page_size = ALIGNED_SIZE(ComputeClusterSize(true, index_meta.leaf_size_cap));
            FatalAssert(pool_size > leaf_page_size, LOG_TAG_BUFFER,
                        "pool_size must be greater than leaf_page_size for IVF_CAPPED index in BufferMgr constructor");
            pool_size = ALIGNED_SIZE(pool_size, leaf_page_size + CACHE_LINE_SIZE);
            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BUFFER,
                    "Initializing BufferMgr with IVF_CAPPED index. leaf_page_size: %zu, pool_size: %zu",
                    leaf_page_size, pool_size);
            _cache = new LocalMemoryPool(leaf_page_size, leaf_page_size, pool_size);
            _cache_meta_container = new CacheMetaContainer(pool_size, leaf_page_size,
                                                           2 * num_user_threads, *_cache);
            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BUFFER,
                    "BufferMgr initialized with IVF_CAPPED index. page_size: %zu, pool_size: %zu",
                    leaf_page_size, pool_size);
        } else {
            FatalAssert(index_meta.type == IndexType::IVF_TREE, LOG_TAG_BUFFER,
                        "Unsupported index type in BufferMgr constructor");
            FatalAssert(index_meta.leaf_size_cap > 1, LOG_TAG_BUFFER,
                        "leaf_size_cap must be greater than 1 for IVF_TREE index in BufferMgr constructor");
            FatalAssert(index_meta.internal_size_cap > 1, LOG_TAG_BUFFER,
                        "internal_size_cap must be greater than 1 for IVF_TREE index in BufferMgr constructor");
            leaf_page_size = ALIGNED_SIZE(ComputeClusterSize(true, index_meta.leaf_size_cap));
            internal_page_size = ALIGNED_SIZE(ComputeClusterSize(false, index_meta.internal_size_cap));
            FatalAssert(pool_size > leaf_page_size + internal_page_size + 2*CACHE_LINE_SIZE,
                        LOG_TAG_BUFFER,
                        "pool_size must be greater than the sum of leaf_page_size and internal_page_size for IVF_TREE index in BufferMgr constructor");
            pool_size = ALIGNED_SIZE(pool_size, std::lcm(leaf_page_size + CACHE_LINE_SIZE,
                                                         internal_page_size + CACHE_LINE_SIZE));
            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BUFFER,
                    "Initializing BufferMgr with IVF_TREE index. leaf_page_size: %zu, internal_page_size: %zu, pool_size: %zu",
                    leaf_page_size, internal_page_size, pool_size);
            _cache = new LocalMemoryPool(internal_page_size, leaf_page_size, pool_size);
            _cache_meta_container = new CacheMetaContainer(pool_size, leaf_page_size, internal_page_size, index_meta.leaf_size_cap,
                                                           2 * num_user_threads, *_cache);
            DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_BUFFER,
                    "BufferMgr initialized with IVF_TREE index. leaf_page_size: %zu, internal_page_size: %zu, pool_size: %zu",
                    leaf_page_size, internal_page_size, pool_size);
        }

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
        status = rdma_mgr->RegisterMemory(_cache->GetBaseAddress(), _cache->GetPoolSize());
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
        IndexInfo index_info;
        rdma_mgr->ReceiveMessage(mnode_id, &index_info, sizeof(index_info));
        FatalAssert(index_info.type == index_meta.type, LOG_TAG_BUFFER,
                    "Received index type does not match expected index type in BufferMgr::Init()");
        FatalAssert(index_info.num_points > 0, LOG_TAG_BUFFER,
                    "Received number of points must be greater than 0 in BufferMgr::Init()");
        index_meta.num_points = index_info.num_points;
        if (index_info.type != IndexType::IVF_TREE) {
            FatalAssert(index_info.type == IndexType::IVF_FLAT || index_info.type == IndexType::IVF_CAPPED, LOG_TAG_BUFFER,
                        "Received unsupported index type in BufferMgr::Init()");
            FatalAssert((index_info.type == IndexType::IVF_FLAT && index_info.ivf_flat_info.num_clusters > 0) ||
                        (index_info.type == IndexType::IVF_CAPPED && index_info.ivf_capped_info.num_clusters > 0 &&
                         index_info.ivf_capped_info.cluster_capacity > 0 &&
                         index_info.ivf_capped_info.cluster_capacity == index_meta.leaf_size_cap), LOG_TAG_BUFFER,
                        "Received number of clusters must be greater than 0 for IVF_FLAT or IVF_CAPPED index in BufferMgr::Init()");
            index_meta.num_levels = 1;
            uint32_t num_clusters = 0;
            if (index_info.type == IndexType::IVF_FLAT) {
                num_clusters = index_info.ivf_flat_info.num_clusters;
                FatalAssert(index_info.ivf_flat_info.num_clusters > 0, LOG_TAG_BUFFER,
                            "Received number of clusters must be greater than 0 for IVF_FLAT index in BufferMgr::Init()");
            } else {
                num_clusters = index_info.ivf_capped_info.num_clusters;
                FatalAssert(index_info.ivf_capped_info.num_clusters > 0 &&
                            index_info.ivf_capped_info.cluster_capacity > 0 &&
                            index_info.ivf_capped_info.cluster_capacity == index_meta.leaf_size_cap, LOG_TAG_BUFFER,
                            "Received number of clusters must be greater than 0 for IVF_CAPPED index in BufferMgr::Init()");
            }
            index_meta.top_centroids.resize(num_clusters);

            for (uint32_t c = 0; c < num_clusters; ++c) {
                ClusterMeta cluster_meta;
                rdma_mgr->ReceiveMessage(mnode_id, &cluster_meta, sizeof(cluster_meta));
                FatalAssert(cluster_meta.remote_size > 0, LOG_TAG_BUFFER,
                                "Received remote size must be greater than 0 for IVF_TREE index in BufferMgr::Init()");
                FatalAssert(cluster_meta.remote_addr != 0, LOG_TAG_BUFFER,
                            "Received remote address cannot be 0 for IVF_TREE index in BufferMgr::Init()");
                FatalAssert(cluster_meta.centroid_id._level == 1, LOG_TAG_BUFFER,
                            "Received centroid ID level does not match expected level for index in BufferMgr::Init()");
                _buffer_map.emplace(
                    cluster_meta.centroid_id,
                    BufferEntry(cluster_meta.centroid_id, cluster_meta.remote_addr, cluster_meta.remote_size,
                                cluster_meta.num_elements, _cache->GetPageSize(true))
                );
                index_meta.top_centroids[c].id = cluster_meta.centroid_id;
            }

            for (uint32_t c = 0; c < num_clusters; ++c) {
                rdma_mgr->ReceiveMessage(mnode_id, &index_meta.top_centroids[c].data,
                                         sizeof(index_meta.top_centroids[c].data));
            }

        } else {
            FatalAssert(index_info.ivf_tree_info.leaf_cluster_capacity == index_meta.leaf_size_cap, LOG_TAG_BUFFER,
                        "Received leaf cluster capacity does not match expected leaf cluster capacity for IVF_TREE index in BufferMgr::Init()");
            FatalAssert(index_info.ivf_tree_info.internal_cluster_capacity == index_meta.internal_size_cap, LOG_TAG_BUFFER,
                        "Received internal cluster capacity does not match expected internal cluster capacity for IVF_TREE index in BufferMgr::Init()");
            rdma_mgr->ReceiveMessage(mnode_id, &index_meta.num_levels, sizeof(index_meta.num_levels));
            FatalAssert(index_meta.num_levels > 0, LOG_TAG_BUFFER,
                        "Received number of levels must be greater than 0 for IVF_TREE index in BufferMgr::Init()");
            for (uint8_t level_idx = index_meta.num_levels - 1; level_idx != UINT8_MAX; --level_idx) {
                uint32_t num_clusters = 0;
                rdma_mgr->ReceiveMessage(mnode_id, &num_clusters, sizeof(num_clusters));
                FatalAssert(num_clusters > 0, LOG_TAG_BUFFER,
                            "Received number of clusters must be greater than 0 for each level of IVF_TREE index in BufferMgr::Init()");
                if (level_idx == index_meta.num_levels  - 1) {
                    index_meta.top_centroids.resize(num_clusters);
                }

                for (uint32_t c = 0; c < num_clusters; ++c) {
                    ClusterMeta cluster_meta;
                    rdma_mgr->ReceiveMessage(mnode_id, &cluster_meta, sizeof(cluster_meta));
                    FatalAssert(cluster_meta.remote_size > 0, LOG_TAG_BUFFER,
                                "Received remote size must be greater than 0 for IVF_TREE index in BufferMgr::Init()");
                    FatalAssert(cluster_meta.remote_addr != 0, LOG_TAG_BUFFER,
                                "Received remote address cannot be 0 for IVF_TREE index in BufferMgr::Init()");
                    FatalAssert(cluster_meta.centroid_id._level == level_idx + 1, LOG_TAG_BUFFER,
                                "Received centroid ID level does not match expected level for IVF_TREE index in BufferMgr::Init()");
                    _buffer_map.emplace(
                        cluster_meta.centroid_id,
                        BufferEntry(cluster_meta.centroid_id, cluster_meta.remote_addr, cluster_meta.remote_size,
                                    cluster_meta.num_elements, _cache->GetPageSize(cluster_meta.centroid_id.IsLeaf()))
                    );
                    if (level_idx == index_meta.num_levels - 1) {
                        index_meta.top_centroids[c].id = cluster_meta.centroid_id;
                    }
                }

                if (level_idx == index_meta.num_levels - 1) {
                    for (uint32_t c = 0; c < num_clusters; ++c) {
                        rdma_mgr->ReceiveMessage(mnode_id, &index_meta.top_centroids[c].data,
                                                sizeof(index_meta.top_centroids[c].data));
                    }
                }
            }
        }
    }

    ~BufferMgr() {
        _cache_meta_container->FreeAll();
        delete _cache_meta_container;
        delete _cache;
    }

    RetStatus ReadFromRemote(std::vector<VectorID>&& cluster_ids, size_t num_pages_to_load) {
        RetStatus status = RetStatus::Success();
        RDMA_Manager* rdma_mgr = RDMA_Manager::GetInstance();
        CHECK_NOT_NULLPTR(rdma_mgr, LOG_TAG_BUFFER);
        NodeID mnode_id = rdma_mgr->GetMemoryNodeID();
        FatalAssert(!cluster_ids.empty(), LOG_TAG_BUFFER, "id set is empty!");
        const bool is_leaf = cluster_ids[0].IsLeaf();
        const SlotType slotType = is_leaf ? SlotType::Leaf : SlotType::Internal;
        bool use_sg = (num_pages_to_load != cluster_ids.size());
        void** local_buffers = new void*[num_pages_to_load];
        size_t num_allocated = 0;
        size_t num_tries = 0;
        size_t num_from_cool = 0;
        size_t num_from_pool = 0;
        size_t bytes_freed_from_cool = 0;
        size_t num_bytes_needed = 0;
        size_t num_pages_freed = 0;
        UNUSED_VARIABLE(num_bytes_needed);
#ifdef ENABLE_MEMORY_STAT_COLLECTION
        for (VectorID cluster_id : cluster_ids) {
            auto it = _buffer_map.find(cluster_id);
            FatalAssert(it != _buffer_map.end(), LOG_TAG_BUFFER,
                        "Cluster ID not found in BufferMgr::ReadFromRemote()");
            BufferEntry& entry = it->second;
            num_bytes_needed += entry.total_size_bytes;
        }
        _memory_stats_lock.Lock(SX_EXCLUSIVE);
#endif
        if (num_pages_to_load > 0) {
            _cache_meta_container->PopulateCoolingList(num_pages_to_load, is_leaf);
        }

        while (num_allocated < num_pages_to_load) {
            size_t num_to_alloc = num_pages_to_load - num_allocated;
            size_t current_allocated =
                    _cache->BatchAllocate(slotType, local_buffers + num_allocated, num_to_alloc,
                                          AllocationFlags{.clear = 0, .non_blocking = 1, .atomic = 0, .unused = 0});
            num_from_pool += current_allocated;
            num_allocated += current_allocated;
            if (num_allocated < num_pages_to_load) {
                num_to_alloc = num_pages_to_load - num_allocated;
                current_allocated =
                    _cache_meta_container->TriggerEviction(num_to_alloc, local_buffers + num_allocated, bytes_freed_from_cool, num_pages_freed, is_leaf);
                num_allocated += current_allocated;
                num_from_cool += current_allocated;
            }

            if ((num_tries > 0) && (num_allocated < num_pages_to_load)) {
                usleep(1);
            }
            ++num_tries;
        }
        threadSelf->UpdateMemoryStats(num_from_cool, num_from_pool, num_tries);

        FatalAssert(num_allocated == num_pages_to_load, LOG_TAG_BUFFER,
                    "Failed to allocate enough pages in BufferMgr::ReadFromRemote()");
#ifdef ENABLE_MEMORY_STAT_COLLECTION
        FatalAssert(num_bytes_needed > 0, LOG_TAG_BUFFER,
                    "num_bytes_needed must be greater than 0 in BufferMgr::ReadFromRemote()");
        if (_memory_stats_tail == nullptr) {
            FatalAssert(_memory_stats_head == nullptr, LOG_TAG_BUFFER,
                        "Memory stats head should be null when tail is null in BufferMgr::ReadFromRemote()");
            FatalAssert(num_from_cool == 0, LOG_TAG_BUFFER,
                        "num_from_cool should be 0 for the first memory stats entry in BufferMgr::ReadFromRemote()");
            FatalAssert(num_pages_freed == 0, LOG_TAG_BUFFER,
                        "num_pages_freed should be 0 for the first memory stats entry in BufferMgr::ReadFromRemote()");
            FatalAssert(bytes_freed_from_cool == 0, LOG_TAG_BUFFER,
                        "bytes_freed_from_cool should be 0 for the first memory stats entry in BufferMgr::ReadFromRemote()");
            if (is_leaf) {
                _memory_stats_head = new MemoryStatsNode(num_allocated, num_bytes_needed, 0, 0);
            } else {
                _memory_stats_head = new MemoryStatsNode(0, 0, num_allocated, num_bytes_needed);
            }
            _memory_stats_tail = _memory_stats_head;
        } else {
            FatalAssert(_memory_stats_head != nullptr, LOG_TAG_BUFFER,
                        "Memory stats head should not be null when adding a new entry in BufferMgr::ReadFromRemote()");
            MemoryStatsNode* new_node = nullptr;
            if (is_leaf) {
                FatalAssert(_memory_stats_tail->num_allocated_pages_leaf >= num_pages_freed, LOG_TAG_BUFFER,
                        "Current pages should be greater than or equal to pages allocated from pool in BufferMgr::ReadFromRemote()");
                size_t _current_pages = _memory_stats_tail->num_allocated_pages_leaf - num_pages_freed + num_from_pool;
                FatalAssert(_memory_stats_tail->num_bytes_in_use_leaf >= bytes_freed_from_cool, LOG_TAG_BUFFER,
                            "Current bytes should be greater than or equal to bytes freed from cool in BufferMgr::ReadFromRemote()");
                size_t _current_bytes = _memory_stats_tail->num_bytes_in_use_leaf - bytes_freed_from_cool + num_bytes_needed;
                new_node = new MemoryStatsNode(_current_pages, _current_bytes, _memory_stats_tail->num_allocated_pages_internal, _memory_stats_tail->num_bytes_in_use_internal);
            } else {
                FatalAssert(_memory_stats_tail->num_allocated_pages_internal >= num_pages_freed, LOG_TAG_BUFFER,
                        "Current pages should be greater than or equal to pages allocated from pool in BufferMgr::ReadFromRemote()");
                size_t _current_pages = _memory_stats_tail->num_allocated_pages_internal - num_pages_freed + num_from_pool;
                FatalAssert(_memory_stats_tail->num_bytes_in_use_internal >= bytes_freed_from_cool, LOG_TAG_BUFFER,
                            "Current bytes should be greater than or equal to bytes freed from cool in BufferMgr::ReadFromRemote()");
                size_t _current_bytes = _memory_stats_tail->num_bytes_in_use_internal - bytes_freed_from_cool + num_bytes_needed;
                new_node = new MemoryStatsNode(_memory_stats_tail->num_allocated_pages_leaf, _memory_stats_tail->num_bytes_in_use_leaf, _current_pages, _current_bytes);
            }
            _memory_stats_tail->next = new_node;
            _memory_stats_tail = new_node;
        }
        _memory_stats_lock.Unlock();
#endif
        /* todo: more efficnet implementation */
        if (use_sg) {
            size_t num_used = 0;
            void*** local_addrs = new void**[cluster_ids.size()];
            uint32_t** sizes = new uint32_t*[cluster_ids.size()];
            uint32_t* num_sge = new uint32_t[cluster_ids.size()];
            uintptr_t* remote_addrs = new uintptr_t[cluster_ids.size()];
            for (size_t i = 0; i < cluster_ids.size(); ++i) {
                FatalAssert(is_leaf == cluster_ids[i].IsLeaf(), LOG_TAG_BUFFER,
                            "All cluster IDs must be of the same type in BufferMgr::ReadFromRemote()");
                auto it = _buffer_map.find(cluster_ids[i]);
                FatalAssert(it != _buffer_map.end(), LOG_TAG_BUFFER,
                            "Cluster ID not found in BufferMgr::ReadFromRemote()");
                BufferEntry& entry = it->second;
                remote_addrs[i] = entry.remote_addr;
                num_sge[i] = entry.num_pages;
                local_addrs[i] = local_buffers + num_used;
                sizes[i] = new uint32_t[entry.num_pages];
                for (size_t p = 0; p < entry.num_pages; ++p) {
                    entry.pages[p] = local_buffers[num_used + p];
                    sizes[i][p] = ComputeSubClusterSize(is_leaf, p == 0,
                                                        entry.num_pages == 1, entry.page_num_elements[p]);
                    FatalAssert(sizes[i][p] > 0, LOG_TAG_BUFFER,
                                "Page size must be greater than 0 in BufferMgr::ReadFromRemote()");
                    FatalAssert(sizes[i][p] <= _cache->GetPageSize(is_leaf), LOG_TAG_BUFFER,
                                "Page size is smaller than number of elements in BufferMgr::ReadFromRemote()");
                }
                num_used += entry.num_pages;
            }
            FatalAssert(num_used == num_pages_to_load, LOG_TAG_BUFFER,
                        "Number of pages to load mismatch in BufferMgr::ReadFromRemote()");
            status = rdma_mgr->RDMASGRead(mnode_id, local_addrs, remote_addrs, sizes, num_sge,
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
            uint32_t* sizes = new uint32_t[cluster_ids.size()];
            for (size_t i = 0; i < cluster_ids.size(); ++i) {
                FatalAssert(is_leaf == cluster_ids[i].IsLeaf(), LOG_TAG_BUFFER,
                            "All cluster IDs must be of the same type in BufferMgr::ReadFromRemote()");
                auto it = _buffer_map.find(cluster_ids[i]);
                FatalAssert(it != _buffer_map.end(), LOG_TAG_BUFFER,
                            "Cluster ID not found in BufferMgr::ReadFromRemote()");
                BufferEntry& entry = it->second;
                remote_addrs[i] = entry.remote_addr;
                sizes[i] = entry.total_size_bytes;
                FatalAssert(sizes[i] <= _cache->GetPageSize(is_leaf) * entry.num_pages, LOG_TAG_BUFFER,
                            "Total cluster size exceeds allocated pages in BufferMgr::ReadFromRemote()");
                FatalAssert(sizes[i] > 0, LOG_TAG_BUFFER,
                            "Cluster size must be greater than 0 in BufferMgr::ReadFromRemote()");
                FatalAssert(entry.num_pages == 1, LOG_TAG_BUFFER,
                            "Cluster with multiple pages must use scatter-gather RDMA read in BufferMgr::ReadFromRemote()");
                FatalAssert(ComputeClusterSize(is_leaf, entry.page_num_elements[0]) == sizes[i], LOG_TAG_BUFFER,
                            "Cluster size does not match number of elements in BufferMgr::ReadFromRemote()");
                entry.pages[0] = local_buffers[i];
            }
            status = rdma_mgr->RDMARead(mnode_id, local_buffers, remote_addrs, sizes,
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
    LocalMemoryPool* _cache;
    CacheMetaContainer* _cache_meta_container;
    std::unordered_map<VectorID, BufferEntry, VectorIDHash> _buffer_map;
    SXSpinLock _poll_lock;
    BlockingQueue<IVFSearchTask*>* _search_task_queue;

#ifdef ENABLE_MEMORY_STAT_COLLECTION
    SXLock _memory_stats_lock;
    MemoryStatsNode* _memory_stats_head = nullptr;
    MemoryStatsNode* _memory_stats_tail = nullptr;
#endif
};

};

#endif