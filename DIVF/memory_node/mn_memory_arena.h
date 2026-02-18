#ifndef MN_MEMORY_ARENA_H_
#define MN_MEMORY_ARENA_H_

#include "debug.h"
#include "common.h"
#include "utils/thread.h"
#include "utils/synchronization.h"

#include <vector>
#include <sys/mman.h>

namespace divftree {

struct ArenaRegion {
    void* base = nullptr;
    size_t len = 0;
    std::atomic<size_t> next_free_offset = 0;

    ArenaRegion() = default;
    ArenaRegion(void* b, size_t l) : base(b), len(l) {
        CHECK_NOT_NULLPTR(b, LOG_TAG_BASIC);
        FatalAssert(l > 0, LOG_TAG_BASIC, "length cannot be 0");
        FatalAssert(ALIGNED(b), LOG_TAG_BASIC, "base must be aligned");
        FatalAssert(ALIGNED(l), LOG_TAG_BASIC, "length must be aligned");
    }
    ArenaRegion(const ArenaRegion& other) = delete;
    ArenaRegion(ArenaRegion&& other) : base(other.base), len(other.len),
                                       next_free_offset(other.next_free_offset.load(std::memory_order_relaxed)) {
        other.base = nullptr;
        other.len = 0;
        other.next_free_offset.store(0, std::memory_order_relaxed);
    }
    inline ArenaRegion& operator=(const ArenaRegion& other) = delete;
    inline ArenaRegion& operator=(ArenaRegion&& other) {
        base = other.base;
        len = other.len;
        next_free_offset.store(other.next_free_offset.load(std::memory_order_relaxed), std::memory_order_relaxed);
        other.base = nullptr;
        other.len = 0;
        other.next_free_offset.store(0, std::memory_order_relaxed);
    }
};

class MemoryArena {
public:
    MemoryArena(size_t page_bytes, size_t initial_num_pages, size_t min_num_pages_per_region) :
        page_size(page_bytes), region_min_pages(min_num_pages_per_region), next_region_num_pages(initial_num_pages) {
        FatalAssert(page_size > 0, LOG_TAG_BASIC, "Page size must be greater than 0 in MemoryArena constructor");
        FatalAssert(ALIGNED(page_size), LOG_TAG_BASIC, "Page size must be aligned to page size in MemoryArena constructor");
        FatalAssert(initial_num_pages >= min_num_pages_per_region, LOG_TAG_BASIC,
                    "Initial number of pages must be at least the minimum number of pages per region in MemoryArena constructor");
        AllocateNewRegion(initial_num_pages);
        next_region_num_pages = std::max(next_region_num_pages / 2, region_min_pages);
    }

    ~MemoryArena() {
        for (ArenaRegion& region : regions) {
            if (munmap(region.base, region.len) != 0) {
                DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_BASIC, "Failed to free memory for MemoryPool. "
                        "errno %d, errno msg: %s", errno, strerror(errno));
            }
        }
        regions.clear();
    }

    void* AllocatePage() {
        FatalAssert(page_size > 0, LOG_TAG_BASIC, "Page size must be greater than 0 in MemoryArena::AllocatePage()");
        FatalAssert(ALIGNED(page_size), LOG_TAG_BASIC, "Page size must be aligned to page size in MemoryArena::AllocatePage()");
        region_allocation_lock.Lock(SX_SHARED);
        bool locked_exclusive = false;
        while (true) {
            FatalAssert(!regions.empty(), LOG_TAG_BASIC, "No regions available in MemoryArena::AllocatePage()");
            ArenaRegion& region = regions.back();
            FatalAssert(region.base != nullptr, LOG_TAG_BASIC, "Region base cannot be null in MemoryArena::AllocatePage()");
            FatalAssert(region.len >= page_size, LOG_TAG_BASIC, "Region length must be at least page size in MemoryArena::AllocatePage()");
            FatalAssert(region.len % page_size == 0, LOG_TAG_BASIC, "Region length must be a multiple of page size in MemoryArena::AllocatePage()");
            FatalAssert(ALIGNED(region.base), LOG_TAG_BASIC, "Region base must be aligned to page size in MemoryArena::AllocatePage()");

            size_t offset = region.next_free_offset.fetch_add(page_size);
            if (offset >= region.len) {
                if (!locked_exclusive) {
                    region.next_free_offset.fetch_sub(page_size);
                    region_allocation_lock.Unlock();
                    region_allocation_lock.Lock(SX_EXCLUSIVE);
                    locked_exclusive = true;
                } else {
                    FatalAssert(next_region_num_pages >= region_min_pages, LOG_TAG_BASIC, "invalid next region num_pages");
                    FatalAssert(region_min_pages > 0, LOG_TAG_BASIC, "each region should at least have a single page");
                    AllocateNewRegion(next_region_num_pages);
                    next_region_num_pages = std::max(next_region_num_pages / 2, region_min_pages);
                    FatalAssert(next_region_num_pages >= region_min_pages, LOG_TAG_BASIC, "invalid next region num_pages");
                }
                continue;
            }
            void* page = (void*)((uintptr_t)region.base + offset);
            FatalAssert(page != nullptr, LOG_TAG_BASIC, "could not allocate page");
            FatalAssert(ALIGNED(page), LOG_TAG_BASIC, "page should be cache-aligned");
            FatalAssert(page >= region.base && (uintptr_t)page + page_size <= (uintptr_t)region.base + region.len,
                        LOG_TAG_BASIC, "out-of-bounds page allocated");
            region_allocation_lock.Unlock();
            return page;
        }
    }

    const std::vector<ArenaRegion>& GetRegions() const {
        return regions;
    }

protected:
    const size_t page_size;
    const size_t region_min_pages;
    size_t next_region_num_pages;
    std::vector<ArenaRegion> regions;
    SXLock region_allocation_lock;

    void AllocateNewRegion(size_t num_pages) {
        FatalAssert(false, LOG_TAG_BASIC, "should nopt be called as it is not supported by the rdma manger");
        if (!regions.empty()) {
            threadSelf->SanityCheckLockHeldInModeByMe(&region_allocation_lock, SX_EXCLUSIVE);
            FatalAssert(regions.back().base != nullptr, LOG_TAG_BASIC, "Region base cannot be null in MemoryArena::AllocateNewRegion()");
            FatalAssert(regions.back().len >= page_size, LOG_TAG_BASIC, "Region length must be at least page size in MemoryArena::AllocateNewRegion()");
            FatalAssert(regions.back().len % page_size == 0, LOG_TAG_BASIC, "Region length must be a multiple of page size in MemoryArena::AllocateNewRegion()");
            FatalAssert(ALIGNED(regions.back().base), LOG_TAG_BASIC, "Region base must be aligned to page size in MemoryArena::AllocateNewRegion()");
            FatalAssert(regions.back().next_free_offset.load(std::memory_order_relaxed) == regions.back().len,
                        LOG_TAG_BASIC, "current region should be fully allocated before allocating a new region");
        }
        FatalAssert(ALIGNED(page_size), LOG_TAG_BASIC, "page_size should be cache_alinged");
#ifdef USE_HUGETLB
        void* new_region_base = mmap64(nullptr, page_size * num_pages, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
#else
        void* new_region_base = mmap64(nullptr, page_size * num_pages, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

        if (new_region_base == MAP_FAILED) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_MEMORY, "Failed to allocate memory for MemoryArena."
                    "errno %d, errno msg: %s", errno, strerror(errno));
        }

        if (new_region_base == nullptr) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_MEMORY, "MemoryArena mmap returned nullptr");
        }

        if (!ALIGNED(new_region_base)) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_MEMORY,
                    "MemoryArena new_region_base is not properly aligned. Requested alignment: %lu, new_region_base address: %p",
                    CACHE_LINE_SIZE, new_region_base);
        }

        regions.emplace_back(new_region_base, num_pages * page_size);
    }
};

};

#endif