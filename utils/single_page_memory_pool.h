#ifndef SINGLE_PAGE_MEMORY_POOL_H_
#define SINGLE_PAGE_MEMORY_POOL_H_

#include <sys/mman.h>
#include <atomic>
#ifdef MEMORY_DEBUG
#include <set>
#include <mutex>
#endif

#include "debug.h"
#include "common.h"

#include "utils/thread.h"
#include "utils/concurrent_datastructures.h"

namespace divftree {

union AllocationFlags {
    struct {
        uint8_t clear : 1; // if the allocated memory should be cleared
        uint8_t non_blocking : 1; // if the allocation should be non-blocking
        uint8_t atomic : 1; // if batch allocation should allocate as much as it can or fail altogether if not enough memory
        uint8_t unused : 5; // reserved for future use
    };
    uint8_t value;
};

class MemoryPool {
public:
    MemoryPool(size_t page_size, size_t pool_size) :
        pageSize(page_size), poolSize(pool_size),
        freePages(pool_size / page_size) {
        FatalAssert(page_size > 0, LOG_TAG_MEMORY, "page_size must be greater than 0");
        FatalAssert(pool_size > 0, LOG_TAG_MEMORY, "pool_size must be greater than 0");
        FatalAssert(ALIGNED(page_size), LOG_TAG_MEMORY, "page_size must be aligned to CACHE_LINE_SIZE");
        FatalAssert(ALIGNED(pool_size), LOG_TAG_MEMORY, "pool_size must be aligned to CACHE_LINE_SIZE");
        FatalAssert(pool_size > page_size,
                    LOG_TAG_MEMORY, "pool_size is too small for the given page_size and alignment");
        FatalAssert(pool_size % page_size == 0,
                    LOG_TAG_MEMORY, "pool_size must be multiple of page_size");

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_MEMORY,
                "Creating MemoryPool with page size %lu and pool size %lu", page_size, pool_size);

#ifdef USE_HUGETLB
        base = mmap64(nullptr, poolSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
#else
        base = mmap64(nullptr, poolSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
        if (base == MAP_FAILED) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_MEMORY, "Failed to allocate memory for MemoryPool."
                    "errno %d, errno msg: %s", errno, strerror(errno));
        }

        if (base == nullptr) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_MEMORY, "MemoryPool mmap returned nullptr");
        }

        if (!ALIGNED(base)) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_MEMORY,
                    "MemoryPool base is not properly aligned. Requested alignment: %lu, base address: %p",
                    CACHE_LINE_SIZE, base);
        }

        for (size_t offset = 0; offset + pageSize <= poolSize; offset += pageSize) {
            void* page_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + offset);
            freePages.Push(page_ptr);
        }

        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_MEMORY,
                "MemoryPool created successfully with %zu pages", poolSize / pageSize);
    }

    ~MemoryPool() {
#ifdef MEMORY_DEBUG
        FatalAssert(allocations.empty(), LOG_TAG_MEMORY,
                    "MemoryPool is being destroyed while some allocations are not freed");
#endif

        if (munmap(base, poolSize) != 0) {
            DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_MEMORY, "Failed to free memory for MemoryPool. "
                    "errno %d, errno msg: %s", errno, strerror(errno));
        }
    }

#ifdef MEMORY_DEBUG
    void AllocationSanityCheck(void* address, size_t size, std::set<std::pair<void*, size_t>>& current_allocations) {
        auto last = current_allocations.upper_bound(std::make_pair(address, size));
        for (auto it = current_allocations.begin(); it != last; ++it) {
            FatalAssert(((*it).first < address) && ((*it).first + (*it).second <= address),
                        LOG_TAG_MEMORY, "Memory corruption detected before allocated slot");
        }
        for (auto it = last; it != current_allocations.end(); ++it) {
            FatalAssert(((*it).first > address) && (address + size <= (*it).first),
                        LOG_TAG_MEMORY, "Memory corruption detected after allocated slot");
        }
    }

    void AllocationSanityCheck(void* address, size_t size) {
        AllocationSanityCheck(address, size, allocations);
    }
#endif

    void* Allocate(AllocationFlags flags = {.value = 0}) {
        void* page_ptr = nullptr;
        bool success = freePages.TryPopHead(page_ptr);

        if (!success && !flags.non_blocking) {
            while (!success) {
                success = freePages.PopHead(page_ptr);
            }
        }

        if (!success) {
            return nullptr;
        }

        FatalAssert(ALIGNED(page_ptr), LOG_TAG_MEMORY, "Page is not properly aligned");
#ifdef MEMORY_DEBUG
        {
            std::lock_guard<std::mutex> lock(allocation_mutex);
            AllocationSanityCheck(page_ptr, pageSize);
            allocations.insert(std::make_pair(page_ptr, pageSize));
        }
#endif
        if (flags.clear) {
            std::memset(page_ptr, 0, pageSize);
        }

        return page_ptr;
    }

    size_t BatchAllocate(void** ptr_array, size_t count,
                         AllocationFlags flags = {.value = 0}) {
        CHECK_NOT_NULLPTR(ptr_array, LOG_TAG_MEMORY);
        FatalAssert(count > 0, LOG_TAG_MEMORY, "Count must be greater than 0");
        memset(ptr_array, 0, sizeof(void*) * count);
        size_t allocated = freePages.TryBatchPopHead(reinterpret_cast<void**>(ptr_array), count);

        if (allocated < count && flags.atomic) {
            FatalAssert(flags.non_blocking, LOG_TAG_MEMORY,
                        "Atomic flag can only be set in non-blocking mode");
            if (allocated > 0) {
                freePages.BatchPush(reinterpret_cast<void**>(ptr_array), allocated);
            }
            return 0;
        }

        if (allocated == count || flags.non_blocking) {
            if (flags.clear) {
                for (size_t i = 0; i < allocated; ++i) {
                    std::memset(ptr_array[i], 0, pageSize);
                }
            }
#ifdef MEMORY_DEBUG
        for (size_t i = 0; i < allocated; ++i) {
                FatalAssert(ALIGNED(ptr_array[i]), LOG_TAG_MEMORY, "Page is not properly aligned");
                {
                    std::lock_guard<std::mutex> lock(allocation_mutex);
                    AllocationSanityCheck(ptr_array[i], pageSize);
                    allocations.insert(std::make_pair(ptr_array[i], pageSize));
                }
            }
#endif
            return allocated;
        }

        FatalAssert(!flags.atomic, LOG_TAG_MEMORY,
                    "Atomic flag can should have been handled before");
        FatalAssert(!flags.non_blocking, LOG_TAG_MEMORY,
                    "Non-blocking flag can should have been handled before");
        while (allocated < count) {
            allocated += freePages.BatchPopHead(reinterpret_cast<void**>(ptr_array) + allocated, count - allocated);
        }
        FatalAssert(allocated == count, LOG_TAG_MEMORY,
                    "After blocking batch allocation, allocated pages should equal requested count");
        if (flags.clear) {
            for (size_t i = 0; i < allocated; ++i) {
                std::memset(ptr_array[i], 0, pageSize);
            }
        }
#ifdef MEMORY_DEBUG
        for (size_t i = 0; i < allocated; ++i) {
            FatalAssert(ALIGNED(ptr_array[i]), LOG_TAG_MEMORY, "Page is not properly aligned");
            {
                std::lock_guard<std::mutex> lock(allocation_mutex);
                AllocationSanityCheck(ptr_array[i], pageSize);
                allocations.insert(std::make_pair(ptr_array[i], pageSize));
            }
        }
#endif
        return allocated;
    }

    void Free(void* ptr) {
        CHECK_NOT_NULLPTR(ptr, LOG_TAG_MEMORY);
        FatalAssert(ALIGNED(ptr), LOG_TAG_MEMORY, "Page is not properly aligned");
#ifdef MEMORY_DEBUG
        {
            std::lock_guard<std::mutex> lock(allocation_mutex);
            auto it = allocations.find(std::make_pair(ptr, pageSize));
            FatalAssert(it != allocations.end(), LOG_TAG_MEMORY, "Double free or corruption detected");
            allocations.erase(it);
        }
#endif
        bool res = freePages.Push(ptr);
        FatalAssert(res, LOG_TAG_MEMORY, "Failed to free page back to MemoryPool");
        UNUSED_VARIABLE(res);
    }

    void BatchFree(void** ptr_array, size_t count) {
        CHECK_NOT_NULLPTR(ptr_array, LOG_TAG_MEMORY);
        FatalAssert(count > 0, LOG_TAG_MEMORY, "Count must be greater than 0");
#ifdef MEMORY_DEBUG
        for (size_t i = 0; i < count; ++i) {
            CHECK_NOT_NULLPTR(ptr_array[i], LOG_TAG_MEMORY);
            FatalAssert(ALIGNED(ptr_array[i]), LOG_TAG_MEMORY, "Page is not properly aligned");
            {
                std::lock_guard<std::mutex> lock(allocation_mutex);
                auto it = allocations.find(std::make_pair(ptr_array[i], pageSize));
                FatalAssert(it != allocations.end(), LOG_TAG_MEMORY, "Double free or corruption detected");
                allocations.erase(it);
            }
        }
#endif
        bool res = freePages.BatchPush(reinterpret_cast<void**>(ptr_array), count);
        FatalAssert(res, LOG_TAG_MEMORY, "Failed to free pages back to MemoryPool");
        UNUSED_VARIABLE(res);
    }

    void* GetBaseAddress() {
        return base;
    }

    size_t GetPageSize() const {
        FatalAssert(pageSize < UINT32_MAX, LOG_TAG_MEMORY,
                    "Page size exceeds UINT32_MAX in MemoryPool::GetPageSize()");
        FatalAssert(pageSize > 0, LOG_TAG_MEMORY,
                    "Page size is zero in MemoryPool::GetPageSize()");
        return pageSize;
    }

    size_t GetPoolSize() const {
        return poolSize;
    }

protected:
    const size_t pageSize;
    const size_t poolSize;

    BlockingQueue<void*> freePages;

    void* base;

#ifdef MEMORY_DEBUG
    std::set<std::pair<void*, size_t>> allocations;
    std::mutex allocation_mutex;
#endif
};

};
#endif