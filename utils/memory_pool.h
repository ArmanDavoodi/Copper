#ifndef MEMORY_POOL_H_
#define MEMORY_POOL_H_

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

union alignas(CACHE_LINE_SIZE) SlotMeta {
    struct {
        uint8_t in_use : 1; // if the slot is in use
        uint8_t type : 1; // type of the slot (leaf or internal)
        uint8_t reserved : 6; // reserved for future use
    };
    char value[CACHE_LINE_SIZE];

    SlotMeta() : value{0} {}
};

enum SlotType : uint8_t {
    Leaf = 0,
    Internal = 1
};

struct alignas(CACHE_LINE_SIZE) Slot {
    SlotMeta meta;
    char data[]; // actual data starts here
};

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
    virtual ~MemoryPool() = default;

    /* will wait until enough memory is available */
    virtual void* Allocate(SlotType type, AllocationFlags flags = {.value = 0}) = 0;
    virtual size_t BatchAllocate(SlotType type, void** ptr_array, size_t count,
                                 AllocationFlags flags = {.value = 0}) = 0;

    virtual void Free(void* ptr) = 0;
    /* will fail if even one is not freed */
    virtual void BatchFree(void** ptr_array, size_t count) = 0;
};

class LocalMemoryPool : public MemoryPool {
public:
    LocalMemoryPool(size_t internal_size, size_t leaf_size, size_t pool_size) :
        leafObjectSize(leaf_size), internalObjectSize(internal_size),
        leafSlotSize(ALIGNED_SIZE(sizeof(SlotMeta)) + ALIGNED_SIZE(leaf_size)),
        internalSlotSize(ALIGNED_SIZE(sizeof(SlotMeta)) + ALIGNED_SIZE(internal_size)), poolSize(pool_size),
        freeLeafSlots((pool_size / 2) / leafSlotSize),
        freeInternalSlots((pool_size / 2) / internalSlotSize) {

        FatalAssert(leaf_size > 0, LOG_TAG_MEMORY, "leaf_size must be greater than 0");
        FatalAssert(internal_size > 0, LOG_TAG_MEMORY, "internal_size must be greater than 0");
        FatalAssert(
            pool_size >= 2 * (ALIGNED_SIZE(std::max(leafSlotSize, internalSlotSize)) +
                              ALIGNED_SIZE(sizeof(SlotMeta))),
            LOG_TAG_MEMORY, "pool_size is too small for the given object_size and alignment");
#ifdef USE_HUGETLB
        base = mmap64(nullptr, poolSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
#else
        base = mmap64(nullptr, poolSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
        if (base == MAP_FAILED) {
            DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_MEMORY, "Failed to allocate memory for LocalMemoryPool."
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

        nextAvailableLeafSlot.store(reinterpret_cast<uintptr_t>(base), std::memory_order_relaxed);
        uintptr_t firstInternalSlotAddr = reinterpret_cast<uintptr_t>(base) + poolSize - internalSlotSize;
        firstInternalSlotAddr -= (firstInternalSlotAddr % CACHE_LINE_SIZE);
        nextAvailableInternalSlot.store(firstInternalSlotAddr, std::memory_order_relaxed);
        FatalAssert(ALIGNED(reinterpret_cast<Slot*>(nextAvailableInternalSlot.load(std::memory_order_relaxed))) &&
                    ALIGNED(reinterpret_cast<Slot*>(nextAvailableInternalSlot.load(std::memory_order_relaxed))->data),
                    LOG_TAG_MEMORY,
                    "First internal slot is not properly aligned");
        FatalAssert(ALIGNED(reinterpret_cast<Slot*>(nextAvailableLeafSlot.load(std::memory_order_relaxed))) &&
                    ALIGNED(reinterpret_cast<Slot*>(nextAvailableLeafSlot.load(std::memory_order_relaxed))->data),
                    LOG_TAG_MEMORY,
                    "First leaf slot is not properly aligned");
    }

    ~LocalMemoryPool() override {
        SANITY_CHECK(
            if (nextAvailableLeafSlot.load(std::memory_order_relaxed) != reinterpret_cast<uintptr_t>(base)) {
                Slot* leafSlot =
                    reinterpret_cast<Slot*>(reinterpret_cast<uintptr_t>(
                        nextAvailableLeafSlot.load(std::memory_order_relaxed)) - leafSlotSize);
                for (; leafSlot != reinterpret_cast<Slot*>(base);
                    leafSlot = reinterpret_cast<Slot*>(reinterpret_cast<uintptr_t>(leafSlot) - leafSlotSize)) {
                    FatalAssert(leafSlot->meta.in_use == 0, LOG_TAG_MEMORY,
                                "MemoryPool is being destroyed while some slots are in use");
                }
            }

            uintptr_t firstInternalSlotAddr = reinterpret_cast<uintptr_t>(base) + poolSize - internalSlotSize;
            firstInternalSlotAddr -= (firstInternalSlotAddr % CACHE_LINE_SIZE);
            if (nextAvailableInternalSlot.load(std::memory_order_relaxed) != firstInternalSlotAddr) {
                Slot* internalSlot =
                    reinterpret_cast<Slot*>(reinterpret_cast<uintptr_t>(
                        nextAvailableInternalSlot.load(std::memory_order_relaxed)) + internalSlotSize);
                for (; internalSlot != reinterpret_cast<Slot*>(firstInternalSlotAddr);
                       internalSlot = reinterpret_cast<Slot*>(reinterpret_cast<uintptr_t>(internalSlot) +
                                                              internalSlotSize)) {
                    FatalAssert(internalSlot->meta.in_use == 0, LOG_TAG_MEMORY,
                                "MemoryPool is being destroyed while some slots are in use");
                }
            }
#ifdef MEMORY_DEBUG
            FatalAssert(allocations.empty(), LOG_TAG_MEMORY,
                        "MemoryPool is being destroyed while some allocations are not freed");
#endif
        );

        if (munmap(base, poolSize) != 0) {
            DIVFLOG(LOG_LEVEL_ERROR, LOG_TAG_MEMORY, "Failed to free memory for LocalMemoryPool. "
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

    void* Allocate(SlotType type, AllocationFlags flags = {.value = 0}) override {
        // return std::aligned_alloc(CACHE_LINE_SIZE,
        //                                (type == SlotType::Leaf ? leafObjectSize : internalObjectSize));
        Slot* slot_ptr = nullptr;
        bool success = false;
        size_t objectSize;
        if (type == SlotType::Leaf) {
            objectSize = leafObjectSize;
            success = freeLeafSlots.TryPopHead(slot_ptr);
            if (!success) {
                slot_ptr = reinterpret_cast<Slot*>(nextAvailableLeafSlot.fetch_add(leafSlotSize));
                uintptr_t boundry = nextAvailableInternalSlot.load(std::memory_order_acquire);
                if (reinterpret_cast<uintptr_t>(slot_ptr) + leafSlotSize > boundry ||
                    reinterpret_cast<uintptr_t>(slot_ptr) > boundry) {
                    nextAvailableLeafSlot.fetch_sub(leafSlotSize);
                    slot_ptr = nullptr;
                } else {
                    success = true;
                }
            } SANITY_CHECK( else {
                FatalAssert(slot_ptr->meta.type == static_cast<uint8_t>(type), LOG_TAG_MEMORY,
                            "Allocated slot type mismatch");
            } )

            if (!success && !flags.non_blocking) {
                while (!success) {
                    success = freeLeafSlots.PopHead(slot_ptr);
                    if (!success && nextAvailableLeafSlot.load(std::memory_order_acquire) +
                        leafSlotSize <= nextAvailableInternalSlot.load(std::memory_order_acquire)) {
                        slot_ptr = reinterpret_cast<Slot*>(nextAvailableLeafSlot.fetch_add(leafSlotSize));
                        uintptr_t boundry = nextAvailableInternalSlot.load(std::memory_order_acquire);
                        if (reinterpret_cast<uintptr_t>(slot_ptr) + leafSlotSize > boundry ||
                            reinterpret_cast<uintptr_t>(slot_ptr) > boundry) {
                            nextAvailableLeafSlot.fetch_sub(leafSlotSize);
                            slot_ptr = nullptr;
                        } else {
                            success = true;
                        }
                    } SANITY_CHECK( else if (success) {
                        FatalAssert(slot_ptr->meta.type == static_cast<uint8_t>(type), LOG_TAG_MEMORY,
                                    "Allocated slot type mismatch");
                    } )
                }
            }
        } else {
            objectSize = internalObjectSize;
            success = freeInternalSlots.TryPopHead(slot_ptr);
            if (!success) {
                slot_ptr = reinterpret_cast<Slot*>(nextAvailableInternalSlot.fetch_sub(internalSlotSize));
                uintptr_t boundry = nextAvailableLeafSlot.load(std::memory_order_acquire);
                if (reinterpret_cast<uintptr_t>(slot_ptr) < boundry ||
                    reinterpret_cast<uintptr_t>(slot_ptr) - internalSlotSize < boundry) {
                    nextAvailableInternalSlot.fetch_add(internalSlotSize);
                    slot_ptr = nullptr;
                } else {
                    success = true;
                }
            } SANITY_CHECK( else {
                FatalAssert(slot_ptr->meta.type == static_cast<uint8_t>(type), LOG_TAG_MEMORY,
                            "Allocated slot type mismatch");
            } )

            if (!success && !flags.non_blocking) {
                while (!success) {
                    success = freeInternalSlots.PopHead(slot_ptr);
                    if (!success && nextAvailableInternalSlot.load(std::memory_order_acquire) - internalSlotSize >=
                        nextAvailableLeafSlot.load(std::memory_order_acquire)) {
                        slot_ptr = reinterpret_cast<Slot*>(nextAvailableInternalSlot.fetch_sub(internalSlotSize));
                        uintptr_t boundry = nextAvailableLeafSlot.load(std::memory_order_acquire);
                        if (reinterpret_cast<uintptr_t>(slot_ptr) < boundry ||
                            reinterpret_cast<uintptr_t>(slot_ptr) - internalSlotSize < boundry) {
                            nextAvailableInternalSlot.fetch_add(internalSlotSize);
                            slot_ptr = nullptr;
                        } else {
                            success = true;
                        }
                    } SANITY_CHECK( else if (success) {
                        FatalAssert(slot_ptr->meta.type == static_cast<uint8_t>(type), LOG_TAG_MEMORY,
                                    "Allocated slot type mismatch");
                    } )
                }
            }
        }

        if (!success) {
            return nullptr;
        }

        FatalAssert(ALIGNED(slot_ptr), LOG_TAG_MEMORY, "Slot is not properly aligned");
        FatalAssert(ALIGNED(slot_ptr->data), LOG_TAG_MEMORY, "Slot data is not properly aligned");
#ifdef MEMORY_DEBUG
        {
            size_t slotSize = type == SlotType::Leaf ? leafSlotSize : internalSlotSize;
            std::lock_guard<std::mutex> lock(allocation_mutex);
            AllocationSanityCheck((void*)(slot_ptr), slotSize);
            allocations.insert(std::make_pair((void*)(slot_ptr), slotSize));
        }
#endif
        FatalAssert(slot_ptr->meta.in_use == 0, LOG_TAG_MEMORY, "Allocated slot is in use!");
        slot_ptr->meta.in_use = 1;
        slot_ptr->meta.type = static_cast<uint8_t>(type);
        if (flags.clear) {
            std::memset(slot_ptr->data, 0, objectSize);
        }

        return slot_ptr->data;
    }

    size_t BatchAllocate(SlotType type, void** ptr_array, size_t count,
                         AllocationFlags flags = {.value = 0}) override {
        CHECK_NOT_NULLPTR(ptr_array, LOG_TAG_MEMORY);
        FatalAssert(count > 0, LOG_TAG_MEMORY, "Count must be greater than 0");
        size_t allocated = 0;
        size_t assigned = 0;
        size_t objectSize;
        if (type == SlotType::Leaf) {
            objectSize = leafObjectSize;
            allocated = freeLeafSlots.TryBatchPopHead(reinterpret_cast<Slot**>(ptr_array), count);
            SANITY_CHECK(
                for (size_t i = 0; i < allocated; ++i) {
                    FatalAssert(((Slot*)(ptr_array[i]))->meta.type == static_cast<uint8_t>(type), LOG_TAG_MEMORY,
                                "Allocated slot type mismatch");
                }
            );

            if (allocated < count) {
                uintptr_t nextInternal = nextAvailableInternalSlot.load(std::memory_order_acquire);
                uintptr_t nextLeaf = nextAvailableLeafSlot.load(std::memory_order_acquire);
                size_t available = (nextInternal > nextLeaf) ? (nextInternal - nextLeaf) / leafSlotSize : 0;
                size_t to_allocate = std::min(available, count - allocated);
                if (to_allocate > 0) {
                    uintptr_t begin = nextAvailableLeafSlot.fetch_add(to_allocate * leafSlotSize);
                    uintptr_t end = begin + to_allocate * leafSlotSize;
                    uintptr_t boundry = nextAvailableInternalSlot.load(std::memory_order_acquire);
                    if (begin >= boundry) {
                        // no available slots
                        begin = 0;
                        end = 0;
                        nextAvailableLeafSlot.fetch_sub(to_allocate * leafSlotSize);
                    } else if (end > boundry) {
                        size_t num_invalid = (end - boundry + leafSlotSize - 1) / leafSlotSize;
                        nextAvailableLeafSlot.fetch_sub(num_invalid * leafSlotSize);
                        end -= num_invalid * leafSlotSize;
                    }

                    FatalAssert(end >= begin, LOG_TAG_MEMORY, "Invalid slot allocation range");
                    FatalAssert((end - begin) % leafSlotSize == 0, LOG_TAG_MEMORY,
                                "Invalid slot allocation range alignment");
                    size_t newly_allocated = (end - begin) / leafSlotSize;
                    FatalAssert(newly_allocated <= (count - allocated), LOG_TAG_MEMORY,
                                "Allocated more slots than requested");
                    for (size_t i = 0; i < newly_allocated; ++i) {
                        ptr_array[allocated + i] = reinterpret_cast<void*>(begin + (i * leafSlotSize));
                    }
                    allocated += newly_allocated;
                }
            }

            FatalAssert(allocated <= count, LOG_TAG_MEMORY,
                        "Allocated more slots than requested");

            if (allocated < count) {
                if (!flags.non_blocking) {
                    SANITY_CHECK(
                        size_t before = allocated;
                    );
                    while (allocated < count) {
                        allocated += freeLeafSlots.BatchPopHead(
                            reinterpret_cast<Slot**>(ptr_array) + allocated, count - allocated);
                    }
                    FatalAssert(allocated == count, LOG_TAG_MEMORY,
                                "After blocking batch allocation, allocated slots should equal requested count");
                    SANITY_CHECK(
                        for (size_t i = before; i < count; ++i) {
                            FatalAssert(((Slot*)(ptr_array[i]))->meta.type == static_cast<uint8_t>(type), LOG_TAG_MEMORY,
                                        "Allocated slot type mismatch");
                        }
                    );
                } else if (flags.atomic) {
                    // Return the already allocated slots back to the pool
                    assigned = allocated;
                    allocated = 0;
                }
            }
        } else {
            objectSize = internalObjectSize;
            allocated = freeInternalSlots.TryBatchPopHead(reinterpret_cast<Slot**>(ptr_array), count);
            SANITY_CHECK(
                for (size_t i = 0; i < allocated; ++i) {
                    FatalAssert(((Slot*)(ptr_array[i]))->meta.type == static_cast<uint8_t>(type), LOG_TAG_MEMORY,
                                "Allocated slot type mismatch");
                }
            );

            if (allocated < count) {
                uintptr_t nextInternal = nextAvailableInternalSlot.load(std::memory_order_acquire);
                uintptr_t nextLeaf = nextAvailableLeafSlot.load(std::memory_order_acquire);
                size_t available = (nextInternal > nextLeaf) ?
                                   (nextInternal - nextLeaf) / internalSlotSize : 0;
                size_t to_allocate = std::min(available, count - allocated);
                if (to_allocate > 0) {
                    uintptr_t begin = nextAvailableInternalSlot.fetch_sub(to_allocate * internalSlotSize);
                    uintptr_t end = begin - (to_allocate * internalSlotSize);
                    uintptr_t boundry = nextAvailableLeafSlot.load(std::memory_order_acquire);
                    if (begin <= boundry) {
                        // no available slots
                        begin = 0;
                        end = 0;
                        nextAvailableInternalSlot.fetch_add(to_allocate * internalSlotSize);
                    } else if (end < boundry) {
                        size_t num_invalid = (boundry - end + internalSlotSize - 1) / internalSlotSize;
                        nextAvailableInternalSlot.fetch_add(num_invalid * internalSlotSize);
                        end += num_invalid * internalSlotSize;
                    }

                    FatalAssert(begin >= end, LOG_TAG_MEMORY, "Invalid slot allocation range");
                    FatalAssert((begin - end) % internalSlotSize == 0, LOG_TAG_MEMORY,
                                "Invalid slot allocation range alignment");
                    size_t newly_allocated = (begin - end) / internalSlotSize;
                    FatalAssert(newly_allocated <= (count - allocated), LOG_TAG_MEMORY,
                                "Allocated more slots than requested");
                    for (size_t i = 0; i < newly_allocated; ++i) {
                        ptr_array[allocated + i] = reinterpret_cast<void*>(begin - (i * internalSlotSize));
                    }
                    allocated += newly_allocated;
                }
            }

            FatalAssert(allocated <= count, LOG_TAG_MEMORY,
                        "Allocated more slots than requested");

            if (allocated < count) {
                if (!flags.non_blocking) {
                    SANITY_CHECK(
                        size_t before = allocated;
                    );
                    while (allocated < count) {
                        allocated += freeInternalSlots.BatchPopHead(
                            reinterpret_cast<Slot**>(ptr_array) + allocated, count - allocated);
                    }
                    FatalAssert(allocated == count, LOG_TAG_MEMORY,
                                "After blocking batch allocation, allocated slots should equal requested count");
                    SANITY_CHECK(
                        for (size_t i = before; i < count; ++i) {
                            FatalAssert(((Slot*)(ptr_array[i]))->meta.type == static_cast<uint8_t>(type), LOG_TAG_MEMORY,
                                        "Allocated slot type mismatch");
                        }
                    );
                } else if (flags.atomic) {
                    // Return the already allocated slots back to the pool
                    assigned = allocated;
                    allocated = 0;
                }
            }
        }

#ifdef MEMORY_DEBUG
        size_t slotSize = type == SlotType::Leaf ? leafSlotSize : internalSlotSize;
#endif

        if (assigned > 0) {
            FatalAssert(allocated == 0, LOG_TAG_MEMORY,
                        "In atomic non-blocking allocation, allocated should be zero if assigned is greater than zero");
#ifdef MEMORY_DEBUG
            std::set<std::pair<void*, size_t>> current_allocations;
#endif
            for (size_t i = 0; i < assigned; ++i) {
                Slot* slot_ptr = reinterpret_cast<Slot*>(ptr_array[i]);
                FatalAssert(ALIGNED(slot_ptr), LOG_TAG_MEMORY, "Slot is not properly aligned");
                FatalAssert(ALIGNED(slot_ptr->data), LOG_TAG_MEMORY, "Slot data is not properly aligned");
#ifdef MEMORY_DEBUG
                {
                    AllocationSanityCheck((void*)(slot_ptr), slotSize);
                    AllocationSanityCheck((void*)(slot_ptr), slotSize, current_allocations);
                    current_allocations.insert(std::make_pair((void*)(slot_ptr), slotSize));
                }
#endif
                FatalAssert(slot_ptr->meta.in_use == 0, LOG_TAG_MEMORY, "Allocated slot is in use!");
                slot_ptr->meta.type = static_cast<uint8_t>(type);
            }
            freeLeafSlots.BatchPush(reinterpret_cast<Slot**>(ptr_array), assigned);
            memset(ptr_array, 0, sizeof(void*) * assigned);
        }

        for (size_t i = 0; i < allocated; ++i) {
            Slot* slot_ptr = reinterpret_cast<Slot*>(ptr_array[i]);
            FatalAssert(ALIGNED(slot_ptr), LOG_TAG_MEMORY, "Slot is not properly aligned");
            FatalAssert(ALIGNED(slot_ptr->data), LOG_TAG_MEMORY, "Slot data is not properly aligned");
#ifdef MEMORY_DEBUG
            {
                std::lock_guard<std::mutex> lock(allocation_mutex);
                AllocationSanityCheck((void*)(slot_ptr), slotSize);
                allocations.insert(std::make_pair((void*)(slot_ptr), slotSize));
            }
#endif
            FatalAssert(slot_ptr->meta.in_use == 0, LOG_TAG_MEMORY, "Allocated slot is in use!");
            slot_ptr->meta.in_use = 1;
            slot_ptr->meta.type = static_cast<uint8_t>(type);
            if (flags.clear) {
                std::memset(slot_ptr->data, 0, objectSize);
            }
            ptr_array[i] = slot_ptr->data;
        }

        return allocated;
    }

    void Free(void* ptr) override {
        CHECK_NOT_NULLPTR(ptr, LOG_TAG_MEMORY);
        // std::free(ptr);
        FatalAssert(ALIGNED(ptr), LOG_TAG_MEMORY, "Slot data is not properly aligned");
        Slot* slot_ptr = reinterpret_cast<Slot*>(
            reinterpret_cast<uintptr_t>(ptr) - ALIGNED_SIZE(sizeof(SlotMeta)));
        FatalAssert(ALIGNED(slot_ptr), LOG_TAG_MEMORY, "Slot is not properly aligned");
        FatalAssert((reinterpret_cast<uintptr_t>(slot_ptr) >= reinterpret_cast<uintptr_t>(base) &&
                    reinterpret_cast<uintptr_t>(slot_ptr) < reinterpret_cast<uintptr_t>(base) + poolSize),
                    LOG_TAG_MEMORY, "Pointer does not belong to this MemoryPool");
        FatalAssert(slot_ptr->data == ptr, LOG_TAG_MEMORY, "Pointer does not point to the start of slot data");
        FatalAssert(slot_ptr->meta.in_use == 1, LOG_TAG_MEMORY, "Double free detected");
        slot_ptr->meta.in_use = 0;
#ifdef MEMORY_DEBUG
        {
            std::lock_guard<std::mutex> lock(allocation_mutex);
            auto it = allocations.find(std::make_pair((void*)(slot_ptr),
                                                      slot_ptr->meta.type == static_cast<uint8_t>(SlotType::Leaf) ?
                                                      leafSlotSize : internalSlotSize));
            FatalAssert(it != allocations.end(), LOG_TAG_MEMORY, "Double free or corruption detected");
            allocations.erase(it);
        }
#endif

        if (slot_ptr->meta.type == static_cast<uint8_t>(SlotType::Leaf)) {
            freeLeafSlots.Push(slot_ptr);
        } else {
            freeInternalSlots.Push(slot_ptr);
        }
    }

    void BatchFree(void** ptr_array, size_t count) override {
        CHECK_NOT_NULLPTR(ptr_array, LOG_TAG_MEMORY);
        FatalAssert(count > 0, LOG_TAG_MEMORY, "Count must be greater than 0");
        SlotType type =
            static_cast<SlotType>((reinterpret_cast<Slot*>(reinterpret_cast<uintptr_t>(ptr_array[0]) -
                                                           ALIGNED_SIZE(sizeof(SlotMeta))))->meta.type);
        for (size_t i = 0; i < count; ++i) {
            CHECK_NOT_NULLPTR(ptr_array[i], LOG_TAG_MEMORY);
            FatalAssert(ALIGNED(ptr_array[i]), LOG_TAG_MEMORY, "Slot data is not properly aligned");
            Slot* slot_ptr = reinterpret_cast<Slot*>(
                reinterpret_cast<uintptr_t>(ptr_array[i]) - ALIGNED_SIZE(sizeof(SlotMeta)));
            FatalAssert(ALIGNED(slot_ptr), LOG_TAG_MEMORY, "Slot is not properly aligned");
            FatalAssert((reinterpret_cast<uintptr_t>(slot_ptr) >= reinterpret_cast<uintptr_t>(base) &&
                        reinterpret_cast<uintptr_t>(slot_ptr) < reinterpret_cast<uintptr_t>(base) + poolSize),
                        LOG_TAG_MEMORY, "Pointer does not belong to this MemoryPool");
            FatalAssert(slot_ptr->data == ptr_array[i], LOG_TAG_MEMORY, "Pointer does not point to the start of slot data");
            FatalAssert(slot_ptr->meta.in_use == 1, LOG_TAG_MEMORY, "Double free detected");
            FatalAssert(slot_ptr->meta.type == static_cast<uint8_t>(type), LOG_TAG_MEMORY,
                        "Allocated slot type mismatch");
            slot_ptr->meta.in_use = 0;
#ifdef MEMORY_DEBUG
            {
                std::lock_guard<std::mutex> lock(allocation_mutex);
                auto it = allocations.find(std::make_pair((void*)(slot_ptr),
                                                          (slot_ptr->meta.type == static_cast<uint8_t>(SlotType::Leaf) ?
                                                          leafSlotSize : internalSlotSize)));
                FatalAssert(it != allocations.end(), LOG_TAG_MEMORY, "Double free or corruption detected");
                allocations.erase(it);
            }
#endif
            ptr_array[i] = reinterpret_cast<void*>(slot_ptr);
        }

        if (type == SlotType::Leaf) {
            freeLeafSlots.BatchPush(reinterpret_cast<Slot**>(ptr_array), count);
        } else {
            freeInternalSlots.BatchPush(reinterpret_cast<Slot**>(ptr_array), count);
        }

        memset(ptr_array, 0, sizeof(void*) * count);
    }

    void* GetBaseAddress() {
        return base;
    }

    size_t GetPageSize(bool is_leaf) const {
        FatalAssert(leafObjectSize < UINT32_MAX, LOG_TAG_MEMORY,
                    "Page size exceeds UINT32_MAX in MemoryPool::GetPageSize()");
        FatalAssert(leafObjectSize > 0, LOG_TAG_MEMORY,
                    "Page size is zero in MemoryPool::GetPageSize()");
        FatalAssert(internalObjectSize < UINT32_MAX, LOG_TAG_MEMORY,
                    "Page size exceeds UINT32_MAX in MemoryPool::GetPageSize()");
        FatalAssert(internalObjectSize > 0, LOG_TAG_MEMORY,
                    "Page size is zero in MemoryPool::GetPageSize()");
        return is_leaf ? leafObjectSize : internalObjectSize;
    }

    size_t GetPoolSize() const {
        return poolSize;
    }

protected:
    const size_t leafObjectSize;
    const size_t internalObjectSize;
    const size_t leafSlotSize;
    const size_t internalSlotSize;
    const size_t poolSize;

    /* this should be handled in the upper layer for eviction */
    // std::atomic<uint64_t> slotsBeingFreed;
    BlockingQueue<Slot*> freeLeafSlots;
    BlockingQueue<Slot*> freeInternalSlots;
    std::atomic<uintptr_t> nextAvailableLeafSlot;
    std::atomic<uintptr_t> nextAvailableInternalSlot;

    void* base;

#ifdef MEMORY_DEBUG
    std::set<std::pair<void*, size_t>> allocations;
    std::mutex allocation_mutex;
#endif
};

};
#endif