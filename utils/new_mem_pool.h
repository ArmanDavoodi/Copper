
#include <cstdint>
#include <cstddef>

#include "utils/thread.h"
#include "utils/synchronization.h"
#include "utils/concurrent_datastructures.h"
#include "utils/btree.h"

#include "common.h"

namespace divftree {

constexpr size_t gcd(size_t a, size_t b) {
    while (b != 0) {
        size_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

constexpr size_t lcm(size_t a, size_t b) {
    return (a == 0 || b == 0) ? 0 : (a / gcd(a, b)) * b;
}

enum class PageState : uint8_t {
    INVALID = 0,
    FREE = 1,
    ALLOCATED = 2,
    ALLOCATING = 3, /* an intermediate state used during allocation or merge */
    RIGHT_COALESCING = 4, /* an intermediate state used during free */
    LEFT_COALESCING = 5, /* an intermediate state used during free */
    DANGLING = 6, /* an intermediate state used during free */
    LOCKED = 7 /* an intermediate state used during read */
};

inline constexpr size_t INVALID_FREE_LIST_INDEX = SIZE_MAX;

struct FreePageList;

/* should be of size 64 bytes */
struct alignas(CACHE_LINE_SIZE) PageHeader {
    std::atomic<PageState> state = PageState::INVALID;
    std::atomic<size_t> readers_count = 0;

    std::atomic<Page*>* list_head = nullptr;
    std::atomic<Page*> next_free_page = nullptr;
    std::atomic<Page*> prev_free_page = nullptr;


    size_t page_index = INVALID_FREE_LIST_INDEX; /* page total size == page_index * min_aloc_size */
    size_t page_total_bytes = 0;
    size_t data_size_bytes = 0; /* not including the header and footer */
};

/* should be of size 64 bytes */
struct PageFooter {
    std::atomic<Page*> start = nullptr;
};

inline constexpr size_t PAGE_DATA_SIZE_ALIGNMENT = lcm(CACHE_LINE_SIZE - (sizeof(PageFooter) % CACHE_LINE_SIZE),
                                                       alignof(PageFooter));

struct alignas(CACHE_LINE_SIZE) Page {
    static_assert(sizeof(PageHeader) == CACHE_LINE_SIZE, "PageHeader size is not equal to CACHE_LINE_SIZE");
    PageHeader header;
    uint8_t data[];
    /* there is also a page footer at the end -> todo: make sure it has correct alignment*/

    static constexpr size_t GetPageSize(size_t data_bytes) {
        FatalAssert(data_bytes > 0, LOG_TAG_MEMORY, "Data bytes must be greater than zero");
        size_t total_size = ALIGNED_SIZE(sizeof(PageHeader)) + ALIGNED_SIZE(data_bytes, PAGE_DATA_SIZE_ALIGNMENT) +
                            sizeof(PageFooter);
        FatalAssert(ALIGNED(total_size, CACHE_LINE_SIZE), LOG_TAG_MEMORY,
                    "next page should be aligned to cache line size");
        return total_size;
    }

    static constexpr size_t GetDataBytes(size_t page_size) {
        if (page_size <= ALIGNED_SIZE(sizeof(PageHeader) +
                         ALIGNED_SIZE(sizeof(PageFooter), PAGE_DATA_SIZE_ALIGNMENT))) {
            return 0;
        }
        FatalAssert(ALIGNED(page_size), CACHE_LINE_SIZE, "half page size is not aligned");
        size_t data_bytes = page_size - ALIGNED_SIZE(sizeof(PageHeader)) - sizeof(PageFooter);
        FatalAssert(ALIGNED(data_bytes, PAGE_DATA_SIZE_ALIGNMENT),
                    LOG_TAG_MEMORY, "data bytes of half page is not correctly aligned");
        return data_bytes;
    }

    inline PageFooter* GetSelfFooter() {
        FatalAssert(ALIGNED(this) && ALIGNED(data), LOG_TAG_MEMORY, "page is not aligned");
        PageFooter* footer = reinterpret_cast<PageFooter*>(data + header.data_size_bytes);
        FatalAssert(ALIGNED(footer, alignof(PageFooter)) &&
                    ALIGNED(footer, CACHE_LINE_SIZE - (sizeof(PageFooter) % CACHE_LINE_SIZE)), LOG_TAG_MEMORY,
                    "page footer is not correctly aligned");
        return footer;
    }

    /* maybe out of bounds! */
    inline Page* GetPageAfter() {
        Page* page = reinterpret_cast<Page*>(reinterpret_cast<uint8_t*>(this) + header.page_total_bytes);
        FatalAssert(ALIGNED(page), LOG_TAG_MEMORY, "next page is not aligned");
        return page;
    }

    /* maybe out of bounds! + the start pointer might change when we read it but there should always be
       a footer there as long as it is not out of bounds*/
    PageFooter* GetPageFooterBefore() {
        FatalAssert(ALIGNED(this) && ALIGNED(data), LOG_TAG_MEMORY, "page is not aligned");
        if (reinterpret_cast<uintptr_t>(this) < sizeof(PageFooter)) {
            return nullptr;
        }

        PageFooter* footer = reinterpret_cast<PageFooter*>(reinterpret_cast<uint8_t*>(this) - sizeof(PageFooter));
        FatalAssert(ALIGNED(footer, alignof(PageFooter)) &&
                    ALIGNED(footer, CACHE_LINE_SIZE - (sizeof(PageFooter) % CACHE_LINE_SIZE)), LOG_TAG_MEMORY,
                    "prev page footer is not correctly aligned");
        return footer;
    }
};

class Allocator {
public:
    Allocator(void* base_ptr, size_t total_size_bytes, size_t min_alloc_size_bytes);
    Allocator(size_t total_size_bytes, size_t min_alloc_size_bytes);
    ~Allocator();

    /* todo batch allocate and batch free */
    /* todo: page index should be uint32_t */
    /* todo: memory allocation and free sanity checks + get stats for how much fragmentation and other stuff we have and implement the remove empty lists that are not accessed frequently logic if needed */
    void* allocate(size_t requested_size_bytes) {
        if (requested_size_bytes == 0) {
            return nullptr;
        }

        size_t min_num_blocks_needed = GetFreeListIndex(requested_size_bytes);
        Page* page = GetFirstFit(min_num_blocks_needed);
        if (page == nullptr) {
            return nullptr;
        }

        if (page->header.page_index > min_num_blocks_needed) {
            SplitPage(page, min_num_blocks_needed);
        }

        FatalAssert(page->header.page_index == min_num_blocks_needed,
                    LOG_TAG_MEMORY, "Allocated page size does not match requested size");
        FatalAssert(page->header.data_size_bytes >= requested_size_bytes,
                    LOG_TAG_MEMORY, "Allocated page data size is less than requested size");
        FatalAssert(page->header.readers_count.load(std::memory_order_acquire) == 0,
                    LOG_TAG_MEMORY, "Allocated page has active readers");
        FatalAssert(page->header.next_free_page.load(std::memory_order_acquire) == nullptr,
                    LOG_TAG_MEMORY, "Allocated page is still in free list");
        FatalAssert(page->header.prev_free_page.load(std::memory_order_acquire) == nullptr,
                    LOG_TAG_MEMORY, "Allocated page is still in free list");
        FatalAssert(page->header.state.load(std::memory_order_acquire) == PageState::ALLOCATING,
                    LOG_TAG_MEMORY, "Allocated page is not in ALLOCATING state");
        page->header.state.store(PageState::ALLOCATED, std::memory_order_release);
        FatalAssert(ALIGNED(page) && ALIGNED(page->data), LOG_TAG_MEMORY, "Allocated page is not aligned");
        return reinterpret_cast<void*>(page->data);
    }

    void free(void* ptr) {
        if (ptr == nullptr) {
            return;
        }

        Page* page = reinterpret_cast<Page*>(reinterpret_cast<uint8_t*>(ptr) - sizeof(PageHeader));
        FatalAssert(ALIGNED(page) && ALIGNED(page->data), LOG_TAG_MEMORY, "Pointer to free is not aligned");
        FatalAssert(page->data == ptr, LOG_TAG_MEMORY, "Pointer to free does not match page data pointer");
        FatalAssert(page->header.state.load(std::memory_order_acquire) == PageState::ALLOCATED,
                    LOG_TAG_MEMORY, "Page to free is not in ALLOCATED state");
        page->header.state.store(PageState::RIGHT_COALESCING, std::memory_order_release);
        FreePage(page);
    }

protected:
    void* const base;
    const size_t total_size;
    const size_t min_alloc_size;
    BPlusTree<uint32_t, Page> free_page_lists;
    SXLock allocator_lock;

    inline constexpr size_t GetFreeListIndex(size_t requested_size_bytes) {
        size_t page_size = Page::GetPageSize(requested_size_bytes);
        if (page_size < min_alloc_size) {
            page_size = min_alloc_size;
        } else if (page_size % min_alloc_size != 0) {
            page_size = ALIGNED_SIZE(page_size, min_alloc_size);
        }
        return page_size / min_alloc_size;
    }

    Page* GetHead(std::atomic<Page*>& head) {
        Page* page = head.load(std::memory_order_acquire);
        while (page != nullptr) {
            PageState expected = PageState::FREE;
            Page* next_page = page->header.next_free_page.load(std::memory_order_acquire);
            if (!page->header.state.compare_exchange_strong(expected, PageState::ALLOCATING)) {
                DIVFTREE_YIELD();
                page = head.load(std::memory_order_acquire);
                continue;
            }

            while (page->header.readers_count.load(std::memory_order_acquire) > 0) {
                DIVFTREE_YIELD();
            }

            head.compare_exchange_strong(page, next_page);
            FatalAssert(page->header.list_head == &head,
                        LOG_TAG_MEMORY, "Page's free list head does not match the current free list");
            next_page->header.prev_free_page.compare_exchange_strong(page, nullptr);
            page->header.list_head = nullptr;
            page->header.next_free_page.store(nullptr, std::memory_order_release);
            page->header.prev_free_page.store(nullptr, std::memory_order_release);
            return page;
        }
        return nullptr;
    }

    inline Page* GetFirstFit(size_t index) {
        auto it = free_page_lists.Get(index);
        auto end = free_page_lists.end();

        while (true) {
            while ((it != end) && ((*it).load(std::memory_order_acquire) == nullptr)) {
                ++it;
            }
            if (it == end) {
                return nullptr;
            }

            Page* page = GetHead(*it);
            if (page != nullptr) {
                return page;
            };
            ++it;
        }
    }

    /* todo: this can cause issues! */
    bool RemoveFromList(Page* page, PageState& page_state) {
        CHECK_NOT_NULLPTR(page, LOG_TAG_MEMORY);
        FatalAssert(ALIGNED(page) && ALIGNED(page->data), LOG_TAG_MEMORY, "Page is not aligned");
        PageState expected = PageState::FREE;
        while (true) {
            if (!(page->header.state.compare_exchange_strong(expected, PageState::DANGLING))) {
                if (expected == PageState::LOCKED) {
                    expected = PageState::FREE;
                    DIVFTREE_YIELD();
                    continue;
                }
                page_state = expected;
                return false;
            }
            break;
        }
        page_state = PageState::DANGLING;

        while (page->header.readers_count.load(std::memory_order_acquire) > 0) {
            DIVFTREE_YIELD();
        }

        Page* next_page = page->header.next_free_page.load(std::memory_order_acquire);
        expected = PageState::FREE;
        while (next_page != nullptr &&
               !(next_page->header.state.compare_exchange_strong(expected, PageState::LOCKED))) {
            DIVFTREE_YIELD();
            next_page = page->header.next_free_page.load(std::memory_order_acquire);
            expected = PageState::FREE;
        }

        Page* prev_page = page->header.prev_free_page.load(std::memory_order_acquire);
        if (prev_page == nullptr) {
            /* this page is head */
            FatalAssert(page->header.list_head != nullptr,
                        LOG_TAG_MEMORY, "Page's free list head is null");
            FatalAssert(page->header.list_head->load(std::memory_order_acquire) == page,
                        LOG_TAG_MEMORY, "Page is not the head of its free list");
            page->header.list_head->store(next_page, std::memory_order_release);
        } else {
            prev_page->header.next_free_page.store(next_page, std::memory_order_release);
        }

        if (next_page != nullptr) {
            next_page->header.prev_free_page.store(prev_page, std::memory_order_release);
            next_page->header.state.store(PageState::FREE, std::memory_order_release);
        }

        page->header.list_head = nullptr;
        page->header.next_free_page.store(nullptr, std::memory_order_release);
        page->header.prev_free_page.store(nullptr, std::memory_order_release);
        return true;
    }

    inline bool CoalesceWithRight(Page* page, PageState next_state) {
        CHECK_NOT_NULLPTR(page, LOG_TAG_MEMORY);
        FatalAssert(ALIGNED(page) && ALIGNED(page->data), LOG_TAG_MEMORY, "Page is not aligned");
        FatalAssert(page->header.state.load(std::memory_order_acquire) == PageState::RIGHT_COALESCING,
                    LOG_TAG_MEMORY, "Page is not in RIGHT_COALESCING state during coalesce");

        Page* right_page = page->GetPageAfter();
        CHECK_NOT_NULLPTR(right_page, LOG_TAG_MEMORY);
        FatalAssert(ALIGNED(right_page) && ALIGNED(right_page->data), LOG_TAG_MEMORY, "Right page is not aligned");
        if (right_page >= reinterpret_cast<Page*>(reinterpret_cast<uint8_t*>(base) + total_size)) {
            page->header.state.store(next_state, std::memory_order_release);
            return false;
        }

        PageState right_state = right_page->header.state.load(std::memory_order_acquire);
        PageState expected = PageState::DANGLING;

        while (!right_page->header.state.compare_exchange_strong(expected, PageState::LEFT_COALESCING)) {
            DIVFTREE_YIELD();
            right_state = right_page->header.state.load(std::memory_order_acquire);
            if (right_state == PageState::FREE) {
                (void)RemoveFromList(right_page, right_state);
            }

            if (right_state == PageState::INVALID ||
                right_state == PageState::ALLOCATED ||
                right_state == PageState::ALLOCATING) {
                page->header.state.store(next_state, std::memory_order_release);
                if (right_state == PageState::INVALID ||
                    right_state == PageState::ALLOCATED ||
                    right_state == PageState::ALLOCATING) {
                        return false;
                }
                page->header.state.store(PageState::RIGHT_COALESCING, std::memory_order_release);
            }
            expected = PageState::DANGLING;
        }

        FatalAssert(right_page->header.state.load(std::memory_order_acquire) == PageState::LEFT_COALESCING,
                    LOG_TAG_MEMORY, "Invalid state!");

        size_t new_page_index = page->header.page_index + right_page->header.page_index;
        size_t new_page_total_bytes = page->header.page_total_bytes + right_page->header.page_total_bytes;
        page->header.page_index = new_page_index;
        page->header.page_total_bytes = new_page_total_bytes;
        page->header.data_size_bytes = Page::GetDataBytes(new_page_total_bytes);
        PageFooter* page_footer = right_page->GetSelfFooter();
        page_footer->start.store(page, std::memory_order_release);
        right_page->header.state.store(PageState::INVALID, std::memory_order_release);
        page->header.state.store(next_state, std::memory_order_release);
        return true;
    }

    inline bool CoalesceWithLeft(Page*& page) {
        CHECK_NOT_NULLPTR(page, LOG_TAG_MEMORY);
        FatalAssert(ALIGNED(page) && ALIGNED(page->data), LOG_TAG_MEMORY, "Page is not aligned");
        FatalAssert(page->header.state.load(std::memory_order_acquire) == PageState::LEFT_COALESCING,
                    LOG_TAG_MEMORY, "Page is not in RIGHT_COALESCING state during coalesce");
        if (page == reinterpret_cast<Page*>(base)) {
            return false;
        }

        Page* left_page = nullptr;
        while (left_page == nullptr) {
            PageFooter* left_footer = page->GetPageFooterBefore();
            CHECK_NOT_NULLPTR(left_footer, LOG_TAG_MEMORY);
            left_page = left_footer->start.load(std::memory_order_acquire);
            CHECK_NOT_NULLPTR(left_page, LOG_TAG_MEMORY);
            FatalAssert(ALIGNED(left_page) && ALIGNED(left_page->data), LOG_TAG_MEMORY, "Left page is not aligned");
            PageState left_state = left_page->header.state.load(std::memory_order_acquire);
            if (left_state == PageState::FREE) {
                RemoveFromList(left_page, left_state);
            }

            if (left_state == PageState::INVALID ||
                left_state == PageState::LOCKED ||
                left_state == PageState::LEFT_COALESCING) {
                left_page = nullptr;
                DIVFTREE_YIELD();
                continue;
            }

            if (left_state == PageState::ALLOCATED ||
                left_state == PageState::ALLOCATING) {
                return false;
            }

            if (left_state == PageState::RIGHT_COALESCING) {
                page->header.state.store(PageState::DANGLING, std::memory_order_release);
                DIVFTREE_YIELD();
                if (page->)
            }
        }
        CHECK_NOT_NULLPTR(left_page, LOG_TAG_MEMORY);
        FatalAssert(ALIGNED(left_page) && ALIGNED(left_page->data), LOG_TAG_MEMORY, "Right page is not aligned");
        if (left_page >= reinterpret_cast<Page*>(reinterpret_cast<uint8_t*>(base) + total_size)) {
            return false;
        }

        PageState right_state = left_page->header.state.load(std::memory_order_acquire);
        PageState expected = PageState::DANGLING;
        if (right_state == PageState::FREE) {
            (void)RemoveFromList(left_page, right_state);
        }

        if (right_state == PageState::INVALID ||
            right_state == PageState::ALLOCATED ||
            right_state == PageState::ALLOCATING) {
            return false;
        }

        while (!left_page->header.state.compare_exchange_strong(expected, PageState::LEFT_COALESCING)) {
            DIVFTREE_YIELD();
            right_state = left_page->header.state.load(std::memory_order_acquire);
            if (right_state == PageState::FREE) {
                (void)RemoveFromList(left_page, right_state);
            }

            if (right_state == PageState::INVALID ||
                right_state == PageState::ALLOCATED ||
                right_state == PageState::ALLOCATING) {
                return false;
            }
            expected = PageState::DANGLING;
        }

        FatalAssert(left_page->header.state.load(std::memory_order_acquire) == PageState::LEFT_COALESCING,
                    LOG_TAG_MEMORY, "Invalid state!");

        size_t new_page_index = page->header.page_index + left_page->header.page_index;
        size_t new_page_total_bytes = page->header.page_total_bytes + left_page->header.page_total_bytes;
        page->header.page_index = new_page_index;
        page->header.page_total_bytes = new_page_total_bytes;
        page->header.data_size_bytes = Page::GetDataBytes(new_page_total_bytes);
        PageFooter* page_footer = left_page->GetSelfFooter();
        page_footer->start.store(page, std::memory_order_release);
        left_page->header.state.store(PageState::INVALID, std::memory_order_release);
        return true;
    }

    inline void FreePage(Page* page, bool check_left_coalesce = true) {
        CHECK_NOT_NULLPTR(page, LOG_TAG_MEMORY);
        FatalAssert(ALIGNED(page) && ALIGNED(page->data), LOG_TAG_MEMORY, "Page is not aligned");
        FatalAssert(page->header.readers_count.load(std::memory_order_acquire) == 0,
                    LOG_TAG_MEMORY, "Page has active readers during free");
        FatalAssert(page->header.state.load(std::memory_order_acquire) == PageState::RIGHT_COALESCING,
                    LOG_TAG_MEMORY, "Page is not in RIGHT_COALESCING state during free");
        Page* right_page = page->GetPageAfter();
        CHECK_NOT_NULLPTR(right_page, LOG_TAG_MEMORY);
        FatalAssert(ALIGNED(right_page) && ALIGNED(right_page->data), LOG_TAG_MEMORY, "Right page is not aligned");
        if (right_page < reinterpret_cast<Page*>(reinterpret_cast<uint8_t*>(base) + total_size)) {
            PageState right_state = right_page->header.state.load(std::memory_order_acquire);
            while (right_state == PageState::RIGHT_COALESCING) {
                DIVFTREE_YIELD();
                right_state = right_page->header.state.load(std::memory_order_acquire);
            }
        }
        page->header.state.store(PageState::LEFT_COALESCING, std::memory_order_release);
    }

    inline void SplitPage(Page* target, size_t desired_num_blocks) {
        CHECK_NOT_NULLPTR(target, LOG_TAG_MEMORY);
        FatalAssert(ALIGNED(target) && ALIGNED(target->data), LOG_TAG_MEMORY, "Target page is not aligned");
        FatalAssert(target->header.page_index > desired_num_blocks,
                    LOG_TAG_MEMORY, "Target page is smaller than desired size");
        FatalAssert(target->header.state.load(std::memory_order_acquire) == PageState::ALLOCATING,
                    LOG_TAG_MEMORY, "Target page is not in ALLOCATING state");

        size_t target_size_bytes = desired_num_blocks * min_alloc_size;
        size_t leftover_index = target->header.page_index - desired_num_blocks;

        Page* leftover = reinterpret_cast<Page*>(reinterpret_cast<uint8_t*>(target) + target_size_bytes);
        FatalAssert(ALIGNED(leftover) && ALIGNED(leftover->data), LOG_TAG_MEMORY, "Leftover page is not aligned");

        leftover->header.page_index = leftover_index;
        leftover->header.page_total_bytes = leftover_index * min_alloc_size;
        leftover->header.data_size_bytes = Page::GetDataBytes(leftover->header.page_total_bytes);
        leftover->header.state.store(PageState::RIGHT_COALESCING, std::memory_order_release);
        leftover->header.readers_count.store(0, std::memory_order_release);
        PageFooter* leftover_footer = leftover->GetSelfFooter();
        leftover_footer->start.store(leftover, std::memory_order_release);

        target->header.page_index = desired_num_blocks;
        target->header.page_total_bytes = target_size_bytes;
        target->header.data_size_bytes = Page::GetDataBytes(target->header.page_total_bytes);
        PageFooter* target_footer = target->GetSelfFooter();
        target_footer->start.store(target, std::memory_order_release);

        FreePage(leftover, false);
    }

    inline size_t GetContainerFreeListIndex(size_t page_size_bytes) {
        FatalAssert(page_size_bytes >= min_alloc_size, LOG_TAG_MEMORY, "Size bytes is less than minimum allocation size");
        FatalAssert(ALIGNED(page_size_bytes) && ALIGNED(page_size_bytes, min_alloc_size), LOG_TAG_MEMORY,
                    "Size bytes is not aligned");
        size_t index = 0;
        size_t block_max = 2;
        size_t num_blocks = page_size_bytes / min_alloc_size;
        while (num_blocks >= block_max) {
            block_max *= 2;
            index++;
        }
        FatalAssert(index < free_page_lists.size(), LOG_TAG_MEMORY, "Free list index out of bounds");
        return index;
    }

    void InsertPage(Page* page) {
        CHECK_NOT_NULLPTR(page, LOG_TAG_MEMORY);
        FatalAssert(ALIGNED(page) && ALIGNED(page->data), LOG_TAG_MEMORY, "Page is not aligned");
        FatalAssert(page->header.page_index != INVALID_FREE_LIST_INDEX &&
                    page->header.page_index < free_page_lists.size(),
                    LOG_TAG_MEMORY, "Page free list index is invalid");
        FatalAssert(page->header.readers_count.load(std::memory_order_acquire) == 0,
                    LOG_TAG_MEMORY, "Page has active readers during free");
        FatalAssert(page->header.next_free_page.load(std::memory_order_acquire) == nullptr,
                    LOG_TAG_MEMORY, "Page next free page is not null during free");
        FatalAssert(page->header.prev_free_page.load(std::memory_order_acquire) == nullptr,
                    LOG_TAG_MEMORY, "Page prev free page is not null during free");
        FatalAssert(page->header.state.load(std::memory_order_acquire) == PageState::DEALLOCATING ||
                    page->header.state.load(std::memory_order_acquire) == PageState::ALLOCATING,
                    LOG_TAG_MEMORY, "Page is not in ALLOCATING state during free");
        page->header.readers_count.store(1, std::memory_order_release); // prevent others from reading during insertion
        page->header.state.store(PageState::FREE, std::memory_order_release);
        page->header.prev_free_page.store(nullptr, std::memory_order_release);

        while (true) {
            Page* head_page = free_page_lists[page->header.page_index].load(std::memory_order_acquire);

            if (head_page == nullptr) {
                page->header.next_free_page.store(nullptr, std::memory_order_release);
                if (free_page_lists[page->header.page_index].compare_exchange_strong(head_page, page)) {
                    break;
                }
                DIVFTREE_YIELD();
                continue;
            }

            PageState expected = PageState::FREE;
            if (!page->header.state.compare_exchange_strong(expected, PageState::LOCKED)) {
                DIVFTREE_YIELD();
                continue;
            }

            page->header.next_free_page.store(head_page, std::memory_order_release);
            if (free_page_lists[page->header.page_index].compare_exchange_strong(head_page, page)) {
                if (head_page != nullptr) {
                    head_page->header.prev_free_page.store(page, std::memory_order_release);
                }
                break;
            }
            DIVFTREE_YIELD();
        }
    }

    Page* FitPageAndReturnLeftOver(Page* target, size_t desired_num_blocks) {
        CHECK_NOT_NULLPTR(target, LOG_TAG_MEMORY);
        FatalAssert(ALIGNED(target) && ALIGNED(target->data), LOG_TAG_MEMORY, "Target page is not aligned");
        FatalAssert(target->header.state.load(std::memory_order_acquire) == PageState::ALLOCATING,
                    LOG_TAG_MEMORY, "Target page is not in ALLOCATING state");
        FatalAssert(target->header.next_free_page.load(std::memory_order_acquire) == nullptr,
                    LOG_TAG_MEMORY, "Target page next free page is not null during split");
        FatalAssert(target->header.prev_free_page.load(std::memory_order_acquire) == nullptr,
                    LOG_TAG_MEMORY, "Target page prev free page is not null during split");
        FatalAssert(target->header.page_total_bytes % min_alloc_size == 0, LOG_TAG_MEMORY,
                    "Invalid page size");
        FatalAssert(target->header.readers_count.load(std::memory_order_acquire) == 0,
                    LOG_TAG_MEMORY, "Target page has active readers during split");
        FatalAssert(desired_num_blocks > 0, LOG_TAG_MEMORY, "Desired number of blocks is zero");
        FatalAssert(target->header.page_total_bytes / min_alloc_size >= desired_num_blocks,
                    LOG_TAG_MEMORY, "Target page is too small to split");

        FatalAssert(target->header.page_index != INVALID_FREE_LIST_INDEX &&
                    target->header.page_index < free_page_lists.size() &&
                    target->header.page_index > 0 &&
                    GetContainerFreeListIndex(target->header.page_total_bytes) == target->header.page_index,
                    LOG_TAG_MEMORY, "Target page invalid index");
        FatalAssert(target->header.state.load(std::memory_order_acquire) == PageState::ALLOCATING,
                    LOG_TAG_MEMORY, "Target page is not in ALLOCATING state");
        if (target->header.page_total_bytes / min_alloc_size == desired_num_blocks) {
            return nullptr;
        }

        size_t target_size_bytes = desired_num_blocks * min_alloc_size;
        size_t target_data_size_bytes = Page::GetDataBytes(target_size_bytes);
        size_t target_index = GetContainerFreeListIndex(target_size_bytes);
        size_t left_over_size_bytes = target->header.page_total_bytes - target_size_bytes;
        size_t left_over_data_size_bytes = Page::GetDataBytes(left_over_size_bytes);
        size_t left_over_index = GetContainerFreeListIndex(left_over_size_bytes);

        FatalAssert(target_size_bytes > 0 && left_over_size_bytes > 0, LOG_TAG_MEMORY,
                    "Split results in zero sized page");
        FatalAssert(target_data_size_bytes > 0 && left_over_data_size_bytes > 0, LOG_TAG_MEMORY,
                    "Split results in zero sized page data");
        FatalAssert(target_index < free_page_lists.size(), LOG_TAG_MEMORY,
                    "Target page has invalid free list index");
        FatalAssert(target_index <= target->header.page_index, LOG_TAG_MEMORY,
                    "Target page has larger free list index than original page");
        FatalAssert(left_over_index < free_page_lists.size(), LOG_TAG_MEMORY,
                    "Left over page has invalid free list index");
        FatalAssert(left_over_index <= target->header.page_index, LOG_TAG_MEMORY,
                    "Left over page has larger free list index than target page");

        Page* free_page = reinterpret_cast<Page*>(reinterpret_cast<uint8_t*>(target) + (target_size_bytes));
        free_page->header.next_free_page.store(nullptr, std::memory_order_release);
        free_page->header.prev_free_page.store(nullptr, std::memory_order_release);
        free_page->header.page_total_bytes = left_over_size_bytes;
        free_page->header.data_size_bytes = left_over_data_size_bytes;
        free_page->header.page_index = left_over_index;
        free_page->header.readers_count.store(0, std::memory_order_release);
        free_page->header.state.store(PageState::ALLOCATING, std::memory_order_release);
        FatalAssert(free_page->GetSelfFooter() == target->GetSelfFooter(),
                    LOG_TAG_MEMORY, "Page footer mismatch after split");
        free_page->GetSelfFooter()->start.store(free_page, std::memory_order_release);

        target->header.page_total_bytes = target_size_bytes;
        target->header.data_size_bytes = target_data_size_bytes;
        target->header.page_index = target_index;
        FatalAssert(target->GetSelfFooter() == free_page->GetPageFooterBefore(),
                    LOG_TAG_MEMORY, "Page footer mismatch after split");
        target->GetSelfFooter()->start.store(target, std::memory_order_release);
        return free_page;
    }

    Page* Coalescing(Page* page) {
        CHECK_NOT_NULLPTR(page, LOG_TAG_MEMORY);
        FatalAssert(ALIGNED(page) && ALIGNED(page->data), LOG_TAG_MEMORY, "Page is not aligned");
        FatalAssert(page->header.page_index != INVALID_FREE_LIST_INDEX &&
                    page->header.page_index < free_page_lists.size(),
                    LOG_TAG_MEMORY, "Page free list index is invalid");
        FatalAssert(page->header.state.load(std::memory_order_acquire) == PageState::ALLOCATED,
                    LOG_TAG_MEMORY, "Target page is not in ALLOCATED state");
        FatalAssert(page->header.next_free_page.load(std::memory_order_acquire) == nullptr,
                    LOG_TAG_MEMORY, "Target page next free page is not null during split");
        FatalAssert(page->header.prev_free_page.load(std::memory_order_acquire) == nullptr,
                    LOG_TAG_MEMORY, "Target page prev free page is not null during split");
        FatalAssert(page->header.readers_count.load(std::memory_order_acquire) == 0,
                    LOG_TAG_MEMORY, "Target page has active readers during merge");

        /* first try to merge with the right neighbour in a blocking manner
           then check if the left neighbour is free and if so merge with that
           as well, otherwise do not block */
        page->header.state.store(PageState::RIGHT_COALESCING, std::memory_order_release);
        Page* next_page = page->GetPageAfter();
        PageState state = next_page->header.state.load(std::memory_order_acquire);
        while (state == PageState::DEALLOCATING || state == PageState::RIGHT_COALESCING) {
            DIVFTREE_YIELD();
            state = next_page->header.state.load(std::memory_order_acquire);
        }

        if (state != PageState::FREE) {
            page->header.state.store(PageState::DEALLOCATING, std::memory_order_release);
            state = next_page->header.state.load(std::memory_order_acquire);
            if ()
        }
    }
};
};