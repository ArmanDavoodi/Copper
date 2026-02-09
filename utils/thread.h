#ifndef THREAD_H_
#define THREAD_H_

#include "configs/support.h"

#include <thread>
#include <cstdint>
#include <unordered_set>
#include <random>
#include <atomic>
#include <mutex>
#include <unistd.h>
#include <sys/syscall.h>

#include "debug.h"

#include "utils/string.h"

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))

#include <immintrin.h>
#define DIVFTREE_YIELD()  _mm_pause()

#elif defined(__i386__) || defined(__x86_64__)

#include <immintrin.h>
#define DIVFTREE_YIELD()  __builtin_ia32_pause()

#elif defined(__aarch64__) || defined(__arm__)

#define DIVFTREE_YIELD()  asm volatile("yield" ::: "memory")

#else
#define DIVFTREE_YIELD()
#endif


// #ifndef DIVF_SEED
// #define DIVF_SEED std::random_device()()
// #endif
namespace divftree {

inline std::random_device::result_type GetRandomSeed() {
    static std::random_device rd;
    static std::mutex rd_mutex;
    std::lock_guard<std::mutex> lock(rd_mutex);
    return rd();
}

enum LockMode : uint8_t  {
    SX_SHARED, SX_EXCLUSIVE
};

enum LockHeld : uint8_t {
    LOCKED, UNLOCKED, LOCK_CHECK_DISABLED
};

String LockModeToString(LockMode mode) {
    switch (mode) {
        case SX_SHARED: return "SHARED";
        case SX_EXCLUSIVE: return "EXCLUSIVE";
        default: return "INVALID";
    }
}

typedef uint64_t DIVFThreadID;
inline constexpr DIVFThreadID INVALID_DIVF_THREAD_ID = UINT64_MAX;

class Thread;

inline thread_local Thread* threadSelf = nullptr;

class Thread {
public:
    Thread(uint32_t random_perc) : centroid_compute_buffer(nullptr),
                                   _parent_id((threadSelf == nullptr) ? INVALID_DIVF_THREAD_ID : threadSelf->ID()),
                                   _id(nextId.fetch_add(1)), _done(true), _safty_net(false), _thrd(nullptr),
                                   _gen(GetRandomSeed()), _gen64(GetRandomSeed()), _uniform_dist(1, random_perc),
                                   _next_task_id(0) {}

    ~Thread() {
        if ((threadSelf == nullptr) || (threadSelf->ID() != _parent_id)) {
            return;
        }

        WaitForThreadToFinish();

        if (JoinableByMe()) {
            _thrd.load(std::memory_order_relaxed)->join();
        }

        if (_thrd.load(std::memory_order_relaxed) != nullptr) {
            delete _thrd.load(std::memory_order_relaxed);
            _thrd.store(nullptr, std::memory_order_relaxed);
        }
    }

    inline void InitDIVFThread() {
        if (_id != 0) {
            uint64_t retry = 0;
            while ((_thrd.load(std::memory_order_acquire) == nullptr) && (retry < MAX_THREAD_RETRY_COUNT)) {
                ++retry;
                DIVFTREE_YIELD();
            }
            if (_thrd.load(std::memory_order_acquire) == nullptr) {
                _thrd.wait(nullptr);
            }
        }

        FatalAssert(threadSelf == nullptr, LOG_TAG_THREAD, "thread should not be inited yet!");
        FatalAssert((_id == 0) || (_thrd.load(std::memory_order_relaxed) != nullptr), LOG_TAG_THREAD,
                    "thread should have started!");
        FatalAssert((_id == 0) || (_thrd.load(std::memory_order_relaxed)->get_id() == std::this_thread::get_id()),
                    LOG_TAG_THREAD, "the thread itself should call this function");
        FatalAssert(_done.load(std::memory_order_relaxed), LOG_TAG_THREAD, "thread should be in the initial state");
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_THREAD, "Initing thread %p - ID:%lu - parent:%lu", this, _id, _parent_id);
        threadSelf = this;

#ifdef HANG_DETECTION
        all_threads_lock.lock();
        all_threads.push_back(this);
        all_threads_lock.unlock();
#endif
        _done.store(false, std::memory_order_release);
    }

    inline void InitDIVFThread(uint16_t dim) {
        InitDIVFThread();
        FatalAssert(centroid_compute_buffer == nullptr, LOG_TAG_THREAD,
                    "centroid compute buffer should not be initialized yet!");
        centroid_compute_buffer = new MVTYPE[dim * 2];
    }

    /* should be the last thing called before exiting the thread */
    inline void DestroyDIVFThread() {
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        FatalAssert(!_done.load(std::memory_order_relaxed), LOG_TAG_THREAD,
                    "thread should not be in the initial state");
        FatalAssert((_id == 0) || (_thrd.load(std::memory_order_relaxed) != nullptr), LOG_TAG_THREAD,
                    "thread should have started!");
        FatalAssert((_id == 0) || (_thrd.load(std::memory_order_relaxed)->get_id() == std::this_thread::get_id()),
                    LOG_TAG_THREAD, "the thread itself should call this function");
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_THREAD, "Destroying thread %p - ID:%lu - parent:%lu", this, _id, _parent_id);
        threadSelf = nullptr;
        if (centroid_compute_buffer != nullptr) {
            delete[] centroid_compute_buffer;
            centroid_compute_buffer = nullptr;
        }
        _safty_net.store(true, std::memory_order_release);
#ifdef HANG_DETECTION
        all_threads_lock.lock();
        for (size_t i = 0; i < all_threads.size(); ++i) {
            if (all_threads[i] == this) {
                all_threads[i] = all_threads.back();
                all_threads.pop_back();
                break;
            }
        }
        all_threads_lock.unlock();
#endif
        _done.store(true, std::memory_order_release);
        _done.notify_all();
        _safty_net.store(false, std::memory_order_release);
    }

    template<class _Callable, class... Args>
    inline void Start(_Callable&& func, Args&&... args) {
        FatalAssert(threadSelf != nullptr, LOG_TAG_THREAD, "caller should be a valid thread!");
        FatalAssert(threadSelf->ID() == _parent_id, LOG_TAG_THREAD, "caller should be parent");
        FatalAssert(_thrd.load(std::memory_order_relaxed) == nullptr, LOG_TAG_THREAD, "thread should be null");
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_THREAD, "Starting thread %p - ID:%lu - parent:%lu", this, _id, _parent_id);
        _thrd.store(new std::thread(std::forward<_Callable>(func), this, (std::forward<Args>(args))...),
                    std::memory_order_release);
        _thrd.notify_one();
    }

    template<class _Callable, class _Object, class... Args>
    inline void StartMemberFunction(_Callable&& func, _Object* obj, Args&&... args) {
        FatalAssert(threadSelf != nullptr, LOG_TAG_THREAD, "caller should be a valid thread!");
        FatalAssert(threadSelf->ID() == _parent_id, LOG_TAG_THREAD, "caller should be parent");
        FatalAssert(_thrd.load(std::memory_order_relaxed) == nullptr, LOG_TAG_THREAD, "thread should be null");
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_THREAD, "Starting thread %p - ID:%lu - parent:%lu", this, _id, _parent_id);
        _thrd.store(new std::thread(std::forward<_Callable>(func), obj, this, (std::forward<Args>(args))...),
                    std::memory_order_release);
        _thrd.notify_one();
    }

    inline void Detach() {
        FatalAssert(threadSelf != nullptr, LOG_TAG_THREAD, "caller should be a valid thread!");
        FatalAssert(threadSelf->ID() == _parent_id, LOG_TAG_THREAD, "caller should be parent");
        FatalAssert(_id != 0, LOG_TAG_THREAD, "cannot detach the main thread!");
        FatalAssert(_thrd.load(std::memory_order_acquire) != nullptr, LOG_TAG_THREAD,
                    "cannot detach a thread which is not started!");
        _thrd.load(std::memory_order_acquire)->detach();
    }

    inline void WaitForThreadToFinish() {
        FatalAssert(threadSelf != nullptr, LOG_TAG_THREAD, "caller should be a valid thread!");
        FatalAssert(threadSelf->ID() != _id, LOG_TAG_THREAD, "caller cannot be self!");
#ifdef LOCK_DEBUG
        FatalAssert(waitingFor == INVALID_DIVF_THREAD_ID, LOG_TAG_THREAD, "we cannot be waiting on two threads!");
        waitingFor = _id;
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_LOCK, "Waiting for thread %p - ID:%lu", this, _id);
#endif
        if (!_done.load(std::memory_order_acquire)) {
            _done.wait(false, std::memory_order_acquire);
        }
        FatalAssert(_done.load(std::memory_order_relaxed), LOG_TAG_THREAD, "thread should be done!");

        while (_safty_net.load(std::memory_order_acquire)) {
            DIVFTREE_YIELD();
        }
        FatalAssert(!_safty_net.load(std::memory_order_acquire), LOG_TAG_THREAD, "safty net should be disabled");
#ifdef LOCK_DEBUG
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_LOCK, "done Waiting for thread %p - ID:%lu", this, _id);
        waitingFor = INVALID_DIVF_THREAD_ID;
#endif
    }

    inline void Join() {
        FatalAssert(JoinableByMe(), LOG_TAG_THREAD, "The thread should be joinable!");
        _thrd.load(std::memory_order_acquire)->join();
    }

    inline bool JoinableByMe() const {
        FatalAssert(threadSelf != nullptr, LOG_TAG_THREAD, "caller thread not inited!");
        if ((_id == 0) || (threadSelf->ID() != _parent_id) || (_thrd.load(std::memory_order_acquire) == nullptr)) {
            return false;
        }
        return _thrd.load(std::memory_order_relaxed)->joinable();
    }

    inline DIVFThreadID ID() const {
        return _id;
    }

    inline void AcquireLockSanityLog(void* lockAddr, LockMode mode) {
        UNUSED_VARIABLE(lockAddr);
        UNUSED_VARIABLE(mode);
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        CHECK_NOT_NULLPTR(lockAddr, LOG_TAG_LOCK);
#ifdef LOCK_DEBUG
        SanityCheckLockNotHeldByMe(lockAddr);
        if (mode == SX_EXCLUSIVE) {
            heldExclusive.insert(lockAddr);
            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_LOCK, "Acquired EXCLUSIVE lock %p", lockAddr);
        } else {
            heldShared.insert(lockAddr);
            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_LOCK, "Acquired SHARED lock %p", lockAddr);
        }
#endif
    }

    inline void DowngradeLockSanityLog(void* lockAddr) {
        UNUSED_VARIABLE(lockAddr);
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        CHECK_NOT_NULLPTR(lockAddr, LOG_TAG_LOCK);
#ifdef LOCK_DEBUG
        SanityCheckLockHeldInModeByMe(lockAddr, SX_EXCLUSIVE);
        heldExclusive.erase(lockAddr);
        heldShared.insert(lockAddr);
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_LOCK, "Downgraded EXCLUSIVE to SHARED lock %p", lockAddr);
#endif
    }

    inline void UpgradeLockSanityLog(void* lockAddr) {
        UNUSED_VARIABLE(lockAddr);
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        CHECK_NOT_NULLPTR(lockAddr, LOG_TAG_LOCK);
#ifdef LOCK_DEBUG
        SanityCheckLockHeldInModeByMe(lockAddr, SX_SHARED);
        heldShared.erase(lockAddr);
        heldExclusive.insert(lockAddr);
        DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_LOCK, "Upgraded SHARED to EXCLUSIVE lock %p", lockAddr);
#endif
    }

    inline void ReleaseLockSanityLog(void* lockAddr, LockMode mode) {
        UNUSED_VARIABLE(lockAddr);
        UNUSED_VARIABLE(mode);
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        CHECK_NOT_NULLPTR(lockAddr, LOG_TAG_LOCK);
#ifdef LOCK_DEBUG
        SanityCheckLockHeldInModeByMe(lockAddr, mode);
        if (mode == SX_EXCLUSIVE) {
            heldExclusive.erase(lockAddr);
            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_LOCK, "Released EXCLUSIVE lock %p", lockAddr);
        } else {
            heldShared.erase(lockAddr);
            DIVFLOG(LOG_LEVEL_DEBUG, LOG_TAG_LOCK, "Released SHARED lock %p", lockAddr);
        }
#endif
    }

    inline uint64_t SanityCheckNumLocksHeldByMe() {
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
#ifdef LOCK_DEBUG
        return heldExclusive.size() + heldShared.size();
#else
        return 0;
#endif
    }

    inline void SanityCheckLockNotHeldInBothModesByMe(void* lockAddr) {
        UNUSED_VARIABLE(lockAddr);
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        CHECK_NOT_NULLPTR(lockAddr, LOG_TAG_LOCK);
#ifdef LOCK_DEBUG
        FatalAssert((heldExclusive.find(lockAddr) == heldExclusive.end()) ||
                    (heldShared.find(lockAddr) == heldShared.end()),
                    LOG_TAG_LOCK, "Lock %p is held in both modes by thread %lu", lockAddr, _id);
#endif
    }

    inline LockHeld LockHeldByMe(void* lockAddr) {
        UNUSED_VARIABLE(lockAddr);
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        CHECK_NOT_NULLPTR(lockAddr, LOG_TAG_LOCK);
#ifdef LOCK_DEBUG
        SanityCheckLockNotHeldInBothModesByMe(lockAddr);
        if ((heldExclusive.find(lockAddr) != heldExclusive.end()) || (heldShared.find(lockAddr) != heldShared.end())) {
            return LockHeld::LOCKED;
        } else {
            return LockHeld::UNLOCKED;
        }
#endif
        return LockHeld::LOCK_CHECK_DISABLED;
    }

    inline LockHeld LockHeldInModeByMe(void* lockAddr, LockMode mode) {
        UNUSED_VARIABLE(lockAddr);
        UNUSED_VARIABLE(mode);
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        CHECK_NOT_NULLPTR(lockAddr, LOG_TAG_LOCK);
#ifdef LOCK_DEBUG
        SanityCheckLockNotHeldInBothModesByMe(lockAddr);
        if (((mode == LockMode::SX_EXCLUSIVE) && (heldExclusive.find(lockAddr) != heldExclusive.end())) ||
            ((mode == LockMode::SX_SHARED) && (heldShared.find(lockAddr) != heldShared.end()))) {
            return LockHeld::LOCKED;
        } else {
            return LockHeld::UNLOCKED;
        }
#endif
        return LockHeld::LOCK_CHECK_DISABLED;
    }

    inline void SanityCheckLockHeldByMe(void* lockAddr) {
        UNUSED_VARIABLE(lockAddr);
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        CHECK_NOT_NULLPTR(lockAddr, LOG_TAG_LOCK);
#ifdef LOCK_DEBUG
        SanityCheckLockNotHeldInBothModesByMe(lockAddr);
        FatalAssert(LockHeldByMe(lockAddr) == LockHeld::LOCKED,
                    LOG_TAG_LOCK, "Lock %p is not held by thread %lu", lockAddr, _id);
#endif
    }

    inline void SanityCheckLockHeldInModeByMe(void* lockAddr, LockMode mode) {
        UNUSED_VARIABLE(lockAddr);
        UNUSED_VARIABLE(mode);
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        CHECK_NOT_NULLPTR(lockAddr, LOG_TAG_LOCK);
#ifdef LOCK_DEBUG
        SanityCheckLockNotHeldInBothModesByMe(lockAddr);
        FatalAssert(LockHeldInModeByMe(lockAddr, mode) == LockHeld::LOCKED,
                    LOG_TAG_LOCK, "Lock %p is not held by thread %lu in mode %hhu", lockAddr, _id, (uint8_t)(mode));
#endif
    }

    inline void SanityCheckLockNotHeldByMe(void* lockAddr) {
        UNUSED_VARIABLE(lockAddr);
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        CHECK_NOT_NULLPTR(lockAddr, LOG_TAG_LOCK);
#ifdef LOCK_DEBUG
        SanityCheckLockNotHeldInBothModesByMe(lockAddr);
        FatalAssert(LockHeldByMe(lockAddr) == LockHeld::UNLOCKED,
                    LOG_TAG_LOCK, "Lock %p is held by thread %lu", lockAddr, _id);
#endif
    }

    inline void SanityCheckLockNotHeldInModeByMe(void* lockAddr, LockMode mode) {
        UNUSED_VARIABLE(lockAddr);
        UNUSED_VARIABLE(mode);
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        CHECK_NOT_NULLPTR(lockAddr, LOG_TAG_LOCK);
#ifdef LOCK_DEBUG
        SanityCheckLockNotHeldInBothModesByMe(lockAddr);
        FatalAssert(LockHeldInModeByMe(lockAddr, mode) == LockHeld::UNLOCKED,
                    LOG_TAG_LOCK, "Lock %p is held by thread %lu in mode %hhu", lockAddr, _id, (uint8_t)(mode));
#endif
    }

    inline void LoopIncrement() {
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
#ifdef HANG_DETECTION
        _iteration_sn.fetch_add(1);
#endif
    }

    inline bool CheckHang() {
#ifdef HANG_DETECTION
        uint64_t current_sn = _iteration_sn.load(std::memory_order_acquire);
        if (current_sn == last_iteration_sn) {
            return true;
        } else {
            last_iteration_sn = current_sn;
            return false;
        }
#else
        return false;
#endif
    }

    /* todo: check if it works better than a random number generator */
    // inline bool PeriodicBinary(size_t target, size_t base = 100) {
    //     FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
    //     FatalAssert(base >= target, LOG_TAG_BASIC, "base cannot be less than target!");
    //     ++rand_num;
    //     return (rand_num % base < target);
    // }

    inline uint32_t UniformRandom(uint32_t range) {
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        FatalAssert(range > 0, LOG_TAG_THREAD, "range cannot be 0!");
        return (_uniform_dist(_gen) % range);
    }

    inline bool UniformBinary(uint32_t rate) {
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        return (_uniform_dist(_gen) <= rate);
    }

    inline uint32_t UniformRange32(uint32_t first, uint32_t second) {
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        FatalAssert(first <= second, LOG_TAG_THREAD, "first cannot be greater than second!");
        return std::uniform_int_distribution<uint32_t>(first, second)(_gen);
    }

    inline uint64_t UniformRange64(uint64_t first, uint64_t second) {
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        FatalAssert(first <= second, LOG_TAG_THREAD, "first cannot be greater than second!");
        return std::uniform_int_distribution<uint64_t>(first, second)(_gen);
    }

    inline std::pair<uint32_t, uint32_t> UniformRangeTwo32(uint32_t first, uint32_t second) {
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        FatalAssert(first <= second, LOG_TAG_THREAD, "first cannot be greater than second!");
        std::uniform_int_distribution<uint32_t> temp_dist(first, second);
        return std::make_pair(temp_dist(_gen), temp_dist(_gen));
    }

    inline std::pair<uint64_t, uint64_t> UniformRangeTwo64(uint64_t first, uint64_t second) {
        FatalAssert(threadSelf == this, LOG_TAG_THREAD, "thread is not inited!");
        FatalAssert(first <= second, LOG_TAG_THREAD, "first cannot be greater than second!");
        std::uniform_int_distribution<uint64_t> temp_dist(first, second);
        return std::make_pair(temp_dist(_gen64), temp_dist(_gen64));
    }

    inline uint64_t GetNextTaskID() {
        return (_next_task_id++);
    }

    inline void UpdatePollStats(bool lock_acquired, size_t num_tasks_polled) {
#ifdef ENABLE_STAT_COLLECTION
        ++cnt_num_polls;
        if (!lock_acquired) {
            ++cnt_unsuccsessful_polls;
        } else if (num_tasks_polled == 0) {
            ++cnt_empty_polls;
        } else {
            cnt_num_remote_reads_polled += num_tasks_polled;
        }
#else
        UNUSED_VARIABLE(lock_acquired);
        UNUSED_VARIABLE(num_tasks_polled);
#endif
    }

    inline void UpdateMemoryStats(size_t num_from_cool, size_t num_from_pool, size_t num_tries) {
#ifdef ENABLE_STAT_COLLECTION
        FatalAssert(num_tries > 0, LOG_TAG_BUFFER, "num_tries should be greater than 0 in BufferMgr::ReadFromRemote()");
        FatalAssert((num_from_cool + num_from_pool) > 0, LOG_TAG_BUFFER,
                    "at least one memory should be allocated in BufferMgr::ReadFromRemote()");
        ++cnt_remote_accesses;
        cnt_tries_to_get_memory += num_tries;
        if (num_tries == 1) {
            ++cnt_single_try;
            cnt_got_memory_from_cool_once += num_from_cool;
            cnt_got_memory_from_pool_once += num_from_pool;
        } else {
            cnt_got_memory_from_cool_multiple += num_from_cool;
            cnt_got_memory_from_pool_multiple += num_from_pool;
        }
#else
        UNUSED_VARIABLE(num_from_cool);
        UNUSED_VARIABLE(num_from_pool);
        UNUSED_VARIABLE(num_tries);
#endif
    }

    /* todo add stats to check how many times do we get anything from pool vs cooling list(i.e eviction) */
    inline void AppendStats(String& stats_str,
                            size_t& total_num_polls,
                            size_t& total_unsuccsessful_polls,
                            size_t& total_empty_polls,
                            size_t& total_num_remote_reads_polled,
                            size_t& total_remote_accesses,
                            size_t& total_tries_to_get_memory,
                            size_t& total_single_try,
                            size_t& total_got_memory_from_cool_once,
                            size_t& total_got_memory_from_pool_once,
                            size_t& total_got_memory_from_cool_multiple,
                            size_t& total_got_memory_from_pool_multiple) {
#ifdef ENABLE_STAT_COLLECTION
        stats_str += String("Thread(%lu): %lu - "
                            "num_polls: %zu(num_fail: %.2f%%(%zu), num_empty %.2f%%(%zu), num_valid: %.2f%%(%zu)), "
                            "num_valid_to_successful_ratio: %.2f%%, "
                            "num_remote_reads_polled: %zu, avg_num_polled_per_valid_polls: %zu, "
                            "num_remote_rdma_reads: %zu(single_alloc: %.2f%%(%zu), multi_alloc: %.2f%%(%zu)), "
                            "num_alloc_tries: %zu, avg_num_alloc_tries_per_req: %zu, "
                            "avg_num_alloc_multi_tries_per_req: %zu, "
                            "num_pages_got: %zu(from_pool: %.2f%%(%zu), from_cooling_list: %.2f%%(%zu))\n",
                            syscall(SYS_gettid), _id, cnt_num_polls,
                            (cnt_num_polls > 0 ? (100.0 * cnt_unsuccsessful_polls / cnt_num_polls) : 0),
                            cnt_unsuccsessful_polls,
                            (cnt_num_polls > 0 ? (100.0 * cnt_empty_polls / cnt_num_polls) : 0), cnt_empty_polls,
                            (cnt_num_polls > 0 ?
                                (100.0 * (cnt_num_polls - cnt_unsuccsessful_polls - cnt_empty_polls) / cnt_num_polls) :
                                0),
                            cnt_num_polls - cnt_unsuccsessful_polls - cnt_empty_polls,
                            (cnt_num_polls > 0 ? (100.0 * (cnt_num_polls - cnt_unsuccsessful_polls - cnt_empty_polls) /
                                                  (cnt_num_polls - cnt_unsuccsessful_polls)) : 0),
                            cnt_num_remote_reads_polled,
                            (((cnt_num_polls > 0) && (cnt_num_polls - cnt_unsuccsessful_polls - cnt_empty_polls) > 0) ?
                             (cnt_num_remote_reads_polled / (cnt_num_polls - cnt_unsuccsessful_polls - cnt_empty_polls))
                             : 0), cnt_remote_accesses,
                            (cnt_remote_accesses > 0 ? (100.0 * cnt_single_try / cnt_remote_accesses) : 0), cnt_single_try,
                            (cnt_remote_accesses > 0 ? (100.0 * (cnt_remote_accesses - cnt_single_try) / cnt_remote_accesses) : 0), (cnt_remote_accesses - cnt_single_try),
                            cnt_tries_to_get_memory,
                            (cnt_remote_accesses > 0 ? (cnt_tries_to_get_memory / cnt_remote_accesses) : 0),
                            (((cnt_remote_accesses > 0) && (cnt_remote_accesses - cnt_single_try > 0)) ?
                                (cnt_tries_to_get_memory / (cnt_remote_accesses - cnt_single_try)) : 0),
                            cnt_got_memory_from_cool_once + cnt_got_memory_from_cool_multiple + cnt_got_memory_from_pool_once + cnt_got_memory_from_pool_multiple,
                            (cnt_got_memory_from_cool_once + cnt_got_memory_from_cool_multiple + cnt_got_memory_from_pool_once + cnt_got_memory_from_pool_multiple > 0) ?
                            ((cnt_got_memory_from_pool_once + cnt_got_memory_from_pool_multiple) * 100.0) / (cnt_got_memory_from_cool_once + cnt_got_memory_from_cool_multiple + cnt_got_memory_from_pool_once + cnt_got_memory_from_pool_multiple) : 0,
                            cnt_got_memory_from_pool_once + cnt_got_memory_from_pool_multiple,
                            (cnt_got_memory_from_cool_once + cnt_got_memory_from_cool_multiple + cnt_got_memory_from_pool_once + cnt_got_memory_from_pool_multiple > 0) ?
                            ((cnt_got_memory_from_cool_once + cnt_got_memory_from_cool_multiple) * 100.0) / (cnt_got_memory_from_cool_once + cnt_got_memory_from_cool_multiple + cnt_got_memory_from_pool_once + cnt_got_memory_from_pool_multiple) : 0,
                            cnt_got_memory_from_cool_once + cnt_got_memory_from_cool_multiple
                        );
        total_num_polls += cnt_num_polls;
        total_unsuccsessful_polls += cnt_unsuccsessful_polls;
        total_empty_polls += cnt_empty_polls;
        total_num_remote_reads_polled += cnt_num_remote_reads_polled;
        total_remote_accesses += cnt_remote_accesses;
        total_tries_to_get_memory += cnt_tries_to_get_memory;
        total_single_try += cnt_single_try;
        total_got_memory_from_cool_once += cnt_got_memory_from_cool_once;
        total_got_memory_from_pool_once += cnt_got_memory_from_pool_once;
        total_got_memory_from_cool_multiple += cnt_got_memory_from_cool_multiple;
        total_got_memory_from_pool_multiple += cnt_got_memory_from_pool_multiple;
#else
        UNUSED_VARIABLE(stats_str);
        UNUSED_VARIABLE(total_num_polls);
        UNUSED_VARIABLE(total_unsuccsessful_polls);
        UNUSED_VARIABLE(total_empty_polls);
        UNUSED_VARIABLE(total_num_remote_reads_polled);
        UNUSED_VARIABLE(total_remote_accesses);
        UNUSED_VARIABLE(total_tries_to_get_memory);
        UNUSED_VARIABLE(total_single_try);
        UNUSED_VARIABLE(total_got_memory_from_cool_once);
        UNUSED_VARIABLE(total_got_memory_from_pool_once);
        UNUSED_VARIABLE(total_got_memory_from_cool_multiple);
        UNUSED_VARIABLE(total_got_memory_from_pool_multiple);
#endif
    }

    /*
     * even if there are only 5 vectors in a layer the probablity of choosing taht cluster is 20%.
     * Therefore, if we retry 21 times it should be highly unlikely that this happens. In case we do not succeed.
     * Now if 80% of the clusters in the layer are unusable(delete/Not stable), (which is not a good case!) the
     * probablity of us failing to get a good cluster after 21 retries is less than 1%.
     */
    inline static constexpr uint64_t MAX_RETRY = 21;
#ifdef HANG_DETECTION
    static inline std::vector<Thread*> all_threads;
    static inline std::mutex all_threads_lock;
#endif
    MVTYPE* centroid_compute_buffer;

protected:
    static inline std::atomic<DIVFThreadID> nextId = 0;
    static constexpr uint64_t MAX_THREAD_RETRY_COUNT = 10000;

    // Thread implementation details
    const DIVFThreadID _parent_id;
    const DIVFThreadID _id;
    std::atomic<bool> _done;
    std::atomic<bool> _safty_net;
    std::atomic<std::thread*> _thrd;
    std::mt19937 _gen;
    std::mt19937_64 _gen64;
    std::uniform_int_distribution<uint32_t> _uniform_dist;
    uint64_t _next_task_id;

#ifdef HANG_DETECTION
    std::atomic<uint64_t> _iteration_sn = 0;
    uint64_t last_iteration_sn = 0;
#endif
    // for randomized algs
    // size_t rand_num = 0;
#ifdef LOCK_DEBUG
    DIVFThreadID waitingFor = INVALID_DIVF_THREAD_ID;
    std::unordered_set<void*> heldShared;
    std::unordered_set<void*> heldExclusive;
#endif

#ifdef ENABLE_STAT_COLLECTION
    size_t cnt_num_polls = 0;
    size_t cnt_unsuccsessful_polls = 0; /* lock failed */
    size_t cnt_empty_polls = 0;
    size_t cnt_num_remote_reads_polled = 0;

    size_t cnt_remote_accesses = 0;
    size_t cnt_tries_to_get_memory = 0;
    size_t cnt_single_try = 0;
    size_t cnt_got_memory_from_cool_once = 0;
    size_t cnt_got_memory_from_pool_once = 0;
    size_t cnt_got_memory_from_cool_multiple = 0;
    size_t cnt_got_memory_from_pool_multiple = 0;
#endif
};

};

#endif