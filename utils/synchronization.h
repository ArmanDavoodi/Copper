#ifndef SYNCHRONIZATION_H_
#define SYNCHRONIZATION_H_

#include "configs/support.h"

#include <shared_mutex>
#include <atomic>
#include <condition_variable>

#include "debug.h"

#include "utils/string.h"
#include "utils/thread.h"
// #include "utils/concurrent_datastructures.h"

namespace divftree {

// TODO: need to recheck atomic load and store for 128 bit data since we may not be able to
// pass the same pointer as two arguments to compare_exchange128.

union alignas(16) atomic_data128 {
    struct Detail {
#if !defined(USE_LANG_ATOMIC) && (defined(__DIVFTREE_ATOMIC128__) || defined(__DIVFTREE_CMPXCHG128__))
        // std::atomic<uint64_t> lo;
        // std::atomic<uint64_t> hi;
        uint64_t lo;
        uint64_t hi;
#else
        std::atomic<__int128_t> raw_atomic;
#endif
    };
    Detail detail;
    __int128_t raw;

    constexpr atomic_data128(__int128_t data) : raw{data} {}

    constexpr atomic_data128() : raw{0} {}
};


inline bool compare_exchange128(atomic_data128* dest, atomic_data128* expected, const atomic_data128* desired,
                                bool relaxed = false) {
    FatalAssert(TYPE_ALIGNED(dest, 16), LOG_TAG_BASIC, "Destination pointer is not 16-byte aligned");
    FatalAssert(TYPE_ALIGNED(expected, 16), LOG_TAG_BASIC, "Expected pointer is not 16-byte aligned");
    FatalAssert(TYPE_ALIGNED(desired, 16), LOG_TAG_BASIC, "Desired pointer is not 16-byte aligned");

#if !defined(USE_LANG_ATOMIC) && defined(__DIVFTREE_CMPXCHG128__)
    bool result;
    if (relaxed) {
        asm volatile
        (
            "lock cmpxchg16b %1\n\t"
            "setz %0"       // on gcc6 and later, use a flag output constraint instead
            : "=q" ( result )
            , "+m" ( *dest )
            , "+d" ( expected->detail.hi )
            , "+a" ( expected->detail.lo )
            : "c" ( desired->detail.hi )
            , "b" ( desired->detail.lo )
            : "cc"
        );
    }
    else {
        asm volatile
        (
            "lock cmpxchg16b %1\n\t"
            "setz %0"       // on gcc6 and later, use a flag output constraint instead
            : "=q" ( result )
            , "+m" ( *dest )
            , "+d" ( expected->detail.hi )
            , "+a" ( expected->detail.lo )
            : "c" ( desired->detail.hi )
            , "b" ( desired->detail.lo )
            : "cc", "memory" // compile-time memory barrier.  Omit if you want memory_order_relaxed compile-time ordering.
        );
    }
    return result; /* May need to use atomic fences later? */
#else
    return dest->detail.raw_atomic.compare_exchange_strong(expected->raw, desired->raw, relaxed ?
                                                           std::memory_order_relaxed : std::memory_order_seq_cst);
#endif
}

// Atomic store: No memory fence/ordering!
inline void atomic_store128(atomic_data128* dest, const atomic_data128* src, bool relaxed = false) {
    FatalAssert(TYPE_ALIGNED(dest, 16), LOG_TAG_BASIC, "Destination pointer is not 16-byte aligned");
    FatalAssert(TYPE_ALIGNED(src, 16), LOG_TAG_BASIC, "Source pointer is not 16-byte aligned");

#if !defined(USE_LANG_ATOMIC) && defined(__DIVFTREE_ATOMIC128__)
    if (relaxed) {
        asm volatile (
            "movdqu %1, %0"
            : "=m" (*dest)
            : "x" (*src)
            : "xmm0"
        );
    }
    else {
        asm volatile (
            "movdqu %1, %0"
            : "=m" (*dest)
            : "x" (*src)
            : "xmm0", "memory"
        );
    }

    // std::atomic_thread_fence(std::memory_order_release);
// #elif defined(__DIVFTREE_CMPXCHG128__)
//     atomic_data128 expected;
//     std::memcpy(&expected, src, 16);
//     while (!compare_exchange128(dest, &expected, src, relaxed)) {
//         DIVFTREE_YIELD();
//     }
#else
    dest->detail.raw_atomic.store(src->raw, relaxed ? std::memory_order_relaxed : std::memory_order_seq_cst);
#endif
}

// Atomic load: No memory fence/ordering!
inline void atomic_load128(const atomic_data128* src, atomic_data128* dest, bool relaxed = false) {
    FatalAssert(TYPE_ALIGNED(src, 16), LOG_TAG_BASIC, "Source pointer is not 16-byte aligned");
    FatalAssert(TYPE_ALIGNED(dest, 16), LOG_TAG_BASIC, "Destination pointer is not 16-byte aligned");

#if !defined(USE_LANG_ATOMIC) && defined(__DIVFTREE_ATOMIC128__)
    // std::atomic_thread_fence(std::memory_order_acquire);

    if (relaxed) {
        asm volatile (
            "movdqu %1, %0"
            : "=x" (*dest)
            : "m" (*src)
            : "xmm0"
        );
    }
    else {
        asm volatile (
            "movdqu %1, %0"
            : "=x" (*dest)
            : "m" (*src)
            : "xmm0", "memory"
        );
    }
// #elif defined(__DIVFTREE_CMPXCHG128__)
//     // std::memcpy(dest, src, 16);
//     compare_exchange128(const_cast<atomic_data128*>(src), dest, src, relaxed);
#else
    dest->raw = src->detail.raw_atomic.load(relaxed ? std::memory_order_relaxed : std::memory_order_seq_cst);
#endif
}

/* Todo: a more efficent implementation */
class SXLock {
public:
    SXLock() : _mode(SX_SHARED), _num_shared_holders(0) {}
    ~SXLock() = default;

    void Lock(LockMode mode) {
        threadSelf->SanityCheckLockNotHeldByMe(this);
        if (mode == SX_SHARED) {
            _m.lock_shared();
            FatalAssert(_mode.load(std::memory_order_acquire) != SX_EXCLUSIVE, LOG_TAG_BASIC,
                        "Cannot lock in SX_SHARED mode when already in SX_EXCLUSIVE mode!");
            _num_shared_holders.fetch_add(1);
        } else {
            FatalAssert(mode == SX_EXCLUSIVE, LOG_TAG_BASIC, "Non Shared Lock should be exclusive!");
            _m.lock();
            FatalAssert(_mode.load(std::memory_order_acquire) == SX_SHARED, LOG_TAG_BASIC,
                        "Cannot lock in SX_EXCLUSIVE mode when it is not free!");
            FatalAssert(_num_shared_holders.load() == 0,
                        LOG_TAG_BASIC, "Cannot lock in SX_EXCLUSIVE mode when there are shared holders!");
            _mode.store(SX_EXCLUSIVE, std::memory_order_release);
        }
        threadSelf->AcquireLockSanityLog(this, mode);
    }

    void Unlock() {
        if (_mode.load(std::memory_order_acquire) == SX_SHARED) {
            threadSelf->SanityCheckLockHeldInModeByMe(this, SX_SHARED);
            FatalAssert(_num_shared_holders.load() > 0,
                        LOG_TAG_BASIC, "Cannot unlock in SX_SHARED mode when no holders!");
            threadSelf->ReleaseLockSanityLog(this, SX_SHARED);
            _num_shared_holders.fetch_sub(1);
            _m.unlock_shared();
        } else {
            threadSelf->SanityCheckLockHeldInModeByMe(this, SX_EXCLUSIVE);
            FatalAssert(_mode.load(std::memory_order_acquire) == SX_EXCLUSIVE, LOG_TAG_BASIC,
                        "Cannot unlock in SX_EXCLUSIVE mode when it is not locked!");
            FatalAssert(_num_shared_holders.load() == 0,
                        LOG_TAG_BASIC, "Cannot unlock in SX_EXCLUSIVE mode when there are shared holders!");
            threadSelf->ReleaseLockSanityLog(this, SX_EXCLUSIVE);
            _mode.store(SX_SHARED, std::memory_order_release);
            _m.unlock();
        }
    }

    bool TryLock(LockMode mode) {
        bool locked = false;
        threadSelf->SanityCheckLockNotHeldByMe(this);
        if (mode == SX_SHARED) {
            locked = _m.try_lock_shared();
            if (locked) {
                FatalAssert(_mode.load(std::memory_order_acquire) != SX_EXCLUSIVE, LOG_TAG_BASIC,
                            "Cannot lock in SX_SHARED mode when already in SX_EXCLUSIVE mode!");
                _num_shared_holders.fetch_add(1);
                threadSelf->AcquireLockSanityLog(this, mode);
            }
        } else {
            FatalAssert(mode == SX_EXCLUSIVE, LOG_TAG_BASIC, "Non Shared Lock should be exclusive!");
            locked = _m.try_lock();
            if (locked) {
                FatalAssert(_mode.load(std::memory_order_acquire) == SX_SHARED, LOG_TAG_BASIC,
                            "Cannot lock in SX_EXCLUSIVE mode when it is not free!");
                FatalAssert(_num_shared_holders.load() == 0,
                            LOG_TAG_BASIC, "Cannot lock in SX_EXCLUSIVE mode when there are shared holders!");
                _mode.store(SX_EXCLUSIVE, std::memory_order_release);
                threadSelf->AcquireLockSanityLog(this, mode);
            }
        }
        return locked;
    }

    /* better naming */
    // bool LockWithBlockCheck(LockMode mode) {
    //     bool blocked = !(TryLock(mode));
    //     if (blocked) {
    //         Lock(mode);
    //     }
    //     return blocked;
    // }

    // void SleepTillSignalled(LockMode mode) {
    //     _signal.store(false);
    //     _signal.wait(false);
    // }

    inline String ToString() const {
        return String("<LockMode=%s, SharedHolders=%lu>",
                      LockModeToString(_mode.load(std::memory_order_acquire)).ToCStr(),
                      _num_shared_holders.load());
    }
protected:
    std::atomic<LockMode> _mode;
    std::atomic<uint64_t> _num_shared_holders;
    std::shared_mutex _m;
};

class SXSpinLock {
public:
    SXSpinLock() : _mode(SX_SHARED), _lock(0) {}
    ~SXSpinLock() = default;

    void Lock(LockMode mode) {
        threadSelf->SanityCheckLockNotHeldByMe(this);
        do {
            while (_lock._data._exclusive_flag.load(std::memory_order_acquire) == LOCKED) {
                DIVFTREE_YIELD();
            }

            if (mode == SX_SHARED) {
                non_atomic_lock res(0);
                res._counter = _lock._counter.fetch_add(1, std::memory_order_seq_cst);
                FatalAssert(_lock._data._shared_counter.load(std::memory_order_acquire) > 0, LOG_TAG_BASIC,
                            "counter overflow!");
                if (res._data._exclusive_flag == LOCKED) {
                    _lock._data._shared_counter.fetch_sub(1, std::memory_order_seq_cst);
                    continue;
                }
                FatalAssert(_mode.load(std::memory_order_acquire) == SX_SHARED, LOG_TAG_BASIC,
                            "Lock mode is exclusive!");
                break;
            }
            else {
                non_atomic_lock expected(0);
                if (!_lock._data._exclusive_flag.compare_exchange_strong(expected._data._exclusive_flag,
                                                                         LOCKED, std::memory_order_seq_cst)) {
                    continue;
                }

                while (_lock._data._shared_counter.load(std::memory_order_acquire) > 0) {
                    DIVFTREE_YIELD();
                }
                FatalAssert(_mode.load(std::memory_order_acquire) == SX_SHARED, LOG_TAG_BASIC,
                            "Lock mode is exclusive!");
                _mode.store(SX_EXCLUSIVE, std::memory_order_release);
                break;
            }
        } while(true);
        threadSelf->AcquireLockSanityLog(this, mode);
    }

    void Unlock() {
        if (_mode.load(std::memory_order_acquire) == SX_SHARED) {
            threadSelf->SanityCheckLockHeldInModeByMe(this, SX_SHARED);
            FatalAssert(_lock._data._shared_counter.load(std::memory_order_acquire) > 0,
                        LOG_TAG_BASIC, "Cannot unlock in SX_SHARED mode when no holders!");
            threadSelf->ReleaseLockSanityLog(this, SX_SHARED);
            _lock._data._shared_counter.fetch_sub(1, std::memory_order_release);
        } else {
            threadSelf->SanityCheckLockHeldInModeByMe(this, SX_EXCLUSIVE);
            FatalAssert(_mode.load(std::memory_order_acquire) == SX_EXCLUSIVE, LOG_TAG_BASIC,
                        "Cannot unlock in SX_EXCLUSIVE mode when it is not locked!");
            FatalAssert(_lock._data._exclusive_flag.load(std::memory_order_acquire) == LOCKED,
                        LOG_TAG_BASIC, "Cannot unlock in SX_EXCLUSIVE mode when it is not locked!");
            threadSelf->ReleaseLockSanityLog(this, SX_EXCLUSIVE);
            _mode.store(SX_SHARED, std::memory_order_release);
            _lock._data._exclusive_flag.store(0, std::memory_order_release);
        }
    }

    /* May return false negative for exclusive lock as we might cas sooner than a FAA but see the result of FAA */
    bool TryLock(LockMode mode) {
        threadSelf->SanityCheckLockNotHeldByMe(this);
        if (mode == SX_SHARED) {
            non_atomic_lock res(0);
            res._counter = _lock._counter.fetch_add(1, std::memory_order_seq_cst);
            FatalAssert(_lock._data._shared_counter.load(std::memory_order_acquire) > 0, LOG_TAG_BASIC,
                        "counter overflow!");
            if (res._data._exclusive_flag == LOCKED) {
                _lock._data._shared_counter.fetch_sub(1, std::memory_order_release);
                return false;
            }
            FatalAssert(_mode.load(std::memory_order_acquire) == SX_SHARED, LOG_TAG_BASIC, "Lock mode is exclusive!");
            threadSelf->AcquireLockSanityLog(this, mode);
            return true;
        }
        else {
            non_atomic_lock expected(0);
            if (!_lock._data._exclusive_flag.compare_exchange_strong(expected._data._exclusive_flag,
                                                                        LOCKED, std::memory_order_seq_cst)) {
                return false;
            }

            if (_lock._data._shared_counter.load(std::memory_order_acquire) > 0) {
                _lock._data._exclusive_flag.store(0, std::memory_order_release);
                return false;
            }

            FatalAssert(_mode.load(std::memory_order_acquire) == SX_SHARED, LOG_TAG_BASIC, "Lock mode is exclusive!");
            _mode.store(SX_EXCLUSIVE, std::memory_order_release);
            threadSelf->AcquireLockSanityLog(this, mode);
            return true;
        }
    }

    inline LockMode CurrentMode() const {
        return _mode.load(std::memory_order_acquire);
    }

    /* better naming */
    bool LockWithBlockCheck(LockMode mode) {
        bool blocked = !(TryLock(mode));
        if (blocked) {
            Lock(mode);
        }
        return blocked;
    }

    inline String ToString() const {
        return String("<LockMode=%s, SharedHolders=%u>",
                      LockModeToString(_mode.load(std::memory_order_acquire)).ToCStr(),
                                       _lock._data._shared_counter.load());
    }
protected:
    constexpr static uint32_t LOCKED = INT32_MIN;
    std::atomic<LockMode> _mode;

    union non_atomic_lock {
        struct {
            uint32_t _shared_counter;
            uint32_t _exclusive_flag;
        } _data;
        uint64_t _counter;

        non_atomic_lock(uint64_t initial): _counter(initial) {}
    };
    union atomic_lock {
        struct Detail {
            std::atomic<uint32_t> _shared_counter;
            std::atomic<uint32_t> _exclusive_flag;
        };
        Detail _data;
        std::atomic<uint64_t> _counter;

        atomic_lock(uint64_t initial): _counter(initial) {}
    };

    atomic_lock _lock;
};

template<LockMode mode>
class LockWrapper {
public:
    LockWrapper(SXLock& lock) : _lock(lock) {}
    ~LockWrapper() = default;

    void lock() {
        _lock.Lock(mode);
    }

    void unlock() {
        _lock.Unlock();
    }
protected:
    SXLock& _lock;
};

class CondVar {
public:
    CondVar() = default;
    ~CondVar() = default;

    void NotifyOne() {
        _cond_var.notify_one();
    }

    void NotifyAll() {
        _cond_var.notify_all();
    }

    template<LockMode mode>
    void Wait(LockWrapper<mode> lock) {
        _cond_var.wait(lock);
    }

protected:
    std::condition_variable_any _cond_var;
};

};

#endif