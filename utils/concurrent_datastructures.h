#ifndef CONCURRENT_DATASTRUCTURES_H_
#define CONCURRENT_DATASTRUCTURES_H_

#include <mutex>
#include <set>
#include <queue>

#include "utils/string.h"

#include "third_party/moodycamel/concurrentqueue/concurrentqueue.h"
#include "third_party/moodycamel/concurrentqueue/blockingconcurrentqueue.h"

#include "utils/synchronization.h"

namespace divftree {

/* Todo better implementations -> lockfree */

template<typename K, typename V, typename CMP, typename HashFunc, uint64_t DefaultBucketSize = 4,
         uint64_t DefaultNumBuckets = 32, float MaxLoadFactor = 4.0f, float MinLoadFactor = 1.0f,
         uint64_t ResizeStep = 2, long double SequentialRandomSearchFactor = ((long double)0.4)>
class ConcurrentHashTable {
protected:

    struct Bucket {
        SXSpinLock lock;
        std::vector<std::pair<K, V>> entries;
        Bucket* next = nullptr;
        bool in_list = false;
    };

    size_t default_bucket_cap;
    std::vector<Bucket> buckets;
    std::atomic<size_t> num_elements;
    std::atomic<size_t> num_non_empty_buckets;
    std::atomic<Bucket*> head;
    std::atomic<bool> rehashing;
    SXLock table_lock;
    CMP compare;
    HashFunc hash_func;

    bool RehashStart(LockMode table_lock_mode) {
        if (!rehashing.compare_exchange_strong(false, true)) {
            return false;
        }

        if (table_lock_mode == SX_UNLOCKED) {
            table_lock.Lock(SX_EXCLUSIVE);
        } else if (table_lock_mode == SX_SHARED) {
            table_lock.Unlock();
            table_lock.Lock(SX_EXCLUSIVE);
        }
        return true;
    }

    void RehashEnd(LockMode end_lock_mode) {
        threadSelf->SanityCheckLockHeldByMe(&table_lock, SX_EXCLUSIVE);
        FatalAssert(rehashing.load(std::memory_order_acquire), LOG_TAG_BASIC,
                    "Rehash should be in progress!");
        if (end_lock_mode == SX_SHARED) {
            table_lock.Unlock();
            table_lock.Lock(SX_SHARED);
            rehashing.store(false, std::memory_order_release);
        } else if (end_lock_mode == SX_UNLOCKED) {
            rehashing.store(false, std::memory_order_release);
            table_lock.Unlock();
        } else {
            rehashing.store(false, std::memory_order_release);
        }
    }

    void Rehash(size_t num_buckets) {
        threadSelf->SanityCheckLockHeldByMe(&table_lock, SX_EXCLUSIVE);
        FatalAssert(rehashing.load(std::memory_order_acquire), LOG_TAG_BASIC,
                    "Rehash should be in progress!");
        FatalAssert(num_buckets > 0, LOG_TAG_BASIC, "num_buckets should be greater than zero!");
        head.store(nullptr, std::memory_order_relaxed);
        std::vector<Bucket> new_buckets;
        new_buckets.resize(num_buckets);
        default_bucket_cap = std::max(DefaultBucketSize,
                                      (ResizeStep * num_elements.load(std::memory_order_relaxed)) / num_buckets);
        for (auto& bucket : buckets) {
            threadSelf->SanityCheckLockNotHeldByMe(&bucket.lock);
            for (auto& entry : bucket.entries) {
                size_t new_bucket_index = hash_func(entry.first) % new_buckets.size();
                if (!new_buckets[new_bucket_index].in_list) {
                    FatalAssert(new_buckets[new_bucket_index].entires.capacity() == 0, LOG_TAG_BASIC,
                                "new bucket should be empty!");
                    FatalAssert(new_buckets[new_bucket_index].next == nullptr, LOG_TAG_BASIC,
                                "new bucket next should be null!");
                    new_buckets[new_bucket_index].in_list = true;
                    new_buckets[new_bucket_index].entries.reserve(default_bucket_cap);
                    new_buckets[new_bucket_index].next = head.load(std::memory_order_relaxed);
                    head.store(&new_buckets[new_bucket_index], std::memory_order_relaxed);
                } else {
                    FatalAssert(new_buckets[new_bucket_index].entries.size() > 0, LOG_TAG_BASIC,
                                "new bucket should not be empty!");
                }
                new_buckets[new_bucket_index].entries.emplace_back(std::move(entry));
            }
            bucket.entries.clear();
        }
        buckets = std::move(new_buckets);
    }

    void LockTableExclusive() {
        while (true) {
            while (rehashing.load(std::memory_order_acquire)) {
                rehashing.wait(true);
            }
            table_lock.Lock(SX_EXCLUSIVE);
            if (!rehashing.load(std::memory_order_acquire)) {
                break;
            }
            table_lock.Unlock();
        }
    }

public:
    void LockTable() {
        while (true) {
            while (rehashing.load(std::memory_order_acquire)) {
                rehashing.wait(true);
            }
            table_lock.Lock(SX_SHARED);
            if (!rehashing.load(std::memory_order_acquire)) {
                break;
            }
            table_lock.Unlock();
        }
    }

    bool TryLockTable() {
        if (rehashing.load(std::memory_order_acquire)) {
            return false;
        }
        bool locked = table_lock.TryLock(SX_SHARED);
        if (locked) {
            if (rehashing.load(std::memory_order_acquire)) {
                table_lock.Unlock();
                return false;
            }
        }
        return locked;
    }

    void UnlockTable() {
        table_lock.Unlock();
    }

    void LockBucket(uint64_t key_hash, LockMode type, bool has_table_lock = false) {
        if (!has_table_lock) {
            LockTable();
        }
        threadSelf->SanityCheckLockHeldInModeByMe(&table_lock, SX_SHARED);

        size_t bucket_index = key_hash % buckets.size();
        buckets[bucket_index].lock.Lock(type);
    }

    bool TryLockBucket(uint64_t key_hash, LockMode type, bool has_table_lock = false) {
        if (!has_table_lock) {
            if (!TryLockTable()) {
                return false;
            }
        }
        threadSelf->SanityCheckLockHeldInModeByMe(&table_lock, SX_SHARED);

        size_t bucket_index = key_hash % buckets.size();
        if (!buckets[bucket_index].lock.TryLock(type)) {
            if (!has_table_lock) {
                UnlockTable();
            }
            return false;
        }
        return true;
    }

    void UnlockBucket(uint64_t key_hash, bool unlock_table = true) {
        size_t bucket_index = key_hash % buckets.size();
        buckets[bucket_index].lock.Unlock();
        if (unlock_table) {
            UnlockTable();
        }
    }

    void LockBucket(const K& key, LockMode type, bool has_table_lock = false) {
        LockBucket(hash_func(key), type, has_table_lock);
    }

    bool TryLockBucket(const K& key, LockMode type, bool has_table_lock = false) {
        return TryLockBucket(hash_func(key), type, has_table_lock);
    }

    void UnlockBucket(const K& key, bool unlock_table = true) {
        UnlockBucket(hash_func(key), unlock_table);
    }

    struct Iterator {
        ConcurrentHashTable* table;
        Bucket* bucket;
        size_t data_index;

        void Discard() {
            if (table != nullptr) {
                if (bucket != nullptr) {
                    if (table->rehashing.load(std::memory_order_acquire)) {
                        size_t num_buckets = table->buckets.size();
                        if (num_buckets > DefaultNumBuckets &&
                            (table->num_elements.load(std::memory_order_relaxed) * MinLoadFactor <= num_buckets)) {
                            while (table->num_elements.load(std::memory_order_relaxed) * MinLoadFactor <= num_buckets) {
                                num_buckets /= ResizeStep;
                                if (num_buckets < DefaultNumBuckets) {
                                    num_buckets = DefaultNumBuckets;
                                    break;
                                }
                                FatalAssert(num_buckets <= table->buckets.size(), LOG_TAG_BASIC,
                                            "num_buckets should be less than current buckets size!");
                            }

                            Rehash(num_buckets);
                        }
                        RehashEnd(SX_UNLOCKED);
                    } else {
                        table->table_lock.Unlock();
                    }
                    bucket = nullptr;
                }
                threadSelf->SanityCheckLockNotHeldByMe(&table->table_lock);
                table = nullptr;
            }
        }

        Iterator(ConcurrentHashTable* t, Bucket* b, size_t d_idx) :
            table{t}, bucket{b}, data_index{d_idx} {
            if (table == nullptr) {
                return;
            }

            if (bucket == nullptr) {
                return;
            }

            threadSelf->SanityCheckLockHeldInModeByMe(&table->table_lock, SX_EXCLUSIVE);
            FatalAssert(!table->rehashing.load(std::memory_order_acquire), LOG_TAG_BASIC,
                        "Rehash should not be in progress!");
            FatalAssert(data_index < bucket->entries.size(), LOG_TAG_BASIC,
                        "data_index out of bounds!");
        }

        ~Iterator() {
            Discard();
        }

        Iterator(const Iterator& other) = delete;
        Iterator& operator=(const Iterator& other) = delete;

        Iterator(Iterator&& other) noexcept :
            table{other.table}, bucket{other.bucket}, data_index{other.data_index} {
            other.table = nullptr;
        }

        Iterator& operator=(Iterator&& other) noexcept {
            if (this != &other) {
                Discard();
                table = other.table;
                bucket = other.bucket;
                data_index = other.data_index;
                other.table = nullptr;
            }
            return *this;
        }

        std::pair<K, V>& operator*() {
            FatalAssert(table != nullptr, LOG_TAG_BASIC, "table is null!");
            FatalAssert(bucket != nullptr, LOG_TAG_BASIC, "bucket is null!");
            threadSelf->SanityCheckLockHeldInModeByMe(&table->table_lock, SX_EXCLUSIVE);
            FatalAssert(data_index < bucket->entries.size(), LOG_TAG_BASIC,
                        "data_index out of bounds!");
            return bucket->entries[data_index];
        }

        Iterator& operator++() {
            FatalAssert(table != nullptr, LOG_TAG_BASIC, "table is null!");
            if (bucket == nullptr) {
                FatalAssert(false, LOG_TAG_BASIC, "Iterator out of bounds!");
                return *this;
            }
            threadSelf->SanityCheckLockHeldInModeByMe(&table->table_lock, SX_EXCLUSIVE);
            FatalAssert(data_index < bucket->entries.size(), LOG_TAG_BASIC,
                        "data_index out of bounds!");

            if (data_index + 1 < bucket->entries.size()) {
                data_index++;
            } else {
                Bucket* prev_bucket = bucket;
                while (bucket != nullptr) {
                    FatalAssert(bucket->in_list, LOG_TAG_BASIC,
                                "bucket in list should be true!");
                    bucket = bucket->next;
                    if (bucket != nullptr) {
                        if (bucket->entries.size() > 0) {
                            data_index = 0;
                            break;
                        } else {
                            FatalAssert(bucket != head.load(std::memory_order_relaxed),
                                        LOG_TAG_BASIC,
                                        "bucket in list should not be head!");
                            bucket->in_list = false;
                            prev_bucket->next = bucket->next;
                            num_non_empty_buckets.fetch_sub(1, std::memory_order_relaxed);
                        }
                    }
                    prev_bucket = bucket;
                }

                if (bucket == nullptr) {
                    // end reached
                    if (table->rehashing.load(std::memory_order_acquire)) {
                        Rehash(table->buckets.size() / ResizeStep);
                        RehashEnd(SX_UNLOCKED);
                    } else {
                        table->table_lock.Unlock();
                    }
                    data_index = 0;
                }

            }

            return *this;
        }

        bool operator==(const Iterator& other) const {
            return (table == other.table) && (bucket == other.bucket) &&
                   (data_index == other.data_index);
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }
    };

    struct ElementHandle {
        ElementHandle() : val(nullptr), blck_idx(0), blck_hash(0) {}
        ~ElementHandle() {
            FatalAssert(val == nullptr, LOG_TAG_BASIC,
                        "ElementHandle should have been released before destruction!");
        }

        ElementHandle(const ElementHandle& other) = delete;
        ElementHandle& operator=(const ElementHandle& other) = delete;

        ElementHandle(ElementHandle&& other) noexcept : val(other.val), blck_idx(other.blck_idx), blck_hash(other.blck_hash) {}
        ElementHandle& operator=(ElementHandle&& other) noexcept {
            if (this != &other) {
                val = other.val;
                blck_idx = other.blck_idx;
                blck_hash = other.blck_hash;
            }
            return *this;
        }

        uint64_t GetHash() const {
            return blck_hash;
        }

        bool IsValid() const {
            return (val != nullptr);
        }

        V& operator*() {
            FatalAssert(val != nullptr, LOG_TAG_BASIC,
                        "Dereferencing invalid ElementHandle!");
            return val->second;
        }

        V& Value() {
            FatalAssert(val != nullptr, LOG_TAG_BASIC,
                        "Accessing value of invalid ElementHandle!");
            return val->second;
        }

        const K& Key() const {
            FatalAssert(val != nullptr, LOG_TAG_BASIC,
                        "Accessing key of invalid ElementHandle!");
            return val->first;
        }

    protected:
        ElementHandle(std::pair<K, V>* v, uint64_t idx, uint64_t hash) : val(v), blck_idx(idx), blck_hash(hash) {}

        uint64_t blck_idx;
        uint64_t blck_hash;
        std::pair<K, V>* val;

        friend class ConcurrentHashTable<K, V, CMP, HashFunc, DefaultBucketSize,
                                      DefaultNumBuckets, MaxLoadFactor, MinLoadFactor,
                                      ResizeStep, SequentialRandomSearchFactor>;
    };

    ConcurrentHashTable() :
        default_bucket_cap(DefaultBucketSize),
        buckets(DefaultNumBuckets),
        num_elements(0),
        num_non_empty_buckets(0),
        head(nullptr),
        rehashing(false) {
        for (auto& bucket : buckets) {
            bucket.entries.reserve(default_bucket_cap);
        }
    }

    ConcurrentHashTable(size_t num_elements_hint) :
        default_bucket_cap(DefaultBucketSize),
        buckets(),
        num_elements(0),
        num_non_empty_buckets(0),
        head(nullptr),
        rehashing(false) {
        size_t num_buckets = DefaultNumBuckets;
        if (num_elements_hint >= num_buckets * MaxLoadFactor) {
            while (num_elements_hint >= num_buckets * MaxLoadFactor) {
                num_buckets *= ResizeStep;
                FatalAssert(num_buckets > DefaultNumBuckets, LOG_TAG_BASIC,
                            "num_buckets should be greater than DefaultNumBuckets!");
            }
        } else if (num_elements_hint * MinLoadFactor <= num_buckets) {
            while (num_elements_hint * MinLoadFactor <= num_buckets) {
                num_buckets /= ResizeStep;
                if (num_buckets < DefaultNumBuckets) {
                    num_buckets = DefaultNumBuckets;
                    break;
                }
                FatalAssert(num_buckets > 0, LOG_TAG_BASIC,
                            "num_buckets should be greater than zero!");
            }

        }
        buckets.resize(num_buckets);
        for (auto& bucket : buckets) {
            bucket.entries.reserve(default_bucket_cap);
        }
    }

    bool Insert(const K& key, const V& value, uint64_t hash, bool has_bucket_lock, bool has_table_lock) {
        FatalAssert(!has_bucket_lock || has_table_lock, LOG_TAG_BASIC,
                    "Cannot have bucket lock without table lock!");
        FatalAssert(hash_func(key) == hash, LOG_TAG_BASIC, "Hash mismatch!");
        FatalAssert(buckets.size() > 0, LOG_TAG_BASIC, "buckets size is zero!");
        if (!has_bucket_lock) {
            LockBucket(hash, SX_EXCLUSIVE, has_table_lock);
        }
        threadSelf->SanityCheckLockHeldInModeByMe(&table_lock, SX_SHARED);
        uint64_t bucket_index = hash % buckets.size();
        threadSelf->SanityCheckLockHeldInModeByMe(&buckets[bucket_index].lock, SX_EXCLUSIVE);

        for (auto& entry_ptr : buckets[bucket_index].entries) {
            if (compare(entry_ptr.first, key) == 0) {
                if (!has_bucket_lock) {
                    UnlockBucket(hash, has_table_lock);
                }
                return false;
            }
        }

        if (!buckets[bucket_index].in_list) {
            FatalAssert(buckets[bucket_index].next == nullptr, LOG_TAG_BASIC,
                        "bucket next should be null!");
            buckets[bucket_index].in_list = true;
            buckets[bucket_index].entries.reserve(default_bucket_cap);
            bool added_to_list = true;
            do {
                if (!added_to_list) {
                    DIVFTREE_YIELD();
                }
                buckets[bucket_index].next = head.load(std::memory_order_acquire);
            } while (!(added_to_list = head.compare_exchange_weak(buckets[bucket_index].next,
                                                                  &buckets[bucket_index])));
            num_non_empty_buckets.fetch_add(1);
        }

        buckets[bucket_index].entries.emplace_back(key, value);
        if (!has_bucket_lock) {
            UnlockBucket(hash, false);
        }

        size_t new_size = num_elements.fetch_add(1) + 1;

        /* if bucket is locked by the user we cannot rehash while guranteeing there is no change to bucket */
        if (has_bucket_lock) {
            return true;
        }

        if (new_size >= buckets.size() * MaxLoadFactor) {
            if (!RehashStart(SX_SHARED)) {
                if (!has_table_lock) {
                    UnlockTable();
                }
                return true;
            }
            size_t num_buckets = buckets.size();
            while (new_size >= num_buckets * MaxLoadFactor) {
                num_buckets *= ResizeStep;
                FatalAssert(num_buckets > buckets.size(), LOG_TAG_BASIC,
                            "num_buckets should increase!");
            }
            Rehash(num_buckets);
            RehashEnd((has_table_lock) ? SX_SHARED : SX_UNLOCKED);
            return true;
        }

        if (!has_table_lock) {
            UnlockTable();
        }

        return true;
    }

    bool Insert(const K& key, const V& value) {
        uint64_t hash = hash_func(key);
        return Insert(key, value, hash, false, false);
    }

    template<typename... Args>
    bool Emplace(const K& key, uint64_t hash, bool has_bucket_lock, bool has_table_lock, Args&&... args) {
        FatalAssert(!has_bucket_lock || has_table_lock, LOG_TAG_BASIC,
                    "Cannot have bucket lock without table lock!");
        FatalAssert(hash_func(key) == hash, LOG_TAG_BASIC, "Hash mismatch!");
        FatalAssert(buckets.size() > 0, LOG_TAG_BASIC, "buckets size is zero!");
        if (!has_bucket_lock) {
            LockBucket(hash, SX_EXCLUSIVE, has_table_lock);
        }
        threadSelf->SanityCheckLockHeldInModeByMe(&table_lock, SX_SHARED);
        uint64_t bucket_index = hash % buckets.size();
        threadSelf->SanityCheckLockHeldInModeByMe(&buckets[bucket_index].lock, SX_EXCLUSIVE);

        for (auto& entry_ptr : buckets[bucket_index].entries) {
            if (compare(entry_ptr.first, key) == 0) {
                if (!has_bucket_lock) {
                    UnlockBucket(hash, has_table_lock);
                }
                return false;
            }
        }

        if (!buckets[bucket_index].in_list) {
            FatalAssert(buckets[bucket_index].next == nullptr, LOG_TAG_BASIC,
                        "bucket next should be null!");
            buckets[bucket_index].in_list = true;
            buckets[bucket_index].entries.reserve(default_bucket_cap);
            bool added_to_list = true;
            do {
                if (!added_to_list) {
                    DIVFTREE_YIELD();
                }
                buckets[bucket_index].next = head.load(std::memory_order_acquire);
            } while (!(added_to_list = head.compare_exchange_weak(buckets[bucket_index].next,
                                                                  &buckets[bucket_index])));
            num_non_empty_buckets.fetch_add(1);
        }

        buckets[bucket_index].entries.emplace_back(key, std::forward<Args>(args)...);
        if (!has_bucket_lock) {
            UnlockBucket(hash, false);
        }

        size_t new_size = num_elements.fetch_add(1) + 1;

        /* if bucket is locked by the user we cannot rehash while guranteeing there is no change to bucket */
        if (has_bucket_lock) {
            return true;
        }

        if (new_size >= buckets.size() * MaxLoadFactor) {
            if (!RehashStart(SX_SHARED)) {
                if (!has_table_lock) {
                    UnlockTable();
                }
                return true;
            }
            size_t num_buckets = buckets.size();
            while (new_size >= num_buckets * MaxLoadFactor) {
                num_buckets *= ResizeStep;
                FatalAssert(num_buckets > buckets.size(), LOG_TAG_BASIC,
                            "num_buckets should increase!");
            }
            Rehash(num_buckets);
            RehashEnd((has_table_lock) ? SX_SHARED : SX_UNLOCKED);
            return true;
        }

        if (!has_table_lock) {
            UnlockTable();
        }

        return true;
    }

    template<typename... Args>
    bool Emplace(const K& key, Args&&... args) {
        uint64_t hash = hash_func(key);
        return Emplace(key, hash, false, false, std::forward<Args>(args)...);
    }

    bool Erase(const K& key, uint64_t hash, bool has_bucket_lock, bool has_table_lock) {
        FatalAssert(!has_bucket_lock || has_table_lock, LOG_TAG_BASIC,
                    "Cannot have bucket lock without table lock!");
        FatalAssert(hash_func(key) == hash, LOG_TAG_BASIC, "Hash mismatch!");
        FatalAssert(buckets.size() > 0, LOG_TAG_BASIC, "buckets size is zero!");
        if (!has_bucket_lock) {
            LockBucket(hash, SX_EXCLUSIVE, has_table_lock);
        }
        threadSelf->SanityCheckLockHeldInModeByMe(&table_lock, SX_SHARED);
        uint64_t bucket_index = hash % buckets.size();
        threadSelf->SanityCheckLockHeldInModeByMe(&buckets[bucket_index].lock, SX_EXCLUSIVE);

        size_t erase_index = buckets[bucket_index].entries.size();
        for (size_t i = 0; i < buckets[bucket_index].entries.size(); ++i) {
            if (compare(buckets[bucket_index].entries[i].first, key) == 0) {
                erase_index = i;
                break;
            }
        }
        if (erase_index == buckets[bucket_index].entries.size()) {
            if (!has_bucket_lock) {
                UnlockBucket(hash, has_table_lock);
            }
            return false;
        }
        FatalAssert(buckets[bucket_index].in_list, LOG_TAG_BASIC,
                    "bucket should be in list when erasing!");
        if (erase_index + 1 < buckets[bucket_index].entries.size()) {
            buckets[bucket_index].entries[erase_index] =
                buckets[bucket_index].entries.back();
        }
        buckets[bucket_index].entries.pop_back();
        if (!has_bucket_lock) {
            UnlockBucket(hash, false);
        }

        size_t new_size = num_elements.fetch_sub(1) - 1;

        /* if bucket is locked by the user we cannot rehash while guranteeing there is no change to bucket */
        if (has_bucket_lock) {
            return true;
        }

        if (new_size * MinLoadFactor <= buckets.size() &&
            buckets.size() > DefaultNumBuckets) {
            if (!RehashStart(SX_SHARED)) {
                if (!has_table_lock) {
                    UnlockTable();
                }
                return true;
            }
            size_t num_buckets = buckets.size();
            while (new_size * MinLoadFactor <= num_buckets) {
                num_buckets /= ResizeStep;
                if (num_buckets < DefaultNumBuckets) {
                    num_buckets = DefaultNumBuckets;
                    break;
                }
                FatalAssert(num_buckets <= buckets.size(), LOG_TAG_BASIC,
                            "num_buckets should decrease!");
            }
            Rehash(num_buckets);
            RehashEnd((has_table_lock) ? SX_SHARED : SX_UNLOCKED);
            return true;
        }

        if (!has_table_lock) {
            UnlockTable();
        }

        return true;
    }

    bool Erase(const K& key) {
        uint64_t hash = hash_func(key);
        return Erase(key, hash, false);
    }

    /* should go back if possible */
    bool Erase(Iterator& it) {
        FatalAssert(it.table == this, LOG_TAG_BASIC, "Iterator table mismatch!");
        FatalAssert(it.bucket != nullptr, LOG_TAG_BASIC, "Iterator bucket is null!");
        threadSelf->SanityCheckLockHeldInModeByMe(&table_lock, SX_EXCLUSIVE);
        Bucket* bucket = it.bucket;
        size_t erase_index = it.data_index;
        FatalAssert(erase_index < bucket->entries.size(), LOG_TAG_BASIC,
                    "Iterator entry not found in bucket!");

        if (erase_index + 1 < bucket->entries.size()) {
            bucket->entries[erase_index] =
                bucket->entries.back();
        }
        bucket->entries.pop_back();

        size_t new_size = num_elements.fetch_sub(1) - 1;

        /* we cannot rehash in this case as it causes the Iterator to become invalid so we let the discard handle it*/
        if (new_size * MinLoadFactor <= buckets.size() &&
            buckets.size() > DefaultNumBuckets) {
            RehashStart(SX_EXCLUSIVE);
        }
        return true;
    }

    Iterator Find(const K& key) {
        uint64_t hash = hash_func(key);
        LockTableExclusive();
        uint64_t bucket_index = hash % buckets.size();
        uint64_t data_index = 0;
        bool found = false;
        for (size_t i = 0; i < buckets[bucket_index].entries.size(); ++i) {
            if (compare(buckets[bucket_index].entries[i].first, key) == 0) {
                data_index = i;
                found = true;
                break;
            }
        }

        if (!found) {
            table_lock.Unlock();
            return Iterator(this, nullptr, 0);
        }

        return Iterator(this, &buckets[bucket_index], data_index);
    }

    ElementHandle Get(const K& key, uint64_t hash, bool has_bucket_lock, bool has_table_lock) {
        FatalAssert(!has_bucket_lock || has_table_lock, LOG_TAG_BASIC,
                    "Cannot have bucket lock without table lock!");
        FatalAssert(hash_func(key) == hash, LOG_TAG_BASIC, "Hash mismatch!");
        FatalAssert(buckets.size() > 0, LOG_TAG_BASIC, "buckets size is zero!");
        if (!has_bucket_lock) {
            LockBucket(hash, SX_SHARED, has_table_lock);
        }
        threadSelf->SanityCheckLockHeldInModeByMe(&table_lock, SX_SHARED);
        uint64_t bucket_index = hash % buckets.size();
        threadSelf->SanityCheckLockHeldByMe(&buckets[bucket_index].lock);

        for (size_t i = 0; i < buckets[bucket_index].entries.size(); ++i) {
            if (compare(buckets[bucket_index].entries[i].first, key) == 0) {
                return ElementHandle(&buckets[bucket_index].entries[i], bucket_index, hash);
            }
        }

        if (!has_bucket_lock) {
            UnlockBucket(hash, has_table_lock);
        }
        return ElementHandle(nullptr, 0, hash);
    }

    ElementHandle Get(const K& key) {
        uint64_t hash = hash_func(key);
        return Get(key, hash, false, false);
    }

    void Release(ElementHandle& handle, bool release_bucket_lock = true, bool release_table_lock = true) {
        FatalAssert(release_bucket_lock || !release_table_lock, LOG_TAG_BASIC,
                    "Cannot release bucket lock without releasing table lock!");
        if (handle.val == nullptr) {
            return;
        }

        threadSelf->SanityCheckLockHeldInModeByMe(&table_lock, SX_SHARED);
        FatalAssert(handle.blck_idx < buckets.size(), LOG_TAG_BASIC,
                    "handle bucket index out of bounds!");
        threadSelf->SanityCheckLockHeldByMe(&buckets[handle.blck_idx].lock);
        handle.val = nullptr;
        handle.blck_idx = 0;
        handle.blck_hash = 0;

        if (release_bucket_lock) {
            buckets[handle.blck_idx].lock.Unlock();
        }
        if (release_table_lock) {
            table_lock.Unlock();
        }
    }

    void Release(ElementHandle& first_handle, ElementHandle& second_handle, bool release_bucket_lock = true,
                                                                            bool release_table_lock = true) {
        FatalAssert(release_bucket_lock || !release_table_lock, LOG_TAG_BASIC,
                    "Cannot release bucket lock without releasing table lock!");
        if (!release_bucket_lock) {
            Release(first_handle, false, false);
            Release(second_handle, false, false);
            return;
        }

        if (first_handle.val == second_handle.val) {
            Release(first_handle, false, false);
            Release(second_handle, true, false);
        } else {
            Release(first_handle, true, false);
            Release(second_handle, true, false);
        }

        if (release_table_lock) {
            table_lock.Unlock();
        }
    }

    bool Contains(const K& key, uint64_t hash, bool has_bucket_lock, bool has_table_lock) {
        FatalAssert(!has_bucket_lock || has_table_lock, LOG_TAG_BASIC,
                    "Cannot have bucket lock without table lock!");
        FatalAssert(hash_func(key) == hash, LOG_TAG_BASIC, "Hash mismatch!");
        FatalAssert(buckets.size() > 0, LOG_TAG_BASIC, "buckets size is zero!");
        if (!has_bucket_lock) {
            LockBucket(hash, SX_SHARED, has_table_lock);
        }
        threadSelf->SanityCheckLockHeldInModeByMe(&table_lock, SX_SHARED);
        uint64_t bucket_index = hash % buckets.size();
        threadSelf->SanityCheckLockHeldByMe(&buckets[bucket_index].lock);

        for (size_t i = 0; i < buckets[bucket_index].entries.size(); ++i) {
            if (compare(buckets[bucket_index].entries[i].first, key) == 0) {
                if (!has_bucket_lock) {
                    UnlockBucket(hash, has_table_lock);
                }
                return true;
            }
        }

        if (!has_bucket_lock) {
            UnlockBucket(hash, has_table_lock);
        }
        return false;
    }

    bool Contains(const K& key) {
        uint64_t hash = hash_func(key);
        return Contains(key, hash, false, false);
    }

    size_t Size() {
        return num_elements.load(std::memory_order_acquire);
    }

    bool Empty() {
        return Size() == 0;
    }

    void Clear() {
        LockTableExclusive();
        for (auto& bucket : buckets) {
            bucket.entries.clear();
            bucket.in_list = false;
            bucket.next = nullptr;
        }
        head.store(nullptr, std::memory_order_relaxed);
        num_elements.store(0, std::memory_order_relaxed);
        default_bucket_cap = DefaultBucketSize;
        buckets.resize(default_bucket_cap);
        UnlockTable();
    }

    size_t BucketCount(bool has_table_lock) {
        if (!has_table_lock) {
            LockTable();
        }
        size_t count = buckets.size();
        if (!has_table_lock) {
            UnlockTable();
        }
        return count;

    }

    Iterator begin() {
        LockTableExclusive();
        Bucket* bucket = head.load(std::memory_order_relaxed);
        while (bucket != nullptr) {
            FatalAssert(bucket->in_list, LOG_TAG_BASIC,
                        "bucket in list should be true!");
            if (bucket->entries.size() > 0) {
                return Iterator(this, bucket, 0);
            } else {
                if (bucket == head.load(std::memory_order_relaxed)) {
                    head.store(bucket->next, std::memory_order_relaxed);
                }
                bucket->in_list = false;
                Bucket* next = bucket->next;
                bucket->next = nullptr;
                bucket = next;
                num_non_empty_buckets.fetch_sub(1, std::memory_order_relaxed);
            }
        }

        table_lock.Unlock();
        return Iterator(this, nullptr, 0);
    }

    Iterator end() {
        return Iterator(this, nullptr, 0);
    }

    void GetRandomElement(ElementHandle& handle, bool has_table_lock = false, ElementHandle* exclude = nullptr) {
        FatalAssert(&handle != exclude, LOG_TAG_BASIC,
                    "handle and exclude should be different!");
        FatalAssert(handle.val == nullptr, LOG_TAG_BASIC,
                    "first handle should be empty!");
        FatalAssert((exclude == nullptr) || (has_table_lock), LOG_TAG_BASIC,
                    "Cannot have exclude without table lock!");
        FatalAssert(exclude == nullptr ||
                    (exclude->val != nullptr &&
                     threadSelf->LockHeldInModeByMe(&buckets[exclude->blck_idx].lock, SX_SHARED) != LockHeld::UNLOCKED),
                    LOG_TAG_BASIC,
                    "exclude handle should be valid if provided!");
        if (!has_table_lock) {
            LockTable();
        }
        threadSelf->SanityCheckLockHeldByMe(&table_lock);

        while (true) {
            if (num_elements.load(std::memory_order_acquire) < 1) {
                if (exclude != nullptr) {
                    Release(*exclude, true, false);
                } else if (!has_table_lock) {
                    UnlockTable();
                }
                return;
            }

            size_t num_buckets = num_non_empty_buckets.load(std::memory_order_acquire);
            if (num_buckets <= 4 || (buckets.size() * SequentialRandomSearchFactor < num_buckets)) {
                size_t num_current_elements = num_elements.load(std::memory_order_acquire);
                if (num_current_elements < 1) {
                    if (exclude != nullptr) {
                        Release(*exclude, true, false);
                    } else if (!has_table_lock) {
                        UnlockTable();
                    }
                    return;
                }
                uint64_t random_val = threadSelf->UniformRange64(0, num_current_elements - 1);
                Bucket* bucket = head.load(std::memory_order_acquire);
                if (bucket == nullptr) {
                    if (exclude != nullptr) {
                        Release(*exclude, true, false);
                    } else if (!has_table_lock) {
                        UnlockTable();
                    }
                    return;
                }

                /* todo: stat collection */
                while (bucket != nullptr) {
                    FatalAssert(bucket->in_list, LOG_TAG_BASIC,
                                "bucket in list should be true!");
                    uint64_t idx = bucket - &buckets[0];
                    if (exclude != nullptr && idx == exclude->blck_idx) {
                        if (random_val < bucket->entries.size()) {
                            if (&bucket->entries[random_val] == exclude->val) {
                                break;
                            }
                            handle.val = &bucket->entries[random_val];
                            handle.blck_idx = idx;
                            handle.blck_hash = hash_func(bucket->entries[random_val].first);
                            return;
                        } else {
                            random_val -= bucket->entries.size();
                        }
                        bucket = bucket->next;
                        continue;
                    }

                    bucket->lock.Lock(SX_SHARED);
                    Bucket* next_bucket = bucket->next;
                    if (random_val < bucket->entries.size()) {
                        handle.val = &bucket->entries[random_val];
                        handle.blck_idx = idx;
                        handle.blck_hash = hash_func(bucket->entries[random_val].first);
                        return;
                    } else {
                        random_val -= bucket->entries.size();
                        bucket->lock.Unlock();
                    }

                    bucket = next_bucket;
                }
            } else {
                uint64_t random_bucket_index;
                random_bucket_index = threadSelf->UniformRange64(0, buckets.size() - 1);
                Bucket* bucket = &buckets[random_bucket_index];
                if (exclude != nullptr && random_bucket_index == exclude->blck_idx) {
                    if (bucket->entries.size() <= 1) {
                        continue;
                    }

                    size_t random_index;
                    do {
                        random_index = threadSelf->UniformRange64(0, bucket->entries.size() - 1);
                    } while (bucket->entries[random_index] == exclude->val);

                    handle.val = &bucket->entries[random_index];
                    handle.blck_idx = random_bucket_index;
                    handle.blck_hash = hash_func(bucket->entries[random_index].first);
                    return;
                }

                bucket->lock.Lock(SX_SHARED);
                if (bucket->entries.size() > 0) {
                    size_t random_index = threadSelf->UniformRange64(0, bucket->entries.size() - 1);
                    handle.val = &bucket->entries[random_index];
                    handle.blck_idx = random_bucket_index;
                    handle.blck_hash = hash_func(bucket->entries[random_index].first);
                } else {
                    bucket->lock.Unlock();
                }
            }

            if (handle.val == nullptr) {
                continue;
            }
            FatalAssert(exclude == nullptr || handle.val != exclude->val, LOG_TAG_BASIC,
                        "handle and exclude should point to different elements!");
            threadSelf->SanityCheckLockHeldByMe(&table_lock);
            threadSelf->SanityCheckLockHeldInModeByMe(&buckets[handle.blck_idx].lock, SX_SHARED);
            return;
        }
    }

    void GetTwoRandomElements(ElementHandle& first, ElementHandle& second, bool has_table_lock = false) {
        FatalAssert(&first != &second, LOG_TAG_BASIC,
                    "first and second handles should be different!");
        FatalAssert(first.val == nullptr, LOG_TAG_BASIC,
                    "first handle should be empty!");
        FatalAssert(second.val == nullptr, LOG_TAG_BASIC,
                    "second handle should be empty!");
        if (!has_table_lock) {
            LockTable();
        }
        threadSelf->SanityCheckLockHeldByMe(&table_lock);

        while (true) {
            if (num_elements.load(std::memory_order_acquire) < 2) {
                if (!has_table_lock) {
                    UnlockTable();
                }
                return;
            }

            size_t num_buckets = num_non_empty_buckets.load(std::memory_order_acquire);
            if (num_buckets <= 4 || (buckets.size() * SequentialRandomSearchFactor < num_buckets)) {
                size_t num_current_elements = num_elements.load(std::memory_order_acquire);
                if (num_current_elements < 2) {
                    if (!has_table_lock) {
                        UnlockTable();
                    }
                    return;
                }
                std::pair<uint64_t, uint64_t> random_values;
                do {
                    random_values = threadSelf->UniformRangeTwo64(0, num_current_elements - 1);
                } while (random_values.first == random_values.second);
                Bucket* bucket = head.load(std::memory_order_acquire);
                if (bucket == nullptr) {
                    if (!has_table_lock) {
                        UnlockTable();
                    }
                    return;
                }

                /* todo: stat collection */
                bool first_locked = false;
                bool second_locked = false;
                while (bucket != nullptr) {
                    FatalAssert(bucket->in_list, LOG_TAG_BASIC,
                                "bucket in list should be true!");

                    bucket->lock.Lock(SX_SHARED);
                    Bucket* next_bucket = bucket->next;
                    if (first.val == nullptr) {
                        if (random_values.first < bucket->entries.size()) {
                            first.val = &bucket->entries[random_values.first];
                            first.blck_idx = bucket - &buckets[0];
                            first.blck_hash = hash_func(bucket->entries[random_values.first].first);
                            first_locked = true;
                        } else {
                            random_values.first -= bucket->entries.size();
                        }
                    }

                    if (second.val == nullptr) {
                        if (random_values.second < bucket->entries.size()) {
                            second.val = &bucket->entries[random_values.second];
                            second.blck_idx = bucket - &buckets[0];
                            second.blck_hash = hash_func(bucket->entries[random_values.second].first);
                            second_locked = true;
                        } else {
                            random_values.second -= bucket->entries.size();
                        }
                    }

                    if (!first_locked && !second_locked) {
                        bucket->lock.Unlock();
                    }

                    bucket = next_bucket;
                }
            } else {
                std::pair<uint64_t, uint64_t> random_bucket_indices;
                random_bucket_indices = threadSelf->UniformRangeTwo64(0, buckets.size() - 1);
                Bucket* first_bucket = &buckets[random_bucket_indices.first];
                first_bucket->lock.Lock(SX_SHARED);
                if (first_bucket->entries.size() > 0) {
                    size_t random_index = threadSelf->UniformRange64(0, first_bucket->entries.size() - 1);
                    first.val = &first_bucket->entries[random_index];
                    first.blck_idx = random_bucket_indices.first;
                    first.blck_hash = hash_func(first_bucket->entries[random_index].first);
                } else {
                    first_bucket->lock.Unlock();
                }
                Bucket* second_bucket = &buckets[random_bucket_indices.second];
                if (first_bucket != second_bucket) {
                    second_bucket->lock.Lock(SX_SHARED);
                } else if (first.val == nullptr) {
                    // both indices are same and first not found so second cannot be found either
                    continue;
                }
                if (second_bucket->entries.size() > 0) {
                    size_t random_index = threadSelf->UniformRange64(0, second_bucket->entries.size() - 1);
                    second.val = &second_bucket->entries[random_index];
                    second.blck_idx = random_bucket_indices.second;
                    second.blck_hash = hash_func(second_bucket->entries[random_index].first);
                } else {
                    second_bucket->lock.Unlock();
                }
            }

            if (first.val != nullptr && second.val != nullptr) {
                return;
            } else if (first.val == nullptr && second.val != nullptr) {
                GetRandomElement(first, true, &second);
            } else if (first.val != nullptr && second.val == nullptr) {
                GetRandomElement(second, true, &first);
            }

            if (first.val == nullptr && second.val == nullptr) {
                continue;
            }
            FatalAssert(first.val != second.val, LOG_TAG_BASIC,
                        "first and second handles should point to different elements!");
            FatalAssert(first.val != nullptr && second.val != nullptr, LOG_TAG_BASIC,
                        "Both handles should be valid!");
            threadSelf->SanityCheckLockHeldByMe(&table_lock);
            threadSelf->SanityCheckLockHeldInModeByMe(&buckets[first.blck_idx].lock, SX_SHARED);
            threadSelf->SanityCheckLockHeldInModeByMe(&buckets[second.blck_idx].lock, SX_SHARED);
            return;
        }
    }

    void RehashIfNeeded() {
        LockTable();
        size_t current_size = num_elements.load(std::memory_order_acquire);
        if (!((current_size * MinLoadFactor <= buckets.size() && buckets.size() > DefaultNumBuckets) ||
              current_size >= buckets.size() * MaxLoadFactor)) {
            UnlockTable();
            return;
        }

        UnlockTable();
        LockTableExclusive();
        current_size = num_elements.load(std::memory_order_relaxed);
        if (!((current_size * MinLoadFactor <= buckets.size() && buckets.size() > DefaultNumBuckets) ||
              current_size >= buckets.size() * MaxLoadFactor)) {
            UnlockTable();
            return;
        }

        if (!RehashStart(SX_EXCLUSIVE)) {
            UnlockTable();
            return;
        }

        size_t num_buckets = buckets.size();
        if (current_size * MinLoadFactor <= num_buckets) {
            while (current_size * MinLoadFactor <= num_buckets) {
                num_buckets /= ResizeStep;
                if (num_buckets < DefaultNumBuckets) {
                    num_buckets = DefaultNumBuckets;
                    break;
                }
                FatalAssert(num_buckets <= buckets.size(), LOG_TAG_BASIC,
                            "num_buckets should be less than current buckets size!");
            }
        } else if (current_size >= num_buckets * MaxLoadFactor) {
            while (current_size >= num_buckets * MaxLoadFactor) {
                num_buckets *= ResizeStep;
                FatalAssert(num_buckets >= buckets.size(), LOG_TAG_BASIC,
                            "num_buckets should be greater than current buckets size!");
            }
        }

        Rehash(num_buckets);
        RehashEnd(SX_EXCLUSIVE);
        UnlockTable();
    }
};

// template<typename T, typename Compare = std::less<T>, typename Alloc = std::allocator<T>>
// class ConcurrentSet {
// public:
//     bool Insert(const T& value) {
//         std::lock_guard<std::mutex> lock(_mutex);
//         if (_set.find(value) != _set.end()) {
//             return false;
//         }
//         _set.insert(value);
//         return true;
//     }

//     bool Erase(const T& value) {
//         std::lock_guard<std::mutex> lock(_mutex);
//         auto it = _set.find(value);
//         if (it == _set.end()) {
//             return false;
//         }
//         _set.erase(it);
//         return true;
//     }

//     bool Contains(const T& value) {
//         std::lock_guard<std::mutex> lock(_mutex);
//         return _set.find(value) != _set.end();
//     }

//     size_t Size() {
//         std::lock_guard<std::mutex> lock(_mutex);
//         return _set.size();
//     }

//     bool Empty() {
//         std::lock_guard<std::mutex> lock(_mutex);
//         return _set.empty();
//     }

//     void Clear() {
//         std::lock_guard<std::mutex> lock(_mutex);
//         _set.clear();
//     }

//     divftree::String ToString(divftree::String (*ConvertToString)(const T&)) {
//         std::lock_guard<std::mutex> lock(_mutex);
//         divftree::String result = divftree::String("ConcurrentSet<size=%lu, elements=[");
//         uint64_t size = _set.size();
//         uint64_t index = 0;
//         for (const T& item : _set) {
//             result += ConvertToString(item);
//             if (index < size - 1) {
//                 result += ", ";
//             }
//             index++;
//         }
//         result += "]>";
//         return result;
//     }
// protected:
//     std::set<T, Compare, Alloc> _set;
//     std::mutex _mutex;
// };

template<typename T>
class BlockingQueue {
public:
    BlockingQueue() : q() {}

    BlockingQueue(size_t initial) : q(initial) {}

    inline bool Push(T&& value) {
        return q.enqueue(std::forward<T>(value));
    }

    inline bool Push(const T& value) {
        return q.enqueue(value);
    }

    inline bool BatchPush(const T* value, size_t size) {
        return q.enqueue_bulk(value, size);
    }

    inline bool PopHead(T& value) {
        return q.wait_dequeue_timed(value, std::chrono::microseconds(10));
    }

    inline bool TryPopHead(T& value) {
        return q.try_dequeue(value);
    }

    inline size_t BatchPopHead(T* arr, size_t size) {
        return q.wait_dequeue_bulk_timed(arr, size, std::chrono::microseconds(10));
    }

    inline size_t TryBatchPopHead(T* arr, size_t size) {
        return q.try_dequeue_bulk(arr, size);
    }

    inline bool Empty() {
        return q.size_approx() == 0;
    }

    inline size_t Size() {
        return q.size_approx();
    }

protected:
    moodycamel::BlockingConcurrentQueue<T> q;
};

};

#endif