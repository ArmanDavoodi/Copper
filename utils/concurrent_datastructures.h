#ifndef CONCURRENT_DATASTRUCTURES_H_
#define CONCURRENT_DATASTRUCTURES_H_

#include <mutex>
#include <set>
#include <queue>

#include "utils/string.h"
#include "utils/synchronization.h"

#include "third_party/moodycamel/concurrentqueue/concurrentqueue.h"
#include "third_party/moodycamel/concurrentqueue/blockingconcurrentqueue.h"

namespace divftree {

/* Todo better implementations -> lockfree */

template<typename K, typename V, typename Hash>
class ConcurrentMultiMap {
public:
    ConcurrentMultiMap(size_t num_buckets, Hash hash = Hash()) : _num_buckets(num_buckets),
                                                                 _hash(hash) {
        FatalAssert(_num_buckets > 0, LOG_TAG_BASIC, "Number of buckets must be greater than 0");
        _data = new std::unordered_map<K, std::vector<V>, Hash>[_num_buckets];
        _locks = new SXSpinLock[_num_buckets];
    }

    ~ConcurrentMultiMap() {
        delete[] _data;
        delete[] _locks;
    }

    void Insert(const K& key, const V& value) {
        size_t hash_value = _hash(key);
        size_t bucket_idx = hash_value % _num_buckets;

        _locks[bucket_idx].Lock(SX_EXCLUSIVE);
        _data[bucket_idx][key].push_back(value);
        _locks[bucket_idx].Unlock();
    }

    void BatchInsert(const K& key, const std::vector<V>& values) {
        size_t hash_value = _hash(key);
        size_t bucket_idx = hash_value % _num_buckets;

        _locks[bucket_idx].Lock(SX_EXCLUSIVE);
        auto& vec = _data[bucket_idx][key];
        vec.insert(vec.end(), values.begin(), values.end());
        _locks[bucket_idx].Unlock();
    }

    void BatchInsert(const K& key, std::vector<V>&& values) {
        size_t hash_value = _hash(key);
        size_t bucket_idx = hash_value % _num_buckets;

        _locks[bucket_idx].Lock(SX_EXCLUSIVE);
        auto& vec = _data[bucket_idx][key];
        if (vec.empty()) {
            vec = std::move(values);
        } else {
            vec.insert(vec.end(), values.begin(), values.end());
        }
        _locks[bucket_idx].Unlock();
    }

    bool Erase(const K& key, std::vector<V>& out_value) {
        size_t hash_value = _hash(key);
        size_t bucket_idx = hash_value % _num_buckets;

        _locks[bucket_idx].Lock(SX_EXCLUSIVE);
        auto it = _data[bucket_idx].find(key);
        if (it == _data[bucket_idx].end()) {
            _locks[bucket_idx].Unlock();
            return false;
        }
        if (out_value.empty()) {
            out_value = std::move(it->second);
        } else {
            out_value.reserve(out_value.size() + it->second.size());
            std::move(it->second.begin(), it->second.end(), std::back_inserter(out_value));
        }
        _data[bucket_idx].erase(it);
        _locks[bucket_idx].Unlock();
        return true;
    }

    std::vector<V> Erase(const K& key) {
        size_t hash_value = _hash(key);
        size_t bucket_idx = hash_value % _num_buckets;

        _locks[bucket_idx].Lock(SX_EXCLUSIVE);
        auto it = _data[bucket_idx].find(key);
        if (it == _data[bucket_idx].end()) {
            _locks[bucket_idx].Unlock();
            return std::vector<V>();
        }
        std::vector<V> out_value = std::move(it->second);
        _data[bucket_idx].erase(it);
        _locks[bucket_idx].Unlock();
        return out_value;
    }

    bool Contains(const K& key) {
        size_t hash_value = _hash(key);
        size_t bucket_idx = hash_value % _num_buckets;

        _locks[bucket_idx].Lock(SX_SHARED);
        auto it = _data[bucket_idx].find(key);
        bool found = ((it != _data[bucket_idx].end()) && (!it->second.empty()));
        _locks[bucket_idx].Unlock();
        return found;
    }

    bool IsEmpty() {
        for (size_t i = 0; i < _num_buckets; ++i) {
            _locks[i].Lock(SX_SHARED);
            if (!_data[i].empty()) {
                _locks[i].Unlock();
                return false;
            }
            _locks[i].Unlock();
        }
        return true;
    }

protected:
    const size_t _num_buckets;
    Hash _hash;

    std::unordered_map<K, std::vector<V>, Hash>* _data;
    SXSpinLock* _locks;
};

template<typename T, typename Compare = std::less<T>, typename Alloc = std::allocator<T>>
class ConcurrentSet {
public:
    bool Insert(const T& value) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (_set.find(value) != _set.end()) {
            return false;
        }
        _set.insert(value);
        return true;
    }

    bool Erase(const T& value) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto it = _set.find(value);
        if (it == _set.end()) {
            return false;
        }
        _set.erase(it);
        return true;
    }

    bool Contains(const T& value) {
        std::lock_guard<std::mutex> lock(_mutex);
        return _set.find(value) != _set.end();
    }

    size_t Size() {
        std::lock_guard<std::mutex> lock(_mutex);
        return _set.size();
    }

    bool Empty() {
        std::lock_guard<std::mutex> lock(_mutex);
        return _set.empty();
    }

    void Clear() {
        std::lock_guard<std::mutex> lock(_mutex);
        _set.clear();
    }

    divftree::String ToString(divftree::String (*ConvertToString)(const T&)) {
        std::lock_guard<std::mutex> lock(_mutex);
        divftree::String result = divftree::String("ConcurrentSet<size=%lu, elements=[");
        uint64_t size = _set.size();
        uint64_t index = 0;
        for (const T& item : _set) {
            result += ConvertToString(item);
            if (index < size - 1) {
                result += ", ";
            }
            index++;
        }
        result += "]>";
        return result;
    }
protected:
    std::set<T, Compare, Alloc> _set;
    std::mutex _mutex;
};

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