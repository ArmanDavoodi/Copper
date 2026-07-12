#ifndef TOP_N_H_
#define TOP_N_H_

#include <algorithm>

#include "debug.h"

namespace divftree {
template<typename T, typename CMP>
class TopN {
protected:
    const size_t capacity;
    CMP cmp;
    std::vector<T> data;

public:

    TopN(const CMP& comparator, size_t cap) : capacity(cap), cmp(comparator) {
        FatalAssert(cap > 0, LOG_TAG_BASIC, "TopN capacity must be greater than 0");
        data.reserve(cap);
    }

    TopN(CMP&& comparator, size_t cap) : capacity(cap), cmp(std::forward<CMP>(comparator)) {
        FatalAssert(cap > 0, LOG_TAG_BASIC, "TopN capacity must be greater than 0");
        data.reserve(cap);
    }

    inline static constexpr void MergeExtracted(std::vector<T>& dest, const std::vector<T>& src, const CMP& cmp,
                                                size_t capacity) {

        FatalAssert(capacity > 0, LOG_TAG_BASIC, "TopN capacity must be greater than 0");
        FatalAssert(dest.size() <= capacity, LOG_TAG_BASIC, "TopN destination size must be less than or equal to capacity");
        FatalAssert(src.size() <= capacity, LOG_TAG_BASIC, "TopN source size must be less than or equal to capacity");
        if (src.empty()) {
            return;
        }

        if (dest.empty()) {
            dest = std::move(src);
            return;
        }

        size_t dest_idx = 0;
        size_t src_idx = 0;
        size_t merged_size = 0;
        std::vector<T> merged;
        merged.reserve(capacity);

        while (merged_size < capacity && dest_idx < dest.size() && src_idx < src.size()) {
            if (cmp(dest[dest_idx], src[src_idx])) {
                merged.push_back(std::move(dest[dest_idx]));
                dest_idx++;
            } else {
                merged.push_back(std::move(src[src_idx]));
                src_idx++;
            }
            merged_size++;
        }

        while (merged_size < capacity && dest_idx < dest.size()) {
            merged.push_back(std::move(dest[dest_idx]));
            dest_idx++;
            merged_size++;
        }

        while (merged_size < capacity && src_idx < src.size()) {
            merged.push_back(std::move(src[src_idx]));
            src_idx++;
            merged_size++;
        }

        dest = std::move(merged);
    }

    inline size_t Size() const {
        return data.size();
    }

    inline bool Empty() const {
        return data.empty();
    }

    inline void reserve(size_t size) {
        data.reserve(size);
    }

    inline void Insert(const T& d) {
        if (data.size() < capacity) {
            data.push_back(d);
            std::push_heap(data.begin(), data.end(), cmp);
            return;
        }

        if (!cmp(d, data.front())) {
            return;
        }

        std::pop_heap(data.begin(), data.end(), cmp);
        data.back() = d;
        std::push_heap(data.begin(), data.end(), cmp);
    }

    inline void Insert(T&& d) {
        if (data.size() < capacity) {
            data.emplace_back(std::forward<T>(d));
            std::push_heap(data.begin(), data.end(), cmp);
            return;
        }

        if (!cmp(d, data.front())) {
            return;
        }

        std::pop_heap(data.begin(), data.end(), cmp);
        data.back() = std::forward<T>(d);
        std::push_heap(data.begin(), data.end(), cmp);
    }

    inline void Clear() {
        data.clear();
    }

    inline void Extract(std::vector<T>& dest, bool reverse = false) {
        std::sort_heap(data.begin(), data.end(), cmp);
        if (reverse) {
            std::reverse(data.begin(), data.end());
        }
        dest = std::move(data);
        Clear();
    }
};

};

#endif