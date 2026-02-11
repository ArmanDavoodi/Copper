#ifndef SORTED_LIST_H_
#define SORTED_LIST_H_

#include <vector>
#include <algorithm>

#include "debug.h"

#include "utils/string.h"

namespace divftree {
template<typename T, typename CMP>
class SortedList {
public:
    typedef std::vector<T>::iterator Iterator;

    SortedList(const CMP& comparator, size_t size = 0) : cmp(comparator) {
        if (size > 0) {
            data.reserve(size);
        }
    }

    SortedList(CMP&& comparator, size_t size = 0) : cmp(std::forward<CMP>(comparator)) {
        if (size > 0) {
            data.reserve(size);
        }
    }

    SortedList(const CMP& comparator, std::vector<T>&& base) : cmp(comparator),
                                                               data(std::forward<std::vector<T>>(base)) {}

    SortedList(CMP&& comparator, std::vector<T>&& base) : cmp(std::forward<CMP>(comparator)),
                                                          data(std::forward<std::vector<T>>(base)) {}

    inline T& operator[](size_t idx) {
        return data[idx];
    }

    inline const T& operator[](size_t idx) const {
        return data[idx];
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
        data.insert(std::lower_bound(data.begin(), data.end(), d, cmp), d);
        SANITY_CHECK({
            for (size_t i = 1; i < data.size(); ++i) {
                FatalAssert(cmp(data[i - 1], data[i]), LOG_TAG_BASIC,
                            "Data is not sorted at index %zu!", i);
            }
        });
    }

    inline void Insert(T&& d) {
        data.insert(std::lower_bound(data.begin(), data.end(), d, cmp), std::forward<T>(d));
        SANITY_CHECK({
            for (size_t i = 1; i < data.size(); ++i) {
                FatalAssert(cmp(data[i - 1], data[i]), LOG_TAG_BASIC,
                            "Data is not sorted at index %zu!", i);
                FatalAssert(data[i - 1].first <= data[i].first, LOG_TAG_BASIC,
                            "Data is not sorted at index %zu!", i);
            }
        });
    }

    inline Iterator Find(const T& d) {
        auto it = std::lower_bound(data.begin(), data.end(), d, cmp);
        if (it == data.end()) {
            return it;
        }

        while (it != data.end() && (cmp(*it, d) == 0) && (*it != d)) {
            ++it;
        }

        if (it != data.end() && *it != d) {
            it = data.end();
        }

        return it;
    }

    inline const Iterator Find(const T& d) const {
        auto it = std::lower_bound(data.begin(), data.end(), d, cmp);
        if (it == data.end()) {
            return it;
        }

        while (it != data.end() && (cmp(*it, d) == 0) && (*it != d)) {
            ++it;
        }

        if (it != data.end() && *it != d) {
            it = data.end();
        }

        return it;
    }

    inline void PopBack() {
        SANITY_CHECK({
            for (size_t i = 1; i < data.size(); ++i) {
                FatalAssert(cmp(data[i - 1], data[i]), LOG_TAG_BASIC,
                            "Data is not sorted at index %zu!", i);
            }
        });
        data.pop_back();
    }

    inline void PopBack(size_t num) {
        if (num == 0) {
            return;
        }

        SANITY_CHECK({
            for (size_t i = 1; i < data.size(); ++i) {
                FatalAssert(cmp(data[i - 1], data[i]), LOG_TAG_BASIC,
                            "Data is not sorted at index %zu!", i);
            }
        });

        if (num >= data.size()) {
            data.clear();
            return;
        }

        data.resize(data.size() - num);
    }

    inline void Erase(Iterator it) {
        data.erase(it);
    }

    inline void Clear() {
        data.clear();
    }

    inline void MergeWith(const SortedList<T, CMP>& other, size_t limit, bool include_duplicates = true) {
        FatalAssert(other.cmp == cmp, LOG_TAG_BASIC, "cannot merge two lists with different comparators!");
        std::vector<T> tmp;
        tmp.reserve(std::min(limit, data.size() + other.Size()));
        size_t i = 0, j = 0;
        while (tmp.size() < limit && i < data.size() && j < other.data.size()) {
            if (!include_duplicates && tmp.size() > 0) {
                while (tmp.back() == data[i]) {
                    i++;
                }

                while (tmp.back() == other.data[j]) {
                    j++;
                }
            }
            if (cmp(data[i], other.data[j])) {
                tmp.push_back(data[i++]);
            } else {
                tmp.push_back(other.data[j++]);
            }
        }

        while (tmp.size() < limit && i < data.size()) {
            if (!include_duplicates && tmp.size() > 0 && tmp.back() == data[i]) {
                i++;
                continue;
            }
            tmp.push_back(data[i++]);
        }

        while (tmp.size() < limit && j < other.data.size()) {
            if (!include_duplicates && tmp.size() > 0 && tmp.back() == other.data[j]) {
                j++;
                continue;
            }
            tmp.push_back(other.data[j++]);
        }

        SANITY_CHECK({
            for (size_t i = 1; i < data.size(); ++i) {
                FatalAssert(cmp(data[i - 1], data[i]), LOG_TAG_BASIC,
                            "Data is not sorted at index %zu!", i);
            }

            for (size_t i = 1; i < other.data.size(); ++i) {
                FatalAssert(cmp(other.data[i - 1], other.data[i]), LOG_TAG_BASIC,
                            "Data is not sorted at index %zu!", i);
            }

            for (size_t i = 1; i < tmp.size(); ++i) {
                FatalAssert(cmp(tmp[i - 1], tmp[i]), LOG_TAG_BASIC,
                            "Data is not sorted at index %zu!", i);
            }
        });

        data.swap(tmp);
    }

    inline Iterator begin() {
        return data.begin();
    }

    inline Iterator end() {
        return data.end();
    }

    inline const Iterator begin() const {
        return data.begin();
    }

    inline const Iterator end() const {
        return data.end();
    }

    inline void Extract(std::vector<T>& dest, bool reverse = false) {
        if (reverse) {
            std::reverse(data.begin(), data.end());
        }
        dest = std::move(data);
    }

    template <String (*ElementToString)(const T&)>
    inline String ToString() {
        String res = "[";
        for (size_t i = 0; i < data.size(); ++i) {
            res += ElementToString(data[i]) + String(i == data.size() - 1 ? "]" : ", ");
        }
        return res;
    }

protected:
    CMP cmp;
    std::vector<T> data;
};

};

#endif