#ifndef VECTOR_DIRECTORY_H_
#define VECTOR_DIRECTORY_H_

#include "common.h"

#include "third_party/xxhash/include/xxhash.hpp"

namespace divftree {

struct IVFVectorInfo {
    VectorID centroid_id = INVALID_VECTOR_ID;
    size_t offset = 0;
    VTYPE* vector = nullptr;
};

struct VectorDirectoryNode {
    IVFVectorID id;
    size_t num_duplicates;
    IVFVectorInfo info;

    VectorDirectoryNode* next_in_chain;
    VectorDirectoryNode* next;
};

class VectorDirectory {
public:
    static constexpr size_t MAX_VECTOR_PER_BUCKET_RATIO = 8; // average number of vectors per bucket
    static constexpr size_t MIN_NUM_BUCKETS = 1024; // minimum number of buckets
    static constexpr size_t RESIZE_STEP = 2; // resize by this factor

    VectorDirectory(size_t expected_num_vectors, uint16_t dim, uint16_t num_locks)
        : dimension(dim), locks(num_locks), size(0), head(nullptr),
          buckets(std::max(MIN_NUM_BUCKETS, expected_num_vectors / MAX_VECTOR_PER_BUCKET_RATIO), nullptr) {
    }

    ~VectorDirectory() {
        VectorDirectoryNode* current = head;
        while(current != nullptr) {
            VectorDirectoryNode* next = current->next;
            delete current;
            current = next;
        }
    }

    void Rehash(size_t new_num_buckets) {
        FatalAssert(false, LOG_TAG_NOT_IMPLEMENTED, "VectorDirectory::Rehash() not currently supported!");
        if (new_num_buckets == buckets.size() || new_num_buckets < MIN_NUM_BUCKETS) {
            return;
        }

        std::vector<VectorDirectoryNode*> new_buckets(new_num_buckets, nullptr);
        VectorDirectoryNode* current = head;
        while(current != nullptr) {
            current->next_in_chain = nullptr;
            size_t bucket_idx = current->id.vector_hash % new_buckets.size();
            VectorDirectoryNode** new_node = &new_buckets[bucket_idx];
            while (*new_node != nullptr) {
                new_node = &((*new_node)->next_in_chain);
            }
            (*new_node) = current;
            current = current->next;
        }
        buckets = std::move(new_buckets);
    }

    inline uint64_t GetVectorHash(const VTYPE* vector) const {
        return xxh::xxhash3<64>(reinterpret_cast<const uint8_t*>(vector), dimension * sizeof(VTYPE));
    }

    inline size_t GetNumBuckets() const {
        return buckets.size();
    }

    inline size_t GetNumLocks() const {
        return locks.size();
    }

    inline size_t GetLockIndex(uint64_t vector_hash) const {
        return (vector_hash % buckets.size()) % locks.size();
    }

    inline IVFVectorID Insert(const VTYPE* vector, bool fail_if_duplicate, bool* is_duplicate = nullptr) {
        return Insert(GetVectorHash(vector), vector, fail_if_duplicate, is_duplicate);
    }

    IVFVectorID Insert(const uint64_t hash, const VTYPE* vector, bool fail_if_duplicate, bool* is_duplicate = nullptr) {
        IVFVectorID vid{.vector_hash = hash, .value = 0};

        size_t bucket_idx = vid.vector_hash % buckets.size();
        VectorDirectoryNode** node = &buckets[bucket_idx];
        while(*node != nullptr) {
            if ((*node)->id.vector_hash == vid.vector_hash) {
                if (DIVF_MEMCMP((*node)->info.vector, vector, dimension * sizeof(VTYPE)) == 0) {
                    if (is_duplicate != nullptr) {
                        *is_duplicate = true;
                    }

                    if (fail_if_duplicate) {
                        return INVALID_IVF_VECTOR_ID;
                    } else {
                        ++((*node)->num_duplicates);
                        ++num_vectors;
                        return ((*node)->id);
                    }
                } else {
                    // hash collision, try next value
                    vid.value++;
                }
            }
            node = &((*node)->next_in_chain);
        }

        *node = new VectorDirectoryNode();
        (*node)->id = vid;
        (*node)->next_in_chain = nullptr;
        (*node)->num_duplicates = 0;
        (*node)->info.centroid_id = VectorID::AsID(INVALID_VECTOR_ID);
        (*node)->info.offset = UINT64_MAX;
        (*node)->info.vector = nullptr;
        (*node)->next = head;
        head = *node;
        ++size;
        ++num_vectors;

        if (size > buckets.size() * MAX_VECTOR_PER_BUCKET_RATIO) {
            Rehash(buckets.size() * RESIZE_STEP);
        }
        if (is_duplicate != nullptr) {
            *is_duplicate = false;
        }
        return vid;
    }

    void Delete(IVFVectorID id) {
        UNUSED_VARIABLE(id);
        DIVFLOG(LOG_LEVEL_PANIC, LOG_TAG_NOT_IMPLEMENTED, "VectorDirectory::Delete() not implemented!");
    }

    IVFVectorInfo* Find(IVFVectorID id, size_t* num_dup = nullptr) {
        size_t bucket_idx = id.vector_hash % buckets.size();
        VectorDirectoryNode* node = buckets[bucket_idx];
        while(node != nullptr) {
            if (node->id.vector_hash == id.vector_hash && node->id.value == id.value) {
                if (num_dup != nullptr) {
                    *num_dup = node->num_duplicates;
                }
                return &(node->info);
            }
            node = node->next_in_chain;
        }
        return nullptr;
    }

    void LockVector(uint64_t vector_hash, LockMode mode) {
        size_t lock_idx = GetLockIndex(vector_hash);
        locks[lock_idx].Lock(mode);
    }

    bool TryLockVector(uint64_t vector_hash, LockMode mode) {
        size_t lock_idx = GetLockIndex(vector_hash);
        return locks[lock_idx].TryLock(mode);
    }

    void UnlockVector(uint64_t vector_hash) {
        size_t lock_idx = GetLockIndex(vector_hash);
        locks[lock_idx].Unlock();
    }

    size_t Size(bool unique = false) const {
        return (unique ? size : num_vectors);
    }

protected:
    const uint16_t dimension;
    std::vector<SXSpinLock> locks;
    size_t size;
    size_t num_vectors;
    VectorDirectoryNode* head;
    std::vector<VectorDirectoryNode*> buckets;
};

};

#endif