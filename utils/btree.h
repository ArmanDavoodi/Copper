#ifndef BTREE_H_
#define BTREE_H_

#include "common.h"

namespace divftree {

class BPlusTreeNode {
public:
    const bool is_leaf;
    std::atomic<bool> locked_exclusive;
    std::atomic<uint8_t> num_readers;
    uint8_t num_keys;

    BPlusTreeNode(bool is_leaf) : is_leaf(is_leaf), locked_exclusive(false), num_readers(0), num_keys(0) {}

    inline bool IsEmpty() const {
        return num_keys == 0;
    }

    virtual uint8_t GetMaxKeys() const = 0;
    virtual uint8_t GetMinKeys() const = 0;

    inline void Lock(LockMode mode) {
        threadSelf->SanityCheckLockNotHeldByMe(this);
        if (mode == SX_EXCLUSIVE) {
            bool expected = false;
            while (!locked_exclusive.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
                expected = false;
                DIVFTREE_YIELD();
            }
            while (num_readers.load(std::memory_order_acquire) > 0) {
                DIVFTREE_YIELD();
            }
        } else {
            while (true) {
                while (locked_exclusive.load(std::memory_order_acquire)) {
                    DIVFTREE_YIELD();
                }
                uint8_t old_readers = num_readers.fetch_add(1, std::memory_order_acquire);
                UNUSED_VARIABLE(old_readers);
                FatalAssert(old_readers < UINT8_MAX, LOG_TAG_BASIC, "reader overflow!");
                if (!locked_exclusive.load(std::memory_order_acquire)) {
                    break;
                }
                num_readers.fetch_sub(1, std::memory_order_release);
            }
        }
        threadSelf->AcquireLockSanityLog(this, mode);
    }

    inline bool UpgradeLock() {
        threadSelf->SanityCheckLockHeldInModeByMe(this, SX_SHARED);
        FatalAssert(num_readers.load(std::memory_order_acquire) >= 1, LOG_TAG_BASIC,
                    "there should be at least one shared reader(self) to upgrade to exclusive");
        bool expected = false;
        if (!locked_exclusive.compare_exchange_strong(expected, true, std::memory_order_acquire)) {
            return false;
        }
        num_readers.fetch_sub(1, std::memory_order_release);
        while (num_readers.load(std::memory_order_acquire) > 0) {
            DIVFTREE_YIELD();
        }
        threadSelf->UpgradeLockSanityLog(this);
        return true;
    }

    inline void DowngradeLock() {
        threadSelf->SanityCheckLockHeldInModeByMe(this, SX_EXCLUSIVE);
        FatalAssert(locked_exclusive.load(std::memory_order_acquire), LOG_TAG_BASIC,
                    "there should be exclusive lock held by self to downgrade to shared");
        FatalAssert(num_readers.load(std::memory_order_acquire) == 0, LOG_TAG_BASIC,
                    "there should be no readers when downgrading to shared");
        uint8_t old_readers = num_readers.fetch_add(1, std::memory_order_acquire);
        UNUSED_VARIABLE(old_readers);
        FatalAssert(old_readers < UINT8_MAX, LOG_TAG_BASIC, "reader overflow!");
        locked_exclusive.store(false, std::memory_order_release);
        threadSelf->DowngradeLockSanityLog(this);
    }

    inline void Unlock(LockMode mode) {
        threadSelf->SanityCheckLockHeldInModeByMe(this, mode);
        if (mode == SX_EXCLUSIVE) {
            FatalAssert(locked_exclusive.load(std::memory_order_acquire), LOG_TAG_BASIC,
                        "Unlocking exclusive lock that is not held!");
            locked_exclusive.store(false, std::memory_order_release);
        } else {
            FatalAssert(num_readers.load(std::memory_order_acquire) > 0, LOG_TAG_BASIC,
                        "Unlocking shared lock that is not held!");
            num_readers.fetch_sub(1, std::memory_order_release);
        }
    }
};

template<typename K>
inline constexpr size_t BPLUSTREE_INTERNAL_DEGREE() {
    static_assert(sizeof(K) <= CACHE_LINE_SIZE - sizeof(BPlusTreeNodeHeader) - 2 * sizeof(uintptr_t),
                  "Key size is too large for BPlusTreeNodeHeader");
    /* Internal nodes have one more pointer than keys and leaf nodes have a pointer to next leaf */
    size_t t = (CACHE_LINE_SIZE - sizeof(BPlusTreeNodeHeader) - sizeof(uintptr_t)) / (sizeof(K) + sizeof(uintptr_t));
    return (t + 1) / 2; // an internal node can have at most 2t children
}

template<typename K>
inline constexpr size_t BPLUSTREE_LEAF_DEGREE() {
    static_assert(sizeof(K) <= CACHE_LINE_SIZE - sizeof(BPlusTreeNodeHeader) - 2 * sizeof(uintptr_t),
                  "Key size is too large for BPlusTreeNodeHeader");
    size_t max_keys_supported = 31;
    size_t size = max_keys_supported * (sizeof(K) + sizeof(uintptr_t)) + sizeof(BPlusTreeNodeHeader) + sizeof(uintptr_t);
    max_keys_supported = (size / CACHE_LINE_SIZE) + (size % CACHE_LINE_SIZE != 0 ? 1 : 0);
    max_keys_supported -= (max_keys_supported % 2 == 0 ? 1 : 0); // make it odd
    return (max_keys_supported + 1) / 2;
}

template<typename K, typename V>
class BPlusTreeLeafNode : public BPlusTreeNode {
public:
    static_assert(sizeof(V) <= sizeof(uintptr_t),
                  "Value size is too large for BPlusTreeNode leaf value storage");
    static inline constexpr uint8_t degree = static_cast<uint8_t>(BPLUSTREE_LEAF_DEGREE<K>());
    static inline constexpr uint8_t MaxKeys = degree * 2 - 1;
    static inline constexpr uint8_t MinKeys = degree - 1;
    static_assert(MaxKeys > 0, "BPlusTreeLeafNode must be able to hold at least one key");
    static_assert(MinKeys > 0, "BPlusTreeLeafNode must be able to hold at least one key");

    BPlusTreeLeafNode<K, V>* next_leaf;
    K keys[MaxKeys];
    V values[MaxKeys];


    BPlusTreeLeafNode() : BPlusTreeNode(true) {}

    inline constexpr uint8_t GetMaxKeys() const override {
        return MaxKeys;
    }

    inline constexpr uint8_t GetMinKeys() const override {
        return MinKeys;
    }

    uint8_t InsertNonFull(const K& key, const V& value) {
        FatalAssert(num_keys < MaxKeys, LOG_TAG_BASIC, "Node is full, cannot insert key");
        for (int8_t i = num_keys - 1; i >= 0; --i) {
            if (keys[i] > key) {
                keys[i + 1] = keys[i];
                values[i + 1] = values[i];
            } else {
                keys[i + 1] = key;
                values[i + 1] = value;
                num_keys++;
                return static_cast<uint8_t>(i + 1);
            }
        }
        FatalAssert(false, LOG_TAG_BASIC, "Key insertion failed");
        return MaxKeys + 1; // Should not reach here
    }

    void Remove(uint8_t index, V& value) {
        FatalAssert(is_leaf, LOG_TAG_BASIC, "RemoveFromLeaf called on non-leaf node");
        FatalAssert(index < num_keys, LOG_TAG_BASIC, "Index out of bounds for removal");
        value = values[index];
        for (uint8_t i = index; i < num_keys - 1; ++i) {
            keys[i] = keys[i + 1];
            values[i] = values[i + 1];
        }
        num_keys--;
    }
};

template<typename K>
class BPlusTreeInternalNode : public BPlusTreeNode {
public:
    static inline constexpr uint8_t degree = static_cast<uint8_t>(BPLUSTREE_INTERNAL_DEGREE<K>());
    static inline constexpr uint8_t MaxKeys = degree * 2 - 1;
    static inline constexpr uint8_t MinKeys = degree - 1;
    static_assert(MaxKeys > 0, "BPlusTreeLeafNode must be able to hold at least one key");

    K keys[MaxKeys];
    BPlusTreeNode* children[MaxKeys + 1];

    BPlusTreeInternalNode() : BPlusTreeNode(false) {}

    inline constexpr uint8_t GetMaxKeys() const override {
        return MaxKeys;
    }

    inline constexpr uint8_t GetMinKeys() const override {
        return MinKeys;
    }

    K GetChildPredecessor(uint8_t child_idx) const {
        FatalAssert(child_idx <= num_keys, LOG_TAG_BASIC, "Index out of bounds for predecessor retrieval");
        const BPlusTreeNode* current = children[child_idx];
        CHECK_NOT_NULLPTR(current, LOG_TAG_BASIC);
        while (!current->is_leaf) {
            FatalAssert(current->num_keys > 0, LOG_TAG_BASIC,
                        "Internal node has no keys during predecessor retrieval");
            current = static_cast<const BPlusTreeInternalNode<K>*>(current)->children[current->num_keys];
            CHECK_NOT_NULLPTR(current, LOG_TAG_BASIC);
        }
        FatalAssert(current->num_keys > 0, LOG_TAG_BASIC,
                    "Leaf node has no keys during predecessor retrieval");
        const BPlusTreeLeafNode<K, void*>* leaf_node = static_cast<const BPlusTreeLeafNode<K, void*>*>(current);
        return leaf_node->keys[leaf_node->num_keys - 1];
    }

    K GetChildSuccessor(uint8_t child_idx) const {
        FatalAssert(!is_leaf, LOG_TAG_BASIC, "GetChildSuccessor called on leaf node");
        FatalAssert(child_idx < num_keys, LOG_TAG_BASIC, "Index out of bounds for successor retrieval");
        const BPlusTreeNode* current = children[child_idx + 1];
        CHECK_NOT_NULLPTR(current, LOG_TAG_BASIC);
        while (!current->is_leaf) {
            FatalAssert(current->num_keys > 0, LOG_TAG_BASIC,
                        "Internal node has no keys during successor retrieval");
            current = static_cast<const BPlusTreeInternalNode<K>*>(current)->children[0];
            CHECK_NOT_NULLPTR(current, LOG_TAG_BASIC);
        }
        FatalAssert(current->num_keys > 0, LOG_TAG_BASIC,
                    "Leaf node has no keys during successor retrieval");
        return static_cast<const BPlusTreeLeafNode<K, void*>*>(current)->keys[0];
    }

    void BorrowLeftForChild(uint8_t child_idx) {
        FatalAssert(!is_leaf, LOG_TAG_BASIC, "BorrowLeftForChild called on leaf node");
        FatalAssert(child_idx > 0 && child_idx < num_keys + 1, LOG_TAG_BASIC,
                    "Index out of bounds for borrowing left");
        BPlusTreeNode* child = children[child_idx];
        BPlusTreeNode* left_sibling = children[child_idx - 1];
        CHECK_NOT_NULLPTR(child, LOG_TAG_BASIC);
        CHECK_NOT_NULLPTR(left_sibling, LOG_TAG_BASIC);

        left_sibling->Lock(SX_EXCLUSIVE);
        child->Lock(SX_EXCLUSIVE);

        if (child->is_leaf) {
            FatalAssert(left_sibling->is_leaf, LOG_TAG_BASIC,
                        "Left sibling and child leaf status mismatch");
            BPlusTreeLeafNode<K, void*>* child_leaf = static_cast<BPlusTreeLeafNode<K, void*>*>(child);
            BPlusTreeLeafNode<K, void*>* left_sibling_leaf = static_cast<BPlusTreeLeafNode<K, void*>*>(left_sibling);
            FatalAssert(left_sibling_leaf->num_keys > 0, LOG_TAG_BASIC,
                        "Left sibling leaf has no keys to borrow from");
            FatalAssert(child_leaf->num_keys < BPlusTreeLeafNode<K, void*>::MaxKeys, LOG_TAG_BASIC,
                        "Child leaf node is full, cannot borrow key");
            // Shift child's keys and children to the right
            for (int8_t i = child_leaf->num_keys - 1; i >= 0; --i) {
                child_leaf->keys[i + 1] = child_leaf->keys[i];
                child_leaf->values[i + 1] = child_leaf->values[i];
            }
            // Move key from parent to child
            child_leaf->keys[0] = keys[child_idx - 1];
            child_leaf->values[0] = left_sibling_leaf->values[left_sibling_leaf->num_keys - 1];
            // Move key from left sibling to parent
            keys[child_idx - 1] = left_sibling_leaf->keys[left_sibling_leaf->num_keys - 1];
            left_sibling_leaf->num_keys--;
            child_leaf->num_keys++;
        } else {
            FatalAssert(!left_sibling->is_leaf, LOG_TAG_BASIC,
                        "Left sibling and child leaf status mismatch");
            BPlusTreeInternalNode<K>* child_internal = static_cast<BPlusTreeInternalNode<K>*>(child);
            BPlusTreeInternalNode<K>* left_sibling_internal = static_cast<BPlusTreeInternalNode<K>*>(left_sibling);
            FatalAssert(left_sibling_internal->num_keys > 0, LOG_TAG_BASIC,
                        "Left sibling internal node has no keys to borrow from");
            FatalAssert(child_internal->num_keys < BPlusTreeInternalNode<K>::MaxKeys,
                        LOG_TAG_BASIC, "Child internal node is full, cannot borrow key");
            // Shift child's keys and children to the right
            for (int8_t i = child_internal->num_keys - 1; i >= 0; --i) {
                child_internal->keys[i + 1] = child_internal->keys[i];
                child_internal->children[i + 2] = child_internal->children[i + 1];
            }
            child_internal->children[1] = child_internal->children[0];
            // Move key from parent to child
            child_internal->keys[0] = keys[child_idx - 1];
            child_internal->children[0] = left_sibling_internal->children[left_sibling_internal->num_keys];
            // Move key from left sibling to parent
            keys[child_idx - 1] = left_sibling_internal->keys[left_sibling_internal->num_keys - 1];
            left_sibling_internal->num_keys--;
            child_internal->num_keys++;
        }

        child->Unlock(SX_EXCLUSIVE);
        left_sibling->Unlock(SX_EXCLUSIVE);
    }

    void BorrowRightForChild(uint8_t child_idx) {
        FatalAssert(!is_leaf, LOG_TAG_BASIC, "BorrowRightForChild called on leaf node");
        FatalAssert(child_idx < num_keys, LOG_TAG_BASIC,
                    "Index out of bounds for borrowing right");
        BPlusTreeNode* child = children[child_idx];
        BPlusTreeNode* right_sibling = children[child_idx + 1];
        CHECK_NOT_NULLPTR(child, LOG_TAG_BASIC);
        CHECK_NOT_NULLPTR(right_sibling, LOG_TAG_BASIC);
        FatalAssert(right_sibling->num_keys > 0, LOG_TAG_BASIC,
                    "Right sibling has no keys to borrow from");

        child->Lock(SX_EXCLUSIVE);
        right_sibling->Lock(SX_EXCLUSIVE);

        if (child->is_leaf) {
            FatalAssert(right_sibling->is_leaf, LOG_TAG_BASIC,
                        "Right sibling and child leaf status mismatch");
            BPlusTreeLeafNode<K, void*>* child_leaf = static_cast<BPlusTreeLeafNode<K, void*>*>(child);
            BPlusTreeLeafNode<K, void*>* right_sibling_leaf = static_cast<BPlusTreeLeafNode<K, void*>*>(right_sibling);
            FatalAssert(child_leaf->num_keys < BPlusTreeLeafNode<K, void*>::MaxKeys, LOG_TAG_BASIC,
                        "Child leaf node is full, cannot borrow key");

            // Move key from parent to child
            child_leaf->keys[child_leaf->num_keys] = keys[child_idx];
            child_leaf->values[child_leaf->num_keys] = right_sibling_leaf->values[0];
            child_leaf->num_keys++;
            // Move key from right sibling to parent
            keys[child_idx] = right_sibling_leaf->keys[0];
            // Shift right sibling's keys and children to the left
            for (uint8_t i = 0; i < right_sibling_leaf->num_keys - 1; ++i) {
                right_sibling_leaf->keys[i] = right_sibling_leaf->keys[i + 1];
                right_sibling_leaf->values[i] = right_sibling_leaf->values[i + 1];
            }
            right_sibling_leaf->num_keys--;
        } else {
            FatalAssert(!right_sibling->is_leaf, LOG_TAG_BASIC,
                        "Right sibling and child leaf status mismatch");
            BPlusTreeInternalNode<K>* child_internal = static_cast<BPlusTreeInternalNode<K>*>(child);
            BPlusTreeInternalNode<K>* right_sibling_internal = static_cast<BPlusTreeInternalNode<K>*>(right_sibling);
            FatalAssert(child_internal->num_keys < BPlusTreeInternalNode<K>::MaxKeys,
                        LOG_TAG_BASIC, "Child internal node is full, cannot borrow key");

            // Move key from parent to child
            child_internal->keys[child_internal->num_keys] = keys[child_idx];
            child_internal->children[child_internal->num_keys + 1] =
                right_sibling_internal->children[0];
            child_internal->num_keys++;
            // Move key from right sibling to parent
            keys[child_idx] = right_sibling_internal->keys[0];
            // Shift right sibling's keys and children to the left
            for (uint8_t i = 0; i < right_sibling_internal->num_keys - 1; ++i) {
                right_sibling_internal->keys[i] = right_sibling_internal->keys[i + 1];
                right_sibling_internal->children[i] = right_sibling_internal->children[i + 1];
            }
            right_sibling_internal->children[right_sibling_internal->num_keys - 1] =
                right_sibling_internal->children[right_sibling_internal->num_keys];
            right_sibling_internal->num_keys--;
        }

        right_sibling->Unlock(SX_EXCLUSIVE);
        child->Unlock(SX_EXCLUSIVE);
    }

    void MergeChildWithRightSibling(uint8_t child_idx) {
        FatalAssert(!is_leaf, LOG_TAG_BASIC, "MergeChildren called on leaf node");
        FatalAssert(child_idx < num_keys, LOG_TAG_BASIC,
                    "Index out of bounds for merging children");
        BPlusTreeNode* left_child = children[child_idx];
        BPlusTreeNode* right_child = children[child_idx + 1];
        CHECK_NOT_NULLPTR(left_child, LOG_TAG_BASIC);
        CHECK_NOT_NULLPTR(right_child, LOG_TAG_BASIC);

        left_child->Lock(SX_EXCLUSIVE);
        right_child->Lock(SX_EXCLUSIVE);

        if (left_child->is_leaf) {
            FatalAssert(right_child->is_leaf, LOG_TAG_BASIC,
                        "Left and right child leaf status mismatch");
            BPlusTreeLeafNode<K, void*>* left_leaf = static_cast<BPlusTreeLeafNode<K, void*>*>(left_child);
            BPlusTreeLeafNode<K, void*>* right_leaf = static_cast<BPlusTreeLeafNode<K, void*>*>(right_child);
            FatalAssert(left_leaf->num_keys + right_leaf->num_keys < BPlusTreeLeafNode<K, void*>::MaxKeys,
                        LOG_TAG_BASIC, "Not enough space to merge children");
            left_leaf->keys[left_leaf->num_keys] = keys[child_idx];
            left_leaf->num_keys++;

            for (uint8_t i = 0; i < right_leaf->num_keys; ++i) {
                left_leaf->keys[left_leaf->num_keys] = right_leaf->keys[i];
                left_leaf->values[left_leaf->num_keys] = right_leaf->values[i];
                left_leaf->num_keys++;
            }
            left_leaf->next_leaf = right_leaf->next_leaf;

        } else {
            FatalAssert(!right_child->is_leaf, LOG_TAG_BASIC,
                        "Left and right child leaf status mismatch");
            BPlusTreeInternalNode<K>* left_internal = static_cast<BPlusTreeInternalNode<K>*>(left_child);
            BPlusTreeInternalNode<K>* right_internal = static_cast<BPlusTreeInternalNode<K>*>(right_child);
            FatalAssert(left_internal->num_keys + right_internal->num_keys < BPlusTreeInternalNode<K>::MaxKeys,
                        LOG_TAG_BASIC, "Not enough space to merge children");
            left_internal->keys[left_internal->num_keys] = keys[child_idx];
            left_internal->num_keys++;
            for (uint8_t i = 0; i < right_internal->num_keys; ++i) {
                left_internal->keys[left_internal->num_keys] = right_internal->keys[i];
                left_internal->children[left_internal->num_keys] = right_internal->children[i];
                left_internal->num_keys++;
            }
            left_internal->children[left_internal->num_keys] =
                right_internal->children[right_internal->num_keys];
        }

        // Remove key and child pointer from parent
        for (uint8_t i = child_idx; i < num_keys - 1; ++i) {
            keys[i] = keys[i + 1];
            children[i + 1] = children[i + 2];
        }
        num_keys--;
        right_child->Unlock(SX_EXCLUSIVE);
        delete right_child;
        left_child->Unlock(SX_EXCLUSIVE);
    }

    void FillChild(uint8_t child_idx) {
        FatalAssert(!is_leaf, LOG_TAG_BASIC, "FillChild called on leaf node");
        FatalAssert(child_idx < num_keys + 1, LOG_TAG_BASIC,
                    "Index out of bounds for filling child");
        if (child_idx != 0 && children[child_idx - 1]->num_keys >= MinKeys + 1) {
            BorrowLeftForChild(child_idx);
        } else if (child_idx != num_keys &&
                   children[child_idx + 1]->num_keys >= MinKeys + 1) {
            BorrowRightForChild(child_idx);
        } else {
            if (child_idx != num_keys) {
                MergeChildWithRightSibling(child_idx);
            } else {
                MergeChildWithRightSibling(child_idx - 1);
            }
        }
    }

    BPlusTreeNode* RemoveKey(uint8_t index, K& removed_key) {
        FatalAssert(!is_leaf, LOG_TAG_BASIC, "RemoveFromInternal called on leaf node");
        FatalAssert(index < num_keys, LOG_TAG_BASIC, "Index out of bounds for removal");
        if (children[index]->num_keys >= MinKeys + 1) {
            K pred_key = GetChildPredecessor(index);
            keys[index] = pred_key;
            removed_key = pred_key;
            return children[index];
        } else if (children[index + 1]->num_keys >= MinKeys + 1) {
            K succ_key = GetChildSuccessor(index);
            keys[index] = succ_key;
            removed_key = succ_key;
            return children[index + 1];
        } else {
            K merge_key = keys[index];
            MergeChildWithRightSibling(index);
            removed_key = merge_key;
            return children[index];
        }
    }
};

template<typename K, typename V>
class BPlusTree {
    enum class InsertionResult : uint8_t {
        EXISTS,
        INSERTED,
        SPLIT_REQUIRED
    };

public:
    BPlusTree() : root(new BPlusTreeLeafNode<K, V>()),
                  leftmost_leaf(root.load(std::memory_order_relaxed)),
                  tree_size(0) {}
    ~BPlusTree() = default;

    class Iterator {
    public:
        Iterator(BPlusTreeLeafNode<K, V>* start_leaf, uint8_t start_index)
            : current_leaf(start_leaf), index(start_index) {
            if (current_leaf != nullptr) {
                threadSelf->SanityCheckLockHeldInModeByMe(current_leaf, SX_SHARED);
            }
        }

        Iterator(const Iterator& it)
            : current_leaf(it.current_leaf), index(it.index) {
            if (current_leaf != nullptr) {
                current_leaf->Lock(SX_SHARED);
            }
        }

        Iterator(Iterator&& it) noexcept
            : current_leaf(it.current_leaf), index(it.index) {
            it.current_leaf = nullptr;
            it.index = 0;
        }

        Iterator& operator=(const Iterator& it) {
            if (this != &it) {
                Release();
                current_leaf = it.current_leaf;
                index = it.index;
                if (current_leaf != nullptr) {
                    current_leaf->Lock(SX_SHARED);
                }
            }
            return *this;
        }

        Iterator& operator=(Iterator&& it) noexcept {
            if (this != &it) {
                Release();
                current_leaf = it.current_leaf;
                index = it.index;
                it.current_leaf = nullptr;
                it.index = 0;
            }
            return *this;
        }

        inline void Release() {
            if (current_leaf != nullptr) {
                threadSelf->SanityCheckLockHeldInModeByMe(current_leaf, SX_SHARED);
                current_leaf->Unlock(SX_SHARED);
                current_leaf = nullptr;
            }
            index = 0;
        }

        ~Iterator() {
            Release();
        }

        inline static Iterator End() {
            return Iterator(nullptr, 0);
        }

        inline bool HasNext() const {
            return current_leaf != nullptr && (index < current_leaf->num_keys || current_leaf->next_leaf != nullptr);
        }

        inline Iterator& operator++() {
            if (current_leaf == nullptr) {
                return *this;
            }

            threadSelf->SanityCheckLockHeldInModeByMe(current_leaf, SX_SHARED);
            index++;
            if (index >= current_leaf->num_keys) {
                if (current_leaf->next_leaf != nullptr) {
                    current_leaf->next_leaf->Lock(SX_SHARED);
                }
                current_leaf->Unlock(SX_SHARED);
                current_leaf = current_leaf->next_leaf;
                index = 0;
            }
            return *this;
        }

        inline bool operator==(const ConstIterator& other) const {
            return current_leaf == other.current_leaf && index == other.index;
        }

        inline bool operator!=(const ConstIterator& other) const {
            return !(*this == other);
        }

        inline bool operator==(const Iterator& other) const {
            return current_leaf == other.current_leaf && index == other.index;
        }

        inline bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

        inline V& operator*() {
            FatalAssert(current_leaf != nullptr, LOG_TAG_BASIC, "Dereferencing end iterator");
            FatalAssert(index < current_leaf->num_keys, LOG_TAG_BASIC, "Index out of bounds in iterator dereference");
            threadSelf->SanityCheckLockHeldInModeByMe(current_leaf, SX_SHARED);
            return current_leaf->values[index];
        }

        inline K Key() const {
            FatalAssert(current_leaf != nullptr, LOG_TAG_BASIC, "Dereferencing end iterator");
            FatalAssert(index < current_leaf->num_keys, LOG_TAG_BASIC, "Index out of bounds in iterator dereference");
            threadSelf->SanityCheckLockHeldInModeByMe(current_leaf, SX_SHARED);
            return current_leaf->keys[index];
        }

    protected:
        BPlusTreeLeafNode<K, V>* current_leaf;
        uint8_t index;
    };

    class ConstIterator {
    public:
        ConstIterator(const BPlusTreeLeafNode<K, V>* start_leaf, uint8_t start_index)
            : current_leaf(start_leaf), index(start_index) {
            if (current_leaf != nullptr) {
                threadSelf->SanityCheckLockHeldInModeByMe(current_leaf, SX_SHARED);
            }
        }

        ConstIterator(const Iterator& it)
            : current_leaf(it.current_leaf), index(it.index) {
            if (current_leaf != nullptr) {
                current_leaf->Lock(SX_SHARED);
            }
        }

        ConstIterator(Iterator&& it) noexcept
            : current_leaf(it.current_leaf), index(it.index) {
            it.current_leaf = nullptr;
            it.index = 0;
        }

        ConstIterator(const ConstIterator& it)
            : current_leaf(it.current_leaf), index(it.index) {
            if (current_leaf != nullptr) {
                current_leaf->Lock(SX_SHARED);
            }
        }

        ConstIterator(ConstIterator&& it) noexcept
            : current_leaf(it.current_leaf), index(it.index) {
            it.current_leaf = nullptr;
            it.index = 0;
        }

        ConstIterator& operator=(const Iterator& it) {
            if (this != &it) {
                Release();
                current_leaf = it.current_leaf;
                index = it.index;
                if (current_leaf != nullptr) {
                    current_leaf->Lock(SX_SHARED);
                }
            }
            return *this;
        }

        ConstIterator& operator=(Iterator&& it) noexcept {
            if (this != &it) {
                Release();
                current_leaf = it.current_leaf;
                index = it.index;
                it.current_leaf = nullptr;
                it.index = 0;
            }
            return *this;
        }

        ConstIterator& operator=(const ConstIterator& it) {
            if (this != &it) {
                Release();
                current_leaf = it.current_leaf;
                index = it.index;
                if (current_leaf != nullptr) {
                    current_leaf->Lock(SX_SHARED);
                }
            }
            return *this;
        }

        ConstIterator& operator=(ConstIterator&& it) noexcept {
            if (this != &it) {
                Release();
                current_leaf = it.current_leaf;
                index = it.index;
                it.current_leaf = nullptr;
                it.index = 0;
            }
            return *this;
        }

        inline void Release() {
            if (current_leaf != nullptr) {
                threadSelf->SanityCheckLockHeldInModeByMe(current_leaf, SX_SHARED);
                current_leaf->Unlock(SX_SHARED);
                current_leaf = nullptr;
            }
            index = 0;
        }

        ~ConstIterator() {
            Release();
        }

        inline static ConstIterator End() {
            return ConstIterator(nullptr, 0);
        }

        inline bool HasNext() const {
            return current_leaf != nullptr && (index < current_leaf->num_keys || current_leaf->next_leaf != nullptr);
        }

        inline ConstIterator& operator++() {
            if (current_leaf == nullptr) {
                return *this;
            }

            threadSelf->SanityCheckLockHeldInModeByMe(current_leaf, SX_SHARED);
            index++;
            if (index >= current_leaf->num_keys) {
                if (current_leaf->next_leaf != nullptr) {
                    current_leaf->next_leaf->Lock(SX_SHARED);
                }
                current_leaf->Unlock(SX_SHARED);
                current_leaf = current_leaf->next_leaf;
                index = 0;
            }
            return *this;
        }

        inline bool operator==(const ConstIterator& other) const {
            return current_leaf == other.current_leaf && index == other.index;
        }

        inline bool operator!=(const ConstIterator& other) const {
            return !(*this == other);
        }

        inline bool operator==(const Iterator& other) const {
            return current_leaf == other.current_leaf && index == other.index;
        }

        inline bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

        inline const V& operator*() const {
            FatalAssert(current_leaf != nullptr, LOG_TAG_BASIC, "Dereferencing end iterator");
            FatalAssert(index < current_leaf->num_keys, LOG_TAG_BASIC, "Index out of bounds in iterator dereference");
            threadSelf->SanityCheckLockHeldInModeByMe(current_leaf, SX_SHARED);
            return current_leaf->values[index];
        }

        inline K Key() const {
            FatalAssert(current_leaf != nullptr, LOG_TAG_BASIC, "Dereferencing end iterator");
            FatalAssert(index < current_leaf->num_keys, LOG_TAG_BASIC, "Index out of bounds in iterator dereference");
            threadSelf->SanityCheckLockHeldInModeByMe(current_leaf, SX_SHARED);
            return current_leaf->keys[index];
        }

    protected:
        BPlusTreeLeafNode<K, V>* current_leaf;
        uint8_t index;
    };

    Iterator GetAndInsertIfNeeded(const K& key, const V& value) {
        BPlusTreeLeafNode<K, V>* leaf = nullptr;
        uint8_t index = 0;
        InsertionResult result = GetAndInsertIfNeededOptimistic(key, value, leaf, index);
        if (result == InsertionResult::SPLIT_REQUIRED) {
            result = GetAndInsertIfNeededPessimistic(key, value, leaf, index);
        }
        FatalAssert(result != InsertionResult::SPLIT_REQUIRED, LOG_TAG_BASIC,
                    "Insertion failed even after pessimistic attempt");
        threadSelf->SanityCheckLockHeldInModeByMe(leaf, SX_SHARED);
        return Iterator(leaf, index);
    }

    Iterator Get(const K& key) {
        BPlusTreeNode* current = GetRoot(SX_SHARED);
        CHECK_NOT_NULLPTR(current, LOG_TAG_BASIC);

        while (!current->is_leaf) {
            threadSelf->SanityCheckLockHeldInModeByMe(current, SX_SHARED);
            BPlusTreeInternalNode<K>* current_internal = static_cast<BPlusTreeInternalNode<K>*>(current);
            uint8_t i = 0;
            while (i < current_internal->num_keys && key > current_internal->keys[i]) {
                i++;
            }
            BPlusTreeNode* child = current_internal->children[i];
            CHECK_NOT_NULLPTR(child, LOG_TAG_BASIC);
            child->Lock(SX_SHARED);
            current->Unlock(SX_SHARED);
            current = child;
        }
        CHECK_NOT_NULLPTR(current, LOG_TAG_BASIC);
        FatalAssert(current->is_leaf, LOG_TAG_BASIC, "Current node must be a leaf");
        threadSelf->SanityCheckLockHeldInModeByMe(current, SX_SHARED);
        BPlusTreeLeafNode<K, V>* leaf_node = static_cast<BPlusTreeLeafNode<K, V>*>(current);

        uint8_t i = 0;
        while (i < leaf_node->num_keys && key > leaf_node->keys[i]) {
            i++;
        }

        if (i < leaf_node->num_keys && key == leaf_node->keys[i]) {
            return Iterator(leaf_node, i);
        }

        leaf_node->Unlock(SX_SHARED);
        return Iterator::End();
    }

    Iterator InclusiveUpperBound(const K& key) {
        BPlusTreeNode* current = GetRoot(SX_SHARED);
        CHECK_NOT_NULLPTR(current, LOG_TAG_BASIC);

        while (!current->is_leaf) {
            threadSelf->SanityCheckLockHeldInModeByMe(current, SX_SHARED);
            BPlusTreeInternalNode<K>* current_internal = static_cast<BPlusTreeInternalNode<K>*>(current);
            uint8_t i = 0;
            while (i < current_internal->num_keys && key > current_internal->keys[i]) {
                i++;
            }
            BPlusTreeNode* child = current_internal->children[i];
            CHECK_NOT_NULLPTR(child, LOG_TAG_BASIC);
            child->Lock(SX_SHARED);
            current->Unlock(SX_SHARED);
            current = child;
        }
        CHECK_NOT_NULLPTR(current, LOG_TAG_BASIC);
        FatalAssert(current->is_leaf, LOG_TAG_BASIC, "Current node must be a leaf");
        threadSelf->SanityCheckLockHeldInModeByMe(current, SX_SHARED);
        BPlusTreeLeafNode<K, V>* leaf_node = static_cast<BPlusTreeLeafNode<K, V>*>(current);

        while (leaf_node != nullptr) {
            uint8_t i = 0;
            while (i < leaf_node->num_keys && key > leaf_node->keys[i]) {
                i++;
            }
            if (i < leaf_node->num_keys) {
                return Iterator(leaf_node, i);
            }
            BPlusTreeLeafNode<K, V>* next_leaf = leaf_node->next_leaf;
            if (next_leaf != nullptr) {
                next_leaf->Lock(SX_SHARED);
            }
            leaf_node->Unlock(SX_SHARED);
            leaf_node = next_leaf;
        }

        return Iterator::End();
    }

    bool Remove(const K& key, V& value) {
        K key_to_remove = key;
        std::vector<BPlusTreeNode*> path;
        path.reserve(32);
        BPlusTreeNode* current = GetRoot(SX_EXCLUSIVE);

        while (!current->is_leaf) {
            threadSelf->SanityCheckLockHeldInModeByMe(current, SX_EXCLUSIVE);
            BPlusTreeInternalNode<K>* current_internal = static_cast<BPlusTreeInternalNode<K>*>(current);
            uint8_t i = 0;
            while (i < current_internal->num_keys && key_to_remove > current_internal->keys[i]) {
                i++;
            }

            if (i < current_internal->num_keys && key_to_remove == current_internal->keys[i]) {
                path.push_back(current);
                current = current_internal->RemoveKey(i, key_to_remove);
                current->Lock(SX_EXCLUSIVE);
                continue;
            }

            BPlusTreeNode* child = current_internal->children[i];
            CHECK_NOT_NULLPTR(child, LOG_TAG_BASIC);

            bool node_fully_traversed = (i == current_internal->num_keys);

            if (child->num_keys == child->GetMinKeys()) {
                current_internal->FillChild(i);
            }

            if (node_fully_traversed && i > current_internal->num_keys) {
                i--;
            }
            path.push_back(current);
            current = current_internal->children[i];
            current->Lock(SX_EXCLUSIVE);
        }
        CHECK_NOT_NULLPTR(current, LOG_TAG_BASIC);
        FatalAssert(current->is_leaf, LOG_TAG_BASIC, "Current node must be a leaf");
        threadSelf->SanityCheckLockHeldInModeByMe(current, SX_EXCLUSIVE);

        BPlusTreeLeafNode<K, V>* current_leaf = static_cast<BPlusTreeLeafNode<K, V>*>(current);
        uint8_t i = 0;
        while (i < current_leaf->num_keys && key_to_remove > current_leaf->keys[i]) {
            i++;
        }

        bool removed;
        if (i < current_leaf->num_keys && key_to_remove == current_leaf->keys[i]) {
            current_leaf->Remove(i, value);
            tree_size--;
            removed = true;
        } else {
            removed = false;
        }

        current_leaf->Unlock(SX_EXCLUSIVE);
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            BPlusTreeNode* node = *it;
            node->Unlock(SX_EXCLUSIVE);
        }
        return removed;
    }

    inline size_t Size() const {
        return tree_size.load(std::memory_order_acquire);
    }

    inline Iterator begin() {
        leftmost_leaf->Lock(SX_SHARED);
        return Iterator(leftmost_leaf, 0);
    }

    inline Iterator end() {
        return Iterator::End();
    }

    inline ConstIterator begin() const {
        leftmost_leaf->Lock(SX_SHARED);
        return ConstIterator(leftmost_leaf, 0);
    }

    inline ConstIterator end() const {
        return ConstIterator::End();
    }

protected:
    std::atomic<BPlusTreeNode*> root;
    BPlusTreeLeafNode<K, V> * const leftmost_leaf;
    std::atomic<size_t> tree_size;


    BPlusTreeNode* GetRoot(LockMode mode) {
        while (true) {
            BPlusTreeNode* current_root = root.load(std::memory_order_acquire);
            CHECK_NOT_NULLPTR(current_root, LOG_TAG_BASIC);
            current_root->Lock(mode);
            if (current_root == root.load(std::memory_order_acquire)) {
                return current_root;
            }
            current_root->Unlock(mode);
            DIVFTREE_YIELD();
        }
    }

    /* both parent and child should be locked in X mode -> child will be unlocked at the end */
    void SplitChild(BPlusTreeInternalNode<K>* parent, uint8_t index) {
        CHECK_NOT_NULLPTR(parent, LOG_TAG_BASIC);
        FatalAssert(!parent->is_leaf, LOG_TAG_BASIC, "Parent node must be an internal node");
        FatalAssert(index < parent->num_keys + 1, LOG_TAG_BASIC, "Index out of bounds for splitting child");
        FatalAssert(parent->num_keys > 0, LOG_TAG_BASIC, "Parent node must have at least one key to split child");
        threadSelf->SanityCheckLockHeldInModeByMe(parent, SX_EXCLUSIVE);
        BPlusTreeNode* child = parent->children[index];
        threadSelf->SanityCheckLockHeldInModeByMe(child, SX_EXCLUSIVE);
        CHECK_NOT_NULLPTR(child, LOG_TAG_BASIC);
        BPlusTreeNode* sibling = nullptr;
        K new_key;
        if (child->is_leaf) {
            BPlusTreeLeafNode<K, V>* child_leaf = static_cast<BPlusTreeLeafNode<K, V>*>(child);
            BPlusTreeLeafNode<K, V>* sibling_leaf = new BPlusTreeLeafNode<K, V>();
            sibling_leaf->Lock(SX_EXCLUSIVE);
            sibling = sibling_leaf;
            sibling_leaf->num_keys = BPlusTreeLeafNode<K, V>::MinKeys;
            for (uint8_t j = 0; j < BPlusTreeLeafNode<K, V>::MinKeys; j++) {
                sibling_leaf->keys[j] = child_leaf->keys[j + BPlusTreeLeafNode<K, V>::MinKeys + 1];
                sibling_leaf->values[j] = child_leaf->values[j + BPlusTreeLeafNode<K, V>::MinKeys + 1];
            }
            sibling_leaf->next_leaf = child_leaf->next_leaf;
            child_leaf->next_leaf = sibling_leaf;
            child_leaf->num_keys = BPlusTreeLeafNode<K, V>::MinKeys;
            new_key = child_leaf->keys[BPlusTreeLeafNode<K, V>::MinKeys];
        } else {
            BPlusTreeInternalNode<K>* child_internal = static_cast<BPlusTreeInternalNode<K>*>(child);
            BPlusTreeInternalNode<K>* sibling_internal = new BPlusTreeInternalNode<K>();
            sibling_internal->Lock(SX_EXCLUSIVE);
            sibling = sibling_internal;
            sibling_internal->num_keys = BPlusTreeInternalNode<K>::MinKeys;
            for (uint8_t j = 0; j < BPlusTreeInternalNode<K>::MinKeys; j++) {
                sibling_internal->keys[j] = child_internal->keys[j + BPlusTreeInternalNode<K>::MinKeys + 1];
                sibling_internal->children[j] = child_internal->children[j + BPlusTreeInternalNode<K>::MinKeys + 1];
            }
            sibling_internal->children[BPlusTreeInternalNode<K>::MinKeys] =
                child_internal->children[BPlusTreeInternalNode<K>::MinKeys * 2 + 1];
            child_internal->num_keys = BPlusTreeInternalNode<K>::MinKeys;
            new_key = child_internal->keys[BPlusTreeInternalNode<K>::MinKeys];
        }

        // Move parent's child pointer and key to make space for sibling
        for (int8_t j = parent->num_keys; j > (int8_t)index; --j) {
            parent->children[j + 1] = parent->children[j];
            parent->keys[j] = parent->keys[j - 1];
        }

        parent->children[index + 1] = sibling;
        parent->keys[index] = new_key;
        parent->num_keys++;
        sibling->Unlock(SX_EXCLUSIVE);
        child->Unlock(SX_EXCLUSIVE);
    }

    InsertionResult GetAndInsertIfNeededOptimistic(const K& key, const V& value,
                                                   BPlusTreeLeafNode<K, V>*& container_leaf, uint8_t& index) {
        FatalAssert(container_leaf == nullptr, LOG_TAG_BASIC,
                    "Container leaf must be null for optimistic insertion");
        BPlusTreeNode* parent = nullptr;
        BPlusTreeNode* current = GetRoot(SX_SHARED);
        CHECK_NOT_NULLPTR(current, LOG_TAG_BASIC);

        while (!current->is_leaf) {
            threadSelf->SanityCheckLockHeldInModeByMe(current, SX_SHARED);
            BPlusTreeInternalNode<K>* current_internal = static_cast<BPlusTreeInternalNode<K>*>(current);
            uint8_t i = 0;
            while (i < current_internal->num_keys && key > current_internal->keys[i]) {
                i++;
            }
            BPlusTreeNode* child = current_internal->children[i];
            CHECK_NOT_NULLPTR(child, LOG_TAG_BASIC);
            child->Lock(SX_SHARED);
            if (parent != nullptr) {
                parent->Unlock(SX_SHARED);
            }
            parent = current_internal;
            current = child;
        }
        CHECK_NOT_NULLPTR(current, LOG_TAG_BASIC);
        FatalAssert(current->is_leaf, LOG_TAG_BASIC, "Current node must be a leaf");
        threadSelf->SanityCheckLockHeldInModeByMe(current, SX_SHARED);
        BPlusTreeLeafNode<K, V>* leaf_node = static_cast<BPlusTreeLeafNode<K, V>*>(current);

        uint8_t i = 0;
        while (i < leaf_node->num_keys && key > leaf_node->keys[i]) {
            i++;
        }

        if (i < leaf_node->num_keys && key == leaf_node->keys[i]) {
            if (parent != nullptr) {
                parent->Unlock(SX_SHARED);
            }
            index = i;
            container_leaf = leaf_node;
            return InsertionResult::EXISTS;
        }

        if (leaf_node->num_keys == BPlusTreeLeafNode<K, V>::MaxKeys) {
            if (parent != nullptr) {
                parent->Unlock(SX_SHARED);
            }
            leaf_node->Unlock(SX_SHARED);
            return InsertionResult::SPLIT_REQUIRED;
        }

        if (!leaf_node->UpgradeLock()) {
            leaf_node->Unlock(SX_SHARED);
            DIVFTREE_YIELD();
            leaf_node->Lock(SX_EXCLUSIVE);

            uint8_t i = 0;
            while (i < leaf_node->num_keys && key > leaf_node->keys[i]) {
                i++;
            }

            if (i < leaf_node->num_keys && key == leaf_node->keys[i]) {
                if (parent != nullptr) {
                    parent->Unlock(SX_SHARED);
                }
                leaf_node->DowngradeLock();
                index = i;
                container_leaf = leaf_node;
                return InsertionResult::EXISTS;
            }

            if (leaf_node->num_keys == BPlusTreeLeafNode<K, V>::MaxKeys) {
                if (parent != nullptr) {
                    parent->Unlock(SX_SHARED);
                }
                leaf_node->Unlock(SX_EXCLUSIVE);
                return InsertionResult::SPLIT_REQUIRED;
            }
        }

        index = leaf_node->InsertNonFull(key, value);
        if (parent != nullptr) {
            parent->Unlock(SX_SHARED);
        }
        leaf_node->DowngradeLock();
        tree_size.fetch_add(1, std::memory_order_acq_rel);
        container_leaf = leaf_node;
        return InsertionResult::INSERTED;
    }

    InsertionResult GetAndInsertIfNeededPessimistic(const K& key, const V& value,
                                                    BPlusTreeNode<K, V>*& container_leaf, uint8_t& index) {
        FatalAssert(container_leaf == nullptr, LOG_TAG_BASIC,
                    "Container leaf must be null for pessimistic insertion");
        std::vector<BPlusTreeNode*> path;
        path.reserve(32);
        BPlusTreeNode* current = GetRoot(SX_EXCLUSIVE);

        if (current->num_keys == current->GetMaxKeys()) {
            BPlusTreeInternalNode<K>* new_root = new BPlusTreeInternalNode<K>();
            new_root->children[0] = root;
            new_root->Lock(SX_EXCLUSIVE);
            root.store(new_root, std::memory_order_release);
            SplitChild(new_root, 0);
            current = new_root;
        }
        CHECK_NOT_NULLPTR(current, LOG_TAG_BASIC);

        while (!current->is_leaf) {
            threadSelf->SanityCheckLockHeldInModeByMe(current, SX_EXCLUSIVE);
            BPlusTreeInternalNode<K>* internal_node = static_cast<BPlusTreeInternalNode<K>*>(current);
            FatalAssert(internal_node->num_keys > 0, LOG_TAG_BASIC,
                        "Internal node has no keys during insertion traversal");
            FatalAssert(internal_node->num_keys < BPlusTreeInternalNode<K>::MaxKeys,
                        LOG_TAG_BASIC, "Internal node is full during insertion traversal");
            uint8_t i = 0;
            while (i < internal_node->num_keys && key > internal_node->keys[i]) {
                i++;
            }
            BPlusTreeNode* child = internal_node->children[i];
            CHECK_NOT_NULLPTR(child, LOG_TAG_BASIC);
            child->Lock(SX_EXCLUSIVE);

            if (child->num_keys == child->GetMaxKeys()) {
                SplitChild(internal_node, i);
                continue;
            }
            path.push_back(current);
            current = internal_node->children[i];
            current->Lock(SX_EXCLUSIVE);
        }
        CHECK_NOT_NULLPTR(current, LOG_TAG_BASIC);
        FatalAssert(current->is_leaf, LOG_TAG_BASIC, "Current node must be a leaf");
        FatalAssert(current->num_keys > 0, LOG_TAG_BASIC,
                        "Internal node has no keys during insertion traversal");
        FatalAssert(current->num_keys < BPlusTreeLeafNode<K, V>::MaxKeys,
                    LOG_TAG_BASIC, "Internal node is full during insertion traversal");
        threadSelf->SanityCheckLockHeldInModeByMe(current, SX_EXCLUSIVE);
        BPlusTreeLeafNode<K, V>* leaf_node = static_cast<BPlusTreeLeafNode<K, V>*>(current);

        uint8_t i = 0;
        while (i < leaf_node->num_keys && key > leaf_node->keys[i]) {
            i++;
        }

        if (i < leaf_node->num_keys && key == leaf_node->keys[i]) {
            leaf_node->DowngradeLock();
            index = i;
            container_leaf = leaf_node;
            for (auto it = path.rbegin(); it != path.rend(); ++it) {
                BPlusTreeNode* node = *it;
                node->Unlock(SX_EXCLUSIVE);
            }
            return InsertionResult::EXISTS;
        }

        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            BPlusTreeNode* node = *it;
            node->Unlock(SX_EXCLUSIVE);
        }

        index = leaf_node->InsertNonFull(key, value);

        leaf_node->DowngradeLock();
        tree_size.fetch_add(1, std::memory_order_acq_rel);
        container_leaf = leaf_node;
        return InsertionResult::INSERTED;
    }
};

};

#endif