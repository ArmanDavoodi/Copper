#ifndef STRING_H_
#define STRING_H_

#include <cstring>
#include <stdarg.h>
#include <stdio.h>
#include <string>

namespace divftree {

class String {
public:
    String() : _data(nullptr), size(0), capacity(0) {}

    String(size_t cap) : _data(cap > 0 ? static_cast<char*>(malloc((cap + 1) * sizeof(char))) : nullptr),
                         size(0), capacity(cap) {
        if (_data != nullptr) {
            _data[0] = 0;
        }
    }

    String(const String& other) : _data(((other.capacity == 0) ?
                                  nullptr : static_cast<char*>(malloc((other.capacity + 1) * sizeof(char))))),
                                  size(other.size), capacity(other.capacity) {
        if (size > 0) {
            memcpy(_data, other._data, sizeof(char) * (size));
        }

        if (other.capacity > 0) {
            _data[size] = 0;
        }
    }

    String(String&& other) : _data(other._data), size(other.size), capacity(other.capacity) {
        other._data = nullptr;
        other.size = 0;
        other.capacity = 0;
    }

    String(const std::string& other) :  _data(((other.size() == 0) ?
                                        nullptr : static_cast<char*>(malloc(((other.size() * 2) + 1) * sizeof(char))))),
                                        size(other.size()), capacity(other.size() * 2) {
        if (size > 0) {
            memcpy(_data, &(other[0]), sizeof(char) * (size));
            _data[size] = 0;
        }
    }

    String(const char* fmt, ...) : _data(nullptr), size(0), capacity(0) {
        if (fmt == nullptr) {
            return;
        }

        va_list argptr;
        va_start(argptr, fmt);

        va_list arg_copy;
        va_copy(arg_copy, argptr);
        int _size = vsnprintf(NULL, 0, fmt, arg_copy);
        va_end(arg_copy);
        if (_size < 0) {
            size = 0;
            return;
        }
        size = _size;
        capacity = size * 2;
        _data = static_cast<char*>(malloc((capacity + 1) * sizeof(char)));

        int num_writen = vsnprintf(_data, size + 1, fmt, argptr);
        va_end(argptr);
        if ((num_writen < 0) || ((size_t)(num_writen) != size)) {
            free(_data);
            _data = nullptr;
            size = 0;
            capacity = 0;
            return;
        }
        _data[size] = 0;
    }

    String& operator=(const String& other) {
        if (other.SameAs(*this)) {
            return *this;
        }

        if (other.size == 0) {
            if (_data != nullptr) {
                size = 0;
                _data[0] = 0;
            }
            return *this;
        }

        if (_data != nullptr) {
            if (capacity < other.size) {
                Reserve(other.size * 2);
            }
        }

        size = other.size;
        memcpy(_data, other._data, sizeof(char) * size);
        _data[size] = 0;

        return *this;
    }

    String& operator=(String&& other) {
        if (other.SameAs(*this)) {
            return *this;
        }

        if (other.size == 0 && other.capacity < capacity) {
            if (other._data != nullptr) {
                free(other._data);
                other._data = nullptr;
            }
            other.capacity = 0;

            size = 0;
            if (_data != nullptr) {
                _data[size] = 0;
            }
            return *this;
        }

        if (_data != nullptr) {
            free(_data);
            _data = nullptr;
            size = 0;
            capacity = 0;
        }

        size = other.size;
        _data = other._data;
        capacity = other.capacity;
        other._data = nullptr;
        other.size = 0;
        other.capacity = 0;

        return *this;
    }

    String& operator=(const char* other) {
        size_t other_size = (other != nullptr) ? strlen(other) : 0;
        if (other_size == 0) {
            size = 0;
            if (_data != nullptr) {
                _data[0] = 0;
            }
            return *this;
        }

        if (capacity > other_size) {
            size = other_size;
            memcpy(_data, other, sizeof(char) * size);
            _data[size] = 0;
            return *this;
        }

        if (_data != nullptr) {
            free(_data);
            _data = nullptr;
        }

        capacity = other_size * 2;
        size = other_size;
        _data = static_cast<char*>(malloc((capacity + 1) * sizeof(char)));
        memcpy(_data, other, sizeof(char) * size);
        _data[size] = 0;
        return *this;
    }

    String& operator=(const std::string& other) {
        return (*this = other.c_str());
    }

    ~String() {
        if (_data != nullptr) {
            free(_data);
            _data = nullptr;
            size = 0;
            capacity = 0;
        }
    }

    inline char& operator[](size_t index) {
        return _data[index];
    }

    inline const char& operator[](size_t index) const {
        return _data[index];
    }

    inline String operator+(const String& other) const {
        size_t new_size = size + other.size;
        size_t new_cap = new_size * 2;
        String res(new_cap);
        res.size = new_size;
        if (res.size == 0) {
            return res;
        }

        size_t offset = 0;
        if (size > 0) {
            memcpy(res._data, _data, sizeof(char) * size);
            offset = size;
        }

        if (other.size > 0) {
            memcpy(res._data + offset, other._data, sizeof(char) * other.size);
        }

        res._data[res.size] = 0;
        return res;
    }

    inline String operator+(String&& other) const {
        if (size == 0) {
            return String(std::move(other));
        }

        if (other.size == 0) {
            return String(*this);
        }

        String res = (*this + other);
        if (other._data != nullptr) {
            free(other._data);
            other._data = nullptr;
        }
        other.size = 0;
        other.capacity = 0;
        return res;
    }

    inline String& operator+=(const String& other) {
        if (other.size == 0) {
            return *this;
        }

        size_t new_size = size + other.size;
        if (capacity < new_size) {
            Reserve(new_size * 2);
        }
        memcpy(_data + size, other._data, sizeof(char) * other.size);
        size = new_size;
        _data[size] = 0;
        return *this;
    }

    inline String& operator+=(String&& other) {
        if (other.size == 0) {
            return *this;
        }

        if (size == 0) {
            return (*this = std::move(other));
        }

        size_t new_size = size + other.size;
        if (capacity < new_size) {
            Reserve(new_size * 2);
        }
        memcpy(_data + size, other._data, sizeof(char) * other.size);
        size = new_size;
        _data[size] = 0;
        return *this;
    }

    inline bool SameAs(const String& other) const {
        return (_data == other._data);
    }

    inline bool operator==(const String& other) const {
        if (SameAs(other)) {
            return true;
        }

        if (other._data == nullptr || _data == nullptr) {
            return false;
        }

        return !(strcmp(_data, other._data));
    }

    inline bool operator!=(const String& other) const {
        return !(*this == other);
    }

    inline bool operator>(const String& other) const {
        if (SameAs(other)) {
            return false;
        }

        if (other._data == nullptr) {
            return true;
        }

        if (_data == nullptr) {
            return false;
        }

        return (strcmp(_data, other._data) > 0);
    }

    inline bool operator<(const String& other) const {
        return (other > *this);
    }

    inline bool operator<=(const String& other) const {
        return ((*this < other) || (*this == other));
    }

    inline bool operator>=(const String& other) const {
        return ((*this > other) || (*this == other));
    }

    inline void Reserve(size_t cap) {
        if (cap <= capacity) {
            return;
        }

        char* new_data = static_cast<char*>(malloc((cap + 1) * sizeof(char)));
        if (_data != nullptr) {
            memcpy(new_data, _data, sizeof(char) * size);
            free(_data);
        }
        _data = new_data;
        capacity = cap;
        _data[size] = 0;
    }

    inline void ShrinkToFit() {
        if (capacity == size) {
            return;
        }

        if (size == 0) {
            if (_data != nullptr) {
                free(_data);
                _data = nullptr;
            }
            capacity = 0;
            return;
        }

        char* new_data = static_cast<char*>(malloc((size + 1) * sizeof(char)));
        memcpy(new_data, _data, sizeof(char) * size);
        free(_data);
        _data = new_data;
        capacity = size;
        _data[size] = 0;
    }

    inline void Clear() {
        size = 0;
        if (_data != nullptr) {
            _data[0] = 0;
        }
    }

    /* new_size of 0 means no change to the vector */
    inline String& FitInBox(size_t new_size) {
        if (new_size == size || new_size == 0) {
            return *this;
        }

        char* new_data = static_cast<char*>(malloc((new_size + 1) * sizeof(char)));
        capacity = new_size;
        if (new_size < size) {
            memcpy(new_data, _data, new_size * sizeof(char));
        }
        else {
            size_t prefix_spaces = (new_size - size) / 2;
            size_t suffix_spaces = new_size - size - prefix_spaces;
            memset(new_data, ' ', prefix_spaces * sizeof(char));
            if (_data != nullptr) {
                memcpy(new_data + prefix_spaces, _data, size * sizeof(char));
            }
            memset(new_data + prefix_spaces + size, ' ', suffix_spaces * sizeof(char));
        }

        if (_data != nullptr) {
            free(_data);
        }
        new_data[new_size] = 0; // Null-terminate the string
        _data = new_data;
        size = new_size;

        return *this;
    }

    const char* ToCStr() const {
        return (_data ? _data : "");
    }
protected:
    char* _data;
    size_t size;
    size_t capacity;
};

};

#endif