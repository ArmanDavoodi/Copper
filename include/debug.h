#ifndef DIVFTREE_DEBUG_H_
#define DIVFTREE_DEBUG_H_

#include <cstdint>

#ifdef TESTING
#undef BUILD
#define BUILD DEBUG
#define FAULT_INJECTION // todo: add to compile time flags

#define TESTABLE friend class UT::Test
#else
#define TESTABLE
#endif

#if !defined(BUILD) || (BUILD!=DEBUG && BUILD!=RELEASE)
#error "BUILD must be defined as DEBUG or RELEASE"
#endif

// might add other modes as well later and consider not completly disabling the logs in release
#if BUILD==DEBUG
#define ENABLE_TEST_LOGGING
#define ENABLE_ASSERTS
#define LOCK_DEBUG
#define LOG_FUNCTION_NAME
#elif BUILD==RELEASE
#define ENABLE_ASSERTS
#endif

// Todo: handle during compile
#ifdef ENABLE_TEST_LOGGING
#define ENABLE_FAULT_LOGGING
#endif

#define UNUSED_VARIABLE(x) (void)(x)

enum LOG_LEVELS : uint8_t {
    LOG_LEVEL_ZERO,
    LOG_LEVEL_PANIC,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_LOG,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_NUM
};

enum LOG_TAG_BITS : uint64_t {
    LOG_TAG_BASIC_BIT,
    LOG_TAG_VECTOR_BIT,
    LOG_TAG_CLUSTER_BIT,
    LOG_TAG_BUFFER_BIT,
    LOG_TAG_DIVFTREE_VERTEX_BIT,
    LOG_TAG_DIVFTREE_BIT,
    LOG_TAG_CLUSTERING_BIT,
    LOG_TAG_MEMORY_BIT,
    LOG_TAG_LOCK_BIT,
    LOG_TAG_THREAD_BIT,
    LOG_TAG_HANG_DETECTOR_BIT,
    LOG_TAG_NOT_IMPLEMENTED_BIT,
    LOG_TAG_TEST_BIT,
    NUM_TAGS
};

#define LOG_TAG_BASIC (1ul << (uint64_t)(LOG_TAG_BITS::LOG_TAG_BASIC_BIT))
#define LOG_TAG_VECTOR (1ul << (uint64_t)(LOG_TAG_BITS::LOG_TAG_VECTOR_BIT))
#define LOG_TAG_CLUSTER (1ul << (uint64_t)(LOG_TAG_BITS::LOG_TAG_CLUSTER_BIT))
#define LOG_TAG_BUFFER (1ul << (uint64_t)(LOG_TAG_BITS::LOG_TAG_BUFFER_BIT))
#define LOG_TAG_DIVFTREE_VERTEX (1ul << (uint64_t)(LOG_TAG_BITS::LOG_TAG_DIVFTREE_VERTEX_BIT))
#define LOG_TAG_DIVFTREE (1ul << (uint64_t)(LOG_TAG_BITS::LOG_TAG_DIVFTREE_BIT))
#define LOG_TAG_CLUSTERING (1ul << (uint64_t)(LOG_TAG_BITS::LOG_TAG_CLUSTERING_BIT))
#define LOG_TAG_MEMORY (1ul << (uint64_t)(LOG_TAG_BITS::LOG_TAG_MEMORY_BIT))
// this is used for all thread locks so do not use it for logs regarding locking and unlocking as it will mess with the tag filter
#define LOG_TAG_LOCK (1ul << (uint64_t)(LOG_TAG_BITS::LOG_TAG_LOCK_BIT))
#define LOG_TAG_THREAD (1ul << (uint64_t)(LOG_TAG_BITS::LOG_TAG_THREAD_BIT))
#define LOG_TAG_HANG_DETECTOR (1ul << (uint64_t)(LOG_TAG_BITS::LOG_TAG_HANG_DETECTOR_BIT))
#define LOG_TAG_NOT_IMPLEMENTED (1ul << (uint64_t)(LOG_TAG_BITS::LOG_TAG_NOT_IMPLEMENTED_BIT))
#define LOG_TAG_TEST (1ul << (uint64_t)(LOG_TAG_BITS::LOG_TAG_TEST_BIT))

#define LOG_TAG_ANY (-1ul)

#ifndef LOG_MIN_LEVEL
#define LOG_MIN_LEVEL (LOG_LEVEL_PANIC)
#endif

#ifndef LOG_LEVEL
#define LOG_LEVEL (LOG_LEVEL_LOG)
#endif

#ifndef LOG_TAG
#define LOG_TAG (LOG_TAG_ANY)
#endif

#ifndef PRINT_CALLSTACK_LEVEL
#define PRINT_CALLSTACK_LEVEL (LOG_LEVEL_ZERO)
#endif

#ifndef PRINT_CALLSTACK_MAX_FRAMES
#define PRINT_CALLSTACK_MAX_FRAMES (10)
#endif

#ifdef PRINT_FUNCTION_NAME_PRETTY
#define FUNCTION_NAME __PRETTY_FUNCTION__
#define FUNCTION_NAME_MAX_SIZE 50
#else
#define FUNCTION_NAME __FUNCTION__
#define FUNCTION_NAME_MAX_SIZE 35
#endif

#define FILE_NAME_MAX_SIZE 64

#define _COLORF_BLACK "\033[0;30m"
#define _COLORF_RED "\033[0;31m"
#define _COLORF_GREEN "\033[0;32m"
#define _COLORF_YELLOW "\033[0;33m"
#define _COLORF_BLUE "\033[0;34m"
#define _COLORF_PURPLE "\033[0;35m"
#define _COLORF_CYAN "\033[0;36m"
#define _COLORF_WHITE "\033[0;37m"
#define _COLORF_RESET "\033[0m"

#ifdef COLORED
#define COLORF_BLACK _COLORF_BLACK
#define COLORF_RED _COLORF_RED
#define COLORF_GREEN _COLORF_GREEN
#define COLORF_YELLOW _COLORF_YELLOW
#define COLORF_BLUE _COLORF_BLUE
#define COLORF_PURPLE _COLORF_PURPLE
#define COLORF_CYAN _COLORF_CYAN
#define COLORF_WHITE _COLORF_WHITE
#define COLORF_RESET _COLORF_RESET
#else
#define COLORF_BLACK
#define COLORF_RED
#define COLORF_GREEN
#define COLORF_YELLOW
#define COLORF_BLUE
#define COLORF_PURPLE
#define COLORF_CYAN
#define COLORF_WHITE
#define COLORF_RESET
#endif

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <assert.h>
#include <thread>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <stdarg.h>
#include <string.h>
#include <atomic>
#include <semaphore>
#include <map>

#include "utils/string.h"
#include "utils/thread.h"

#ifdef __clang__
#include <experimental/source_location>
    using SourceLoc = std::experimental::source_location;
#else
#include <source_location>
    using SourceLoc = std::source_location;
#endif


#ifdef OS_THREAD_ID
#undef OS_THREAD_ID
#endif

#ifdef THREAD_ID
#undef THREAD_ID
#endif

#ifdef DIVF_THREAD_ID
#undef DIVF_THREAD_ID
#endif

#ifdef _WIN32
#include <windows.h> // For GetCurrentThreadId
using os_thread_id = DWORD;

#define OS_THREAD_ID ((::divftree::os_thread_id)(GetCurrentThreadId()))

std::string print_callstack() {
    /* Not implemented */
    return "";
}

#else // Assume POSIX (Linux, macOS, etc.)
#include <unistd.h>  // For syscall
#include <sys/syscall.h> // For SYS_gettid (Linux)
// On macOS, pthread_threadid_np could be used, but gettid is more common
// for what GDB shows on Linux-like systems.
// For a truly portable solution you might need #ifdef __APPLE__ and
// use pthread_self() and a debugger-specific mapping or pthread_getthreadid_np()

#include <execinfo.h>
#include <cxxabi.h>

using os_thread_id = pid_t; // pid_t is typically int or long

// SYS_gettid returns the kernel thread ID (LWP ID)
#define OS_THREAD_ID ((os_thread_id)(syscall(SYS_gettid)))

std::string print_callstack() {
    return "";
    // std::string callstack = "[";
    // void* frames[PRINT_CALLSTACK_MAX_FRAMES];
    // int numFrames = backtrace(frames, PRINT_CALLSTACK_MAX_FRAMES);

    // // Get symbolic names
    // std::unique_ptr<char*, decltype(&free)> symbols(
    //     backtrace_symbols(frames, numFrames),
    //     free
    // );

    // if (!symbols) {
    //     callstack += "Error retrieving callstack]";
    //     return callstack;
    // }

    // // Iterate through the symbols and demangle C++ names
    // for (int i = 0; i < numFrames; ++i) {
    //     std::string symbolStr = symbols.get()[i];

    //     // Attempt to demangle C++ names
    //     size_t start_mangled = symbolStr.find('(');
    //     size_t end_mangled = symbolStr.find('+', start_mangled);

    //     if (start_mangled != std::string::npos && end_mangled != std::string::npos) {
    //         std::string mangled_name = symbolStr.substr(start_mangled + 1, end_mangled - (start_mangled + 1));
    //         int status;
    //         std::unique_ptr<char, decltype(&free)> demangled_name(
    //             abi::__cxa_demangle(mangled_name.c_str(), 0, 0, &status),
    //             free
    //         );

    //         if (status == 0 && demangled_name) {
    //             // Successfully demangled
    //             callstack += symbolStr.substr(0, start_mangled + 1) + demangled_name.get() +
    //                          symbolStr.substr(end_mangled);
    //         }
    //         else {
    //             // Demangling failed or not a mangled name
    //             callstack += symbolStr;
    //         }
    //     }
    //     else {
    //         callstack += symbolStr;
    //     }
    //     if (i != numFrames - 1) {
    //         callstack += ", ";
    //     }
    // }

    // return callstack + "]";
}

#endif
namespace divftree {

using thread_id = unsigned long;
#define THREAD_ID ((::divftree::thread_id)(OS_THREAD_ID))
#define DIVF_THREAD_ID ((::divftree::threadSelf != nullptr ? ::divftree::threadSelf->ID() :\
                                                           ::divftree::INVALID_DIVF_THREAD_ID))

#define TYPE_ALIGNED(ptr, alignment) \
    ((((uintptr_t)(ptr)) % (alignment)) == 0)

struct Log_Msg {
    char *_msg = nullptr;

    Log_Msg(const char* msg, ...) {
        va_list argptr;
        va_start(argptr, msg);
        va_list arg_copy;
        va_copy(arg_copy, argptr);
        int num_writen = vsnprintf(NULL, 0, msg, arg_copy);
        va_end(arg_copy);
        if (num_writen < 0) {
            sprintf(_msg, "Error %d in logging: %s", errno, strerror(errno));
            va_end(argptr);
            return;
        }
        // _msg = new char[num_writen + 1];
        _msg = static_cast<char*>(malloc((num_writen + 1) * sizeof(char)));
        num_writen = vsnprintf(_msg, num_writen + 1, msg, argptr);
        va_end(argptr);

        if (num_writen < 0) {
            sprintf(_msg, "Error %d in logging: %s", errno, strerror(errno));
        }
    }

    Log_Msg(const Log_Msg& other) {
        if (other._msg == nullptr) {
            _msg = nullptr;
            return;
        }
        size_t size = strlen(other._msg);
        if (size == 0) {
            _msg = nullptr;
            return;
        }
        // _msg = new char[size + 1];
        _msg = static_cast<char*>(malloc((size + 1) * sizeof(char)));
        _msg[0] = 0;
        strcpy(_msg, other._msg);
    }

    Log_Msg(Log_Msg&& other) : _msg(other._msg) {
        other._msg = nullptr;
    }

    ~Log_Msg() {
        if (_msg != nullptr) {
            free(_msg);
            _msg = nullptr;
        }
    }
};

inline void nsleep(uint64_t nsec) {
    std::this_thread::sleep_for(std::chrono::nanoseconds(nsec));
}

inline void usleep(uint64_t usec) {
    std::this_thread::sleep_for(std::chrono::microseconds(usec));
}

inline void msleep(uint64_t msec) {
    std::this_thread::sleep_for(std::chrono::milliseconds(msec));
}

inline void sleep(uint64_t sec) {
    std::this_thread::sleep_for(std::chrono::seconds(sec));
}

inline void timetostr(const std::chrono::_V2::system_clock::time_point _time, char* mbstr)
{
    std::time_t _t = std::chrono::system_clock::to_time_t(_time);
    auto _m = _time - std::chrono::time_point_cast<std::chrono::seconds>(_time);
    size_t num_chars = std::strftime(mbstr, 100, "%F:%T", std::localtime(&_t));
    sprintf(mbstr + num_chars, ".%lu", std::chrono::duration_cast<std::chrono::microseconds>(_m).count());
}

inline const char* leveltostr(LOG_LEVELS level, bool fault_checking = false)
{
    switch (level)
    {
    case LOG_LEVEL_PANIC:
        return COLORF_PURPLE (fault_checking ? "  FAULT_CHECKING(PANIC)  " : "          PANIC          ") COLORF_RESET;
    case LOG_LEVEL_ERROR:
        return COLORF_RED    (fault_checking ? "  FAULT_CHECKING(ERROR)  " : "          ERROR          ") COLORF_RESET;
    case LOG_LEVEL_WARNING:
        return COLORF_YELLOW (fault_checking ? " FAULT_CHECKING(WARNING) " : "         WARNING         ") COLORF_RESET;
    case LOG_LEVEL_LOG:
        return COLORF_CYAN   (fault_checking ? "   FAULT_CHECKING(LOG)   " : "           LOG           ") COLORF_RESET;
    case LOG_LEVEL_DEBUG:
        return COLORF_GREEN  (fault_checking ? "  FAULT_CHECKING(DEBUG)  " : "          DEBUG          ") COLORF_RESET;
    default:
        return               (fault_checking ? "FAULT_CHECKING(UNDEFINED)" : "        UNDEFINED        ");
    }
}

inline const char* tagtostr(uint64_t tag)
{
    switch (tag)
    {
    case LOG_TAG_BASIC:
        return "     Basic     ";
    case LOG_TAG_VECTOR:
        return "    Vector     ";
    case LOG_TAG_CLUSTER:
        return "  Vector Set   ";
    case LOG_TAG_BUFFER:
        return "    Buffer     ";
    case LOG_TAG_DIVFTREE_VERTEX:
        return "DIVFTree Vertex";
    case LOG_TAG_DIVFTREE:
        return " Vector Index  ";
    case LOG_TAG_CLUSTERING:
        return "     Core      ";
    case LOG_TAG_MEMORY:
        return "    Memory     ";
    case LOG_TAG_LOCK:
        return "     Lock      ";
    case LOG_TAG_THREAD:
        return "    Thread     ";
    case LOG_TAG_NOT_IMPLEMENTED:
        return "Not Implemented";
    case LOG_TAG_HANG_DETECTOR:
        return " Hang Detector ";
    case LOG_TAG_TEST:
        return "     Test      ";
    default:
        return "   Undefined   ";
    }
}

inline constexpr bool Pass_Min_Level(LOG_LEVELS level) {
    return ((uint8_t)(level) <= LOG_MIN_LEVEL);
}

inline constexpr bool Pass_Tag(uint64_t tag) {
    return ((LOG_TAG & tag) != 0);
}

inline constexpr bool Pass_Level(LOG_LEVELS level) {
    return ((uint8_t)(level) <= (uint8_t)(LOG_LEVEL));
}

inline constexpr bool Pass_CallStack_Level(LOG_LEVELS level) {
    return ((uint8_t)(level) <= (uint8_t)(PRINT_CALLSTACK_LEVEL));
}

/* todo: Make it threadlocal */
namespace debug {
inline bool * fault_checking = nullptr;

inline void* _lock_ptr = nullptr; // pointer to an SXSpinLock which is used when we need to change log_file
inline char output_log_path[256];
inline uint64_t next_log_sn = 0;
inline FILE * output_log = stdout;

class FaultCheckingExc : public std::exception  {};
};

inline void Log(LOG_LEVELS level, uint64_t tag, const Log_Msg& msg,
                const std::chrono::_V2::system_clock::time_point _time,
                const char* file_name, const char* func_name, size_t line,
                thread_id thread_id, uint64_t divf_thread_id) {
    char time_str[100];
    timetostr(_time, time_str); // todo add coloring if needed
    UNUSED_VARIABLE(func_name);
    if (::divftree::debug::fault_checking != nullptr) {
#ifdef ENABLE_FAULT_LOGGING
#ifdef LOG_FUNCTION_NAME
        fprintf(::divftree::debug::output_log, "%s | %s | %s | %s | %s | Thread(%lu): %lu | Message: %s\n",
            leveltostr(level, true), tagtostr(tag), time_str,
            ::divftree::String("%s:%lu", file_name, line).Fit(FILE_NAME_MAX_SIZE).ToCStr(),
            ::divftree::String("%s", func_name).Fit(FUNCTION_NAME_MAX_SIZE).ToCStr(),
            thread_id, divf_thread_id, msg._msg);
#else
        fprintf(::divftree::debug::output_log, "%s | %s | %s | %s | Thread(%lu): %lu | Message: %s\n",
            leveltostr(level, true), tagtostr(tag), time_str,
            ::divftree::String("%s:%lu", file_name, line).Fit(FILE_NAME_MAX_SIZE).ToCStr(),
            thread_id, divf_thread_id, msg._msg);
#endif
        fflush(::divftree::debug::output_log);
#endif
        if (level == LOG_LEVEL_PANIC) {
            *(::divftree::debug::fault_checking) = true;
            throw ::divftree::debug::FaultCheckingExc{};
        } else {
            return; // do not log the message again
        }
    }

    if (Pass_CallStack_Level(level)) {
        std::string callstack = print_callstack();
#ifdef LOG_FUNCTION_NAME
        fprintf(::divftree::debug::output_log, "%s | %s | %s | %s | %s | Thread(%lu): %lu | Callstack=%s | Message: %s\n",
            leveltostr(level), tagtostr(tag), time_str,
            ::divftree::String("%s:%lu", file_name, line).Fit(FILE_NAME_MAX_SIZE).ToCStr(),
            ::divftree::String("%s", func_name).Fit(FUNCTION_NAME_MAX_SIZE).ToCStr(),
            thread_id, divf_thread_id, callstack.c_str(), msg._msg);
#else
        fprintf(::divftree::debug::output_log, "%s | %s | %s | %s | Thread(%lu): %lu | Callstack=%s | Message: %s\n",
            leveltostr(level), tagtostr(tag), time_str,
            ::divftree::String("%s:%lu", file_name, line).Fit(FILE_NAME_MAX_SIZE).ToCStr(),
            thread_id, divf_thread_id, callstack.c_str(), msg._msg);
#endif
    }
    else {
#ifdef LOG_FUNCTION_NAME
        fprintf(::divftree::debug::output_log, "%s | %s | %s | %s | %s | Thread(%lu): %lu | Message: %s\n",
            leveltostr(level), tagtostr(tag), time_str,
            ::divftree::String("%s:%lu", file_name, line).Fit(FILE_NAME_MAX_SIZE).ToCStr(),
            ::divftree::String("%s", func_name).Fit(FUNCTION_NAME_MAX_SIZE).ToCStr(),
            thread_id, divf_thread_id, msg._msg);
#else
        fprintf(::divftree::debug::output_log, "%s | %s | %s | %s | Thread(%lu): %lu | Message: %s\n",
            leveltostr(level), tagtostr(tag), time_str,
            ::divftree::String("%s:%lu", file_name, line).Fit(FILE_NAME_MAX_SIZE).ToCStr(),
            thread_id, divf_thread_id, msg._msg);
#endif

    }
    fflush(::divftree::debug::output_log);
    if (level == LOG_LEVEL_PANIC) {
        ::divftree::sleep(1); // wait to make sure everything is flushed
        abort();
    }
}

};

#ifdef ENABLE_TEST_LOGGING

#define DIVFLOG_ELINE() \
    do {\
        fprintf(::divftree::debug::output_log, "\n");\
        fflush(::divftree::debug::output_log);\
    } while(0)

#define DIVFLOG(level, tag, msg, ...) \
    do {\
        if constexpr (::divftree::Pass_Min_Level((level)) || ::divftree::Pass_Tag((tag))) {\
            if constexpr (::divftree::Pass_Level((level))){\
                ::divftree::Log((level), (tag), (::divftree::Log_Msg((msg) __VA_OPT__(,) __VA_ARGS__)), \
                            std::chrono::system_clock::now(),\
                            __FILE__, FUNCTION_NAME, __LINE__, THREAD_ID, DIVF_THREAD_ID);\
            }\
        }\
    } while(0)

#define DIVFLOG_IF_TRUE(cond, level, tag, msg, ...) \
    do {\
        if ((cond)){\
            char _TMP_DEBUG[sizeof((#cond))+sizeof((msg))+19] = "Condition True(" #cond "): ";\
            strcat(_TMP_DEBUG+sizeof((#cond)), (msg));\
            DIVFLOG((level), (tag),  _TMP_DEBUG __VA_OPT__(,) __VA_ARGS__);\
        }\
    } while(0)

#define DIVFLOG_IF_FALSE(cond, level, tag, msg, ...) \
    do {\
        if (!(cond)){\
            char _TMP_DEBUG[sizeof((#cond))+sizeof((msg))+20] = "Condition False(" #cond "): ";\
            strcat(_TMP_DEBUG+sizeof((#cond)), (msg));\
            DIVFLOG((level), (tag),  _TMP_DEBUG __VA_OPT__(,) __VA_ARGS__);\
        }\
    } while(0)

// todo add unlikely to assert conditions
#ifdef ENABLE_ASSERTS
#define SANITY_CHECK(codeBlock...) codeBlock
#define FatalAssert(cond, tag, msg, ...) \
    do {\
        if (!(cond)){\
            if (sizeof((msg)) == 0){\
                DIVFLOG(LOG_LEVEL_PANIC, (tag),  "Assertion \'" #cond "\' Failed.");\
            }\
            else{\
                char _TMP_DEBUG[sizeof((#cond))+sizeof((msg))+22] = "Assertion \'" #cond "\' Failed: ";\
                strcat(_TMP_DEBUG+sizeof((#cond)), (msg));\
                DIVFLOG(LOG_LEVEL_PANIC, (tag),  _TMP_DEBUG __VA_OPT__(,) __VA_ARGS__);\
            }\
        }\
    } while(0)

#ifdef ASSERT_ERROR_PANIC
#define ErrorAssert(cond, tag, msg, ...) FatalAssert((cond), (tag), (msg)__VA_OPT__(,) __VA_ARGS__)
#else
#define ErrorAssert(cond, tag, msg, ...) \
    do {\
        if (!(cond)) {\
            if (sizeof((msg)) == 0){\
                DIVFLOG(LOG_LEVEL_ERROR, (tag),  "Assertion \'" #cond "\' Failed.");\
            }\
            else{\
                char _TMP_DEBUG[sizeof((#cond))+sizeof((msg))+22] = "Assertion \'" #cond "\' Failed: ";\
                strcat(_TMP_DEBUG+sizeof((#cond)), (msg));\
                DIVFLOG(LOG_LEVEL_ERROR, (tag),  _TMP_DEBUG __VA_OPT__(,) __VA_ARGS__);\
            }\
        }\
    } while(0)
#endif

#define FaultAssert(statement, condtion, tag, msg, ...) \
    do {\
        bool _FAULTY = false; \
        ::divftree::debug::fault_checking = &_FAULTY;\
        try { \
            (statement);\
        } \
        catch (const ::divftree::debug::FaultCheckingExc& e) {\
            _FAULTY = true;\
        } \
        catch (const std::exception& e) {\
            _FAULTY = true;\
            DIVFLOG(LOG_LEVEL_WARNING, (tag),  "Fault Assertion \'" #statement \
                 "\' got exception: %s", e.what());\
        }\
        catch (...) {\
            _FAULTY = true;\
            DIVFLOG(LOG_LEVEL_WARNING, (tag),  "Fault Assertion \'" #statement \
                 "\' got unkown exception.");\
        }\
        ::divftree::debug::fault_checking = nullptr;\
        if (!(_FAULTY)) {\
            condtion = false;\
            char _ASSERT_TMP_DEBUG[sizeof((#statement))+sizeof((msg))+23] = "\'" #statement "\' No Errors Occured. ";\
            strcat(_ASSERT_TMP_DEBUG+sizeof((#statement)+22), (msg));\
            ErrorAssert(false, (tag), (_ASSERT_TMP_DEBUG) __VA_OPT__(,) __VA_ARGS__);\
        }\
    } while(0)

#else
#define SANITY_CHECK(codeBlock...)
#define FatalAssert(cond, tag, msg, ...)
#define ErrorAssert(cond, tag, msg, ...)
#define FaultAssert(statement, tag, msg, ...)
#endif
#else

#define SANITY_CHECK(codeBlock...)
#define DIVFLOG_ELINE()
#define DIVFLOG(level, tag, msg, ...)
#define DIVFLOG_IF_TRUE(cond, level, tag, msg, ...)
#define DIVFLOG_IF_FALSE(cond, level, tag, msg, ...)

#ifdef ENABLE_ASSERTS
#define FatalAssert(cond, tag, msg, ...) assert((cond))
#else
#define FatalAssert(cond, tag, msg, ...)
#endif

#ifdef ASSERT_ERROR_PANIC
#define ErrorAssert(cond, tag, msg, ...) FatalAssert((cond), (tag), (msg)__VA_OPT__(,) __VA_ARGS__)
#else
#define ErrorAssert(cond, tag, msg, ...)
#endif
#endif

#ifdef FAULT_INJECTION
static inline std::map<std::string, std::binary_semaphore> FI_MAP;
#define FAULT_INJECTION_INIT(name) FI_MAP.try_emplace((name), 0)
#define FAULT_INJECTION_DELETE(name) FI_MAP.erase((name))
#define FAULT_INJECTION_WAIT(name) FI_MAP[(name)].acquire()
#define FAULT_INJECTION_SIGNAL(name) FI_MAP[(name)].release()
// todo:
// todo: add backslash at the end of lines
// #define FAULT_INJECTION_WAIT_UNTIL(name, duration) FI_MAP[(name)].try_acquire_for((duration))
// #define FAULT_INJECTION_WAIT_UNTIL_ERROR(name, duration, tag, msg, ...)
//     do {
//         if (sizeof((msg)) == 0){
//             DIVFLOG(LOG_LEVEL_ERROR, (tag),  "Fault Injection timeout error on \'%s\' after waiting for %lu %s.", );
//         }
//         else{
//             char _TMP_DEBUG[sizeof((#cond))+sizeof((msg))+22] = "Assertion \'" #cond "\' Failed: ";
//             strcat(_TMP_DEBUG+sizeof((#cond)), (msg));
//             DIVFLOG(LOG_LEVEL_ERROR, (tag),  _TMP_DEBUG __VA_OPT__(,) __VA_ARGS__);
//         }
//     } while(0)
// #define FAULT_INJECTION_WAIT_UNTIL_PANIC(name, duration) FI_MAP[(name)].try_acquire_for((duration))
#else
#define FAULT_INJECTION_INIT(name)
#define FAULT_INJECTION_DELETE(name)
#define FAULT_INJECTION_WAIT(name)
#define FAULT_INJECTION_SIGNAL(name)
#endif

#define CHECK_NOT_NULLPTR(ptr, tag) \
    FatalAssert((ptr) != nullptr, (tag), "Pointer is nullptr")

#define DIVF_TEXT_TO_STR(T) #T
#define DIVF_MACRO_TO_STR(T) DIVF_TEXT_TO_STR(T)



namespace divftree {
    constexpr inline int MemCMP(const void* ptr1, const void* ptr2, size_t num) {
        const uint8_t* p1 = static_cast<const uint8_t*>(ptr1);
        const uint8_t* p2 = static_cast<const uint8_t*>(ptr2);
        for (size_t i = 0; i < num; ++i) {
            if (p1[i] != p2[i]) {
                return (p1[i] < p2[i]) ? -1 : 1;
            }
        }
        return 0;
    }

    constexpr inline void MemSET(void* ptr, uint8_t value, size_t num) {
        uint8_t* p = static_cast<uint8_t*>(ptr);
        for (size_t i = 0; i < num; ++i) {
            p[i] = value;
        }
    }

    constexpr inline void MemCOPY(void* dest, const void* src, size_t num) {
        uint8_t* d = static_cast<uint8_t*>(dest);
        const uint8_t* s = static_cast<const uint8_t*>(src);
        for (size_t i = 0; i < num; ++i) {
            d[i] = s[i];
        }
    }
};
// #define MEM_FUN_DEBUG
#ifdef MEM_FUN_DEBUG
#define DIVF_MEMCMP(ptr1, ptr2, num) ::divftree::MemCMP((ptr1), (ptr2), (num))
#define DIVF_MEMSET(ptr, value, num) ::divftree::MemSET((ptr), (value), (num))
#define DIVF_MEMCOPY(dest, src, num) ::divftree::MemCOPY((dest), (src), (num))
#define BARRIER(barrier, point_name) \
    do {\
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_THREAD, "Thread reached point '%s'.", (point_name));\
        (barrier).arrive_and_wait();\
        DIVFLOG(LOG_LEVEL_LOG, LOG_TAG_THREAD, "Thread passed point '%s'.", (point_name));\
    } while(0)

#else
#define DIVF_MEMCMP(ptr1, ptr2, num) memcmp((ptr1), (ptr2), (num))
#define DIVF_MEMSET(ptr, value, num) memset((ptr), (value), (num))
#define DIVF_MEMCOPY(dest, src, num) memcpy((dest), (src), (num))
#define BARRIER(barrier, point_name) \
    do {\
        (barrier).arrive_and_wait();\
    } while(0)
#endif
#endif