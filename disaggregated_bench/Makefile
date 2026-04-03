# Makefile for compiling the benchmark

# todo adding necessary flags for gdb at least when mode is debug
# todo changing the core location after making gdb work

# ================================
# Configurable build options
# ================================

ENABLE_GDB       ?= 1# Enables -g3
ENABLE_ASAN      ?= 1# Enables AddressSanitizer
ENABLE_DEBUGINFO ?= 1# Enables -rdynamic, -fno-omit-frame-pointer and debug-friendly flags
ENABLE_OPT       ?= 0# Enables optimization (-O3 vs -O0)

# ================================
# Common
# ================================

ROOT := $(shell dirname $(CURDIR))
DIVFTREE_PATH := $(ROOT)/include
INCLUDE_PATHS := -I$(DIVFTREE_PATH) -I$(ROOT)

BUILD_DIR := $(ROOT)/disaggregated_bench/build
MN_OBJ_DIR := $(BUILD_DIR)/obj/memory
MN_BIN_DIR := $(BUILD_DIR)/bin/memory
CN_OBJ_DIR := $(BUILD_DIR)/obj/compute
CN_BIN_DIR := $(BUILD_DIR)/bin/compute

CN_SRC_DIR := $(ROOT)/disaggregated_bench/compute_node
MN_SRC_DIR := $(ROOT)/disaggregated_bench/memory_node
$(info Compute Node source dir: $(CN_SRC_DIR))
$(info Memory Node source dir: $(MN_SRC_DIR))

CXX := g++
WARNING_FLAGS := -Wall -Wextra -pedantic -Wno-pedantic -Wno-write-strings -Wno-pointer-arith
CXXFLAGS := $(WARNING_FLAGS) -std=c++20 $(INCLUDE_PATHS) -march=native -mcx16 -MMD -MP -libverbs -pthread
LDFLAGS := -latomic -ldl -libverbs -pthread

# ================================
# Conditional configuration
# ================================
# Optimization level
ifeq ($(ENABLE_OPT),1)
    CXXFLAGS += -O3
else
    CXXFLAGS += -O0
endif

# Debug info
ifeq ($(ENABLE_DEBUGINFO),1)
    CXXFLAGS += -fno-omit-frame-pointer
    LDFLAGS  += -fno-omit-frame-pointer -rdynamic
endif

ifeq ($(ENABLE_GDB),1)
    CXXFLAGS += -g3
endif

# AddressSanitizer
ifeq ($(ENABLE_ASAN),1)
    CXXFLAGS += -fsanitize=address
    LDFLAGS  += -fsanitize=address
endif

# Add custom defines if they are provided
ifneq ($(filter -D%,$(DEFINES)),)
    CXXFLAGS += $(filter -D%,$(DEFINES))
endif

CN_CXXFLAGS := $(CXXFLAGS) -DDIVF_SELF_NODE_TYPE=COMPUTE_NODE_IDX
MN_CXXFLAGS := $(CXXFLAGS) -DDIVF_SELF_NODE_TYPE=MEMORY_NODE_IDX

CN_TESTS = $(wildcard $(CN_SRC_DIR)/*.cpp)
$(info Found CN test files: $(CN_TESTS))
MN_TESTS = $(wildcard $(MN_SRC_DIR)/*.cpp)
$(info Found MN test files: $(MN_TESTS))

MN_OBJS := $(patsubst $(MN_SRC_DIR)/%.cpp, $(MN_OBJ_DIR)/%.o, $(MN_TESTS))
CN_OBJS := $(patsubst $(CN_SRC_DIR)/%.cpp, $(CN_OBJ_DIR)/%.o, $(CN_TESTS))
MN_DEPS := $(patsubst $(MN_SRC_DIR)/%.cpp, $(MN_OBJ_DIR)/%.d, $(MN_TESTS))
CN_DEPS := $(patsubst $(CN_SRC_DIR)/%.cpp, $(CN_OBJ_DIR)/%.d, $(CN_TESTS))
MN_EXES := $(patsubst $(MN_SRC_DIR)/%.cpp, $(MN_BIN_DIR)/%, $(MN_TESTS))
CN_EXES := $(patsubst $(CN_SRC_DIR)/%.cpp, $(CN_BIN_DIR)/%, $(CN_TESTS))

all: $(MN_EXES) $(CN_EXES)

$(MN_BIN_DIR)/%: $(MN_OBJ_DIR)/%.o | $(MN_BIN_DIR)
	$(CXX) $(LDFLAGS) -o $@ $<

$(MN_OBJ_DIR)/%.o: $(MN_SRC_DIR)/%.cpp | $(MN_OBJ_DIR)
	$(CXX) $(MN_CXXFLAGS) -c $< -o $@

$(MN_BIN_DIR): $(BUILD_DIR)
	mkdir -p $(MN_BIN_DIR)

$(MN_OBJ_DIR): $(BUILD_DIR)
	mkdir -p $(MN_OBJ_DIR)

$(CN_BIN_DIR)/%: $(CN_OBJ_DIR)/%.o | $(CN_BIN_DIR)
	$(CXX) $(LDFLAGS) -o $@ $<

$(CN_OBJ_DIR)/%.o: $(CN_SRC_DIR)/%.cpp | $(CN_OBJ_DIR)
	$(CXX) $(CN_CXXFLAGS) -c $< -o $@

$(CN_BIN_DIR): $(BUILD_DIR)
	mkdir -p $(CN_BIN_DIR)

$(CN_OBJ_DIR): $(BUILD_DIR)
	mkdir -p $(CN_OBJ_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)


# Include dependency files (if they exist)
-include $(DEPS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean