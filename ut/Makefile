# Makefile for compiling unit tests in the ut folder

# todo adding necessary flags for gdb at least when mode is debug
# todo changing the core location after making gdb work

ROOT = ..
COPPER_PATH = $(ROOT)/include

BUILD_DIR = $(ROOT)/ut/build
OBJ_DIR := $(BUILD_DIR)/obj
BIN_DIR := $(BUILD_DIR)/bin

SRC_DIR = $(ROOT)/ut

CXX = g++
CXXFLAGS = -std=c++20 -I$(COPPER_PATH) -Wall -Wextra -pedantic -g -MMD -MP

# Add custom defines if they are provided
ifneq ($(filter -D%,$(DEFINES)),)
    CXXFLAGS += $(filter -D%,$(DEFINES))
endif

TESTS = $(wildcard $(SRC_DIR)/*.cpp)

OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(TESTS))
DEPS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.d, $(TESTS))
EXES = $(patsubst $(SRC_DIR)/%.cpp, $(BIN_DIR)/%, $(TESTS))

all: $(EXES)

$(BIN_DIR)/%: $(OBJ_DIR)/%.o | $(BIN_DIR)
	$(CXX) $(LDFLAGS) -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BIN_DIR): $(BUILD_DIR)
	mkdir -p $(BIN_DIR)

$(OBJ_DIR): $(BUILD_DIR)
	mkdir -p $(OBJ_DIR)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)


# Include dependency files (if they exist)
-include $(DEPS)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean

# I am writing a c++ project and I am using Makefile for building the project. In the project I am using some defined macros and I want to add them to the program each time I am building it using the makefile. For example I want to do something like make -DFLAG -DVAR=5 and pass it as defined macro to my code. How can I do this?