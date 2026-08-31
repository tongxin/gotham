CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -fPIC

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LIB_NAME := libgotham.dylib
  SHARED_FLAGS := -dynamiclib
else ifeq ($(OS),Windows_NT)
  LIB_NAME := libgotham.dll
  SHARED_FLAGS := -shared
else
  LIB_NAME := libgotham.so
  SHARED_FLAGS := -shared
endif

BUILD := build
LIB := $(BUILD)/$(LIB_NAME)

all: $(LIB)

$(LIB): sim.cpp sim.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(SHARED_FLAGS) -o $@ sim.cpp

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)

.PHONY: all clean
