SHELL := /bin/bash

CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?=

SRC_DIR := .
TARGET := $(SRC_DIR)/ngram

SRCS := \
	$(SRC_DIR)/main.cpp \
	$(SRC_DIR)/cmd_count.cpp \
	$(SRC_DIR)/cmd_process.cpp \
	$(SRC_DIR)/cmd_batch.cpp \
	$(SRC_DIR)/cmd_build_vocab.cpp \
	$(SRC_DIR)/cmd_ngram_input_gen.cpp

OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(SRC_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -c $< -o $@

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

-include $(DEPS)
