BUILD_DIR := build
RWKV_BACKEND ?= cuda
CMAKE ?= cmake
CMAKE_BUILD_PARALLEL_LEVEL ?= 32

.PHONY: all configure build benchmark backend clean

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR) --fresh -DRWKV_BACKEND=$(RWKV_BACKEND)

build: configure
	$(CMAKE) --build ./$(BUILD_DIR) --parallel $(CMAKE_BUILD_PARALLEL_LEVEL) --target rwkv_backend_support benchmark rwkv_lightning

benchmark: build

backend: build
	./$(BUILD_DIR)/rwkv_lightning

clean:
	$(CMAKE) --build ./$(BUILD_DIR) --target clean
