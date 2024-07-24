# Variables
CC = clang++
LLVM_PATH := /Users/wewe5215/Desktop/llvm-project/build
RISCV := /opt/riscv
SPIKE   := $(RISCV)/bin/spike
RISCV_PK := $(RISCV)/riscv64-unknown-elf/bin/pk
XNN_PACK := /Users/wewe5215/Desktop/AI_template_RVV_backend/3rdparty/XNNPACK
XNN_BUILD := /Users/wewe5215/Desktop/AI_template_RVV_backend/3rdparty/XNNPACK/build
XNN_LIB_DIR = /Users/wewe5215/Desktop/AI_template_RVV_backend/3rdparty/XNNPACK/build/local
VLEN="256"
CLANG := $(LLVM_PATH)/bin/clang++
CFLAGS :=
# CFLAGS += -I$(XNN_PACK)/include
# CFLAGS += -I$(XNN_BUILD)/pthreadpool-source/include
# CFLAGS += -D_POSIX_C_SOURCE=199309L # error: 'CLOCK_MONOTONIC' undeclared
CFLAGS += -O3 -std=c++17 -fPIC -Wall -Wconversion -fno-strict-aliasing -fvisibility=hidden
# CFLAGS += --target=riscv64-unknown-elf
# CFLAGS += --sysroot=$(RISCV)/riscv64-unknown-elf
# CFLAGS += --gcc-toolchain=$(RISCV)
# CFLAGS += -menable-experimental-extensions
# CFLAGS += -march=rv64gcv1_zfh_zvfh
# CFLAGS += -mrvv-vector-bits=$(VLEN)
# CFLAGS += -stdlib=libstdc++
# CFLAGS += -I/opt/riscv/riscv64-unknown-elf/include
# CFLAGS += -I/opt/riscv/riscv64-unknown-elf/include/c++/12.2.0
# CFLAGS += -I/opt/riscv/riscv64-unknown-elf/include/c++/12.2.0/riscv64-unknown-elf
# CFLAGS += -I/opt/riscv/riscv64-unknown-elf/include/c++/12.2.0/backward
# CFLAGS += -I$(LLVM_PATH)/include/c++/v1
# CFLAGS += -I$(LLVM_PATH)/lib/clang/$(shell $(LLVM_PATH)/bin/clang++ -dumpversion)/include
# LDFLAGS := -L$(RISCV)/lib -lc++ -lc++abi -lm -lpthread -ldl
# Additional Includes for XNNPACK
CFLAGS += -I$(XNN_PACK)/include
CFLAGS += -L$(XNN_LIB_DIR) $(XNN_LIB_DIR)/*.a
CFLAGS += -L$(XNN_LIB_DIR)/cpuinfo
CFLAGS += -L$(XNN_LIB_DIR)/kleidiai
CFLAGS += -L$(XNN_LIB_DIR)/pthreadpool
CFLAGS += -lxnnpack -lkleidiai -lpthreadpool -lcpuinfo -lpthread
obj_files = conv2d_bias_add_relu_0.o model_container_base.o model_interface.o debug_utility.o model_container.o

# Pattern Rules
%.o : %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

%.o : %.bin
	ld -r -b binary -o $@ $< && objcopy --rename-section .data=.lrodata,alloc,load,readonly,data,contents $@ $@

# Targets
.PHONY: all clean clean_constants
all: libtest.so

libtest.so: $(obj_files)
	$(CC) -shared $(CFLAGS) -o $@ $(obj_files)
# $(CC) -shared $(CFLAGS) -o $@ $(obj_files)

run:
# $(CLANG) $(CFLAGS) -o standalone standalone.cpp -L. -ltest
# $(SPIKE) $(PK) standalone
	$(CC) $(CFLAGS) -o standalone standalone.cpp -L. -ltest
	./standalone

clean:
	rm -f *.o libtest.so

clean_constants:
	rm -f constants.bin
