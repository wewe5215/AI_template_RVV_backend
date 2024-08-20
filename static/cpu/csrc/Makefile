# Variables
CC = clang++
XNN_PACK := /home/riscv/Desktop/AI_template_RVV_backend/3rdparty/XNNPACK
XNN_BUILD := /home/riscv/Desktop/AI_template_RVV_backend/3rdparty/XNNPACK/build
XNN_LIB_DIR = /home/riscv/Desktop/AI_template_RVV_backend/3rdparty/XNNPACK/build/local
VLEN="256"
CLANG := clang++-17
CFLAGS :=
CFLAGS += -D_POSIX_C_SOURCE=199309L # error: 'CLOCK_MONOTONIC' undeclared
CFLAGS += -O3 -std=c++17 -fPIC -Wall -Wconversion -fno-strict-aliasing -fvisibility=hidden
# Additional Includes for XNNPACK
CFLAGS += -I$(XNN_PACK)/include
CFLAGS += -I$(XNN_LIB_DIR)/pthreadpool-source/include
LDFLAGS := -L$(XNN_LIB_DIR) $(XNN_LIB_DIR)/*.a
LDFLAGS += -L$(XNN_LIB_DIR)/cpuinfo
LDFLAGS += -L$(XNN_LIB_DIR)/kleidiai
LDFLAGS += -L$(XNN_LIB_DIR)/pthreadpool
LDFLAGS += -lXNNPACK -lpthreadpool -lcpuinfo -lpthread -g
obj_files = conv2d_bias_add_relu_0.o model_container_base.o model_interface.o debug_utility.o model_container.o

# Pattern Rules
%.o : %.cpp
	$(CLANG) $(CFLAGS) -c -o $@ $<

%.o : %.bin
	ld -r -b binary -o $@ $< && objcopy --rename-section .data=.lrodata,alloc,load,readonly,data,contents $@ $@

# Targets
.PHONY: all clean clean_constants
all: libtest.so

libtest.so: $(obj_files)
	$(CLANG) -shared $(CFLAGS) -o $@ $(obj_files) $(LDFLAGS)
# $(CC) -shared $(CFLAGS) -o $@ $(obj_files)

run:
# $(CLANG) $(CFLAGS) -o standalone standalone.cpp -L. -ltest
# $(SPIKE) $(PK) standalone
	$(CLANG) $(CFLAGS) -g -o standalone standalone.cpp -L. -ltest

clean:
	rm -f *.o libtest.so

clean_constants:
	rm -f constants.bin
