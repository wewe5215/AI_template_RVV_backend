
CC = clang++-17
XNN_PACK := /home/riscv/Desktop/AI_template_RVV_backend/3rdparty/XNNPACK
XNN_BUILD := /home/riscv/Desktop/AI_template_RVV_backend/3rdparty/XNNPACK/build
XNN_LIB_DIR = /home/riscv/Desktop/AI_template_RVV_backend/3rdparty/XNNPACK/build/local
CFLAGS = -fPIC -Wconversion -fno-strict-aliasing -fvisibility=hidden -O3 -std=c++17 -v -I$(XNN_PACK)/include -I$(XNN_LIB_DIR)/pthreadpool-source/include -L$(XNN_LIB_DIR) $(XNN_LIB_DIR)/*.a -L$(XNN_LIB_DIR)/cpuinfo -L$(XNN_LIB_DIR)/kleidiai -L$(XNN_LIB_DIR)/pthreadpool -lXNNPACK -lpthreadpool -lcpuinfo -lpthread -g
fPIC_flag = -fPIC
obj_files = conv2d_bias_relu_0.obj max_pool2d_1.obj conv2d_bias_2.obj conv2d_bias_relu_3.obj conv2d_bias_relu_4.obj conv2d_bias_add_relu_5.obj conv2d_bias_relu_6.obj conv2d_bias_relu_12.obj conv2d_bias_13.obj conv2d_bias_relu_14.obj conv2d_bias_add_relu_15.obj conv2d_bias_relu_16.obj conv2d_bias_relu_17.obj conv2d_bias_relu_25.obj conv2d_bias_26.obj conv2d_bias_relu_27.obj conv2d_bias_add_relu_28.obj conv2d_bias_relu_29.obj conv2d_bias_relu_30.obj conv2d_bias_relu_44.obj conv2d_bias_45.obj conv2d_bias_relu_46.obj conv2d_bias_add_relu_47.obj conv2d_bias_relu_48.obj conv2d_bias_relu_49.obj avg_pool2d_54.obj gemm_rcr_bias_56.obj model_container_base.obj model_container.obj model_interface.obj debug_utility.obj

%.obj : %.cpp
	$(CC) $(CFLAGS) -fsanitize=address -c -o $@ $<
%.obj : %.bin
	ld -r -b binary -o $@ $<

.PHONY: all clean clean_constants
all: test.so

test.so: $(obj_files)
	$(CC) -shared $(fPIC_flag) $(CFLAGS) -o $@ $(obj_files) -L$(XNN_LIB_DIR) $(XNN_LIB_DIR)/*.a -L$(XNN_LIB_DIR)/cpuinfo -L$(XNN_LIB_DIR)/kleidiai -L$(XNN_LIB_DIR)/pthreadpool -lXNNPACK -lpthreadpool -lcpuinfo -lpthread -g

run:
	$(CC) $(CFLAGS) -fsanitize=undefined -g -o standalone standalone.cpp ./test.so

clean:
	rm -f *.obj test.so

clean_constants:
	rm -f constants.bin
