# Variables
CC = g++
CFLAGS = -O3 -std=c++17 -fPIC -Wall -Wconversion -fno-strict-aliasing -fvisibility=hidden

obj_files = transposed_conv2d_bias_relu_3.o model_container_base.o model_interface.o debug_utility.o model_container.o

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

run:
	g++ -O3 -std=c++17 -o standalone standalone.cpp -L. -ltest
	./standalone

clean:
	rm -f *.o libtest.so

clean_constants:
	rm -f constants.bin
