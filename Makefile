CC ?= cc
CFLAGS ?= -O3 -ffast-math -march=native -Wall -Wextra -std=c99 -D_GNU_SOURCE -fopenmp
OBJCFLAGS ?= -O3 -ffast-math -march=native -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread -fopenmp
UNAME_S := $(shell uname -s)
NATIVE_LDLIBS := $(LDLIBS)
METAL_SRCS := $(wildcard metal/*.metal)

# HIP support (enable with USE_HIP=1)
USE_HIP ?= 0

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_metal.o
NATIVE_CORE_OBJS = ds4_native.o
else
CFLAGS += -DDS4_NO_METAL
CORE_OBJS = ds4.o
NATIVE_CORE_OBJS = ds4_native.o
METAL_LDLIBS := $(LDLIBS)
endif

# HIP configuration (ROCm) - use gcc with HIP libs
ifeq ($(USE_HIP),1)
CFLAGS += -DDS4_HIP -DGGML_HIP -D__HIP_PLATFORM_AMD__
HIP_LDLIBS = -lamdhip64 -lrocblas -lrocsolver

# Use standard gcc with HIP libraries (no kernel compilation for now)
# TODO: Add separate .hip file compilation for GPU kernels
HIP_CFLAGS = -DDS4_HIP -DGGML_HIP -D__HIP_PLATFORM_AMD__
endif

.PHONY: all clean test hip hip-build

all: ds4 ds4-server

# HIP build targets (Linux only, requires ROCm)
ifeq ($(USE_HIP),1)
HIP_CFLAGS := -DDS4_HIP -DGGML_HIP -D__HIP_PLATFORM_AMD__

ds4-hip: CFLAGS += $(HIP_CFLAGS)
ds4-hip: ds4_hip.o ds4_cli.o linenoise.o ds4.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS) $(HIP_LDLIBS)

ds4_hip.o: ds4_hip.c ds4.h ds4_hip.h
	$(CC) $(CFLAGS) $(HIP_CFLAGS) -c -o $@ ds4_hip.c

.PHONY: hip-build
endif

ifeq ($(UNAME_S),Darwin)
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(NATIVE_LDLIBS)
else
ds4: ds4_cli.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4-server: ds4_server.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_native: ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli_native.o linenoise.o $(NATIVE_CORE_OBJS) $(LDLIBS)
endif

ds4.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_cli.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_server.o: ds4_server.c ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_native.o: ds4.c ds4.h ds4_metal.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4.c

ds4_cli_native.o: ds4_cli.c ds4.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_METAL -c -o $@ ds4_cli.c

ds4_metal.o: ds4_metal.m ds4_metal.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_test: ds4_test.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_test.o $(CORE_OBJS) $(METAL_LDLIBS)

test: ds4_test
	./ds4_test

clean:
	rm -f ds4 ds4-server ds4_native ds4_server_test ds4_test *.o
