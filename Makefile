# 目标与工具链（板端 aarch64 Linux，直接用系统 g++）
CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++14 -Wall -Wextra
# RK3568 为 Cortex-A55（ARMv8-A），aarch64 下 NEON 默认可用，无需 -mfpu
ARCHFLAGS ?= -march=armv8-a

# OpenCL 头文件由 scripts/fetch_opencl_headers.sh 拉取
#OPENCL_HEADER_DIR ?= third_party/OpenCL-Headers
#INCLUDES := -Isrc -I$(OPENCL_HEADER_DIR)
INCLUDES := -Isrc

# 板端 Mali 驱动提供 libOpenCL.so；若不在默认搜索路径，用 OPENCL_LIB_DIR 指定
OPENCL_LIB_DIR ?=
LDFLAGS  := $(if $(OPENCL_LIB_DIR),-L$(OPENCL_LIB_DIR),)
LDLIBS   := -lOpenCL -lpthread -lrt

TARGET := conv_bench
SRCS   := src/main.cpp src/utils.cpp src/conv_cpu.cpp src/conv_cl.cpp
OBJS   := $(SRCS:.cpp=.o)
DEPS   := $(OBJS:.o=.d)

KERNEL_SRC := kernels/conv2d.cl

.PHONY: all headers run clean distclean check-env

all: $(TARGET)

# 拉取 OpenCL 头文件（只需执行一次）
headers:
	@bash scripts/fetch_opencl_headers.sh

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INCLUDES) -MMD -MP -c $< -o $@

# conv_cl.cpp 依赖 CL/cl.h，头文件缺失时给出明确提示
#src/conv_cl.o: | $(OPENCL_HEADER_DIR)/CL/cl.h

#$(OPENCL_HEADER_DIR)/CL/cl.h:
#	@echo "缺少 OpenCL 头文件：$@"
#	@echo "请先执行: make headers   （或手动放置 OpenCL-Headers 到 $(OPENCL_HEADER_DIR)）"
#	@exit 1

run: $(TARGET)
	./$(TARGET) $(ARGS)

# 快速自检：打印设备信息并跑一次最小规模（不写 CSV）
check-env: $(TARGET)
	./$(TARGET) --size 64 --iters 1 --warmup 0 --csv ""

clean:
	rm -f $(OBJS) $(DEPS) $(TARGET)

distclean: clean
	rm -f results.csv

-include $(DEPS)
