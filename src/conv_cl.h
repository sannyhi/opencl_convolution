#pragma once
// OpenCL 端封装：设备/上下文/队列/program/kernel/buffer，以及两层卷积的计时执行
// 按 OpenCL 1.2 API 编写，适配 RK3568 板端 Mali 驱动（libOpenCL.so）
#include <string>
#include <vector>

#include "conv_params.h"
#include "utils.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

struct ClTiming {
    double kernelMs = 0.0;  // profiling 事件得到的 kernel 纯耗时
    double wallMs = 0.0;    // 含排入队列到 clFinish 的墙钟耗时
};

class OpenCLRunner {
public:
    ~OpenCLRunner();

    // 初始化：选 GPU 设备、建上下文与队列、编译 kernel 文件
    // 失败时 err 给出可读原因
    bool init(const std::string& kernelPath, std::string& err);

    bool available() const { return ready_; }
    void printDeviceInfo() const;
    const std::string& buildLog() const { return buildLog_; }

    // 计时执行两层卷积：conv1 输出常驻 GPU 作为 conv2 输入
    // warmup 次预热后跑 iters 次，返回每层平均/最小 kernel 耗时与墙钟耗时
    bool runPipeline(const ConvShape& s1, const ConvShape& s2,
                     const float* in, const float* w1, const float* w2,
                     bool useVec4, int warmup, int iters,
                     Timing& t1kernel, Timing& t2kernel, Timing& wallPerIter,
                     std::string& err);

    // 执行一遍并分段抽样回读，与参考结果比对（避免整张 500MB+ 回读）
    bool verifyPipeline(const ConvShape& s1, const ConvShape& s2,
                        const float* in, const float* w1, const float* w2,
                        bool useVec4, const float* ref1, const float* ref2,
                        size_t segments, size_t sampleElems, double relTol, double absTol,
                        DiffResult& diff1, DiffResult& diff2, std::string& err);

private:
    bool buildProgram(const std::string& kernelPath, std::string& err);
    bool enqueueLayer(const ConvShape& s, cl_mem inBuf, cl_mem wBuf, cl_mem outBuf,
                      bool useVec4, cl_event* event);
    bool kernelMsOf(cl_event evt, double& ms);
    void release();

    bool ready_ = false;
    cl_platform_id platform_ = nullptr;
    cl_device_id device_ = nullptr;
    cl_context ctx_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program program_ = nullptr;
    cl_kernel kernelScalar_ = nullptr;
    cl_kernel kernelVec4_ = nullptr;
    std::string buildLog_;
    std::string deviceName_;
    std::string deviceVersion_;
    cl_uint computeUnits_ = 0;
    size_t maxWorkGroup_ = 1;
    size_t preferredMultiple_ = 1;
};
