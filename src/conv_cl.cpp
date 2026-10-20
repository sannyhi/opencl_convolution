#include "conv_cl.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

std::string clError(const std::string& what, cl_int err) {
    std::ostringstream os;
    os << what << " 失败 (OpenCL error " << err << ")";
    return os.str();
}

bool readFile(const std::string& path, std::string& out) {
    std::ifstream f(path.c_str(), std::ios::in | std::ios::binary);
    if (!f.is_open()) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return !out.empty();
}

size_t roundUp(size_t v, size_t m) {
    if (m <= 1) return v;
    return ((v + m - 1) / m) * m;
}

}  // namespace

OpenCLRunner::~OpenCLRunner() {
    release();
}

void OpenCLRunner::release() {
    if (kernelScalar_) { clReleaseKernel(kernelScalar_); kernelScalar_ = nullptr; }
    if (kernelVec4_) { clReleaseKernel(kernelVec4_); kernelVec4_ = nullptr; }
    if (program_) { clReleaseProgram(program_); program_ = nullptr; }
    if (queue_) { clReleaseCommandQueue(queue_); queue_ = nullptr; }
    if (ctx_) { clReleaseContext(ctx_); ctx_ = nullptr; }
    ready_ = false;
}

bool OpenCLRunner::init(const std::string& kernelPath, std::string& err) {
    cl_int status = CL_SUCCESS;

    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (status != CL_SUCCESS || numPlatforms == 0) {
        err = "未找到 OpenCL 平台（驱动未安装？）";
        return false;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    // 优先 GPU 设备
    bool found = false;
    for (cl_uint p = 0; p < numPlatforms && !found; ++p) {
        cl_device_id dev = nullptr;
        if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 1, &dev, nullptr) == CL_SUCCESS) {
            platform_ = platforms[p];
            device_ = dev;
            found = true;
        }
    }
    if (!found) {
        // 回退：任意设备（便于在只有 CPU 设备的环境里验证流程）
        for (cl_uint p = 0; p < numPlatforms && !found; ++p) {
            cl_device_id dev = nullptr;
            if (clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 1, &dev, nullptr) == CL_SUCCESS) {
                platform_ = platforms[p];
                device_ = dev;
                found = true;
            }
        }
        if (found) {
            std::cerr << "[warn] 未找到 GPU 设备，已回退到非 GPU 设备，性能数据无参考意义\n";
        }
    }
    if (!found) {
        err = "未找到可用的 OpenCL 设备";
        return false;
    }

    char name[256] = {0};
    char version[256] = {0};
    clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(version), version, nullptr);
    deviceName_ = name;
    deviceVersion_ = version;
    clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &computeUnits_, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroup_, nullptr);

    ctx_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &status);
    if (status != CL_SUCCESS) {
        err = clError("clCreateContext", status);
        return false;
    }

    // OpenCL 1.2 接口（板端驱动通常只支持 1.2）
    queue_ = clCreateCommandQueue(ctx_, device_, CL_QUEUE_PROFILING_ENABLE, &status);
    if (status != CL_SUCCESS) {
        err = clError("clCreateCommandQueue", status);
        return false;
    }

    if (!buildProgram(kernelPath, err)) return false;

    ready_ = true;
    return true;
}

bool OpenCLRunner::buildProgram(const std::string& kernelPath, std::string& err) {
    std::string source;
    if (!readFile(kernelPath, source)) {
        err = "无法读取 kernel 文件: " + kernelPath;
        return false;
    }

    const char* srcPtr = source.c_str();
    const size_t srcLen = source.size();
    cl_int status = CL_SUCCESS;
    program_ = clCreateProgramWithSource(ctx_, 1, &srcPtr, &srcLen, &status);
    if (status != CL_SUCCESS) {
        err = clError("clCreateProgramWithSource", status);
        return false;
    }

    status = clBuildProgram(program_, 1, &device_, "", nullptr, nullptr);

    // 无论成功与否都取一次日志，便于板端排错
    size_t logSize = 0;
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    if (logSize > 1) {
        std::vector<char> log(logSize, 0);
        clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        buildLog_.assign(log.data());
    }

    if (status != CL_SUCCESS) {
        err = "kernel 编译失败:\n" + buildLog_;
        return false;
    }

    kernelScalar_ = clCreateKernel(program_, "conv2d_nchw", &status);
    if (status != CL_SUCCESS) {
        err = clError("clCreateKernel(conv2d_nchw)", status);
        return false;
    }
    kernelVec4_ = clCreateKernel(program_, "conv2d_nchw_vec4", &status);
    if (status != CL_SUCCESS) {
        err = clError("clCreateKernel(conv2d_nchw_vec4)", status);
        kernelVec4_ = nullptr;
        return false;
    }

    // work-group 尺寸推导依据
    size_t mult = 0;
    if (clGetKernelWorkGroupInfo(kernelScalar_, device_, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                 sizeof(size_t), &mult, nullptr) == CL_SUCCESS && mult > 0) {
        preferredMultiple_ = mult;
    }
    return true;
}

void OpenCLRunner::printDeviceInfo() const {
    std::cout << "\n================ 设备信息 ================\n";
    std::cout << "Device        : " << deviceName_ << "\n";
    std::cout << "OpenCL        : " << deviceVersion_ << "\n";
    std::cout << "Compute units : " << computeUnits_ << "\n";
    std::cout << "Max work-group: " << maxWorkGroup_
              << "   preferred multiple: " << preferredMultiple_ << "\n";
    if (!buildLog_.empty() && buildLog_.find_first_not_of(" \t\r\n") != std::string::npos) {
        std::cout << "Build log     :\n" << buildLog_ << "\n";
    }
}

bool OpenCLRunner::enqueueLayer(const ConvShape& s, cl_mem inBuf, cl_mem wBuf, cl_mem outBuf,
                                bool useVec4, cl_event* event) {
    cl_kernel kern = useVec4 ? kernelVec4_ : kernelScalar_;
    if (kern == nullptr) return false;

    cl_int status = CL_SUCCESS;
    cl_int in_h = s.in_h, in_w = s.in_w, in_c = s.in_c;
    cl_int out_h = s.out_h(), out_w = s.out_w(), out_c = s.out_c;
    cl_int k = s.k, pad = s.pad, stride = s.stride;

    status |= clSetKernelArg(kern, 0, sizeof(cl_mem), &inBuf);
    status |= clSetKernelArg(kern, 1, sizeof(cl_mem), &wBuf);
    status |= clSetKernelArg(kern, 2, sizeof(cl_mem), &outBuf);
    status |= clSetKernelArg(kern, 3, sizeof(cl_int), &in_h);
    status |= clSetKernelArg(kern, 4, sizeof(cl_int), &in_w);
    status |= clSetKernelArg(kern, 5, sizeof(cl_int), &in_c);
    status |= clSetKernelArg(kern, 6, sizeof(cl_int), &out_h);
    status |= clSetKernelArg(kern, 7, sizeof(cl_int), &out_w);
    status |= clSetKernelArg(kern, 8, sizeof(cl_int), &out_c);
    status |= clSetKernelArg(kern, 9, sizeof(cl_int), &k);
    status |= clSetKernelArg(kern, 10, sizeof(cl_int), &pad);
    status |= clSetKernelArg(kern, 11, sizeof(cl_int), &stride);
    if (status != CL_SUCCESS) return false;

    const int oh = s.out_h();
    const int ow = s.out_w();
    const int oc = s.out_c;

    size_t global[3];
    global[0] = useVec4 ? (size_t)((ow + 3) / 4) : (size_t)ow;
    global[1] = (size_t)oh;
    global[2] = (size_t)oc;

    // work-group：按 preferred multiple 取 >=8 的合适值，并受 max work-group 限制
    size_t m = preferredMultiple_ > 0 ? preferredMultiple_ : 1;
    size_t local[3];
    local[0] = roundUp((size_t)8, m);
    if (local[0] > maxWorkGroup_) local[0] = maxWorkGroup_;
    if (local[0] > global[0]) local[0] = global[0];
    if (local[0] == 0) local[0] = 1;
    local[1] = (local[0] * 4 <= maxWorkGroup_ && global[1] >= 4) ? 4 : 1;
    local[2] = 1;

    global[0] = roundUp(global[0], local[0]);
    global[1] = roundUp(global[1], local[1]);
    global[2] = roundUp(global[2], local[2]);

    status = clEnqueueNDRangeKernel(queue_, kern, 3, nullptr, global, local, 0, nullptr, event);
    return status == CL_SUCCESS;
}

bool OpenCLRunner::kernelMsOf(cl_event evt, double& ms) {
    if (evt == nullptr) return false;
    cl_ulong start = 0, end = 0;
    cl_int s1 = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, nullptr);
    cl_int s2 = clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, nullptr);
    if (s1 != CL_SUCCESS || s2 != CL_SUCCESS || end <= start) return false;
    ms = (double)(end - start) * 1e-6;  // ns -> ms
    return true;
}

bool OpenCLRunner::runPipeline(const ConvShape& s1, const ConvShape& s2,
                               const float* in, const float* w1, const float* w2,
                               bool useVec4, int warmup, int iters,
                               Timing& t1kernel, Timing& t2kernel, Timing& wallPerIter,
                               std::string& err) {
    if (!ready_) {
        err = "OpenCL 未初始化";
        return false;
    }
    if (useVec4 && kernelVec4_ == nullptr) {
        err = "vec4 kernel 不可用";
        return false;
    }

    cl_int status = CL_SUCCESS;
    cl_mem inBuf = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  s1.inBytes(), (void*)in, &status);
    if (status != CL_SUCCESS) { err = clError("创建输入 buffer（内存不足？可降 --size）", status); return false; }

    cl_mem w1Buf = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  s1.weightBytes(), (void*)w1, &status);
    if (status != CL_SUCCESS) { err = clError("创建 conv1 权重 buffer", status); clReleaseMemObject(inBuf); return false; }

    cl_mem w2Buf = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  s2.weightBytes(), (void*)w2, &status);
    if (status != CL_SUCCESS) { err = clError("创建 conv2 权重 buffer", status); clReleaseMemObject(inBuf); clReleaseMemObject(w1Buf); return false; }

    // conv1 输出常驻 GPU，直接作为 conv2 输入
    cl_mem midBuf = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, s1.outBytes(), nullptr, &status);
    if (status != CL_SUCCESS) {
        err = clError("创建 conv1 输出 buffer（内存不足？可降 --size）", status);
        clReleaseMemObject(inBuf); clReleaseMemObject(w1Buf); clReleaseMemObject(w2Buf);
        return false;
    }

    cl_mem out2Buf = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    s2.outBytes(), nullptr, &status);
    if (status != CL_SUCCESS) {
        err = clError("创建 conv2 输出 buffer（内存不足？可降 --size）", status);
        clReleaseMemObject(inBuf); clReleaseMemObject(w1Buf); clReleaseMemObject(w2Buf);
        clReleaseMemObject(midBuf);
        return false;
    }

    std::vector<double> k1, k2, wall;
    const int total = (warmup > 0 ? warmup : 0) + (iters > 0 ? iters : 1);

    for (int i = 0; i < total; ++i) {
        const bool timed = i >= warmup;
        cl_event e1 = nullptr, e2 = nullptr;

        Clock::time_point t0 = Clock::now();
        bool ok = enqueueLayer(s1, inBuf, w1Buf, midBuf, useVec4, &e1);
        if (ok) ok = enqueueLayer(s2, midBuf, w2Buf, out2Buf, useVec4, &e2);
        clFinish(queue_);
        Clock::time_point t1 = Clock::now();

        if (!ok) {
            err = "kernel 入队失败";
            if (e1) clReleaseEvent(e1);
            if (e2) clReleaseEvent(e2);
            clReleaseMemObject(inBuf); clReleaseMemObject(w1Buf); clReleaseMemObject(w2Buf);
            clReleaseMemObject(midBuf); clReleaseMemObject(out2Buf);
            return false;
        }

        if (timed) {
            double ms1 = 0.0, ms2 = 0.0;
            if (kernelMsOf(e1, ms1)) k1.push_back(ms1); else k1.push_back(elapsedMs(t0, t1) * 0.5);
            if (kernelMsOf(e2, ms2)) k2.push_back(ms2); else k2.push_back(elapsedMs(t0, t1) * 0.5);
            wall.push_back(elapsedMs(t0, t1));
        }
        if (e1) clReleaseEvent(e1);
        if (e2) clReleaseEvent(e2);
    }

    t1kernel = summarize(k1);
    t2kernel = summarize(k2);
    wallPerIter = summarize(wall);

    clReleaseMemObject(inBuf);
    clReleaseMemObject(w1Buf);
    clReleaseMemObject(w2Buf);
    clReleaseMemObject(midBuf);
    clReleaseMemObject(out2Buf);
    return true;
}

bool OpenCLRunner::verifyPipeline(const ConvShape& s1, const ConvShape& s2,
                                  const float* in, const float* w1, const float* w2,
                                  bool useVec4, const float* ref1, const float* ref2,
                                  size_t segments, size_t sampleElems, double relTol, double absTol,
                                  DiffResult& diff1, DiffResult& diff2, std::string& err) {
    if (!ready_) {
        err = "OpenCL 未初始化";
        return false;
    }

    cl_int status = CL_SUCCESS;
    cl_mem inBuf = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  s1.inBytes(), (void*)in, &status);
    cl_mem w1Buf = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  s1.weightBytes(), (void*)w1, &status);
    cl_mem w2Buf = clCreateBuffer(ctx_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  s2.weightBytes(), (void*)w2, &status);
    cl_mem midBuf = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, s1.outBytes(), nullptr, &status);
    cl_mem out2Buf = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                    s2.outBytes(), nullptr, &status);
    if (status != CL_SUCCESS || !inBuf || !w1Buf || !w2Buf || !midBuf || !out2Buf) {
        err = clError("创建 buffer（内存不足？可降 --size）", status);
        if (inBuf) clReleaseMemObject(inBuf);
        if (w1Buf) clReleaseMemObject(w1Buf);
        if (w2Buf) clReleaseMemObject(w2Buf);
        if (midBuf) clReleaseMemObject(midBuf);
        if (out2Buf) clReleaseMemObject(out2Buf);
        return false;
    }

    bool ok = enqueueLayer(s1, inBuf, w1Buf, midBuf, useVec4, nullptr);
    if (ok) ok = enqueueLayer(s2, midBuf, w2Buf, out2Buf, useVec4, nullptr);
    clFinish(queue_);
    if (!ok) {
        err = "kernel 入队失败";
        clReleaseMemObject(inBuf); clReleaseMemObject(w1Buf); clReleaseMemObject(w2Buf);
        clReleaseMemObject(midBuf); clReleaseMemObject(out2Buf);
        return false;
    }

    // 分段抽样回读比对，避免 800MB 级整张量回读
    auto sampleAndCompare = [&](cl_mem buf, size_t totalElems, const float* ref, DiffResult& diff) {
        const size_t segs = std::max<size_t>(1, segments);
        const size_t chunk = std::max<size_t>(1, totalElems / segs);  // 均分，段间不重叠
        for (size_t seg = 0; seg < segs; ++seg) {
            const size_t off = chunk * seg;
            size_t cnt = std::min(sampleElems, std::min(chunk, totalElems - off));
            if (cnt == 0) continue;
            std::vector<float> got(cnt);
            cl_int s = clEnqueueReadBuffer(queue_, buf, CL_TRUE, off * sizeof(float),
                                           cnt * sizeof(float), got.data(), 0, nullptr, nullptr);
            if (s != CL_SUCCESS) continue;
            DiffResult d = compare(got.data(), ref + off, cnt, relTol, absTol, off);
            mergeDiff(diff, d);
        }
    };

    DiffResult d1, d2;
    sampleAndCompare(midBuf, s1.outElems(), ref1, d1);
    sampleAndCompare(out2Buf, s2.outElems(), ref2, d2);
    diff1 = d1;
    diff2 = d2;

    clReleaseMemObject(inBuf);
    clReleaseMemObject(w1Buf);
    clReleaseMemObject(w2Buf);
    clReleaseMemObject(midBuf);
    clReleaseMemObject(out2Buf);
    return true;
}
