#pragma once
// 卷积形状描述：CPU 与 OpenCL 共用，布局统一为 NCHW / FP32
//   input : [in_c][in_h][in_w]
//   weight: [out_c][in_c][k][k]
//   output: [out_c][out_h][out_w]
#include <cstddef>

struct ConvShape {
    int in_h  = 1024;
    int in_w  = 1024;
    int in_c  = 3;
    int out_c = 64;
    int k     = 3;    // 卷积核边长
    int pad   = 1;
    int stride = 1;

    int out_h() const { return (in_h + 2 * pad - k) / stride + 1; }
    int out_w() const { return (in_w + 2 * pad - k) / stride + 1; }

    size_t inElems()  const { return (size_t)in_h * (size_t)in_w * (size_t)in_c; }
    size_t outElems() const { return (size_t)out_h() * (size_t)out_w() * (size_t)out_c; }
    size_t weightElems() const { return (size_t)out_c * (size_t)in_c * (size_t)k * (size_t)k; }

    size_t inBytes()     const { return inElems() * sizeof(float); }
    size_t outBytes()    const { return outElems() * sizeof(float); }
    size_t weightBytes() const { return weightElems() * sizeof(float); }

    // 2 * out_h * out_w * out_c * in_c * k * k
    double gflops() const {
        return 2.0 * (double)outElems() * (double)in_c * (double)k * (double)k / 1e9;
    }

    bool valid() const {
        return in_h > 0 && in_w > 0 && in_c > 0 && out_c > 0 &&
               k > 0 && pad >= 0 && stride > 0 &&
               (in_h + 2 * pad - k) >= 0 && (in_w + 2 * pad - k) >= 0;
    }
};

// 默认两层配置：1024x1024x3 -> 8 -> 16，3x3 / pad=1 / stride=1
inline ConvShape makeConv1(int size) {
    ConvShape s;
    s.in_h = s.in_w = size;
    s.in_c = 3;
    s.out_c = 8;
    s.k = 3;
    s.pad = 1;
    s.stride = 1;
    return s;
}

inline ConvShape makeConv2(const ConvShape& conv1) {
    ConvShape s;
    s.in_h = conv1.out_h();
    s.in_w = conv1.out_w();
    s.in_c = conv1.out_c;
    s.out_c = 16;
    s.k = 3;
    s.pad = 1;
    s.stride = 1;
    return s;
}
