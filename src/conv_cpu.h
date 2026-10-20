#pragma once
// CPU 端卷积实现（NCHW / FP32）
//   conv_naive: 朴素多重循环，作为性能下限基线
//   conv_neon : ARM NEON SIMD 优化（沿输出宽度 float4 向量化）
// 非 ARM 平台编译时 conv_neon 自动退化为 conv_naive。
#include "conv_params.h"

void conv_naive(const ConvShape& s, const float* in, const float* weight, float* out);
void conv_neon(const ConvShape& s, const float* in, const float* weight, float* out);

// 当前编译目标是否支持 NEON
bool neonAvailable();
