#include "conv_cpu.h"

#include <algorithm>
#include <cstddef>

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#define CONV_HAS_NEON 1
#include <arm_neon.h>
#else
#define CONV_HAS_NEON 0
#endif

bool neonAvailable() {
#if CONV_HAS_NEON
    return true;
#else
    return false;
#endif
}

// 朴素实现：oc -> ic -> kh -> kw -> oy -> ox
void conv_naive(const ConvShape& s, const float* in, const float* weight, float* out) {
    const int OH = s.out_h();
    const int OW = s.out_w();
    const int IH = s.in_h;
    const int IW = s.in_w;
    const int K = s.k;
    const int P = s.pad;
    const int S = s.stride;

    const size_t inPlane = (size_t)IH * (size_t)IW;
    const size_t outPlane = (size_t)OH * (size_t)OW;

    for (int oc = 0; oc < s.out_c; ++oc) {
        float* dstPlane = out + (size_t)oc * outPlane;
        std::fill(dstPlane, dstPlane + outPlane, 0.0f);

        for (int ic = 0; ic < s.in_c; ++ic) {
            const float* srcPlane = in + (size_t)ic * inPlane;
            const float* w = weight + ((size_t)oc * (size_t)s.in_c + (size_t)ic) * (size_t)K * (size_t)K;

            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    const float wv = w[(size_t)kh * K + kw];
                    if (wv == 0.0f) continue;

                    for (int oy = 0; oy < OH; ++oy) {
                        const int iy = oy * S - P + kh;
                        if (iy < 0 || iy >= IH) continue;
                        const float* srcRow = srcPlane + (size_t)iy * IW;
                        float* dstRow = dstPlane + (size_t)oy * OW;

                        for (int ox = 0; ox < OW; ++ox) {
                            const int ix = ox * S - P + kw;
                            if (ix < 0 || ix >= IW) continue;
                            dstRow[ox] += srcRow[ix] * wv;
                        }
                    }
                }
            }
        }
    }
}

#if CONV_HAS_NEON
// NEON 实现：内层沿输出宽度 ox 用 float4 展开，一次算 4 个输出点
static void conv_neon_impl(const ConvShape& s, const float* in, const float* weight, float* out) {
    const int OH = s.out_h();
    const int OW = s.out_w();
    const int IH = s.in_h;
    const int IW = s.in_w;
    const int K = s.k;
    const int P = s.pad;
    const int S = s.stride;

    const size_t inPlane = (size_t)IH * (size_t)IW;
    const size_t outPlane = (size_t)OH * (size_t)OW;

    for (int oc = 0; oc < s.out_c; ++oc) {
        float* dstPlane = out + (size_t)oc * outPlane;
        std::fill(dstPlane, dstPlane + outPlane, 0.0f);

        for (int ic = 0; ic < s.in_c; ++ic) {
            const float* srcPlane = in + (size_t)ic * inPlane;
            const float* w = weight + ((size_t)oc * (size_t)s.in_c + (size_t)ic) * (size_t)K * (size_t)K;

            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    const float wv = w[(size_t)kh * K + kw];
                    const float32x4_t wvec = vdupq_n_f32(wv);

                    for (int oy = 0; oy < OH; ++oy) {
                        const int iy = oy * S - P + kh;
                        if (iy < 0 || iy >= IH) continue;

                        const float* srcRow = srcPlane + (size_t)iy * IW;
                        float* dstRow = dstPlane + (size_t)oy * OW;

                        int ox = 0;
                        if (S == 1) {
                            // 输入 4 点连续且落在图内时才走向量路径（pad 边界交给标量尾部）
                            for (; ox + 4 <= OW; ox += 4) {
                                const int base = ox - P + kw;
                                if (base < 0 || base + 4 > IW) break;
                                float32x4_t v = vld1q_f32(srcRow + base);       // 非对齐加载
                                float32x4_t acc = vld1q_f32(dstRow + ox);
                                acc = vmlaq_f32(acc, v, wvec);                  // acc += v * w
                                vst1q_f32(dstRow + ox, acc);
                            }
                        }

                        // 标量尾部：覆盖向量未处理的剩余点
                        for (; ox < OW; ++ox) {
                            const int ix = ox * S - P + kw;
                            if (ix < 0 || ix >= IW) continue;
                            dstRow[ox] += srcRow[ix] * wv;
                        }
                    }
                }
            }
        }
    }
}
#endif

void conv_neon(const ConvShape& s, const float* in, const float* weight, float* out) {
#if CONV_HAS_NEON
    if (s.stride == 1) {
        conv_neon_impl(s, in, weight, out);
        return;
    }
#endif
    // 非 ARM 平台或 stride != 1：退化为朴素实现
    conv_naive(s, in, weight, out);
}
