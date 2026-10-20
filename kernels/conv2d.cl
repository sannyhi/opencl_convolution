// 两层卷积的 OpenCL kernel（NCHW / FP32，OpenCL C 1.2）
//
// 布局：
//   in  : [in_c][in_h][in_w]
//   w   : [out_c][in_c][k][k]
//   out : [out_c][out_h][out_w]
//
// conv2d_nchw      : 标量版，每个 work-item 计算 1 个输出点，global = (out_w, out_h, out_c)
// conv2d_nchw_vec4 : 向量版，每个 work-item 计算同一行连续 4 个输出点，
//                    global = (ceil(out_w/4), out_h, out_c)
// 两个 kernel 均带边界检查，global 尺寸向上取整不会越界。

__kernel void conv2d_nchw(__global const float* in,
                          __global const float* w,
                          __global float* out,
                          const int in_h,
                          const int in_w,
                          const int in_c,
                          const int out_h,
                          const int out_w,
                          const int out_c,
                          const int k,
                          const int pad,
                          const int stride) {
    const int ox = get_global_id(0);
    const int oy = get_global_id(1);
    const int oc = get_global_id(2);
    if (ox >= out_w || oy >= out_h || oc >= out_c) return;

    const int in_plane = in_h * in_w;
    const int out_plane = out_h * out_w;
    float sum = 0.0f;

    for (int ic = 0; ic < in_c; ++ic) {
        __global const float* inp = in + ic * in_plane;
        __global const float* wp = w + (oc * in_c + ic) * k * k;

        for (int kh = 0; kh < k; ++kh) {
            const int iy = oy * stride - pad + kh;
            if (iy < 0 || iy >= in_h) continue;
            __global const float* row = inp + iy * in_w;

            for (int kw = 0; kw < k; ++kw) {
                const int ix = ox * stride - pad + kw;
                if (ix < 0 || ix >= in_w) continue;
                sum += row[ix] * wp[kh * k + kw];
            }
        }
    }

    out[oc * out_plane + oy * out_w + ox] = sum;
}

__kernel void conv2d_nchw_vec4(__global const float* in,
                               __global const float* w,
                               __global float* out,
                               const int in_h,
                               const int in_w,
                               const int in_c,
                               const int out_h,
                               const int out_w,
                               const int out_c,
                               const int k,
                               const int pad,
                               const int stride) {
    const int gx = get_global_id(0);
    const int oy = get_global_id(1);
    const int oc = get_global_id(2);
    const int ox = gx * 4;
    if (oy >= out_h || oc >= out_c || ox >= out_w) return;

    const int in_plane = in_h * in_w;
    const int out_plane = out_h * out_w;
    __global float* dst = out + oc * out_plane + oy * out_w;

    float4 sum = (float4)(0.0f);

    for (int ic = 0; ic < in_c; ++ic) {
        __global const float* inp = in + ic * in_plane;
        __global const float* wp = w + (oc * in_c + ic) * k * k;

        for (int kh = 0; kh < k; ++kh) {
            const int iy = oy * stride - pad + kh;
            if (iy < 0 || iy >= in_h) continue;
            __global const float* row = inp + iy * in_w;

            for (int kw = 0; kw < k; ++kw) {
                const float wv = wp[kh * k + kw];
                const int base = ox * stride - pad + kw;

                // 仅当 4 个点全部落在图内且 stride 为 1 时走向量路径
                if (stride == 1 && ox + 4 <= out_w && base >= 0 && base + 4 <= in_w) {
                    float4 v = vload4(0, row + base);
                    sum += v * (float4)(wv);
                } else {
                    // 边界：逐点处理
                    for (int t = 0; t < 4; ++t) {
                        const int oxt = ox + t;
                        if (oxt >= out_w) break;
                        const int ix = oxt * stride - pad + kw;
                        if (ix < 0 || ix >= in_w) continue;
                        const float v = row[ix] * wv;
                        if (t == 0) sum.s0 += v;
                        else if (t == 1) sum.s1 += v;
                        else if (t == 2) sum.s2 += v;
                        else sum.s3 += v;
                    }
                }
            }
        }
    }

    if (ox + 4 <= out_w) {
        vstore4(sum, 0, dst + ox);
    } else {
        float sv[4];
        sv[0] = sum.s0;
        sv[1] = sum.s1;
        sv[2] = sum.s2;
        sv[3] = sum.s3;
        for (int t = 0; t < 4; ++t) {
            if (ox + t < out_w) dst[ox + t] = sv[t];
        }
    }
}
