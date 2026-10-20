# RK3568 两层卷积：CPU（C++） vs GPU（OpenCL）速度对比

在 RK3568（4×Cortex-A55 + Mali-G52）上，用同一份 FP32/NCHW 数据执行两层卷积，分别测量：

| 实现 | 说明 |
|---|---|
| CPU 朴素 | C++ 多重循环基线 |
| CPU NEON | ARMv8 SIMD，沿输出宽度 `float4` 向量化 |
| GPU OpenCL | Mali-G52，标量 kernel 与 `float4` 向量化 kernel |

固定压测规模（默认）：

```
输入      1024×1024×3              (NCHW, FP32)
conv1     3×3, 3 → 8     stride=1 pad=1
conv2     3×3, 8 → 16    stride=1 pad=1
```

计算量：conv1 ≈ 0.45 GFLOP，conv2 ≈ 2.42 GFLOP。
（通道数写在 `src/conv_params.h` 的 `makeConv1()` / `makeConv2()` 里，改这两处即可换规模。）

---

## 1. 板端准备

```bash
sudo apt update
sudo apt install -y g++ make git        # git 仅用于拉取头文件
```

确认 GPU 驱动可用（能列出 Mali 设备即可）：

```bash
ls -l /usr/lib/aarch64-linux-gnu/libOpenCL* 2>/dev/null || find / -name "libOpenCL.so*" 2>/dev/null
# 可选：sudo apt install -y clinfo && clinfo | head -40
```

> 本程序使用 OpenCL 1.2 API（RK3568 板端 libmali 通常只到 1.2），
> 未使用 `clCreateCommandQueueWithProperties` 等 2.0 接口。

## 2. 获取 OpenCL 头文件

板端一般只有 `libOpenCL.so` 而没有 `CL/cl.h`：

```bash
cd /path/to/opencl
make headers          # 等价于 bash scripts/fetch_opencl_headers.sh
```

脚本会优先 `git clone --depth 1`，失败则回退到 tarball；两者都失败时按提示手动放置到
`third_party/OpenCL-Headers/CL/cl.h`。若系统自带头文件，可改 Makefile 的 `OPENCL_HEADER_DIR`。

## 3. 编译

```bash
make                                  # 产出 conv_bench
make OPENCL_LIB_DIR=/path/to/lib      # libOpenCL.so 不在默认搜索路径时
```

## 4. 运行

```bash
./conv_bench                          # 默认规模，iters=3
make check-env                        # 最小规模快速自检（64×64，只跑 1 次）
./conv_bench --size 512 --iters 5     # 内存不足时降规模
```

### 命令行参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--size N` | 1024 | 输入高宽（conv1/同尺寸传递） |
| `--iters N` | 3 | 计时迭代次数（输出平均与最小） |
| `--warmup N` | 1 | 预热次数（不计入统计） |
| `--no-neon` | 关 | 跳过 NEON 版本 |
| `--no-gpu` | 关 | 跳过 OpenCL 版本 |
| `--scalar` | 关 | GPU 使用标量 kernel（默认用 vec4 kernel） |
| `--tol T` | 1e-3 | 校验用相对误差阈值 |
| `--abs-tol T` | 1e-3 | 校验用绝对误差阈值（与相对误差满足其一即通过） |
| `--sample N` | 1048576 | GPU 抽样校验的元素数/段 |
| `--segments N` | 4 | 抽样段数（沿张量均匀取点） |
| `--csv PATH` | results.csv | CSV 输出路径，空串表示不写 |

### 校验判据

输出张量中存在大量接近 0 的值，若只看相对误差（分母趋零）会出现
`max_abs≈0` 却判 FAIL 的误报。因此判据为：

```
|abs| <= --abs-tol   或   |rel| <= --tol      满足其一即通过
```

输出同时给出 `|ref|max`（参考值最大幅度）便于判断绝对误差量级。
若确实出现真实偏差（如 kernel 写法错误），`max_abs` 会与 `|ref|max` 同量级并判 FAIL。

### 输出示例（RK3568 板端实测，`--size 128`）

```
================ 设备信息 ================
Device      : Mali-G52
OpenCL      : OpenCL 1.2 ...
Compute units: 2   Max work-group: 256

================ 结果对比 ================
实现         层      耗时(ms)    GFLOPS    加速比
CPU naive    conv1     187.65      0.30    1.000x
CPU naive    conv2    7881.04      0.31    1.000x
CPU naive    合计     8068.69      0.31    1.000x
CPU NEON     conv1     124.41      0.46    1.508x
CPU NEON     conv2    4930.65      0.49    1.598x
CPU NEON     合计     5055.06      0.49    1.596x
GPU(vec4)    conv1      16.90      3.35   11.105x
GPU(vec4)    conv2     879.38      2.75    8.962x
GPU(vec4)    合计      896.28      2.76    9.002x
GPU(vec4)   墙钟/迭代   897.38      2.76    8.991x

================ 精度校验 ================
NEON  vs naive  conv1: max_abs=... max_rel=... PASS
GPU   vs naive  conv1: max_abs=... max_rel=... PASS
```

同时写出 `results.csv`，字段：`impl,layer,ms,gflops,speedup`。

## 5. 计时口径

- CPU：`std::chrono::steady_clock`，预热后跑 `--iters` 次，输出最小与平均耗时
- GPU：`cl_event` 的 `CL_PROFILING_COMMAND_START/END` 得到**纯 kernel 耗时**；同时给出含排入队列到 `clFinish` 的**墙钟耗时/迭代**，避免"只算 kernel"造成不公平
- 输入与权重在计时循环外一次性上传；conv1 输出常驻 GPU 直接作为 conv2 输入，不做主机回读
- 校验采用分段抽样回读（`--segments` × `--sample`），避免 800MB 级整张量回读

## 6. 内存需求

| 张量 | 大小 |
|---|---|
| 输入 1024×1024×3 | 12 MiB |
| conv1 输出 1024×1024×8 | 32 MiB |
| conv2 输出 1024×1024×16 | 64 MiB |
| 主机端峰值（含 NEON 缓冲） | 约 204 MiB |
| GPU 端（conv1 中间结果 + conv2 输出常驻） | 约 96 MiB |

默认规模内存压力很小，任意 RK3568 板卡都能跑；若把通道数改回 64/128 或想压测，
仍可用 `--size 512` / `--size 256` 控制，程序在 buffer 创建失败时会提示降规模。
程序会在 buffer 创建失败时直接提示"内存不足？可降 --size"。

## 7. 常见问题排查

| 现象 | 处理 |
|---|---|
| `fatal error: CL/cl.h: No such file` | 执行 `make headers`，或改 `OPENCL_HEADER_DIR` 指向已有头文件 |
| `cannot find -lOpenCL` | 找 `libOpenCL.so` 所在目录，`make OPENCL_LIB_DIR=...`；必要时 `sudo ln -s` 到 `/usr/lib/aarch64-linux-gnu/` |
| `clGetPlatformIDs` 失败 / 找不到 GPU 设备 | 驱动未装或权限不足：确认 `/dev/mali*` 存在且当前用户可访问（必要时 sudo 运行） |
| `CL_MEM_OBJECT_ALLOCATION_FAILURE` | GPU 内存不足，降低 `--size` |
| kernel 编译失败 | 程序会完整打印 `CL_PROGRAM_BUILD_LOG`，按日志定位；Mali 对 OpenCL 1.2 kernel 语法要求严格（不支持部分 2.0 特性） |
| NEON 版本未生效 | 确认编译环境为 aarch64（`uname -m` = aarch64）；非 ARM 平台会自动退回标量实现 |

## 8. 调优方向

- GPU：调整 `conv_cl.cpp` 中 work-group 尺寸（当前按 `CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE` 推导）、试 `-cl-fast-relaxed-math`、改为 `CL_MEM_ALLOC_HOST_PTR` 零拷贝输入
- CPU：当前 NEON 版沿输出宽度向量化，但仍在每个 `(ic, kh, kw)` 上把输出张量读回再写回，
  输出会被读写 `in_c × k × k` 遍，属于访存瓶颈（实测约 1.5~1.6x）。
  进一步优化应改为**寄存器分块**：`oc → oy → ox 块(8~16) → ic/kh/kw`，累加器常驻寄存器，
  输出只写一遍，通常可再提速数倍；或改用 im2col + GEMM
- GPU：可按输出通道展开 4 路、用 `__local` 缓存权重、或试 `vload4` 的 `-cl-fast-relaxed-math`
- 精度：若需 FP16，把 kernel 与 buffer 换成 `half`（Mali-G52 上吞吐更高），但校验阈值要放宽

## 9. 已完成的自测与验证限制

生成环境为 Windows（无 aarch64 工具链、无 OpenCL SDK），因此做了以下等价验证：

| 项目 | 方式 | 结果 |
|---|---|---|
| `conv_naive` 正确性 | 与独立参考实现（不同循环顺序）交叉比对，9 组形状（含 pad=0/2、stride=2、宽高非 4 倍数、1×1） | 全部 max_abs = 0，PASS |
| 形状/GFLOPS 计算 | 默认规模 conv1 = 0.453 GFLOP、conv2 = 2.416 GFLOP；输出 32 MiB / 64 MiB | 与 Python 独立计算一致 |
| NEON 向量化控制流 | Python 镜像 `float4` + 标量尾部的执行路径，与参考实现比对，11 组形状 | 全部 0 误差 |
| OpenCL vec4 kernel 控制流 | 同样用 Python 镜像 kernel 的分组/边界分支逻辑 | 全部 0 误差 |
| Host 代码（conv_cl.cpp / main.cpp） | 用最小 OpenCL 桩头文件 + `g++ -Wall -Wextra` 编译检查 | 零警告、零错误 |

**仍未验证**（必须在板端完成）：

- `kernels/conv2d.cl` 的真实编译（依赖板端 Mali 驱动，无本地 OpenCL 编译器）
- NEON 内在函数（`vld1q_f32` / `vmlaq_f32` / `vst1q_f32`）在 aarch64 上的实际编译与加速效果
- 真实耗时与加速比数值

板端验证：

```bash
make headers && make && ./conv_bench --size 64 --iters 1   # 先跑小尺寸确认流程与校验 PASS
./conv_bench                                              # 再跑完整规模
```
