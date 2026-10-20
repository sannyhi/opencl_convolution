// RK3568 两层卷积：CPU（朴素 / NEON） vs GPU（OpenCL）性能对比
//   ./conv_bench [--size 1024] [--iters 3] [--warmup 1] [--no-neon] [--no-gpu]
//                [--scalar] [--tol 1e-3] [--sample N] [--segments N] [--csv path]
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "conv_cl.h"
#include "conv_cpu.h"
#include "conv_params.h"
#include "utils.h"

namespace {

struct Options {
    int size = 1024;
    int iters = 3;
    int warmup = 1;
    bool enableNeon = true;
    bool enableGpu = true;
    bool useVec4 = true;
    double tol = 1e-3;      // 相对误差阈值
    double absTol = 1e-3;   // 绝对误差阈值（两者满足其一即通过）
    size_t sample = 1u << 20;  // 每段抽样元素数
    size_t segments = 4;       // 抽样段数
    std::string csv = "results.csv";
    std::string kernelPath = "kernels/conv2d.cl";
};

void usage() {
    std::cout <<
        "用法: conv_bench [选项]\n"
        "  --size N        输入高宽（默认 1024）\n"
        "  --iters N       计时迭代次数（默认 3）\n"
        "  --warmup N      预热次数（默认 1）\n"
        "  --no-neon       跳过 NEON 版本\n"
        "  --no-gpu        跳过 OpenCL 版本\n"
        "  --scalar        GPU 使用标量 kernel（默认 vec4）\n"
        "  --tol T         校验相对误差阈值（默认 1e-3）\n"
        "  --abs-tol T     校验绝对误差阈值（默认 1e-3，与相对误差满足其一即通过）\n"
        "  --sample N      每段抽样元素数（默认 1048576）\n"
        "  --segments N    抽样段数（默认 4）\n"
        "  --csv PATH      CSV 输出路径（默认 results.csv，空串不写）\n"
        "  --kernel PATH   kernel 文件路径（默认 kernels/conv2d.cl）\n"
        "  -h/--help       显示帮助\n";
}

bool parseInt(const char* v, int& out) {
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (end == v || x <= 0) return false;
    out = (int)x;
    return true;
}

Options parseArgs(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](void) -> std::string {
            return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
        };
        if (a == "--size") parseInt(next().c_str(), opt.size);
        else if (a == "--iters") parseInt(next().c_str(), opt.iters);
        else if (a == "--warmup") { if (!parseInt(next().c_str(), opt.warmup)) opt.warmup = 0; }
        else if (a == "--no-neon") opt.enableNeon = false;
        else if (a == "--no-gpu") opt.enableGpu = false;
        else if (a == "--scalar") opt.useVec4 = false;
        else if (a == "--tol") opt.tol = std::atof(next().c_str());
        else if (a == "--abs-tol") opt.absTol = std::atof(next().c_str());
        else if (a == "--sample") opt.sample = (size_t)std::atoll(next().c_str());
        else if (a == "--segments") opt.segments = (size_t)std::atoll(next().c_str());
        else if (a == "--csv") opt.csv = next();
        else if (a == "--kernel") opt.kernelPath = next();
        else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        else std::cerr << "[warn] 未知参数: " << a << "\n";
    }
    return opt;
}

struct LayerTiming {
    Timing conv1;
    Timing conv2;
    double totalMs() const { return conv1.bestMs + conv2.bestMs; }
};

// CPU 计时：预热后跑 iters 次，分别记录 conv1 / conv2 耗时
template <typename Fn>
LayerTiming benchCpu(const ConvShape& s1, const ConvShape& s2,
                     const float* in, const float* w1, const float* w2,
                     float* mid, float* out, int warmup, int iters, Fn conv) {
    for (int i = 0; i < warmup; ++i) {
        conv(s1, in, w1, mid);
        conv(s2, mid, w2, out);
    }

    std::vector<double> t1, t2;
    for (int i = 0; i < iters; ++i) {
        Clock::time_point a0 = Clock::now();
        conv(s1, in, w1, mid);
        Clock::time_point a1 = Clock::now();
        conv(s2, mid, w2, out);
        Clock::time_point a2 = Clock::now();
        t1.push_back(elapsedMs(a0, a1));
        t2.push_back(elapsedMs(a1, a2));
    }

    LayerTiming lt;
    lt.conv1 = summarize(t1);
    lt.conv2 = summarize(t2);
    return lt;
}

// base1 / base2 为朴素实现对应层的耗时，加速比逐层计算
void addRows(std::vector<ResultRow>& rows, const std::string& impl,
             const LayerTiming& lt, const ConvShape& s1, const ConvShape& s2,
             double base1, double base2) {
    const double g1 = s1.gflops();
    const double g2 = s2.gflops();
    const double gTotal = g1 + g2;
    const double total = lt.totalMs();
    rows.push_back({impl, "conv1", lt.conv1.bestMs, gflopsPerSec(g1, lt.conv1.bestMs), speedup(base1, lt.conv1.bestMs)});
    rows.push_back({impl, "conv2", lt.conv2.bestMs, gflopsPerSec(g2, lt.conv2.bestMs), speedup(base2, lt.conv2.bestMs)});
    rows.push_back({impl, "合计", total, gflopsPerSec(gTotal, total), speedup(base1 + base2, total)});
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parseArgs(argc, argv);

    ConvShape s1 = makeConv1(opt.size);
    ConvShape s2 = makeConv2(s1);
    if (!s1.valid() || !s2.valid()) {
        std::cerr << "[error] 非法卷积参数\n";
        return 1;
    }

    std::cout << "================ 配置 ================\n";
    std::cout << "输入      : " << s1.in_h << "x" << s1.in_w << "x" << s1.in_c
              << "  (" << humanBytes(s1.inBytes()) << ")\n";
    std::cout << "conv1     : " << s1.in_c << " -> " << s1.out_c << ", " << s1.k << "x" << s1.k
              << ", pad=" << s1.pad << ", stride=" << s1.stride
              << "  输出 " << s1.out_h() << "x" << s1.out_w() << "x" << s1.out_c
              << " (" << humanBytes(s1.outBytes()) << ")\n";
    std::cout << "conv2     : " << s2.in_c << " -> " << s2.out_c << ", " << s2.k << "x" << s2.k
              << ", pad=" << s2.pad << ", stride=" << s2.stride
              << "  输出 " << s2.out_h() << "x" << s2.out_w() << "x" << s2.out_c
              << " (" << humanBytes(s2.outBytes()) << ")\n";
    std::cout << "计算量    : conv1 " << std::fixed << std::setprecision(2) << s1.gflops()
              << " GFLOP, conv2 " << s2.gflops() << " GFLOP\n";
    std::cout << "迭代      : warmup=" << opt.warmup << ", iters=" << opt.iters << "\n";

    // ---------------- 数据准备（CPU 与 GPU 共用同一份输入/权重） ----------------
    std::vector<float> input(s1.inElems());
    std::vector<float> w1(s1.weightElems());
    std::vector<float> w2(s2.weightElems());
    fillRandom(input.data(), input.size(), 20240601u, 1.0f);
    fillRandom(w1.data(), w1.size(), 19700101u, 0.5f);
    fillRandom(w2.data(), w2.size(), 20240102u, 0.5f);

    std::vector<float> midNaive(s1.outElems());   // conv1 输出（naive），同时作为 conv2 输入
    std::vector<float> outNaive(s2.outElems());   // conv2 输出（naive）
    std::vector<float> scratch1, scratch2;        // NEON / 校验用

    std::vector<ResultRow> rows;
    double baseMs = 0.0;

    // ---------------- CPU 朴素 ----------------
    std::cout << "\n[1/3] CPU 朴素实现 ..." << std::endl;
    LayerTiming naive = benchCpu(s1, s2, input.data(), w1.data(), w2.data(),
                                 midNaive.data(), outNaive.data(),
                                 opt.warmup, opt.iters, conv_naive);
    baseMs = naive.totalMs();
    const double base1 = naive.conv1.bestMs;
    const double base2 = naive.conv2.bestMs;
    addRows(rows, "CPU naive", naive, s1, s2, base1, base2);

    // ---------------- CPU NEON ----------------
    LayerTiming neon;
    DiffResult neonDiff1, neonDiff2;
    bool neonChecked = false;
    if (opt.enableNeon && neonAvailable()) {
        std::cout << "[2/3] CPU NEON 实现 ..." << std::endl;
        scratch1.assign(s1.outElems(), 0.0f);
        scratch2.assign(s2.outElems(), 0.0f);
        neon = benchCpu(s1, s2, input.data(), w1.data(), w2.data(),
                        scratch1.data(), scratch2.data(),
                        opt.warmup, opt.iters, conv_neon);
        addRows(rows, "CPU NEON", neon, s1, s2, base1, base2);

        // 立即校验并释放缓冲，避免与 GPU 缓冲叠加导致内存峰值过高
        conv_neon(s1, input.data(), w1.data(), scratch1.data());
        conv_neon(s2, midNaive.data(), w2.data(), scratch2.data());
        neonDiff1 = compare(scratch1.data(), midNaive.data(), s1.outElems(), opt.tol, opt.absTol);
        neonDiff2 = compare(scratch2.data(), outNaive.data(), s2.outElems(), opt.tol, opt.absTol);
        neonChecked = true;
        scratch1.clear(); scratch1.shrink_to_fit();
        scratch2.clear(); scratch2.shrink_to_fit();
    } else {
        std::cout << "[2/3] 跳过 NEON（未启用或当前平台非 ARM）" << std::endl;
    }

    // ---------------- GPU OpenCL ----------------
    LayerTiming gpu;
    Timing gpuWall;
    bool gpuOk = false;
    DiffResult gpuDiff1, gpuDiff2;

    if (opt.enableGpu) {
        std::cout << "[3/3] OpenCL GPU 实现 ..." << std::endl;
        OpenCLRunner runner;
        std::string err;
        if (!runner.init(opt.kernelPath, err)) {
            std::cerr << "[warn] OpenCL 初始化失败: " << err << "\n";
        } else {
            runner.printDeviceInfo();
            Timing t1, t2, wall;
            std::string runErr;
            if (!runner.runPipeline(s1, s2, input.data(), w1.data(), w2.data(),
                                    opt.useVec4, opt.warmup, opt.iters, t1, t2, wall, runErr)) {
                std::cerr << "[warn] GPU 执行失败: " << runErr << "\n";
            } else {
                gpu.conv1 = t1;
                gpu.conv2 = t2;
                gpuWall = wall;
                gpuOk = true;

                const std::string impl = opt.useVec4 ? "GPU(vec4)" : "GPU(scalar)";
                const double g1 = s1.gflops();
                const double g2 = s2.gflops();
                rows.push_back({impl, "conv1", gpu.conv1.bestMs, gflopsPerSec(g1, gpu.conv1.bestMs), speedup(base1, gpu.conv1.bestMs)});
                rows.push_back({impl, "conv2", gpu.conv2.bestMs, gflopsPerSec(g2, gpu.conv2.bestMs), speedup(base2, gpu.conv2.bestMs)});
                rows.push_back({impl, "合计", gpu.totalMs(), gflopsPerSec(g1 + g2, gpu.totalMs()), speedup(baseMs, gpu.totalMs())});
                rows.push_back({impl, "墙钟/迭代", gpuWall.bestMs, gflopsPerSec(g1 + g2, gpuWall.bestMs), speedup(baseMs, gpuWall.bestMs)});

                // 精度校验：分段抽样回读
                std::string verr;
                if (!runner.verifyPipeline(s1, s2, input.data(), w1.data(), w2.data(),
                                           opt.useVec4, midNaive.data(), outNaive.data(),
                                           opt.segments, opt.sample, opt.tol, opt.absTol,
                                           gpuDiff1, gpuDiff2, verr)) {
                    std::cerr << "[warn] GPU 校验失败: " << verr << "\n";
                }
            }
        }
    } else {
        std::cout << "[3/3] 跳过 GPU" << std::endl;
    }

    printResultTable(rows);

    // ---------------- 精度校验汇总 ----------------
    auto reportDiff = [&](const char* tag, const char* layer, const DiffResult& d, size_t sampled) {
        std::cout << std::scientific << std::setprecision(3)
                  << tag << "  " << layer
                  << ": max_abs=" << d.max_abs
                  << "  max_rel=" << d.max_rel
                  << "  |ref|max=" << d.max_ref_abs
                  << "  " << (d.pass ? "PASS" : "FAIL");
        if (sampled > 0 && sampled != d.compared) {
            std::cout << "  (抽样 " << d.compared << "/" << sampled << " 元素)";
        }
        std::cout << "\n";
    };

    std::cout << "\n================ 精度校验 ================\n";
    std::cout << "判据: |abs| <= " << std::scientific << std::setprecision(1) << opt.absTol
              << "  或  |rel| <= " << opt.tol << "  （满足其一即通过）\n";
    if (neonChecked) {
        reportDiff("NEON vs naive", "conv1", neonDiff1, 0);
        reportDiff("NEON vs naive", "conv2", neonDiff2, 0);
    }
    if (gpuOk) {
        reportDiff("GPU  vs naive", "conv1", gpuDiff1, s1.outElems());
        reportDiff("GPU  vs naive", "conv2", gpuDiff2, s2.outElems());
    }

    writeCsv(opt.csv, rows);
    return 0;
}
