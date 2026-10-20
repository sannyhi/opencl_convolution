#pragma once
// 通用工具：计时、随机数据、结果比对、统计、表格与 CSV 输出
#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

inline double elapsedMs(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

// 固定种子随机填充，保证 CPU 与 GPU 使用完全相同的输入
void fillRandom(float* data, size_t n, unsigned seed, float scale = 1.0f);

struct DiffResult {
    double max_abs = 0.0;      // 最大绝对误差
    double max_rel = 0.0;      // 最大相对误差
    double max_ref_abs = 0.0;  // 参考值最大幅度，用于判断绝对误差量级
    long long first_bad = -1;  // 首个超限位置（全局索引），-1 表示无
    size_t compared = 0;       // 实际比对元素数
    bool pass = true;
};

// 比较 a 与 b。
// 判据：绝对误差 <= absTol  或  相对误差 <= relTol  即视为通过。
// 只用相对误差会在参考值接近 0 时误判（分母趋零），故两者取"或"。
DiffResult compare(const float* a, const float* b, size_t n,
                   double relTol = 1e-3, double absTol = 1e-3,
                   size_t globalOffset = 0);

// 合并多个抽样段的比对结果
void mergeDiff(DiffResult& dst, const DiffResult& src);

struct Timing {
    double bestMs = 0.0;  // 最小耗时（受调度抖动影响最小）
    double avgMs  = 0.0;  // 平均耗时
    size_t count  = 0;
};

Timing summarize(const std::vector<double>& samplesMs);

// 耗时小于 1 微秒视为无效（避免除出天文数字）
inline double gflopsPerSec(double gflop, double ms) {
    return ms > 1e-3 ? gflop / (ms / 1000.0) : 0.0;
}

inline double speedup(double baseMs, double ms) {
    return (ms > 1e-3 && baseMs > 1e-3) ? baseMs / ms : 0.0;
}

std::string humanBytes(size_t bytes);

// 结果表：name / layer / ms / gflops / speedup
struct ResultRow {
    std::string impl;
    std::string layer;
    double ms = 0.0;
    double gflops = 0.0;
    double speedup = 0.0;
};

void printResultTable(const std::vector<ResultRow>& rows);
bool writeCsv(const std::string& path, const std::vector<ResultRow>& rows);
