#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>

void fillRandom(float* data, size_t n, unsigned seed, float scale) {
    // 使用固定种子的 mt19937，保证多次运行 / CPU 与 GPU 输入一致
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(gen);
    }
}

DiffResult compare(const float* a, const float* b, size_t n, double relTol, double absTol,
                   size_t globalOffset) {
    DiffResult r;
    r.compared = n;
    for (size_t i = 0; i < n; ++i) {
        const float va = a[i];
        const float vb = b[i];
        const double diff = (double)va - (double)vb;
        const double absDiff = diff >= 0 ? diff : -diff;
        const double denom = std::max(std::fabs((double)va), 1e-6);
        const double relDiff = absDiff / denom;

        if (!(absDiff == absDiff) || !(relDiff == relDiff)) {  // NaN
            r.pass = false;
            if (r.first_bad < 0) r.first_bad = (long long)(globalOffset + i);
            continue;
        }
        if (absDiff > r.max_abs) r.max_abs = absDiff;
        if (relDiff > r.max_rel) r.max_rel = relDiff;
        if (std::fabs((double)va) > r.max_ref_abs) r.max_ref_abs = std::fabs((double)va);

        // 绝对误差达标 或 相对误差达标 均视为通过
        const bool absOk = absDiff <= absTol;
        const bool relOk = relDiff <= relTol;
        if (!absOk && !relOk) {
            r.pass = false;
            if (r.first_bad < 0) r.first_bad = (long long)(globalOffset + i);
        }
    }
    return r;
}

void mergeDiff(DiffResult& dst, const DiffResult& src) {
    dst.max_abs = std::max(dst.max_abs, src.max_abs);
    dst.max_rel = std::max(dst.max_rel, src.max_rel);
    dst.max_ref_abs = std::max(dst.max_ref_abs, src.max_ref_abs);
    dst.compared += src.compared;
    if (!src.pass) {
        dst.pass = false;
        if (dst.first_bad < 0) dst.first_bad = src.first_bad;
    }
}

Timing summarize(const std::vector<double>& samplesMs) {
    Timing t;
    if (samplesMs.empty()) return t;
    double sum = 0.0;
    double best = std::numeric_limits<double>::max();
    for (double v : samplesMs) {
        sum += v;
        if (v < best) best = v;
    }
    t.bestMs = best;
    t.avgMs = sum / (double)samplesMs.size();
    t.count = samplesMs.size();
    return t;
}

std::string humanBytes(size_t bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB"};
    double v = (double)bytes;
    size_t u = 0;
    while (v >= 1024.0 && u < 3) {
        v /= 1024.0;
        ++u;
    }
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1f %s", v, units[u]);
    return std::string(buf);
}

void printResultTable(const std::vector<ResultRow>& rows) {
    std::cout << "\n================ 结果对比 ================\n";
    std::cout << std::left << std::setw(12) << "实现"
              << std::setw(10) << "层"
              << std::right << std::setw(14) << "耗时(ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(12) << "加速比" << "\n";
    std::cout << std::string(60, '-') << "\n";
    for (const ResultRow& r : rows) {
        std::cout << std::left << std::setw(12) << r.impl
                  << std::setw(10) << r.layer
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(14) << r.ms
                  << std::setw(12) << r.gflops
                  << std::setw(11) << (r.speedup > 0.0 ? std::to_string(r.speedup) + "x" : "-")
                  << "\n";
    }
    std::cout << std::string(60, '-') << "\n";
}

static std::string csvEscape(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

bool writeCsv(const std::string& path, const std::vector<ResultRow>& rows) {
    if (path.empty()) return true;
    std::ofstream f(path.c_str(), std::ios::out | std::ios::trunc);
    if (!f.is_open()) {
        std::cerr << "[warn] 无法写入 CSV: " << path << "\n";
        return false;
    }
    f << "impl,layer,ms,gflops,speedup\n";
    for (const ResultRow& r : rows) {
        f << csvEscape(r.impl) << "," << csvEscape(r.layer) << ","
          << std::fixed << std::setprecision(3) << r.ms << ","
          << std::setprecision(3) << r.gflops << ","
          << std::setprecision(3) << r.speedup << "\n";
    }
    f.close();
    std::cout << "[info] 结果已写入 " << path << "\n";
    return true;
}
