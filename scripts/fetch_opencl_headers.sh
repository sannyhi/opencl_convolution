#!/usr/bin/env bash
# 下载 Khronos OpenCL-Headers 到 third_party/OpenCL-Headers
# RK3568 板端通常只有 libOpenCL.so（Mali 驱动），缺少 CL/cl.h 等头文件。
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/OpenCL-Headers"
TMP="$ROOT/third_party/.download"

if [ -d "$DEST/CL" ] && [ -f "$DEST/CL/cl.h" ]; then
    echo "[headers] 已存在: $DEST/CL/cl.h"
    exit 0
fi

mkdir -p "$(dirname "$DEST")"

clone_by_git() {
    command -v git >/dev/null 2>&1 || return 1
    echo "[headers] git clone --depth 1 KhronosGroup/OpenCL-Headers"
    rm -rf "$DEST"
    git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers.git "$DEST"
}

download_tarball() {
    local url="https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/heads/main.tar.gz"
    echo "[headers] 下载 tarball: $url"
    rm -rf "$TMP" "$DEST"
    mkdir -p "$TMP"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$TMP/headers.tar.gz" || return 1
    elif command -v wget >/dev/null 2>&1; then
        wget -q -O "$TMP/headers.tar.gz" "$url" || return 1
    else
        return 1
    fi
    tar -xzf "$TMP/headers.tar.gz" -C "$TMP" || return 1
    mv "$TMP"/OpenCL-Headers-* "$DEST" || return 1
    rm -rf "$TMP"
}

if clone_by_git || download_tarball; then
    if [ -f "$DEST/CL/cl.h" ]; then
        echo "[headers] 完成: $DEST/CL/cl.h"
        exit 0
    fi
fi

cat <<'EOF'
[headers] 自动下载失败，请手动准备头文件：
  1) 在能上网的机器下载 https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/heads/main.tar.gz
  2) 拷贝到板子并解压，使目录结构为 third_party/OpenCL-Headers/CL/cl.h
  3) 若板端系统已自带头文件（如 /usr/include/CL/cl.h），可直接改 Makefile 的 INCLUDES
EOF
exit 1
