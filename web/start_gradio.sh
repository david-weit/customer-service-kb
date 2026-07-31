#!/bin/sh
# 启动 Gradio：若 7860 已被占用则先结束占用进程，再激活 .venv 并启动。
# 可用: sh web/start_gradio.sh  或  bash web/start_gradio.sh
set -eu

PORT="${GRADIO_SERVER_PORT:-7860}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${PROJECT_ROOT}"

kill_port() {
  port="$1"
  pids=""

  if command -v lsof >/dev/null 2>&1; then
    pids=$(lsof -tiTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true)
  elif command -v fuser >/dev/null 2>&1; then
    pids=$(fuser -n tcp "${port}" 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' || true)
  elif command -v ss >/dev/null 2>&1; then
    pids=$(
      ss -lptn "sport = :${port}" 2>/dev/null \
        | grep -oE 'pid=[0-9]+' \
        | cut -d= -f2 \
        | sort -u || true
    )
  else
    # python:slim 等精简镜像：用 python 扫 /proc 找监听端口的 pid
    pids=$(
      python3 - "${port}" <<'PY' 2>/dev/null || true
import os, re, sys
port = int(sys.argv[1])
needle = f"{port:04X}"
inodes = set()
for path in ("/proc/net/tcp", "/proc/net/tcp6"):
    if not os.path.exists(path):
        continue
    with open(path) as f:
        next(f, None)
        for line in f:
            parts = line.split()
            if len(parts) < 10:
                continue
            local = parts[1]
            state = parts[3]
            inode = parts[9]
            if state != "0A":  # LISTEN
                continue
            if local.rsplit(":", 1)[-1].upper() == needle:
                inodes.add(inode)
if not inodes:
    sys.exit(0)
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    fd_dir = f"/proc/{pid}/fd"
    try:
        for fd in os.listdir(fd_dir):
            try:
                target = os.readlink(f"{fd_dir}/{fd}")
            except OSError:
                continue
            m = re.match(r"socket:\[(\d+)\]", target)
            if m and m.group(1) in inodes:
                print(pid)
                break
    except OSError:
        continue
PY
    )
  fi

  pids=$(echo "${pids}" | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')
  # 去掉首尾空格后判断是否为空
  trimmed=$(echo "${pids}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  if [ -z "${trimmed}" ]; then
    echo "端口 ${port} 空闲。"
    return 0
  fi

  echo "端口 ${port} 被占用，结束进程: ${trimmed}"
  # shellcheck disable=SC2086
  kill ${trimmed} 2>/dev/null || true
  sleep 1
  # shellcheck disable=SC2086
  kill -9 ${trimmed} 2>/dev/null || true
  echo "已清理端口 ${port}。"
}

kill_port "${PORT}"

if [ ! -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
  echo "未找到 ${PROJECT_ROOT}/.venv，请先创建虚拟环境。" >&2
  exit 1
fi

# shellcheck disable=SC1091
. "${PROJECT_ROOT}/.venv/bin/activate"
exec python "${PROJECT_ROOT}/web/gradio_app.py"
