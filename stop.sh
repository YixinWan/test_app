#!/bin/bash
# 停止脚本：优先根据 pid 文件停止，否则按端口停止
set -euo pipefail

PORT=8000
PID_FILE="uvicorn.pid"

kill_by_pid_file() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi

  local pid
  pid="$(cat "$PID_FILE" || true)"
  if [[ -z "$pid" ]]; then
    rm -f "$PID_FILE"
    return 1
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    echo "PID $pid not running. Cleaning $PID_FILE."
    rm -f "$PID_FILE"
    return 1
  fi

  echo "Stopping server by PID: $pid"
  kill "$pid" || true

  # 等待最多 5 秒优雅退出
  for _ in {1..50}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "Stopped."
      return 0
    fi
    sleep 0.1
  done

  echo "Graceful stop timed out; force killing PID: $pid"
  kill -9 "$pid" || true
  rm -f "$PID_FILE"
  echo "Stopped (forced)."
  return 0
}

kill_by_port() {
  if ! command -v lsof >/dev/null 2>&1; then
    echo "lsof not found; cannot stop by port $PORT."
    echo "Install it (e.g. apt: sudo apt-get install lsof) or stop via pid."
    return 1
  fi

  local pids
  pids="$(lsof -t -i:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    echo "No process is listening on port $PORT."
    return 0
  fi

  echo "Stopping server by port $PORT (PID(s): $pids)"
  # shellcheck disable=SC2086
  kill $pids || true

  sleep 0.5
  pids="$(lsof -t -i:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Force killing remaining PID(s): $pids"
    # shellcheck disable=SC2086
    kill -9 $pids || true
  fi

  rm -f "$PID_FILE"
  echo "Stopped."
}

if kill_by_pid_file; then
  exit 0
fi

kill_by_port
