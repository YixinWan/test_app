#!/bin/bash
# 启动脚本：后台运行 FastAPI 服务，支持代码热更新

# 检查端口 8000 是否被占用
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "Warning: Port 8000 is already in use. Attempting to restart..."
    kill $(lsof -t -i:8000)
    sleep 1
fi

echo "Starting uvicorn server with reload..."
# 使用 nohup 后台运行，日志输出到 app.log
nohup python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload > app.log 2>&1 &

# 记录 PID，便于 stop.sh 关闭
echo $! > uvicorn.pid

echo "Server is running in background."
echo "View logs with: tail -f app.log"
echo "Stop server with: ./stop.sh"
