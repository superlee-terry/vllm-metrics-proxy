#!/usr/bin/env bash
#
# start.sh — 启动 vLLM Metrics Proxy，代理 localhost:11434 的 vLLM 实例
#
# 用法:
#   ./start.sh              # 默认代理端口 11435
#   ./start.sh 8080         # 自定义代理端口
#   PROXY_PORT=9000 ./start.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 配置 ----
export VLLM_UPSTREAM="${VLLM_UPSTREAM:-http://localhost:11434}"
export PROXY_PORT="${1:-${PROXY_PORT:-11435}}"
export DB_PATH="${DB_PATH:-${SCRIPT_DIR}/metrics.db}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

# ---- 颜色 ----
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}================================================${NC}"
echo -e "${CYAN}  vLLM Metrics Proxy${NC}"
echo -e "${CYAN}================================================${NC}"
echo ""
echo -e "  上游 vLLM:  ${GREEN}${VLLM_UPSTREAM}${NC}"
echo -e "  代理监听:   ${GREEN}http://0.0.0.0:${PROXY_PORT}${NC}"
echo -e "  数据库:     ${DB_PATH}"
echo -e "  日志级别:   ${LOG_LEVEL}"
echo ""
echo -e "  Dashboard:  ${GREEN}http://localhost:${PROXY_PORT}/${NC}"
echo -e "  API:        ${GREEN}http://localhost:${PROXY_PORT}/v1/...${NC}"
echo ""

# ---- 检查上游 ----
if curl -sf --connect-timeout 3 "${VLLM_UPSTREAM}/v1/models" > /dev/null 2>&1; then
    echo -e "  上游状态:   ${GREEN}✓ 可达${NC}"
else
    echo -e "  上游状态:   ${YELLOW}⚠ 不可达（vLLM 可能还在加载模型，代理仍会正常启动）${NC}"
fi

echo -e "${CYAN}------------------------------------------------${NC}"
echo ""

# ---- 启动 ----
exec python -m vllm_metrics_proxy
