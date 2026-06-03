# vLLM Metrics Proxy

透明反向代理，部署在 vLLM 推理引擎前方，自动采集每个请求的性能指标（延迟、Token 用量、缓存命中率、投机解码统计），存入 SQLite，并提供实时 Web Dashboard 展示引擎状态与活跃请求管理。

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)
![License MIT](https://img.shields.io/badge/license-MIT-brightgreen)

## 架构

```
客户端 → Nginx (HTTPS) → FastAPI Proxy → httpx → vLLM (OpenAI API)
                              │
                              ├─ 采集 per-request 指标 → SQLite
                              ├─ Snapshot/measure Prometheus counter deltas
                              ├─ 活跃请求追踪 + 取消 API
                              └─ 实时引擎状态 Dashboard
```

## 功能特性

### Per-Request 指标采集

| 指标 | 来源 | 说明 |
|------|------|------|
| 输入/输出 Token | API response `usage` | prompt_tokens, completion_tokens |
| 缓存命中 Token | API response + Prometheus delta | prefix cache 命中数 |
| Prefill T/s | 计算得出 | prompt_tokens / ttft_seconds |
| TTFT | Streaming 首字延迟 | 首个 content token 到达时间 |
| Decode T/s | 计算得出 | completion_tokens / generate_time |
| 草稿接受率 | Prometheus counter delta | spec_accepted / spec_draft |
| 总耗时 | 计时 | 端到端延迟 (ms) |
| 推理 Token | API response | reasoning_tokens |

### 引擎实时监控

从 vLLM Prometheus `/metrics` 端点实时获取：

- 引擎状态（Awake / Sleeping）
- KV Cache 使用率
- 投机解码（Speculative Decoding）接受率
- 队列请求数（Running / Waiting）
- 引擎全局缓存命中率

### 活跃请求管理

- 实时展示当前正在处理的请求（2s 轮询）
- 显示请求 ID、模型、类型（Stream/Sync）、已用时间
- 一键断开请求（客户端收到 499 错误）

### 透明代理

- 客户端只需修改目标端口，请求体不修改
- 自动为 Streaming 请求注入 `stream_options: {"include_usage": true}` 获取 usage 数据
- 支持 GET/POST/PUT/DELETE/PATCH/OPTIONS 所有 HTTP 方法
- 代理 `/health`、`/ping`、`/version`、`/openapi.json` 等工具端点

## 快速开始

### 安装

```bash
git clone https://github.com/superlee-terry/vllm-metrics-proxy.git
cd vllm-metrics-proxy
pip install -e .
```

### 启动

```bash
# 方式一：使用启动脚本（推荐）
./start.sh [port]

# 方式二：环境变量启动
VLLM_UPSTREAM=http://localhost:11434 PROXY_PORT=8080 vllm-metrics-proxy
```

启动后：
- API 代理：`http://localhost:8080/v1/...`
- Dashboard：`http://localhost:8080/`

### Nginx 反代配置

```nginx
server {
    listen 443 ssl;
    server_name vllm.yourdomain.com;

    ssl_certificate     /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `VLLM_UPSTREAM` | `http://localhost:8001` | vLLM 上游地址 |
| `PROXY_PORT` | `8000` | 代理监听端口 |
| `DB_PATH` | `./metrics.db` | SQLite 数据库文件路径 |
| `LOG_LEVEL` | `INFO` | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| `AUTH_ENABLED` | `false` | 是否启用 API Key 认证（`true`/`false`） |
| `ADMIN_TOKEN` | `""` (空) | 管理员口令，保护 Key 创建/修改/删除操作（空=不限） |

## API 端点

### 代理端点

| 路由 | 方法 | 说明 |
|------|------|------|
| `/v1/{path}` | GET, POST, PUT, DELETE, PATCH, OPTIONS | OpenAI API 全量转发，采集指标 |
| `/health` | GET | 健康检查（透传到 vLLM） |
| `/ping` | GET, POST | Ping（透传） |
| `/version` | GET | 版本信息（透传） |
| `/openapi.json` | GET | OpenAPI 文档（透传） |

### Dashboard API

| 路由 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Dashboard 页面 |
| `/api/health` | GET | 代理健康状态 |
| `/api/summary?since=1h` | GET | 汇总统计 + 按 Model 分组 |
| `/api/requests?since=1h&limit=50&offset=0` | GET | 分页请求日志 |
| `/api/engine-stats` | GET | 实时引擎状态（Prometheus） |
| `/api/active-requests` | GET | 活跃请求列表 |
| `/api/active-requests/{id}/cancel` | POST | 断开指定请求 |
| `/api/keys` | GET | 列出所有 API Key（掩码显示） |
| `/api/keys` | POST | 创建 API Key |
| `/api/keys/{key_id}` | DELETE | 删除 API Key |
| `/api/keys/{key_id}` | PATCH | 更新 API Key（启用/禁用、修改过期时间） |

### API Key 认证

启用认证后，所有 `/v1/*` 代理请求必须携带有效的 API Key。Dashboard 和健康检查端点不受影响。

#### 启用认证

```bash
AUTH_ENABLED=true VLLM_UPSTREAM=http://localhost:11434 PROXY_PORT=8080 vllm-metrics-proxy
```

#### 创建 API Key

通过 Dashboard UI 或 API 创建：

```bash
# 创建永久 Key
curl -X POST http://localhost:8080/api/keys \
  -H "Content-Type: application/json" \
  -d '{"name": "my-app"}'

# 创建 30 天有效期的 Key
curl -X POST http://localhost:8080/api/keys \
  -H "Content-Type: application/json" \
  -d '{"name": "temp-key", "expires_in": "30d"}'
```

创建成功后返回完整 Key（**只显示一次**）：

```json
{"id": "550e8400-e29b-41d4-a716-446655440000", "name": "my-app", "expired_at": null, "enabled": 1}
```

#### 使用 API Key

支持两种传递方式：

```bash
# 方式一：Authorization: Bearer
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer 550e8400-e29b-41d4-a716-446655440000" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.6-27b", "messages": [{"role": "user", "content": "Hi"}]}'

# 方式二：X-API-Key
curl http://localhost:8080/v1/chat/completions \
  -H "X-API-Key: 550e8400-e29b-41d4-a716-446655440000" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.6-27b", "messages": [{"role": "user", "content": "Hi"}]}'
```

#### 有效期格式

| 格式 | 含义 |
|------|------|
| 不传 / `null` | 永不过期 |
| `"1h"` | 1 小时 |
| `"24h"` | 24 小时 |
| `"7d"` | 7 天 |
| `"30d"` | 30 天 |
| `"90d"` | 90 天 |

#### 管理 API Key

设置了 `ADMIN_TOKEN` 后，创建/修改/删除 Key 需要携带管理员口令：

```bash
# 创建 Key（带管理口令）
curl -X POST http://localhost:8080/api/keys \
  -H "Content-Type: application/json" \
  -H "X-Admin-Token: your-secret" \
  -d '{"name": "my-app"}'
```

```bash
# 列出所有 Key（ID 已掩码，无需管理口令）
curl http://localhost:8080/api/keys

# 禁用 Key
curl -X PATCH http://localhost:8080/api/keys/{key_id} \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'

# 删除 Key
curl -X DELETE http://localhost:8080/api/keys/{key_id}
```

## Dashboard 界面

- **汇总卡片**：总请求数、总 Token、平均 TTFT、平均 Prefill T/s、平均 Decode T/s
- **引擎状态卡片**：引擎状态、KV Cache 使用率、投稿接受率、队列请求、缓存命中率
- **Model 分组表**：按模型维度的请求量和平均指标
- **活跃请求面板**：实时显示处理中的请求，支持一键断开
- **请求日志表**：11 列详细指标，分页加载，5 秒自动刷新
- **API Keys 面板**：创建/禁用/删除 API Key，创建时一次性显示完整 Key

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/ -v

# 启动（指定端口）
./start.sh 8080
```

## 项目结构

```
vllm-metrics-proxy/
├── vllm_metrics_proxy/
│   ├── __main__.py          # CLI 入口
│   ├── config.py            # 环境变量配置
│   ├── auth.py              # API Key 认证 + CRUD
│   ├── proxy.py             # 核心代理逻辑 + 活跃请求追踪
│   ├── metrics.py           # 指标计算
│   ├── vllm_metrics.py      # Prometheus 解析 + counter delta
│   ├── db.py                # SQLite 数据层
│   └── routes/
│       ├── proxy.py         # /v1/* 代理路由 + 工具端点
│       └── dashboard.py     # Dashboard API + Key 管理
├── static/
│   └── index.html           # 单页 Dashboard UI
├── tests/                   # pytest 测试
├── docs/superpowers/        # 设计文档
├── start.sh                 # 启动脚本
└── pyproject.toml
```

## 依赖

- Python 3.11+
- FastAPI >= 0.115
- uvicorn[standard] >= 0.34
- httpx >= 0.28
- aiosqlite >= 0.21
- pydantic-settings >= 2.7

## License

[MIT](LICENSE)
