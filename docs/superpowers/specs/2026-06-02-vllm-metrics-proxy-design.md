# vLLM Metrics Proxy — Design Spec

**Date**: 2026-06-02
**Status**: Implemented
**Project**: `/mnt/data/vllm-metrics-proxy/` (独立于 club-3090)

## Goal

Build a transparent reverse proxy that sits between clients and vLLM, recording per-request metrics (latency, token usage, cache ratio, speculative decoding stats) into SQLite and exposing them via a Web UI dashboard — with real-time engine monitoring and active request management.

## Architecture

```
客户端 → Nginx (HTTPS:8443) → FastAPI (port 8080) → httpx.AsyncClient → vLLM (port 11434)
                    │
                    ├─ 解析响应，写 SQLite
                    ├─ Snapshot/measure Prometheus counter deltas
                    ├─ Serve Web UI Dashboard
                    └─ Active request tracking + cancel API
```

- **Transparent**: clients only change the base URL; request body is never modified (except auto-injecting `stream_options`).
- **Zero-intrusion**: no patches to vLLM or changes to its config.
- **Streaming-aware**: parses SSE chunks to capture TTFT and real-time token throughput.
- **Prometheus-integrated**: snapshots vLLM `/metrics` counters before/after each request for per-request spec decode and cache stats.
- **Observable**: tracks in-flight requests in memory, exposes active request list and cancel API.

## Tech Stack

| Package | Purpose |
|---|---|
| `fastapi` | Web framework + static file serving |
| `uvicorn` | ASGI server |
| `httpx` | Async HTTP forwarding (streaming support) |
| `aiosqlite` | Async SQLite operations |
| `pydantic-settings` | Environment variable config |

## Configuration (env vars)

| Variable | Default | Description |
|---|---|---|
| `VLLM_UPSTREAM` | `http://localhost:8001` | vLLM upstream address |
| `PROXY_PORT` | `8000` | Proxy listen port |
| `DB_PATH` | `./metrics.db` | SQLite file path |
| `LOG_LEVEL` | `INFO` | Log level |

## Project Structure

```
/mnt/data/vllm-metrics-proxy/
├── vllm_metrics_proxy/
│   ├── __init__.py
│   ├── __main__.py          # CLI entrypoint (python -m vllm_metrics_proxy)
│   ├── main.py              # FastAPI app factory + startup (lifespan)
│   ├── config.py            # Env vars / config (pydantic-settings)
│   ├── proxy.py             # Core proxy logic (forward + metrics collection + active tracking)
│   ├── metrics.py           # Metric computation (latency, speed, cache ratio, total_tps)
│   ├── vllm_metrics.py      # Prometheus parser + counter delta tracker + engine stats
│   ├── db.py                # SQLite connection management + init + queries
│   └── routes/
│       ├── __init__.py
│       ├── proxy.py         # /v1/* API forwarding + /health, /ping, /version, /openapi.json
│       └── dashboard.py     # Web UI routes (static HTML + API)
├── static/
│   └── index.html           # Single-page dashboard (vanilla JS + CSS, dark theme)
├── tests/
│   ├── test_metrics.py      # Unit tests for metric computation
│   ├── test_db.py           # Unit tests for DB operations
│   ├── test_proxy.py        # Integration tests for proxy routes
│   └── test_dashboard.py    # Integration tests for dashboard API
├── docs/
│   └── superpowers/         # Design spec + implementation plan
├── start.sh                 # Startup script (accepts port as first arg)
├── pyproject.toml
└── README.md
```

## Proxy Logic

### Request Flow

1. Client sends request to proxy (`/v1/*`).
2. Proxy generates `request_id` (uuid4) and records `start_time`.
3. Proxy registers request in `active_requests` (in-memory dict).
4. **Prometheus snapshot**: fetches vLLM `/metrics` to snapshot counters (spec_decode, prefix_cache, generation_tokens).
5. Proxy forwards request body via `httpx.AsyncClient` to vLLM upstream.
   - For streaming requests: auto-injects `stream_options: {"include_usage": true}` if not present, updates `content-length`.
   - Uses original HTTP method (GET/POST/PUT/DELETE/etc.), not hardcoded POST.
6. **If streaming**: proxy parses SSE chunks:
   - First non-empty `choices[0].delta.content` chunk → compute `ttft_ms`.
   - Each chunk is transparently forwarded to client as SSE.
   - Final chunk (before `data: [DONE]`) contains `usage` → extract token counts.
   - After `[DONE]`: measure Prometheus counter deltas, compute all metrics, write to SQLite, unregister from active_requests.
7. **If non-streaming**: wait for complete JSON response, parse `response.usage`, measure counter deltas, compute metrics, write to SQLite, unregister.
8. Return response to client.
9. **Filtering**: requests without a `model` field (e.g. `/v1/models`, health checks) are not recorded.

### SSE Streaming Details

vLLM streaming format:
```
data: {"id":"...","choices":[...],"usage":null}
data: {"id":"...","choices":[...],"usage":null}
data: {"id":"...","choices":[],"usage":{"prompt_tokens":100,"completion_tokens":50,...}}
data: [DONE]
```

- **TTFT**: time from request sent to first non-empty `choices[0].delta.content` chunk.
- **Prompt speed**: `prompt_tokens / (ttft_ms / 1000)` tok/s.
- **Completion speed**: `completion_tokens / ((latency_ms - ttft_ms) / 1000)` tok/s (streaming); `completion_tokens / (latency_ms / 1000)` (non-streaming).
- **stream_options injection**: vLLM doesn't include `usage` in streaming chunks by default; proxy auto-injects `stream_options: {"include_usage": true}`.

### Prometheus Counter Delta Tracking

Per-request metrics not available in API response (speculative decoding, prefix caching) are obtained via Prometheus counter deltas:

1. **Before request**: `snapshot_counters()` fetches vLLM `/metrics` and stores current values of:
   - `vllm:spec_decode_num_draft_tokens_total`
   - `vllm:spec_decode_num_accepted_tokens_total`
   - `vllm:prefix_cache_queries_total`
   - `vllm:prefix_cache_hits_total`
   - `vllm:prompt_tokens_cached_total`
   - `vllm:generation_tokens_total`

2. **After request**: `measure_counter_deltas()` fetches `/metrics` again, computes deltas.
   - Only returns if there was actual activity (prevents cross-request contamination).
   - Fills in `cached_tokens` and `completion_tokens` if not available from API response.

3. **Global state**: uses module-level `_prev_counters` dict (not per-request). Works because vLLM processes sequentially for the most part, and the activity check prevents stale reads.

### Non-Streaming

- `latency_ms = end_time - start_time`.
- `ttft_ms = latency_ms` (no streaming TTFT).
- All token data from `response.usage`, supplemented by Prometheus deltas.

### Active Request Tracking

- In-memory `active_requests` dict keyed by `request_id`.
- Registered when proxy_request starts, unregistered on completion/error.
- Exposed via `GET /api/active-requests`.
- Cancel endpoint: `POST /api/active-requests/{request_id}/cancel` sets a cancellation flag.
- Streaming requests check flag each iteration; cancelled requests get `[DONE]` + error logged.
- Non-streaming requests: cancel closes the httpx client connection.

### Error Handling

| Scenario | Behavior |
|---|---|
| vLLM unreachable | Return 502 + log error |
| vLLM returns 4xx/5xx | Pass through error code, no metric recorded |
| `usage` field missing | Record request with null token fields; supplement from Prometheus deltas |
| SQLite write failure | Log error, do not affect response |
| Request cancelled | Close upstream connection; return appropriate status to client |

## Route Coverage

### API Proxy Routes (`/v1/*`)

| Route | Methods | Notes |
|---|---|---|
| `/v1/{path:path}` | GET, POST, PUT, DELETE, PATCH, OPTIONS | Full API forwarding with metrics capture |

### Utility Passthrough Routes

| Route | Methods | Notes |
|---|---|---|
| `/health` | GET | Forwarded to vLLM (no metrics) |
| `/ping` | GET, POST | Forwarded to vLLM (no metrics) |
| `/version` | GET | Forwarded to vLLM (no metrics) |
| `/openapi.json` | GET | Forwarded to vLLM (no metrics) |

These use `_proxy_passthrough()` which does raw Response forwarding without metrics recording.

### Dashboard API Routes

| Route | Method | Purpose |
|---|---|---|
| `GET /` | GET | Serve `index.html` |
| `GET /api/health` | GET | Proxy health status |
| `GET /api/summary?since=1h` | GET | Summary cards + model breakdown |
| `GET /api/requests?since=1h&limit=50&offset=0` | GET | Paginated request log |
| `GET /api/engine-stats` | GET | Real-time vLLM engine stats from Prometheus |
| `GET /api/active-requests` | GET | List of currently in-flight requests |
| `POST /api/active-requests/{id}/cancel` | POST | Cancel an active request |

## Data Model

### SQLite Schema (single table `requests`)

```sql
CREATE TABLE requests (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),

    -- Request info
    model           TEXT,
    stream          INTEGER NOT NULL DEFAULT 0,

    -- Token usage (from vLLM response.usage + Prometheus deltas)
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    cached_tokens   INTEGER,
    reasoning_tokens INTEGER,

    -- Latency (ms)
    latency_ms      REAL,
    ttft_ms         REAL,

    -- Computed metrics
    prompt_speed    REAL,       -- prompt_tokens / (ttft_ms / 1000) tok/s
    completion_speed REAL,     -- completion_tokens / generate_time tok/s
    total_tps       REAL,      -- (prompt + completion) / total_latency tok/s
    cached_ratio    REAL,      -- cached_tokens / prompt_tokens

    -- Speculative decoding (from Prometheus counter deltas)
    spec_draft_tokens   INTEGER,
    spec_accepted_tokens INTEGER,

    -- Status
    status          TEXT NOT NULL DEFAULT 'success',
    error_message   TEXT
);

CREATE INDEX idx_requests_created_at ON requests(created_at);
CREATE INDEX idx_requests_model ON requests(model);
CREATE INDEX idx_requests_status ON requests(status);
```

## Web UI Dashboard

### Layout

- **Header**: title + connection indicator + time range selector (1h / 6h / 24h / 7d / all).
- **Summary cards** (row 1): total requests, total tokens, avg TTFT, avg Prefill T/s, avg Decode T/s.
- **Engine stats cards** (row 2): engine state (Awake/Sleeping), KV cache usage %, spec decode accept rate, queue requests (running/waiting), engine cache hit rate.
- **Model breakdown table**: per-model count, avg TTFT, avg Prefill T/s, avg Decode T/s, cache hit ratio.
- **Active requests panel**: list of currently in-flight requests with elapsed time and cancel button.
- **Request log table**: paginated list of recent requests with 11 columns.
- **Auto-refresh**: polls API every 5 seconds silently.

### Request Log Columns

| Column | Data Source | Format |
|---|---|---|
| 时间 | `created_at` | HH:MM:SS |
| Model | `model` | Truncated to 14 chars |
| 输入 | `prompt_tokens` | Number |
| 输出 | `completion_tokens` | Number |
| 缓存 | `cached_tokens` | Green if > 0 |
| Prefill T/s | `prompt_speed` | tok/s |
| TTFT | `ttft_ms` | ms or s |
| Decode T/s | `completion_speed` | tok/s |
| 草稿接受 | `spec_accepted_tokens` / `spec_draft_tokens` | accepted/total (rate%) |
| 耗时 | `latency_ms` | ms or s |
| 状态 | `status` | ✓/✗ |

### Engine Stats Cards

| Card | Data Source | Notes |
|---|---|---|
| 引擎状态 | `vllm:engine_sleep_state{sleep_state="awake"}` | Awake (green) / Sleeping (yellow) |
| KV Cache 使用率 | `vllm:kv_cache_usage_perc` | 0–100% |
| 投稿接受率 | spec accepted / draft totals | Cumulative rate |
| 队列请求 | `vllm:num_requests_running` + `vllm:num_requests_waiting` | running/waiting format |
| 引擎缓存命中率 | `vllm:prompt_tokens_cached_total` / `vllm:prompt_tokens_total` | Cumulative rate |

### Dashboard Display Rules

- Non-streaming requests: TTFT shows total latency (not streaming-specific).
- Error requests: status column shows `✗`, hover reveals `error_message`.
- Null values: displayed as `--`.

## Deployment

### Startup

```bash
./start.sh [port]           # e.g., ./start.sh 8080
# or
VLLM_UPSTREAM=http://localhost:11434 PROXY_PORT=8080 python -m vllm_metrics_proxy
```

The `start.sh` script:
- Exports all config env vars (required for Python subprocess)
- Accepts port as first argument: `./start.sh 8080`
- Checks upstream reachability before starting
- Uses `exec python -m vllm_metrics_proxy` for signal forwarding

### Nginx Reverse Proxy

```nginx
server {
    listen 8443 ssl;
    server_name vllm.superlee.site;
    ssl_certificate     /path/to/superlee.site_wildcard.pem;
    ssl_certificate_key /path/to/superlee.site_wildcard.key;

    location / {
        proxy_pass http://172.17.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;           # SSE streaming support
        proxy_cache off;
        proxy_read_timeout 300s;       # Long timeout for LLM inference
        proxy_send_timeout 300s;
    }
}
```

## Out of Scope (Future)

- Authentication / access control.
- Multi-vLLM-backend load balancing.
- Persistent alerting / notifications.
- Historical data retention/cleanup (configurable TTL).
- Rate limiting per client.
