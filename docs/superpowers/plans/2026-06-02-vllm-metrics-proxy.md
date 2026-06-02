# vLLM Metrics Proxy — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a transparent reverse proxy that records per-request vLLM metrics (latency, tokens, cache ratio, speculative decoding stats) into SQLite with a Web UI dashboard, real-time engine monitoring, and active request management.

**Architecture:** FastAPI receives `/v1/*` requests, forwards them to vLLM via `httpx.AsyncClient`, parses streaming/non-streaming responses for usage data, snapshots Prometheus counters for per-request spec decode and cache stats, computes metrics, writes to SQLite, and tracks in-flight requests. A single-page dashboard reads from SQLite and Prometheus via API endpoints.

**Tech Stack:** Python 3.11+, FastAPI, uvicorn, httpx, aiosqlite, pydantic-settings, vanilla JS/CSS

---

## File Map

| File | Responsibility |
|---|---|
| `vllm_metrics_proxy/config.py` | Env var loading, `Settings` class |
| `vllm_metrics_proxy/metrics.py` | Pure functions: latency, speed, cache ratio, total_tps |
| `vllm_metrics_proxy/vllm_metrics.py` | Prometheus parser, counter delta tracker, engine stats |
| `vllm_metrics_proxy/db.py` | SQLite init, insert, query helpers |
| `vllm_metrics_proxy/proxy.py` | Core proxy logic: forward + collect metrics + active request tracking |
| `vllm_metrics_proxy/routes/proxy.py` | `/v1/*` route registration + utility passthrough routes |
| `vllm_metrics_proxy/routes/dashboard.py` | Dashboard page + API endpoints |
| `vllm_metrics_proxy/main.py` | App factory, lifespan (DB init), mount routes |
| `vllm_metrics_proxy/__main__.py` | CLI entrypoint |
| `static/index.html` | Single-page dashboard UI (dark theme) |
| `start.sh` | Startup script |
| `tests/` | pytest-asyncio test suite |

---

### Task 1: Project scaffolding + config ✅

- [x] Create project directory, pyproject.toml, config.py
- [x] Install dependencies, verify import

---

### Task 2: Data models + DB layer ✅

- [x] Implement db.py with SCHEMA, INSERT_SQL, init_db, insert_request, get_requests, get_summary, get_summary_by_model
- [x] Tests pass (test_db.py)

---

### Task 3: Metrics computation ✅

- [x] Implement metrics.py with compute_metrics and parse_since
- [x] Tests pass (test_metrics.py)

---

### Task 4: Core proxy logic ✅

- [x] Implement proxy.py with streaming + non-streaming handlers
- [x] HTTP method forwarding (not hardcoded POST)
- [x] Request filtering (skip requests without model)
- [x] Tests pass (test_proxy.py)

---

### Task 5: Dashboard API endpoints ✅

- [x] Implement dashboard.py with /api/summary, /api/requests, /api/health
- [x] Tests pass (test_dashboard.py)

---

### Task 6: Web UI (static HTML dashboard) ✅

- [x] Single-page dashboard with dark theme
- [x] Summary cards, model breakdown, request log table
- [x] Auto-refresh (5s polling)

---

### Task 7: CLI entrypoint + production run ✅

- [x] __main__.py with uvicorn.run
- [x] start.sh script with port argument and upstream check

---

### Task 8: Streaming test + README ✅

- [x] Streaming proxy integration test
- [x] README.md

---

### Task 9: Prometheus counter delta tracking ✅

**Files:** `vllm_metrics_proxy/vllm_metrics.py` (new), `vllm_metrics_proxy/proxy.py`

- [x] Implement `_parse_prometheus()` — regex-based Prometheus text format parser
- [x] Implement `snapshot_counters()` — snapshot spec_decode, prefix_cache, cached_tokens, generation_tokens
- [x] Implement `measure_counter_deltas()` — compute deltas since last snapshot with activity check
- [x] Integrate into proxy.py: snapshot before request, measure after request
- [x] Supplement cached_tokens and completion_tokens from Prometheus if API response lacks them

---

### Task 10: Engine stats API + Dashboard panel ✅

**Files:** `vllm_metrics_proxy/vllm_metrics.py`, `vllm_metrics_proxy/routes/dashboard.py`, `static/index.html`

- [x] Implement `fetch_engine_stats()` — structured engine stats from Prometheus
- [x] Add `GET /api/engine-stats` endpoint
- [x] Dashboard: engine stats cards row (state, KV cache, spec accept rate, queue, cache hit rate)

---

### Task 11: Utility endpoint proxying ✅

**Files:** `vllm_metrics_proxy/routes/proxy.py`

- [x] Forward `/health`, `/ping` (GET+POST), `/version`, `/openapi.json` to vLLM
- [x] Use `_proxy_passthrough()` for raw Response forwarding without metrics recording

---

### Task 12: DB schema extensions ✅

**Files:** `vllm_metrics_proxy/db.py`, `vllm_metrics_proxy/metrics.py`

- [x] Add `total_tps`, `spec_draft_tokens`, `spec_accepted_tokens` columns
- [x] Update SCHEMA, INSERT_SQL, all test fixtures

---

### Task 13: Dashboard UI enhancements ✅

**Files:** `static/index.html`

- [x] Enhanced request log: input tokens, output tokens, cached tokens, Prefill T/s, Decode T/s, spec decode accept, latency, status
- [x] Engine stats cards with real-time data
- [x] Connection status indicator

---

### Task 14: Request log column adjustment ✅

**Files:** `static/index.html`

- [x] Add TTFT column after Prefill T/s
- [x] Remove "总 TPS" column (not meaningful; users care about TTFT + decode speed)

---

### Task 15: Active request monitoring + cancel API 🔄

**Files:** `vllm_metrics_proxy/proxy.py`, `vllm_metrics_proxy/routes/dashboard.py`, `static/index.html`

- [ ] Implement in-memory `active_requests` tracking (register on start, unregister on complete)
- [ ] Add `GET /api/active-requests` endpoint
- [ ] Add `POST /api/active-requests/{id}/cancel` endpoint
- [ ] Implement cancellation: set flag, streaming generator checks flag, close upstream connection
- [ ] Dashboard: active requests panel with model, start time, elapsed time, cancel button
- [ ] Auto-refresh active requests panel (every 2s or with main poll)

---

## Summary

| Task | What | Status |
|---|---|---|
| 1 | Scaffolding + config | ✅ |
| 2 | Data models + DB | ✅ |
| 3 | Metrics computation | ✅ |
| 4 | Core proxy (forwarding) | ✅ |
| 5 | Dashboard API | ✅ |
| 6 | Web UI | ✅ |
| 7 | Entrypoint + production run | ✅ |
| 8 | Streaming test + README | ✅ |
| 9 | Prometheus counter delta tracking | ✅ |
| 10 | Engine stats API + panel | ✅ |
| 11 | Utility endpoint proxying | ✅ |
| 12 | DB schema extensions | ✅ |
| 13 | Dashboard UI enhancements | ✅ |
| 14 | Request log column adjustment | ✅ |
| 15 | Active request monitoring + cancel | 🔄 |
