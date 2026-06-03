# API Key Authentication & Token Management

**Date:** 2026-06-03
**Status:** Approved

## Overview

Add API key validation to the vLLM Metrics Proxy for `/v1/*` proxy endpoints, with a token generation and management system supporting configurable expiry.

## Requirements

| Item | Decision |
|------|----------|
| Protected endpoints | `/v1/*` proxy endpoints only |
| Key transport | `Authorization: Bearer <key>` + `X-API-Key: <key>` (both supported) |
| Storage | SQLite (reuse existing `metrics.db`) |
| Key format | UUID v4 (e.g., `550e8400-e29b-41d4-a716-446655440000`) |
| Expiry | Specified at creation time (e.g., `30d`, `90d`, `null` = never) |
| Management interface | REST API + Dashboard UI |
| Admin auth | None (internal network deployment) |
| Enable/disable | Environment variable `AUTH_ENABLED=true` |

## Architecture: FastAPI Dependency Injection

Use FastAPI `Depends` to inject authentication into `/v1/*` routes. A standalone `auth.py` module handles verification logic. This is the least intrusive approach — only `routes/proxy.py` function signatures change.

### Why not middleware?

- Middleware runs before route matching, requiring manual path parsing and whitelisting.
- FastAPI `Depends` provides clean separation, unified error responses (401/403 JSON), and easy test mocking.
- Dashboard and health endpoints remain completely unaffected.

## Data Model

### `api_keys` table

```sql
CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,       -- UUID v4 (the API key itself)
    name        TEXT NOT NULL,          -- human-readable label (e.g., "frontend-app")
    expired_at  TEXT,                   -- ISO 8601 expiry, NULL = never expires
    created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    enabled     INTEGER NOT NULL DEFAULT 1  -- 0=disabled, 1=enabled
);
CREATE INDEX IF NOT EXISTS idx_api_keys_expired_at ON api_keys(expired_at);
```

### Design decisions

- **Key as primary key**: UUID v4 is the key itself — no separate hash column. Simplifies lookup.
- `expired_at = NULL` means permanent (never expires).
- `enabled` field supports soft-disable (reversible, no data loss).
- Key is returned in full only on creation. List endpoints show masked version (e.g., `550e****4400`).

## Auth Module: `vllm_metrics_proxy/auth.py`

### `verify_api_key(request: Request) -> str`

A FastAPI `Depends` dependency. Returns `key_id` on success, raises `HTTPException` on failure.

### Verification flow

```
1. Check AUTH_ENABLED env var
   └─ false → return sentinel (skip auth)
2. Extract key from request.headers:
   ├─ Authorization: Bearer <key>
   └─ X-API-Key: <key>
   └─ neither present → 401 "Missing or invalid API key"
3. SELECT * FROM api_keys WHERE id = ?
   └─ not found → 401 "Missing or invalid API key"
4. Check enabled == 1
   └─ no → 403 "API key has been disabled"
5. Check expired_at IS NULL OR expired_at > now
   └─ expired → 401 "API key has expired"
6. Return key_id
```

### Config extension

```python
# vllm_metrics_proxy/config.py — add:
auth_enabled: bool = False  # AUTH_ENABLED
```

## Route Changes

### `routes/proxy.py` — minimal change

```python
from vllm_metrics_proxy.auth import verify_api_key

@router.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def forward_v1(request: Request, key_id: str = Depends(verify_api_key)):
    return await proxy_request(request)
```

Single `Depends` parameter added. No changes to existing proxy logic.

### `routes/dashboard.py` — new management API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/keys` | Create key (`{name, expires_in?: "30d"|"90d"|null}`) |
| `GET` | `/api/keys` | List all keys (masked) |
| `DELETE` | `/api/keys/{key_id}` | Delete key |
| `PATCH` | `/api/keys/{key_id}` | Update key (enable/disable, change expiry) |

No auth required on management endpoints (internal network).

## Expiry Parsing

`expires_in` parameter formats:

| Input | Meaning |
|-------|---------|
| `null` / omitted | Never expires |
| `"30d"` | 30 days from now |
| `"90d"` | 90 days from now |
| `"1h"` | 1 hour from now |
| `"7d"` | 7 days from now |

Parser: regex `^(\d+)(h|d)$` → compute future ISO 8601 timestamp.

## Dashboard UI

New **"API Keys"** panel/section in existing dashboard:

- **Create form**: name input + expiry dropdown + create button
- **Key list table**: name, created_at, expired_at, status (enabled/disabled/expired), actions (disable/enable/delete)
- **One-time key reveal**: after creation, show full key with copy-to-clipboard prompt (never shown again)
- Consistent dark theme matching existing dashboard style

## Error Response Format

```json
// 401 — missing or invalid key
{"detail": "Missing or invalid API key"}

// 401 — expired key
{"detail": "API key has expired"}

// 403 — disabled key
{"detail": "API key has been disabled"}
```

Uses FastAPI `HTTPException` for standard JSON error responses.

## New Files

| File | Purpose |
|------|---------|
| `vllm_metrics_proxy/auth.py` | Key verification dependency + DB operations |
| `tests/test_auth.py` | Unit + integration tests for auth |

## Modified Files

| File | Changes |
|------|---------|
| `vllm_metrics_proxy/config.py` | Add `auth_enabled` setting |
| `vllm_metrics_proxy/db.py` | Add `api_keys` schema, CRUD functions |
| `vllm_metrics_proxy/routes/proxy.py` | Add `Depends(verify_api_key)` to route |
| `vllm_metrics_proxy/routes/dashboard.py` | Add key management API endpoints |
| `static/index.html` | Add API Keys management UI panel |

## Test Strategy

### `test_auth.py`

- `verify_api_key` scenarios: no key, invalid key, expired, disabled, valid
- `AUTH_ENABLED=false` bypass
- Both header formats (`Authorization: Bearer` and `X-API-Key`)
- `expires_in` parsing: `30d`, `90d`, `1h`, `null`, invalid input
- Key CRUD: create, list (masked), delete, update (enable/disable)
- Integration: authenticated `/v1/chat/completions` → 200; unauthenticated with `AUTH_ENABLED=true` → 401; unauthenticated with `AUTH_ENABLED=false` → passes through
