# API Key Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add API key validation to `/v1/*` proxy endpoints with token generation, expiry management, and Dashboard UI.

**Architecture:** FastAPI `Depends` injection into `routes/proxy.py`. Standalone `auth.py` module handles verification + DB CRUD for keys. SQLite `api_keys` table in existing DB. Dashboard gets a new "API Keys" section.

**Tech Stack:** FastAPI, aiosqlite, pydantic-settings, vanilla JS (existing stack — no new dependencies)

**Design Spec:** `docs/superpowers/specs/2026-06-03-api-key-auth-design.md`

---

### Task 1: Config — add `auth_enabled` setting

**Files:**
- Modify: `vllm_metrics_proxy/config.py`

- [ ] **Step 1: Add `auth_enabled` field to Settings**

In `vllm_metrics_proxy/config.py`, add one field:

```python
class Settings(BaseSettings):
    vllm_upstream: str = "http://localhost:8001"
    proxy_port: int = 8000
    db_path: str = "./metrics.db"
    log_level: str = "INFO"
    auth_enabled: bool = False  # AUTH_ENABLED env var

    model_config = {"env_prefix": ""}
```

- [ ] **Step 2: Run existing tests to verify no breakage**

Run: `cd /mnt/data/vllm-metrics-proxy && python -m pytest tests/ -v`
Expected: All 33 tests pass

- [ ] **Step 3: Commit**

```bash
git add vllm_metrics_proxy/config.py
git commit -m "feat(auth): add auth_enabled config setting"
```

---

### Task 2: Database — `api_keys` table and CRUD functions

**Files:**
- Modify: `vllm_metrics_proxy/db.py`
- Test: `tests/test_auth.py`

- [ ] **Step 1: Write the failing test for DB schema and CRUD**

Create `tests/test_auth.py`:

```python
import pytest
import pytest_asyncio
import aiosqlite
from vllm_metrics_proxy.db import init_db
from vllm_metrics_proxy.auth import create_api_key, get_api_key, list_api_keys, delete_api_key, update_api_key, mask_key


@pytest_asyncio.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    await init_db(path)
    return path


@pytest.mark.asyncio
async def test_api_keys_table_created(db_path):
    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='api_keys'"
        ) as cur:
            row = await cur.fetchone()
            assert row is not None


@pytest.mark.asyncio
async def test_create_api_key_no_expiry(db_path):
    key = await create_api_key(db_path, name="test-key")
    assert key["name"] == "test-key"
    assert key["expired_at"] is None
    assert key["enabled"] == 1
    assert "id" in key
    # id should be a valid UUID v4
    import uuid
    uuid.UUID(key["id"])


@pytest.mark.asyncio
async def test_create_api_key_with_expiry(db_path):
    key = await create_api_key(db_path, name="temp-key", expires_in="30d")
    assert key["expired_at"] is not None
    assert key["name"] == "temp-key"


@pytest.mark.asyncio
async def test_get_api_key(db_path):
    created = await create_api_key(db_path, name="my-key")
    fetched = await get_api_key(db_path, created["id"])
    assert fetched is not None
    assert fetched["id"] == created["id"]
    assert fetched["name"] == "my-key"


@pytest.mark.asyncio
async def test_get_api_key_not_found(db_path):
    fetched = await get_api_key(db_path, "nonexistent-id")
    assert fetched is None


@pytest.mark.asyncio
async def test_list_api_keys_masked(db_path):
    await create_api_key(db_path, name="key-a")
    await create_api_key(db_path, name="key-b")
    keys = await list_api_keys(db_path)
    assert len(keys) == 2
    for k in keys:
        assert k["masked_id"] == mask_key(k["id"])
        # masked_id should not equal the full id
        assert k["masked_id"] != k["id"]


@pytest.mark.asyncio
async def test_delete_api_key(db_path):
    created = await create_api_key(db_path, name="to-delete")
    await delete_api_key(db_path, created["id"])
    fetched = await get_api_key(db_path, created["id"])
    assert fetched is None


@pytest.mark.asyncio
async def test_update_api_key_disable(db_path):
    created = await create_api_key(db_path, name="toggle-key")
    await update_api_key(db_path, created["id"], enabled=False)
    fetched = await get_api_key(db_path, created["id"])
    assert fetched["enabled"] == 0

    # Re-enable
    await update_api_key(db_path, created["id"], enabled=True)
    fetched = await get_api_key(db_path, created["id"])
    assert fetched["enabled"] == 1


@pytest.mark.asyncio
async def test_mask_key():
    # UUID v4 format: 8-4-4-4-12 chars
    masked = mask_key("550e8400-e29b-41d4-a716-446655440000")
    assert masked == "550e****4000"
    assert len(masked) == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/data/vllm-metrics-proxy && python -m pytest tests/test_auth.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_api_key' from 'vllm_metrics_proxy.auth'`

- [ ] **Step 3: Add `api_keys` schema to `db.py` and create `auth.py` with CRUD functions**

In `vllm_metrics_proxy/db.py`, append the `api_keys` table to the existing `SCHEMA` string:

```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    ...existing schema unchanged...
);

CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status);

CREATE TABLE IF NOT EXISTS api_keys (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    expired_at  TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
    enabled     INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_api_keys_expired_at ON api_keys(expired_at);
"""
```

Create `vllm_metrics_proxy/auth.py`:

```python
from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone

import aiosqlite

from vllm_metrics_proxy.config import settings


# ---- Key masking ----

def mask_key(key_id: str) -> str:
    """Mask a UUID: keep first 4 and last 4 chars, replace middle with ****."""
    if len(key_id) < 8:
        return key_id
    return key_id[:4] + "****" + key_id[-4:]


# ---- Expiry parsing ----

_EXPIRY_RE = re.compile(r"^(\d+)(h|d)$")


def parse_expires_in(expires_in: str | None) -> str | None:
    """Parse '30d', '1h', etc. into ISO 8601 datetime string. None means never expire."""
    if expires_in is None:
        return None
    m = _EXPIRY_RE.match(expires_in)
    if not m:
        raise ValueError(f"Invalid expires_in format: {expires_in!r}. Use '30d', '1h', etc.")
    value = int(m.group(1))
    unit = m.group(2)
    now = datetime.now(timezone.utc)
    if unit == "h":
        delta = timedelta(hours=value)
    else:
        delta = timedelta(days=value)
    return now + delta


# ---- DB CRUD ----

def _api_key_db(db_path: str) -> str:
    """Return db_path, ensuring api_keys schema exists."""
    return db_path


async def create_api_key(
    db_path: str, *, name: str, expires_in: str | None = None,
) -> dict:
    """Create a new API key. Returns the full key record (with plaintext id)."""
    key_id = str(uuid.uuid4())
    expired_at = parse_expires_in(expires_in)
    expired_at_str = expired_at.isoformat() if expired_at else None

    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO api_keys (id, name, expired_at) VALUES (?, ?, ?)",
            (key_id, name, expired_at_str),
        )
        await conn.commit()

    return {
        "id": key_id,
        "name": name,
        "expired_at": expired_at_str,
        "enabled": 1,
    }


async def get_api_key(db_path: str, key_id: str) -> dict | None:
    """Fetch a single API key by its id."""
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT * FROM api_keys WHERE id = ?", (key_id,)
        ) as cur:
            row = await cur.fetchone()
            if row is None:
                return None
            return dict(row)


async def list_api_keys(db_path: str) -> list[dict]:
    """List all API keys with masked ids."""
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT * FROM api_keys ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["masked_id"] = mask_key(d["id"])
                result.append(d)
            return result


async def delete_api_key(db_path: str, key_id: str) -> bool:
    """Delete an API key. Returns True if deleted, False if not found."""
    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(
            "DELETE FROM api_keys WHERE id = ?", (key_id,)
        )
        await conn.commit()
        return cursor.rowcount > 0


async def update_api_key(
    db_path: str, key_id: str,
    *,
    enabled: bool | None = None,
    expires_in: str | None = None,
) -> bool:
    """Update an API key's enabled status and/or expiry. Returns True if updated."""
    fields: list[str] = []
    params: list = []

    if enabled is not None:
        fields.append("enabled = ?")
        params.append(1 if enabled else 0)

    if expires_in is not None:
        expired_at = parse_expires_in(expires_in)
        fields.append("expired_at = ?")
        params.append(expired_at.isoformat() if expired_at else None)

    if not fields:
        return False

    params.append(key_id)
    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(
            f"UPDATE api_keys SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        await conn.commit()
        return cursor.rowcount > 0


# ---- Verification (FastAPI Depends) ----

_NO_AUTH_SENTINEL = "__no_auth__"


def _extract_key_from_request(headers: dict) -> str | None:
    """Extract API key from Authorization: Bearer or X-API-Key header."""
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    x_key = headers.get("x-api-key")
    if x_key:
        return x_key.strip()
    return None


async def verify_api_key(request: Request) -> str:
    """FastAPI Depends dependency. Returns key_id on success, raises HTTPException on failure."""
    from fastapi import HTTPException
    from starlette.requests import Request

    if not settings.auth_enabled:
        return _NO_AUTH_SENTINEL

    key_id = _extract_key_from_request(dict(request.headers))
    if not key_id:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")

    db_path = request.app.state.db_path
    key = await get_api_key(db_path, key_id)
    if key is None:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")

    if not key["enabled"]:
        raise HTTPException(status_code=403, detail="API key has been disabled")

    if key["expired_at"] is not None:
        expired_at = datetime.fromisoformat(key["expired_at"])
        if datetime.now(timezone.utc) > expired_at:
            raise HTTPException(status_code=401, detail="API key has expired")

    return key_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/data/vllm-metrics-proxy && python -m pytest tests/test_auth.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run full test suite to check no breakage**

Run: `cd /mnt/data/vllm-metrics-proxy && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add vllm_metrics_proxy/db.py vllm_metrics_proxy/auth.py tests/test_auth.py
git commit -m "feat(auth): add api_keys table, CRUD functions, and auth verification"
```

---

### Task 3: Wire auth into proxy routes

**Files:**
- Modify: `vllm_metrics_proxy/routes/proxy.py`
- Test: `tests/test_auth.py` (add integration tests)

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_auth.py`:

```python
from httpx import ASGITransport, AsyncClient
from vllm_metrics_proxy.main import create_app
from vllm_metrics_proxy.db import init_db
from unittest.mock import AsyncMock, MagicMock, patch


@pytest_asyncio.fixture
async def app_no_auth(tmp_path):
    """App with auth disabled (default)."""
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)
    return create_app(db_path=db_path)


@pytest_asyncio.fixture
async def app_with_auth(tmp_path):
    """App with auth enabled and a test key."""
    from vllm_metrics_proxy.config import Settings
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)
    app_settings = Settings(auth_enabled=True)
    app = create_app(settings_override=app_settings, db_path=db_path)
    return app, db_path


@pytest.mark.asyncio
async def test_no_auth_bypass(app_no_auth):
    """When auth is disabled, /v1/* requests pass without API key."""
    mock_response = {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    with patch("vllm_metrics_proxy.proxy.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.__aenter__.return_value = mock_instance
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_resp.headers = {"content-type": "application/json"}
        mock_instance.request = AsyncMock(return_value=mock_resp)
        MockClient.return_value = mock_instance

        transport = ASGITransport(app=app_no_auth)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
            )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_enabled_rejects_no_key(app_with_auth):
    """When auth is enabled, /v1/* requests without API key get 401."""
    app, _ = app_with_auth
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
        )
    assert resp.status_code == 401
    assert "API key" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_auth_enabled_rejects_invalid_key(app_with_auth):
    """When auth is enabled, invalid API key gets 401."""
    app, _ = app_with_auth
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": "Bearer invalid-uuid"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_auth_enabled_accepts_valid_key_bearer(app_with_auth):
    """Valid key via Authorization: Bearer is accepted."""
    app, db_path = app_with_auth
    from vllm_metrics_proxy.auth import create_api_key
    key = await create_api_key(db_path, name="test-key")

    mock_response = {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    with patch("vllm_metrics_proxy.proxy.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.__aenter__.return_value = mock_instance
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_resp.headers = {"content-type": "application/json"}
        mock_instance.request = AsyncMock(return_value=mock_resp)
        MockClient.return_value = mock_instance

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": f"Bearer {key['id']}"},
            )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_enabled_accepts_valid_key_x_header(app_with_auth):
    """Valid key via X-API-Key header is accepted."""
    app, db_path = app_with_auth
    from vllm_metrics_proxy.auth import create_api_key
    key = await create_api_key(db_path, name="test-key")

    mock_response = {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "model": "test-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    with patch("vllm_metrics_proxy.proxy.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.__aenter__.return_value = mock_instance
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_resp.headers = {"content-type": "application/json"}
        mock_instance.request = AsyncMock(return_value=mock_resp)
        MockClient.return_value = mock_instance

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
                headers={"X-API-Key": key["id"]},
            )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_auth_rejects_disabled_key(app_with_auth):
    """Disabled key gets 403."""
    app, db_path = app_with_auth
    from vllm_metrics_proxy.auth import create_api_key, update_api_key
    key = await create_api_key(db_path, name="test-key")
    await update_api_key(db_path, key["id"], enabled=False)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {key['id']}"},
        )
    assert resp.status_code == 403
    assert "disabled" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_auth_rejects_expired_key(app_with_auth):
    """Expired key gets 401."""
    app, db_path = app_with_auth
    from vllm_metrics_proxy.auth import create_api_key
    # expires_in=-1d → already expired
    key = await create_api_key(db_path, name="temp-key", expires_in="-1d")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
            headers={"Authorization": f"Bearer {key['id']}"},
        )
    assert resp.status_code == 401
    assert "expired" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_dashboard_no_auth_required(app_with_auth):
    """Dashboard endpoints are not affected by auth."""
    app, _ = app_with_auth
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
    assert resp.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/data/vllm-metrics-proxy && python -m pytest tests/test_auth.py -v -k "test_no_auth_bypass or test_auth_enabled_rejects_no_key"`
Expected: FAIL — `/v1/*` routes don't have `Depends(verify_api_key)` yet, so they still pass through

- [ ] **Step 3: Add `Depends(verify_api_key)` to proxy route**

Modify `vllm_metrics_proxy/routes/proxy.py`:

```python
from fastapi import APIRouter, Depends, Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from vllm_metrics_proxy.auth import verify_api_key
from vllm_metrics_proxy.proxy import proxy_request

router = APIRouter()


@router.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"], response_model=None)
async def forward_v1(request: Request, key_id: str = Depends(verify_api_key)):
    return await proxy_request(request)
```

Only two changes: import `Depends` and `verify_api_key`, add `key_id` parameter.

- [ ] **Step 4: Run auth tests to verify they pass**

Run: `cd /mnt/data/vllm-metrics-proxy && python -m pytest tests/test_auth.py -v`
Expected: All auth tests PASS

- [ ] **Step 5: Run full test suite — verify existing proxy tests still pass**

The existing `tests/test_proxy.py` tests use `create_app(db_path=db_path)` which defaults to `auth_enabled=False`, so they should continue to pass unchanged.

Run: `cd /mnt/data/vllm-metrics-proxy && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add vllm_metrics_proxy/routes/proxy.py tests/test_auth.py
git commit -m "feat(auth): wire verify_api_key dependency into /v1/* proxy routes"
```

---

### Task 4: Key management API endpoints

**Files:**
- Modify: `vllm_metrics_proxy/routes/dashboard.py`
- Test: `tests/test_auth.py` (add API tests)

- [ ] **Step 1: Write the failing test for key management endpoints**

Append to `tests/test_auth.py`:

```python
@pytest_asyncio.fixture
async def app_with_keys(tmp_path):
    """App with DB initialized for key management tests."""
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)
    app = create_app(db_path=db_path)
    return app, db_path


@pytest.mark.asyncio
async def test_create_key_api(app_with_keys):
    app, _ = app_with_keys
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/keys", json={"name": "my-app"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "my-app"
    assert data["expired_at"] is None
    assert "id" in data
    import uuid
    uuid.UUID(data["id"])


@pytest.mark.asyncio
async def test_create_key_api_with_expiry(app_with_keys):
    app, _ = app_with_keys
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/keys", json={"name": "temp", "expires_in": "7d"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["expired_at"] is not None


@pytest.mark.asyncio
async def test_create_key_api_invalid_expiry(app_with_keys):
    app, _ = app_with_keys
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/keys", json={"name": "bad", "expires_in": "xyz"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_keys_api(app_with_keys):
    app, _ = app_with_keys
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/api/keys", json={"name": "key-a"})
        await client.post("/api/keys", json={"name": "key-b"})
        resp = await client.get("/api/keys")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["keys"]) == 2
    # Keys should be masked in listing
    for k in data["keys"]:
        assert "masked_id" in k
        assert "id" not in k  # full id should NOT be in listing


@pytest.mark.asyncio
async def test_delete_key_api(app_with_keys):
    app, _ = app_with_keys
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/api/keys", json={"name": "del-me"}).json()
        resp = await client.delete(f"/api/keys/{created['id']}")
    assert resp.status_code == 200

    # Verify deleted
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/keys")
    assert len(resp.json()["keys"]) == 0


@pytest.mark.asyncio
async def test_delete_key_api_not_found(app_with_keys):
    app, _ = app_with_keys
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/api/keys/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_key_api_disable_enable(app_with_keys):
    app, _ = app_with_keys
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post("/api/keys", json={"name": "toggle"}).json()
        # Disable
        resp = await client.patch(f"/api/keys/{created['id']}", json={"enabled": False})
        assert resp.status_code == 200
        # Verify in list
        keys = await client.get("/api/keys").json()
        assert keys["keys"][0]["enabled"] == 0
        # Re-enable
        resp = await client.patch(f"/api/keys/{created['id']}", json={"enabled": True})
        assert resp.status_code == 200
        keys = await client.get("/api/keys").json()
        assert keys["keys"][0]["enabled"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/data/vllm-metrics-proxy && python -m pytest tests/test_auth.py -v -k "test_create_key_api or test_list_keys_api or test_delete_key_api or test_update_key_api"`
Expected: FAIL — 404 endpoints not found

- [ ] **Step 3: Add key management endpoints to dashboard routes**

Add to `vllm_metrics_proxy/routes/dashboard.py`. Add imports at the top and new endpoints:

```python
from fastapi import APIRouter, Request
from starlette.responses import FileResponse, JSONResponse

from vllm_metrics_proxy.auth import create_api_key, delete_api_key, list_api_keys, update_api_key
from vllm_metrics_proxy.config import settings
from vllm_metrics_proxy.db import get_requests, get_requests_count, get_summary, get_summary_by_model
from vllm_metrics_proxy.metrics import parse_since
from vllm_metrics_proxy.gpu_stats import fetch_gpu_stats
from vllm_metrics_proxy.vllm_metrics import fetch_engine_stats
from vllm_metrics_proxy.proxy import (
    get_active_requests,
    cancel_active_request,
    register_active_request,
    unregister_active_request,
)
```

Then append these endpoints after the existing ones:

```python
# ---- API Key Management ----

@router.post("/api/keys")
async def create_key(request: Request):
    body = await request.json()
    name = body.get("name")
    if not name:
        return JSONResponse(status_code=400, content={"detail": "name is required"})
    expires_in = body.get("expires_in")
    try:
        key = await create_api_key(request.app.state.db_path, name=name, expires_in=expires_in)
        return key
    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})


@router.get("/api/keys")
async def list_keys(request: Request):
    keys = await list_api_keys(request.app.state.db_path)
    # Remove full id from listing, keep masked_id only
    safe_keys = []
    for k in keys:
        safe = {
            "masked_id": k["masked_id"],
            "name": k["name"],
            "expired_at": k["expired_at"],
            "created_at": k["created_at"],
            "enabled": k["enabled"],
        }
        safe_keys.append(safe)
    return {"keys": safe_keys}


@router.delete("/api/keys/{key_id}")
async def remove_key(request: Request, key_id: str):
    deleted = await delete_api_key(request.app.state.db_path, key_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"detail": "API key not found"})
    return {"status": "deleted", "key_id": key_id}


@router.patch("/api/keys/{key_id}")
async def patch_key(request: Request, key_id: str):
    body = await request.json()
    enabled = body.get("enabled")
    expires_in = body.get("expires_in")
    if enabled is None and expires_in is None:
        return JSONResponse(status_code=400, content={"detail": "no fields to update"})
    try:
        updated = await update_api_key(
            request.app.state.db_path, key_id,
            enabled=enabled, expires_in=expires_in,
        )
    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})
    if not updated:
        return JSONResponse(status_code=404, content={"detail": "API key not found"})
    return {"status": "updated", "key_id": key_id}
```

- [ ] **Step 4: Run key management tests**

Run: `cd /mnt/data/vllm-metrics-proxy && python -m pytest tests/test_auth.py -v -k "test_create_key_api or test_list_keys_api or test_delete_key_api or test_update_key_api"`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /mnt/data/vllm-metrics-proxy && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add vllm_metrics_proxy/routes/dashboard.py tests/test_auth.py
git commit -m "feat(auth): add key management API endpoints (CRUD)"
```

---

### Task 5: Dashboard UI — API Keys management panel

**Files:**
- Modify: `static/index.html`

- [ ] **Step 1: Add API Keys section to dashboard HTML**

Insert a new section **before** the "Model Breakdown" section (before `<!-- Model Breakdown -->`). This section goes after the "Active Requests" section.

Add CSS styles for the key management form and table (in the `<style>` block, before the responsive media queries):

```css
/* API Keys Management */
.keys-form {
  display: flex;
  gap: 10px;
  align-items: flex-end;
  margin-bottom: 12px;
}

.keys-form input, .keys-form select {
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  color: var(--text-primary);
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 13px;
  font-family: var(--font-sans);
  outline: none;
}

.keys-form input:focus, .keys-form select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(31, 111, 235, 0.3);
}

.keys-form input { flex: 1; min-width: 160px; }

.create-btn {
  background: var(--green);
  color: #fff;
  border: none;
  padding: 6px 16px;
  border-radius: 6px;
  font-size: 13px;
  font-family: var(--font-sans);
  font-weight: 500;
  cursor: pointer;
  transition: opacity 0.2s;
  white-space: nowrap;
}
.create-btn:hover { opacity: 0.85; }

.key-reveal {
  background: var(--bg-tertiary);
  border: 1px solid var(--green);
  border-radius: 6px;
  padding: 12px 16px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.key-reveal-code {
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--green);
  word-break: break-all;
  flex: 1;
}

.key-reveal-hint {
  color: var(--text-secondary);
  font-size: 12px;
}

.key-action-btn {
  background: none;
  border: 1px solid var(--border);
  color: var(--text-secondary);
  padding: 3px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-family: var(--font-sans);
  cursor: pointer;
  transition: all 0.2s;
}
.key-action-btn:hover { background: var(--bg-tertiary); color: var(--text-primary); }
.key-action-btn.danger:hover { border-color: var(--red); color: var(--red); }

.key-enabled { color: var(--green); font-weight: 600; }
.key-disabled { color: var(--red); font-weight: 600; }
.key-expired { color: var(--yellow); font-weight: 600; }
```

Insert the HTML section after the Active Requests `</div>` and before Model Breakdown:

```html
  <!-- API Keys Management -->
  <div class="section" id="keysSection">
    <div class="section-header">
      <span class="section-title">API Keys</span>
      <span class="section-badge" id="keysCount">0</span>
    </div>
    <div class="keys-form">
      <input type="text" id="keyNameInput" placeholder="Key 名称 (如 frontend-app)" maxlength="64">
      <select id="keyExpirySelect">
        <option value="">永不过期</option>
        <option value="1h">1 小时</option>
        <option value="24h">24 小时</option>
        <option value="7d">7 天</option>
        <option value="30d">30 天</option>
        <option value="90d">90 天</option>
      </select>
      <button class="create-btn" onclick="createKey()">创建 Key</button>
    </div>
    <div class="key-reveal" id="keyReveal" style="display:none;">
      <div>
        <div class="key-reveal-code" id="keyRevealCode"></div>
        <div class="key-reveal-hint">请立即复制，此 Key 只显示一次</div>
      </div>
      <button class="key-action-btn" onclick="copyRevealedKey()">复制</button>
    </div>
    <div class="requests-table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Key ID</th>
            <th>名称</th>
            <th>创建时间</th>
            <th>过期时间</th>
            <th>状态</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody id="keysTbody">
          <tr><td colspan="6" class="empty-state">加载中...</td></tr>
        </tbody>
      </table>
    </div>
  </div>
```

- [ ] **Step 2: Add JavaScript for key management**

Add the following functions inside the IIFE in the `<script>` block, after the existing `cancelReq` function and before the `refresh` function:

```javascript
  // --- API Keys Management ---

  var keysTbody = document.getElementById('keysTbody');
  var keysCountEl = document.getElementById('keysCount');
  var keyRevealEl = document.getElementById('keyReveal');
  var keyRevealCodeEl = document.getElementById('keyRevealCode');
  var keyNameInput = document.getElementById('keyNameInput');
  var keyExpirySelect = document.getElementById('keyExpirySelect');
  var revealedKey = null;

  function getKeyStatus(key) {
    if (key.enabled === 0) return { label: '已禁用', cls: 'key-disabled' };
    if (key.expired_at) {
      var exp = new Date(key.expired_at);
      if (exp < new Date()) return { label: '已过期', cls: 'key-expired' };
      return { label: '有效', cls: 'key-enabled' };
    }
    return { label: '有效', cls: 'key-enabled' };
  }

  function formatKeyExpiry(expiredAt) {
    if (!expiredAt) return '<span class="na">永不过期</span>';
    var d = new Date(expiredAt);
    return d.getFullYear() + '-'
      + String(d.getMonth() + 1).padStart(2, '0') + '-'
      + String(d.getDate()).padStart(2, '0') + ' '
      + String(d.getHours()).padStart(2, '0') + ':'
      + String(d.getMinutes()).padStart(2, '0');
  }

  async function refreshKeys() {
    try {
      var data = await fetchJSON('/api/keys');
      var keys = data.keys || [];
      keysCountEl.textContent = keys.length;

      if (keys.length === 0) {
        keysTbody.innerHTML = '<tr><td colspan="6" class="empty-state">暂无 API Key</td></tr>';
        return;
      }

      var html = '';
      for (var i = 0; i < keys.length; i++) {
        var k = keys[i];
        var status = getKeyStatus(k);
        var toggleLabel = k.enabled ? '禁用' : '启用';
        html += '<tr>'
          + '<td style="color:var(--text-secondary)">' + escapeHtml(k.masked_id) + '</td>'
          + '<td>' + escapeHtml(k.name) + '</td>'
          + '<td class="time-col">' + formatTime(k.created_at) + '</td>'
          + '<td>' + formatKeyExpiry(k.expired_at) + '</td>'
          + '<td><span class="' + status.cls + '">' + status.label + '</span></td>'
          + '<td>'
          +   '<button class="key-action-btn" onclick="toggleKey(\'' + escapeHtml(k.masked_id) + '\',' + !k.enabled + ')">' + toggleLabel + '</button> '
          +   '<button class="key-action-btn danger" onclick="deleteKey(\'' + escapeHtml(k.masked_id) + '\')">删除</button>'
          + '</td>'
          + '</tr>';
      }
      keysTbody.innerHTML = html;
    } catch (e) {
      console.error('Failed to load keys:', e);
    }
  }

  window.createKey = async function() {
    var name = keyNameInput.value.trim();
    if (!name) { alert('请输入 Key 名称'); return; }
    var expiresIn = keyExpirySelect.value || null;
    var body = { name: name };
    if (expiresIn) body.expires_in = expiresIn;

    try {
      var resp = await fetch('/api/keys', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        var err = await resp.json();
        alert('创建失败: ' + (err.detail || resp.status));
        return;
      }
      var data = await resp.json();
      revealedKey = data.id;
      keyRevealCodeEl.textContent = data.id;
      keyRevealEl.style.display = 'flex';
      keyNameInput.value = '';
      keyExpirySelect.value = '';
      refreshKeys();
    } catch (e) {
      alert('创建失败: ' + e.message);
    }
  };

  window.copyRevealedKey = function() {
    if (!revealedKey) return;
    navigator.clipboard.writeText(revealedKey).then(function() {
      keyRevealEl.style.display = 'none';
      revealedKey = null;
    });
  };

  window.toggleKey = async function(maskedId, enable) {
    if (!confirm((enable ? '启用' : '禁用') + '该 Key？')) return;
    // We need to find the real key_id - unfortunately list API only returns masked.
    // For toggle/delete, we need the full key_id. We'll store masked→id mapping.
    var keyId = keyMaskToId[maskedId];
    if (!keyId) { alert('找不到对应 Key'); return; }
    try {
      await fetch('/api/keys/' + encodeURIComponent(keyId), {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled: enable }),
      });
      refreshKeys();
    } catch (e) {
      alert('操作失败: ' + e.message);
    }
  };

  window.deleteKey = async function(maskedId) {
    if (!confirm('确认删除该 Key？此操作不可撤销。')) return;
    var keyId = keyMaskToId[maskedId];
    if (!keyId) { alert('找不到对应 Key'); return; }
    try {
      var resp = await fetch('/api/keys/' + encodeURIComponent(keyId), { method: 'DELETE' });
      if (!resp.ok) { alert('删除失败'); return; }
      refreshKeys();
    } catch (e) {
      alert('删除失败: ' + e.message);
    }
  };

  var keyMaskToId = {};

  // Override refreshKeys to build mask→id mapping from list response
  var _origRefreshKeys = refreshKeys;
  refreshKeys = async function() {
    try {
      var data = await fetchJSON('/api/keys');
      var keys = data.keys || [];
      // Build mask→id mapping - we need the full IDs for delete/toggle
      // But the list API only returns masked_id... we need to fix this.
      // For now, use masked_id as the identifier (it's unique enough for UI)
      keysCountEl.textContent = keys.length;
      if (keys.length === 0) {
        keysTbody.innerHTML = '<tr><td colspan="6" class="empty-state">暂无 API Key</td></tr>';
        return;
      }
      var html = '';
      for (var i = 0; i < keys.length; i++) {
        var k = keys[i];
        var status = getKeyStatus(k);
        var toggleLabel = k.enabled ? '禁用' : '启用';
        html += '<tr>'
          + '<td style="color:var(--text-secondary)">' + escapeHtml(k.masked_id) + '</td>'
          + '<td>' + escapeHtml(k.name) + '</td>'
          + '<td class="time-col">' + formatTime(k.created_at) + '</td>'
          + '<td>' + formatKeyExpiry(k.expired_at) + '</td>'
          + '<td><span class="' + status.cls + '">' + status.label + '</span></td>'
          + '<td>'
          +   '<button class="key-action-btn" onclick="toggleKey(\'' + escapeHtml(k.masked_id) + '\',' + !k.enabled + ')">' + toggleLabel + '</button> '
          +   '<button class="key-action-btn danger" onclick="deleteKey(\'' + escapeHtml(k.masked_id) + '\')">删除</button>'
          + '</td>'
          + '</tr>';
      }
      keysTbody.innerHTML = html;
    } catch (e) {
      console.error('Failed to load keys:', e);
    }
  };
```

**IMPORTANT:** The list API currently hides the full `id`. For delete/toggle to work, the list endpoint needs to also return the full `id` (just not displayed in the UI). Go back to `routes/dashboard.py` and update the `list_keys` endpoint to include `id` in the response:

```python
@router.get("/api/keys")
async def list_keys(request: Request):
    keys = await list_api_keys(request.app.state.db_path)
    safe_keys = []
    for k in keys:
        safe = {
            "id": k["id"],  # needed for delete/toggle
            "masked_id": k["masked_id"],
            "name": k["name"],
            "expired_at": k["expired_at"],
            "created_at": k["created_at"],
            "enabled": k["enabled"],
        }
        safe_keys.append(safe)
    return {"keys": safe_keys}
```

Then update the JavaScript to use `k.id` for operations but only display `k.masked_id`:

```javascript
  // Replace the refreshKeys function body to use k.id for operations
  var keyMaskToId = {};

  // Patch refreshKeys to build mapping
  refreshKeys = async function() {
    try {
      var data = await fetchJSON('/api/keys');
      var keys = data.keys || [];
      keyMaskToId = {};
      keysCountEl.textContent = keys.length;
      if (keys.length === 0) {
        keysTbody.innerHTML = '<tr><td colspan="6" class="empty-state">暂无 API Key</td></tr>';
        return;
      }
      var html = '';
      for (var i = 0; i < keys.length; i++) {
        var k = keys[i];
        keyMaskToId[k.masked_id] = k.id;
        var status = getKeyStatus(k);
        var toggleLabel = k.enabled ? '禁用' : '启用';
        html += '<tr>'
          + '<td style="color:var(--text-secondary)">' + escapeHtml(k.masked_id) + '</td>'
          + '<td>' + escapeHtml(k.name) + '</td>'
          + '<td class="time-col">' + formatTime(k.created_at) + '</td>'
          + '<td>' + formatKeyExpiry(k.expired_at) + '</td>'
          + '<td><span class="' + status.cls + '">' + status.label + '</span></td>'
          + '<td>'
          +   '<button class="key-action-btn" onclick="toggleKey(\'' + escapeHtml(k.masked_id) + '\',' + !k.enabled + ')">' + toggleLabel + '</button> '
          +   '<button class="key-action-btn danger" onclick="deleteKey(\'' + escapeHtml(k.masked_id) + '\')">删除</button>'
          + '</td>'
          + '</tr>';
      }
      keysTbody.innerHTML = html;
    } catch (e) {
      console.error('Failed to load keys:', e);
    }
  };
```

Add `refreshKeys()` to the initial load and polling — add to `startPolling`:

```javascript
  // In startPolling(), add:
  // Refresh keys on init (no polling needed — it's low-frequency admin action)
  refreshKeys();
```

And in the existing `refresh()` function's try block, before `setConnStatus('')`, add a `refreshKeys()` call, or better — just call it once at init after `startPolling()`.

- [ ] **Step 3: Visually verify the dashboard**

Run: `cd /mnt/data/vllm-metrics-proxy && AUTH_ENABLED=true python -m vllm_metrics_proxy` (if possible), or verify the HTML renders correctly.

- [ ] **Step 4: Run full test suite**

Run: `cd /mnt/data/vllm-metrics-proxy && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add static/index.html vllm_metrics_proxy/routes/dashboard.py
git commit -m "feat(auth): add API Keys management panel to dashboard UI"
```

---

### Task 6: Update README and start.sh

**Files:**
- Modify: `README.md`
- Modify: `start.sh`

- [ ] **Step 1: Add auth documentation to README**

Add a section to `README.md` documenting:

1. How to enable auth (`AUTH_ENABLED=true`)
2. How to create keys via API or Dashboard
3. How to use keys with clients (`Authorization: Bearer` or `X-API-Key`)
4. Key expiry formats
5. Environment variable reference for `AUTH_ENABLED`

- [ ] **Step 2: Update start.sh to support AUTH_ENABLED**

Ensure `start.sh` passes through `AUTH_ENABLED` if set in the environment.

- [ ] **Step 3: Run full test suite one final time**

Run: `cd /mnt/data/vllm-metrics-proxy && python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add README.md start.sh
git commit -m "docs: add API key authentication usage guide"
```
