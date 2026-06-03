import pytest
import pytest_asyncio
import aiosqlite
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch

from vllm_metrics_proxy.db import init_db
from vllm_metrics_proxy.main import create_app
from vllm_metrics_proxy.config import Settings
from vllm_metrics_proxy.auth import (
    create_api_key,
    get_api_key,
    list_api_keys,
    delete_api_key,
    update_api_key,
    mask_key,
)


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
    masked = mask_key("550e8400-e29b-41d4-a716-446655440000")
    assert masked == "550e****0000"
    assert len(masked) == 12


# ---- Integration tests: auth wired into proxy routes ----

MOCK_COMPLETION = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "model": "test-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}


def _mock_httpx_client():
    """Return a patched httpx.AsyncClient that returns a 200 JSON response."""
    mock_instance = AsyncMock()
    mock_instance.__aenter__.return_value = mock_instance
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = MOCK_COMPLETION
    mock_resp.headers = {"content-type": "application/json"}
    mock_instance.request = AsyncMock(return_value=mock_resp)
    return mock_instance


@pytest_asyncio.fixture
async def app_no_auth(tmp_path):
    """App with auth disabled (default)."""
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)
    return create_app(db_path=db_path)


@pytest_asyncio.fixture
async def app_with_auth(tmp_path):
    """App with auth enabled and a test key."""
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)
    app_settings = Settings(auth_enabled=True)
    app = create_app(settings_override=app_settings, db_path=db_path)
    return app, db_path


@pytest.mark.asyncio
async def test_no_auth_bypass(app_no_auth):
    """When auth is disabled, /v1/* requests pass without API key."""
    with patch("vllm_metrics_proxy.proxy.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client()
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
    key = await create_api_key(db_path, name="test-key")

    with patch("vllm_metrics_proxy.proxy.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client()
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
    key = await create_api_key(db_path, name="test-key")

    with patch("vllm_metrics_proxy.proxy.httpx.AsyncClient") as MockClient:
        MockClient.return_value = _mock_httpx_client()
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
    key = await create_api_key(db_path, name="temp-key", expires_in="1d")
    # Manually set expired_at to the past
    import aiosqlite as _sqlite
    from datetime import datetime, timezone, timedelta
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    async with _sqlite.connect(db_path) as conn:
        await conn.execute("UPDATE api_keys SET expired_at = ? WHERE id = ?", (past, key["id"]))
        await conn.commit()

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
