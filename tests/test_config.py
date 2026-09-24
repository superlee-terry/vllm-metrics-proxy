"""Tests for the runtime-config maintenance API (/api/config)."""
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from vllm_metrics_proxy.main import create_app
from vllm_metrics_proxy.config import Settings
from vllm_metrics_proxy.config_manager import (
    CONFIG_ITEMS, validate_payload,
)
from vllm_metrics_proxy.auth import create_admin_token


def make_settings():
    return Settings(
        vllm_upstream="http://localhost:11434",
        proxy_port=8000,
        db_path="/tmp/unused.db",
        log_level="ERROR",
        auth_enabled=False,
        dashboard_password="testpw",
    )


# ---- validate_payload unit tests ----

def test_validate_ok():
    out = validate_payload({"loop_window_size": 30, "loop_detection_enabled": "false"})
    assert out == {"loop_window_size": 30, "loop_detection_enabled": False}


def test_validate_range_rejected():
    with pytest.raises(ValueError):
        validate_payload({"loop_window_size": 9999})


def test_validate_unknown_key_rejected():
    with pytest.raises(ValueError):
        validate_payload({"nope": 1})


def test_validate_float_ok():
    out = validate_payload({"request_timeout_seconds": 300})
    assert out["request_timeout_seconds"] == 300


# ---- API tests ----

@ pytest_asyncio.fixture
async def app_and_db(tmp_path):
    import asyncio
    db_path = str(tmp_path / "t.db")
    from vllm_metrics_proxy.db import init_db
    await init_db(db_path)
    app = create_app(settings_override=make_settings(), db_path=db_path)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        headers = {"X-Admin-Token": create_admin_token("testpw")}
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield app, client, db_path, headers


@pytest.mark.asyncio
async def test_get_config_returns_all_items(app_and_db):
    _app, client, _db, headers = app_and_db
    resp = await client.get("/api/config", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert set(data["settings"].keys()) == set(CONFIG_ITEMS.keys())
    for v in data["settings"].values():
        assert "label" in v and "current" in v and "source" in v and "default" in v


@pytest.mark.asyncio
async def test_put_config_persists_and_returns(app_and_db):
    _app, client, db_path, headers = app_and_db
    resp = await client.put("/api/config", json={"loop_window_size": 42}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["updated"] == ["loop_window_size"]

    # persisted to DB
    from vllm_metrics_proxy.db import get_setting
    assert (await get_setting(db_path, "loop_window_size")) == "42"

    # reflected in subsequent GET
    resp2 = await client.get("/api/config", headers=headers)
    assert resp2.json()["settings"]["loop_window_size"]["current"] == 42


@pytest.mark.asyncio
async def test_put_config_reverts_to_default(app_and_db):
    _app, client, db_path, headers = app_and_db
    await client.put("/api/config", json={"loop_window_size": 42}, headers=headers)
    resp = await client.put("/api/config", json={"loop_window_size": None}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["reverted"] == ["loop_window_size"]

    from vllm_metrics_proxy.db import get_setting
    assert (await get_setting(db_path, "loop_window_size")) is None

    resp2 = await client.get("/api/config", headers=headers)
    assert resp2.json()["settings"]["loop_window_size"]["current"] == 20  # code default


@pytest.mark.asyncio
async def test_put_config_invalid_returns_422(app_and_db):
    _app, client, _db, headers = app_and_db
    resp = await client.put("/api/config", json={"loop_window_size": 9999}, headers=headers)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_config_page_served(app_and_db):
    _app, client, _db, _headers = app_and_db
    resp = await client.get("/config")
    assert resp.status_code == 200
    assert "配置维护" in resp.text
