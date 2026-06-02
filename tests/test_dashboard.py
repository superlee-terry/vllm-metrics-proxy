import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from vllm_metrics_proxy.main import create_app
from vllm_metrics_proxy.db import init_db, insert_request


@pytest_asyncio.fixture
async def app_with_data(tmp_path):
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)
    app = create_app(db_path=db_path)

    for i in range(5):
        await insert_request(db_path, {
            "id": f"req-{i}",
            "model": "qwen3.6-27b" if i < 3 else "gemma-4-31b",
            "stream": True,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cached_tokens": 80,
            "reasoning_tokens": None,
            "latency_ms": 1000.0,
            "ttft_ms": 40.0,
            "prompt_speed": 2500.0,
            "completion_speed": 52.1,
            "cached_ratio": 0.8,
            "status": "success",
            "error_message": None,
        })
    return app, db_path


@pytest.mark.asyncio
async def test_health_endpoint(app_with_data):
    app, _ = app_with_data
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_summary_endpoint(app_with_data):
    app, _ = app_with_data
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/summary?since=1h")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_requests"] == 5
    assert data["total_prompt_tokens"] == 500
    assert len(data["by_model"]) == 2


@pytest.mark.asyncio
async def test_requests_endpoint_pagination(app_with_data):
    app, _ = app_with_data
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/requests?since=1h&limit=2&offset=0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 5
    assert len(data["requests"]) == 2


@pytest.mark.asyncio
async def test_summary_all_time(app_with_data):
    app, _ = app_with_data
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/summary?since=all")
    assert resp.status_code == 200
    assert resp.json()["total_requests"] == 5
