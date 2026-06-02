import pytest
import pytest_asyncio
import aiosqlite
from vllm_metrics_proxy.db import init_db, insert_request, get_requests, get_summary


@pytest_asyncio.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    await init_db(path)
    return path


@pytest.mark.asyncio
async def test_init_db_creates_table(db_path):
    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='requests'"
        ) as cur:
            row = await cur.fetchone()
            assert row is not None


@pytest.mark.asyncio
async def test_insert_and_retrieve_request(db_path):
    record = {
        "id": "test-123",
        "model": "qwen3.6-27b",
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
    }
    await insert_request(db_path, record)

    rows = await get_requests(db_path, limit=10, offset=0)
    assert len(rows) == 1
    assert rows[0]["id"] == "test-123"
    assert rows[0]["model"] == "qwen3.6-27b"


@pytest.mark.asyncio
async def test_get_summary(db_path):
    for i in range(3):
        await insert_request(db_path, {
            "id": f"req-{i}",
            "model": "qwen3.6-27b",
            "stream": True,
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cached_tokens": 80,
            "reasoning_tokens": None,
            "latency_ms": 1000.0 + i * 100,
            "ttft_ms": 40.0,
            "prompt_speed": 2500.0,
            "completion_speed": 52.1,
            "cached_ratio": 0.8,
            "status": "success",
            "error_message": None,
        })
    summary = await get_summary(db_path, since_hours=1)
    assert summary["total_requests"] == 3
    assert summary["avg_ttft_ms"] == 40.0
