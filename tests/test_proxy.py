import pytest
import pytest_asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import ASGITransport, AsyncClient
from vllm_metrics_proxy.main import create_app
from vllm_metrics_proxy.db import init_db


@pytest_asyncio.fixture
async def app(tmp_path):
    db_path = str(tmp_path / "test.db")
    await init_db(db_path)
    return create_app(db_path=db_path)


@pytest.mark.asyncio
async def test_non_streaming_proxy_forwards_and_records(app, tmp_path):
    """Non-streaming request is forwarded, response returned, metrics recorded."""
    mock_response = {
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "model": "qwen3.6-27b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_tokens": 13,
        },
    }

    with patch("vllm_metrics_proxy.proxy.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.__aenter__.return_value = mock_instance
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_response
        mock_resp.headers = {"content-type": "application/json"}
        mock_instance.post = AsyncMock(return_value=mock_resp)
        MockClient.return_value = mock_instance

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "qwen3.6-27b", "messages": [{"role": "user", "content": "Hi"}]},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "qwen3.6-27b"

    # Verify DB has the record
    from vllm_metrics_proxy.db import get_requests
    rows = await get_requests(str(tmp_path / "test.db"))
    assert len(rows) == 1
    assert rows[0]["model"] == "qwen3.6-27b"
    assert rows[0]["prompt_tokens"] == 10
    assert rows[0]["completion_tokens"] == 3


@pytest.mark.asyncio
async def test_upstream_error_passes_through(app):
    """vLLM 4xx errors are transparently passed, no metrics recorded."""
    with patch("vllm_metrics_proxy.proxy.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.__aenter__.return_value = mock_instance
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {"error": "bad request"}
        mock_resp.headers = {"content-type": "application/json"}
        mock_instance.post = AsyncMock(return_value=mock_resp)
        MockClient.return_value = mock_instance

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
            )

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_upstream_unreachable_returns_502(app):
    """When vLLM is unreachable, proxy returns 502."""
    with patch("vllm_metrics_proxy.proxy.httpx.AsyncClient") as MockClient:
        mock_instance = AsyncMock()
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.post = AsyncMock(side_effect=Exception("connection refused"))
        MockClient.return_value = mock_instance

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "test", "messages": [{"role": "user", "content": "Hi"}]},
            )

    assert resp.status_code == 502
