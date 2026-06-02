from __future__ import annotations

import json
import logging
import time
import uuid

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from vllm_metrics_proxy.config import settings
from vllm_metrics_proxy.db import insert_request
from vllm_metrics_proxy.metrics import compute_metrics

logger = logging.getLogger(__name__)


async def proxy_request(request: Request) -> JSONResponse | StreamingResponse:
    """Forward a request to vLLM upstream and record metrics."""
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()
    db_path = request.app.state.db_path

    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")

    try:
        payload = json.loads(body)
        stream = payload.get("stream", False)
        model = payload.get("model")
    except (json.JSONDecodeError, AttributeError):
        stream = False
        model = None

    upstream = settings.vllm_upstream.rstrip("/")
    upstream_url = f"{upstream}{request.url.path}"

    headers = dict(request.headers)
    headers.pop("host", None)

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            if stream:
                return await _handle_streaming(
                    client, upstream_url, headers, content_type, body,
                    request_id, start_time, model, db_path,
                )
            else:
                return await _handle_non_streaming(
                    client, upstream_url, headers, content_type, body,
                    request_id, start_time, model, db_path,
                )
    except Exception as exc:
        logger.error("Proxy error [%s]: %s", request_id, exc)
        return JSONResponse(
            status_code=502,
            content={"error": f"upstream unavailable: {exc}"},
        )


async def _handle_non_streaming(
    client: httpx.AsyncClient,
    upstream_url: str,
    headers: dict,
    content_type: str,
    body: bytes,
    request_id: str,
    start_time: float,
    model: str | None,
    db_path: str,
) -> JSONResponse:
    resp = await client.post(
        upstream_url, content=body, headers=headers,
    )

    if resp.status_code >= 400:
        return JSONResponse(
            status_code=resp.status_code,
            content=resp.json() if "json" in resp.headers.get("content-type", "") else {"error": resp.text},
        )

    data = resp.json()
    end_time = time.monotonic()
    latency_ms = (end_time - start_time) * 1000.0

    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    cached_tokens = None
    reasoning_tokens = None

    details = usage.get("prompt_tokens_details") or {}
    cached_tokens = details.get("cached_tokens")

    comp_details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = comp_details.get("reasoning_tokens")

    record = compute_metrics(
        request_id=request_id,
        model=model,
        stream=False,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        latency_ms=latency_ms,
        ttft_ms=None,
    )

    await _save_record(record, db_path)
    return JSONResponse(status_code=200, content=data)


async def _handle_streaming(
    client: httpx.AsyncClient,
    upstream_url: str,
    headers: dict,
    content_type: str,
    body: bytes,
    request_id: str,
    start_time: float,
    model: str | None,
    db_path: str,
) -> StreamingResponse:
    async def stream_generator():
        prompt_tokens = None
        completion_tokens = None
        cached_tokens = None
        reasoning_tokens = None
        ttft_ms = None
        first_content_seen = False

        async with client.stream(
            "POST", upstream_url, content=body, headers=headers,
        ) as resp:
            if resp.status_code >= 400:
                error_body = await resp.aread()
                yield JSONResponse(
                    status_code=resp.status_code,
                    content={"error": error_body.decode()},
                ).body
                return

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    if line.strip():
                        yield f"{line}\n\n"
                    continue

                data_str = line[6:]

                if data_str.strip() == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break

                yield f"{line}\n\n"

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if not first_content_seen:
                    choices = chunk.get("choices") or []
                    for choice in choices:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            ttft_ms = (time.monotonic() - start_time) * 1000.0
                            first_content_seen = True
                            break

                usage = chunk.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens")
                    completion_tokens = usage.get("completion_tokens")
                    details = usage.get("prompt_tokens_details") or {}
                    cached_tokens = details.get("cached_tokens")
                    comp_details = usage.get("completion_tokens_details") or {}
                    reasoning_tokens = comp_details.get("reasoning_tokens")

        end_time = time.monotonic()
        latency_ms = (end_time - start_time) * 1000.0

        record = compute_metrics(
            request_id=request_id,
            model=model,
            stream=True,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            latency_ms=latency_ms,
            ttft_ms=ttft_ms,
        )

        await _save_record(record, db_path)

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _save_record(record: dict, db_path: str) -> None:
    try:
        await insert_request(db_path, record)
    except Exception as exc:
        logger.error("Failed to save record %s: %s", record.get("id"), exc)
