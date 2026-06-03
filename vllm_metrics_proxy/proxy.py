from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import AsyncGenerator

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

from vllm_metrics_proxy.config import settings
from vllm_metrics_proxy.db import insert_request
from vllm_metrics_proxy.auth import get_api_key as _get_api_key
from vllm_metrics_proxy.metrics import compute_metrics
from vllm_metrics_proxy.vllm_metrics import snapshot_counters, measure_counter_deltas

logger = logging.getLogger(__name__)

# ---- Active request tracking ----

_active_requests: dict[str, dict] = {}
_cancel_flags: set[str] = set()
_live_streams: dict[str, httpx.Response] = {}  # request_id -> httpx Response for force-cancel


def register_active_request(request_id: str, model: str | None, stream: bool, api_key_name: str = "") -> None:
    """Register a request as active (in-flight)."""
    _active_requests[request_id] = {
        "id": request_id,
        "model": model,
        "stream": stream,
        "start_time": time.monotonic(),
        "api_key_name": api_key_name,
    }


def unregister_active_request(request_id: str) -> None:
    """Remove a request from active tracking."""
    _active_requests.pop(request_id, None)
    _cancel_flags.discard(request_id)
    _live_streams.pop(request_id, None)


def cancel_active_request(request_id: str) -> bool:
    """Cancel an active request by closing its upstream stream."""
    if request_id not in _active_requests:
        return False
    _cancel_flags.add(request_id)
    # Force-close the httpx stream to unblock the reader immediately
    resp = _live_streams.pop(request_id, None)
    if resp:
        try:
            resp.close()
        except Exception:
            pass
    return True


def get_active_requests() -> list[dict]:
    """Return list of currently active requests with elapsed time."""
    now = time.monotonic()
    result = []
    for req_id, info in _active_requests.items():
        elapsed_ms = (now - info["start_time"]) * 1000.0
        result.append({
            "id": req_id,
            "model": info["model"],
            "stream": info["stream"],
            "elapsed_ms": round(elapsed_ms, 0),
            "cancelled": req_id in _cancel_flags,
            "api_key_name": info.get("api_key_name", ""),
        })
    return result


async def proxy_request(request: Request, key_id: str | None = None) -> JSONResponse | StreamingResponse:
    """Forward a request to vLLM upstream and record metrics."""
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()
    db_path = request.app.state.db_path

    # Normalize sentinel — don't store when auth is disabled
    if key_id == "__no_auth__":
        key_id = None

    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")

    try:
        payload = json.loads(body)
        stream = payload.get("stream", False)
        model = payload.get("model")
    except (json.JSONDecodeError, AttributeError):
        stream = False
        model = None

    logger.debug(
        "PROXY [%s] %s %s model=%r stream=%r body_len=%d",
        request_id[:8], request.method, request.url.path, model, stream, len(body),
    )

    upstream = settings.vllm_upstream.rstrip("/")
    upstream_url = f"{upstream}{request.url.path}"

    headers = dict(request.headers)
    headers.pop("host", None)

    # Inject stream_options to get usage data in streaming chunks
    if stream and isinstance(payload, dict):
        stream_opts = payload.get("stream_options")
        if not stream_opts or not stream_opts.get("include_usage"):
            payload["stream_options"] = {"include_usage": True}
            body = json.dumps(payload).encode()
            content_type = "application/json"
            headers["content-length"] = str(len(body))

    # Snapshot Prometheus counters before the request
    await snapshot_counters(settings.vllm_upstream)

    # Resolve key name for active request display
    api_key_name = ""
    if key_id:
        key_row = await _get_api_key(db_path, key_id)
        if key_row:
            api_key_name = key_row.get("name", "")

    # Register active request
    register_active_request(request_id, model, stream, api_key_name=api_key_name)

    try:
        if stream:
            # Streaming: reader task handles unregister in its own finally
            return _handle_streaming(
                request.method, upstream_url, headers, content_type, body,
                request_id, start_time, model, db_path, key_id,
            )
        else:
            # Non-streaming: ensure unregister always runs (success, error, exception)
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                    return await _handle_non_streaming(
                        client, request.method, upstream_url, headers, content_type, body,
                        request_id, start_time, model, db_path, key_id,
                    )
            finally:
                unregister_active_request(request_id)
    except Exception as exc:
        # Already unregistered in finally above for non-stream; for stream this
        # shouldn't normally hit (errors handled inside reader task)
        logger.error("Proxy error [%s]: %s", request_id, exc)
        return JSONResponse(
            status_code=502,
            content={"error": f"upstream unavailable: {exc}"},
        )


async def _handle_non_streaming(
    client: httpx.AsyncClient,
    method: str,
    upstream_url: str,
    headers: dict,
    content_type: str,
    body: bytes,
    request_id: str,
    start_time: float,
    model: str | None,
    db_path: str,
    key_id: str | None = None,
) -> JSONResponse:
    resp = await client.request(
        method, upstream_url, content=body, headers=headers,
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

    try:
        deltas = await measure_counter_deltas(settings.vllm_upstream)
    except Exception:
        deltas = None

    if deltas:
        if cached_tokens is None:
            cached_tokens = deltas.get("cached_tokens")
        if completion_tokens is None:
            completion_tokens = deltas.get("generation_tokens")

    spec_draft_tokens = deltas["spec_draft"] if deltas else None
    spec_accepted_tokens = deltas["spec_accepted"] if deltas else None

    record = compute_metrics(
        request_id=request_id,
        model=model,
        stream=False,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
        latency_ms=latency_ms,
        ttft_ms=latency_ms,
        spec_draft_tokens=spec_draft_tokens,
        spec_accepted_tokens=spec_accepted_tokens,
    )

    record["api_key_id"] = key_id
    await _save_record(record, db_path)
    return JSONResponse(status_code=200, content=data)


def _handle_streaming(
    method: str,
    upstream_url: str,
    headers: dict,
    content_type: str,
    body: bytes,
    request_id: str,
    start_time: float,
    model: str | None,
    db_path: str,
    key_id: str | None = None,
) -> StreamingResponse:
    # Use an asyncio.Queue to decouple upstream reading from client delivery.
    # A background task reads from vLLM and puts chunks into the queue.
    # The ASGI generator reads from the queue and yields to the client.
    # When either side disconnects, the other side detects it and cleanup runs
    # reliably via the background task's try/finally.
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    finished = asyncio.Event()

    async def _upstream_reader():
        """Read from upstream vLLM, push chunks to queue. Runs as background task."""
        prompt_tokens = None
        completion_tokens = None
        cached_tokens = None
        reasoning_tokens = 0
        ttft_ms = None
        first_output_seen = False
        was_cancelled = False

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                async with client.stream(
                    method, upstream_url, content=body, headers=headers,
                ) as resp:
                    if resp.status_code >= 400:
                        error_body = await resp.aread()
                        queue.put_nowait(
                            json.dumps({"error": {"message": error_body.decode(), "type": "upstream_error", "code": resp.status_code}})
                        )
                        return

                    # Register stream for force-cancel support
                    _live_streams[request_id] = resp

                    async for line in resp.aiter_lines():
                        # Check cancellation flag (set by cancel or stream close)
                        if request_id in _cancel_flags:
                            was_cancelled = True
                            queue.put_nowait(
                                json.dumps({"error": {"message": "request cancelled by operator", "type": "server_error", "code": 499}})
                            )
                            queue.put_nowait("[DONE]")
                            break

                        if not line.startswith("data: "):
                            if line.strip():
                                queue.put_nowait(line)
                            continue

                        data_str = line[6:]

                        if data_str.strip() == "[DONE]":
                            queue.put_nowait("[DONE]")
                            break

                        # Forward raw SSE line to client
                        queue.put_nowait(line)

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        if not first_output_seen:
                            choices = chunk.get("choices") or []
                            for choice in choices:
                                delta = choice.get("delta") or {}
                                # TTFT = first output token (reasoning OR content)
                                if delta.get("content") or delta.get("reasoning"):
                                    ttft_ms = (time.monotonic() - start_time) * 1000.0
                                    first_output_seen = True
                                    break

                        # Count reasoning tokens from chunks
                        choices = chunk.get("choices") or []
                        for choice in choices:
                            delta = choice.get("delta") or {}
                            if delta.get("reasoning"):
                                reasoning_tokens += 1

                        usage = chunk.get("usage")
                        if usage:
                            prompt_tokens = usage.get("prompt_tokens")
                            completion_tokens = usage.get("completion_tokens")
                            details = usage.get("prompt_tokens_details") or {}
                            cached_tokens = details.get("cached_tokens")
                            # vLLM may provide reasoning_tokens in details; prefer that
                            comp_details = usage.get("completion_tokens_details") or {}
                            api_reasoning = comp_details.get("reasoning_tokens")
                            if api_reasoning is not None:
                                reasoning_tokens = api_reasoning
        except Exception as exc:
            logger.error("Stream reader error [%s]: %s", request_id, exc)
            # Put a sentinel so the client-side generator can exit
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
            return
        finally:
            # Save metrics FIRST (before unregister) — ensure DB write
            # happens regardless of what follows.
            try:
                end_time = time.monotonic()
                latency_ms = (end_time - start_time) * 1000.0

                try:
                    deltas = await measure_counter_deltas(settings.vllm_upstream)
                except Exception:
                    deltas = None

                if deltas:
                    if cached_tokens is None:
                        cached_tokens = deltas.get("cached_tokens")
                    if completion_tokens is None:
                        completion_tokens = deltas.get("generation_tokens")

                spec_draft_tokens = deltas["spec_draft"] if deltas else None
                spec_accepted_tokens = deltas["spec_accepted"] if deltas else None

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
                    spec_draft_tokens=spec_draft_tokens,
                    spec_accepted_tokens=spec_accepted_tokens,
                )

                record["api_key_id"] = key_id

                if was_cancelled:
                    record["status"] = "cancelled"
                    record["error_message"] = "cancelled by operator"

                await _save_record(record, db_path)
                logger.debug("PROXY [%s] stream saved to DB model=%s prompt=%s output=%s",
                             request_id[:8], model, prompt_tokens, completion_tokens)
            except Exception as exc:
                logger.error("Stream metrics recording error [%s]: %s", request_id, exc)
            finally:
                # Unregister LAST — guaranteed to run even if save fails
                unregister_active_request(request_id)
                finished.set()

    async def stream_generator() -> AsyncGenerator[str, None]:
        """ASGI generator — reads from queue and yields SSE to client."""
        # Launch the upstream reader as a background task
        reader_task = asyncio.create_task(_upstream_reader())

        try:
            while True:
                try:
                    # Use a short timeout so we can detect client disconnect
                    # via CancelledError from ASGI
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Check if reader task is done
                    if finished.is_set() or reader_task.done():
                        break
                    continue

                if item is None:
                    # Error sentinel — stream ended abnormally
                    break

                if item == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break

                if item.startswith("{"):
                    # JSON error from upstream
                    yield f"data: {item}\n\n"
                    break

                # Normal SSE line — item already contains "data: ..." from upstream
                yield f"{item}\n\n"
        except asyncio.CancelledError:
            # Client disconnected — signal reader to stop via cancel flag
            # (reader checks it each loop iteration and breaks naturally).
            _cancel_flags.add(request_id)
        finally:
            # Do NOT await reader_task here — when the generator is cancelled,
            # any await in this finally also gets CancelledError, which orphans
            # the reader and interrupts its save. The reader task runs
            # independently and completes cleanup on its own.
            pass

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _save_record(record: dict, db_path: str) -> None:
    """Save a metrics record. Skips requests without a model (e.g. /v1/models)."""
    if not record.get("model"):
        return
    try:
        await insert_request(db_path, record)
    except Exception as exc:
        logger.error("Failed to save record %s: %s", record.get("id"), exc)
