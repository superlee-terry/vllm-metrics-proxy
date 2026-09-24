from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
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


from vllm_metrics_proxy import loop_rules

# --- Loop detection: re-exported from ``loop_rules`` for back-compat ---------
# The actual detector implementations now live in ``vllm_metrics_proxy/loop_rules.py``
# as an *ordered, DB-backed rule list* (the maintenance page can add / edit /
# delete / reorder / enable-disable rules, and they apply live without restart).
# The streaming hot path below calls ``loop_rules.evaluate_loop_rules``.  These
# names are re-exported so existing tests / ad-hoc scripts keep working.
_COMMON_PUNCT = loop_rules._COMMON_PUNCT
_is_json_array_null_pattern = loop_rules._is_json_array_null_pattern
_is_structural_sequence = loop_rules._is_structural_sequence
_are_repetitions_dense = loop_rules._are_repetitions_dense
_punctuation_spam = loop_rules._punctuation_spam


def _detect_loop(window: list[str], repeat_threshold: int, min_tail_match: int) -> tuple[bool, str]:
    """Legacy two-strategy detector (kept for tests / ad-hoc scripts).

    Runs the ``tail_match`` then ``chunk_repeat`` rules with the supplied
    thresholds.  The live streaming path no longer calls this — it calls
    ``loop_rules.evaluate_loop_rules`` against the DB-backed rule list.
    """
    hit, reason = loop_rules._detect_tail_match(
        window,
        {"min_match": min_tail_match, "min_len": 5, "min_distinct": 2},
    )
    if hit:
        return True, reason
    return loop_rules._detect_chunk_repeat(
        window,
        {"threshold": repeat_threshold, "min_len": 10, "recent": 10},
    )


def _is_anthropic_request(original_path: str) -> bool:
    """Check if the original request was in Anthropic /v1/messages format."""
    return original_path in ("/v1/messages", "/v1/v1/messages")


def _transform_openai_to_anthropic(data: dict) -> dict:
    """Transform OpenAI chat completion response to Anthropic messages format."""
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content_text = message.get("content") or ""
    reasoning_text = message.get("reasoning") or ""
    
    # Build content blocks - include reasoning if present
    content_blocks = []
    if reasoning_text:
        content_blocks.append({"type": "text", "text": reasoning_text})
    if content_text:
        content_blocks.append({"type": "text", "text": content_text})
    if not content_blocks:
        content_blocks = [{"type": "text", "text": ""}]
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        func = tc.get("function") or {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "input": json.loads(func.get("arguments", "{}")),
        })
    
    # Determine stop_reason
    finish_reason = choice.get("finish_reason") or "stop"
    if finish_reason == "tool_calls":
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    
    usage = data.get("usage") or {}
    
    return {
        "id": data.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": data.get("model", ""),
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _transform_stream_chunk_to_anthropic(
    chunk: dict, is_first: bool, is_last: bool, usage: dict | None
) -> list[str]:
    """Transform an OpenAI streaming chunk to Anthropic SSE events.
    
    Returns a list of SSE lines to emit.
    """
    lines = []
    
    if is_first:
        # Emit message_start
        msg_start = {
            "type": "message_start",
            "message": {
                "id": "",
                "type": "message",
                "role": "assistant",
                "model": chunk.get("model", ""),
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        lines.append(f"data: {json.dumps(msg_start)}\n")
        
        # Emit content_block_start
        cb_start = {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
        lines.append(f"data: {json.dumps(cb_start)}\n")
    
    # Emit content_block_delta for each text chunk
    choice = (chunk.get("choices") or [{}])[0]
    delta = choice.get("delta") or {}
    text = delta.get("content") or ""
    if text:
        cb_delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
        lines.append(f"data: {json.dumps(cb_delta)}\n")
    
    # Emit tool_use blocks if present
    tool_calls = delta.get("tool_calls") or []
    for tc in tool_calls:
        func = tc.get("function") or {}
        cb_start_tool = {
            "type": "content_block_start",
            "index": len(lines),
            "content_block": {
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": func.get("name", ""),
                "input": {},
            },
        }
        lines.append(f"data: {json.dumps(cb_start_tool)}\n")
        if func.get("arguments"):
            cb_delta_tool = {
                "type": "content_block_delta",
                "index": len(lines) - 1,
                "delta": {"type": "input_json_delta", "partial_json": func["arguments"]},
            }
            lines.append(f"data: {json.dumps(cb_delta_tool)}\n")
    
    if is_last:
        # Emit message_delta with usage
        usage_data = usage or chunk.get("usage") or {}
        msg_delta = {
            "type": "message_delta",
            "delta": {
                "stop_reason": "end_turn",
                "stop_sequence": None,
            },
            "usage": {
                "input_tokens": usage_data.get("prompt_tokens", 0),
                "output_tokens": usage_data.get("completion_tokens", 0),
            },
        }
        lines.append(f"data: {json.dumps(msg_delta)}\n")
        lines.append("data: [DONE]\n")
    
    return lines


def _normalize_upstream_path(path: str) -> str:
    """Normalize incoming paths for multi-client compatibility.
    
    Handles:
    - Claude Code MCP: /v1/v1/messages → /v1/chat/completions (Anthropic→OpenAI)
    - OpenAI format: /v1/chat/completions → /v1/chat/completions (passthrough)
    - vLLM native: /v1/completions, /v1/embeddings → passthrough
    - Ollama format: /v1/generate, /v1/chat → /v1/chat/completions
    """
    # Fix double /v1 prefix (Claude Code MCP sends base_url=/v1 + path=/v1/messages)
    if path == "/v1/v1/messages":
        return "/v1/chat/completions"
    if path.startswith("/v1/v1/"):
        # General case: /v1/v1/X → /v1/X
        return "/" + path.split("/v1/v1/", 1)[1]
    
    # Anthropic /v1/messages → OpenAI /v1/chat/completions
    if path == "/v1/messages":
        return "/v1/chat/completions"
    
    # Ollama-style paths → OpenAI format
    if path == "/v1/generate":
        return "/v1/completions"
    if path == "/v1/chat":
        return "/v1/chat/completions"
    
    # Everything else: passthrough
    return path


def _transform_body_for_upstream(
    original_path: str,
    normalized_path: str,
    payload: dict,
    body: bytes,
) -> tuple[bytes, dict, str]:
    """Transform Ollama-style request body to OpenAI/vLLM format.
    
    When the path was normalized (e.g. /v1/generate → /v1/completions),
    the request body also needs transformation because Ollama and OpenAI
    use different field names.
    """
    transformed = dict(payload)
    
    # Ollama /v1/generate → OpenAI /v1/completions
    if original_path == "/v1/generate":
        # Ollama uses 'prompt', OpenAI completions uses 'prompt' too - mostly compatible
        # But Ollama may have 'images' field which vLLM doesn't expect
        transformed.pop("images", None)
        transformed.pop("format", None)  # Ollama-specific JSON format field
        # Map Ollama options to OpenAI params
        opts = transformed.pop("options", {})
        if opts:
            for key in ("temperature", "top_p", "top_k", "max_tokens", "seed",
                        "frequency_penalty", "presence_penalty", "stop", "repeat_penalty"):
                if key in opts:
                    transformed.setdefault(key, opts[key])
    
    # Ollama /v1/chat → OpenAI /v1/chat/completions
    elif original_path == "/v1/chat":
        # Ollama chat uses 'messages' which is the same as OpenAI - mostly compatible
        transformed.pop("format", None)
        # Map Ollama options
        opts = transformed.pop("options", {})
        if opts:
            for key in ("temperature", "top_p", "top_k", "max_tokens", "seed",
                        "frequency_penalty", "presence_penalty", "stop", "repeat_penalty"):
                if key in opts:
                    transformed.setdefault(key, opts[key])
        # Ollama 'keep_alive' is Ollama-specific
        transformed.pop("keep_alive", None)
        # Ollama 'stream_options' might differ
        if "stream_options" not in transformed:
            pass  # Will be handled by stream_options injection below
    
    # Anthropic /v1/messages → OpenAI /v1/chat/completions
    # Claude Code sends: role: system/user/assistant, max_tokens, tools (Anthropic format)
    # vLLM needs: role: user/assistant only, max_tokens, tools (OpenAI format)
    if normalized_path == "/v1/chat/completions" and (
        original_path in ("/v1/messages", "/v1/v1/messages")
    ):
        # --- Strip system messages, merge into first user message ---
        messages = transformed.get("messages", [])
        system_content = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                # Extract text content from system message
                if isinstance(msg.get("content"), str):
                    system_content += msg["content"]
                elif isinstance(msg.get("content"), list):
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_content += block.get("text", "")
                elif isinstance(msg.get("content"), str):
                    system_content += msg["content"]
            else:
                filtered_messages.append(msg)
        
        # Prepend system content to first user message
        if system_content:
            for i, msg in enumerate(filtered_messages):
                if msg.get("role") == "user":
                    if isinstance(msg.get("content"), str):
                        msg["content"] = system_content + "\n" + msg["content"]
                    elif isinstance(msg.get("content"), list):
                        if system_content:
                            msg["content"].insert(0, {"type": "text", "text": system_content})
                    break
        
        transformed["messages"] = filtered_messages
        
        # --- Map Anthropic fields to OpenAI ---
        # max_tokens → max_tokens (same name, OK)
        # stop → stop (same, OK)
        # temperature/top_p → same
        # tools: Anthropic {name, description, input_schema} → OpenAI {type:function, function:{name, description, parameters}}
        tools = transformed.get("tools")
        if tools:
            openai_tools = []
            for tool in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {}),
                    }
                })
            transformed["tools"] = openai_tools
        
        # Remove Anthropic-specific fields
        transformed.pop("metadata", None)
        transformed.pop("system", None)  # Anthropic top-level system param
        transformed.pop("stream_options", None)  # Will be re-added later
    
    # Anthropic /v1/messages → OpenAI /v1/chat/completions
    if _is_anthropic_request(original_path):
        # --- Strip system messages, merge into first user message ---
        messages = transformed.get("messages", [])
        system_content = ""
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                if isinstance(msg.get("content"), str):
                    system_content += msg["content"] + "\n"
                elif isinstance(msg.get("content"), list):
                    for block in msg["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            system_content += block.get("text", "") + "\n"
            else:
                filtered_messages.append(msg)
        
        # Prepend system content to first user message
        if system_content.strip():
            for i, msg in enumerate(filtered_messages):
                if msg.get("role") == "user":
                    if isinstance(msg.get("content"), str):
                        msg["content"] = system_content.strip() + "\n" + msg["content"]
                    elif isinstance(msg.get("content"), list):
                        msg["content"].insert(0, {"type": "text", "text": system_content.strip()})
                    break
        transformed["messages"] = filtered_messages
        
        # --- Convert Anthropic tool_use/tool_result to OpenAI tool_calls/tool ---
        # Anthropic: assistant sends {"type": "tool_use", "id": "...", "name": "...", "input": {...}}
        # OpenAI: assistant sends tool_calls: [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}]
        # Anthropic: user sends {"type": "tool_result", "tool_use_id": "...", "content": "..."}
        # OpenAI: tool role message: {"role": "tool", "tool_call_id": "...", "content": "..."}
        converted_messages = []
        for msg in filtered_messages:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                content_blocks = msg["content"]
                tool_calls = []
                remaining_content = []
                for block in content_blocks:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
                    else:
                        remaining_content.append(block)
                if tool_calls:
                    converted_messages.append({
                        "role": "assistant",
                        "content": remaining_content[0].get("text", "") if remaining_content and isinstance(remaining_content[0], dict) and remaining_content[0].get("type") == "text" else (remaining_content if remaining_content else None),
                        "tool_calls": tool_calls,
                    })
                else:
                    converted_messages.append(msg)
            elif msg.get("role") == "user" and isinstance(msg.get("content"), list):
                # Check for tool_result blocks
                tool_results = []
                remaining_content = []
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        tool_results.append(block)
                    else:
                        remaining_content.append(block)
                if tool_results:
                    # Split into: non-tool content as user message, then tool responses as tool messages
                    if remaining_content:
                        converted_messages.append({
                            "role": "user",
                            "content": remaining_content,
                        })
                    for tr in tool_results:
                        content_val = tr.get("content", "")
                        if isinstance(content_val, list):
                            # Extract text from content blocks
                            texts = [b.get("text", "") for b in content_val if isinstance(b, dict) and b.get("type") == "text"]
                            content_val = " ".join(texts)
                        converted_messages.append({
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id", ""),
                            "content": content_val,
                        })
                else:
                    converted_messages.append(msg)
            else:
                converted_messages.append(msg)
        transformed["messages"] = converted_messages
        
        # Map Anthropic-specific fields
        if "max_tokens" in transformed:
            transformed["max_completion_tokens"] = transformed.pop("max_tokens")
        # Remove Anthropic-specific fields vLLM doesn't understand
        transformed.pop("metadata", None)
        transformed.pop("system", None)  # Anthropic top-level system param
        transformed.pop("thinking", None)
        transformed.pop("prompt_caching", None)
    
    # Double /v1 prefix - no body transformation needed, just path fix
    # (Claude Code MCP uses OpenAI-compatible body format already)
    
    new_body = json.dumps(transformed).encode()
    return new_body, transformed, "application/json"


async def _resolve_key_from_headers(request: Request, db_path: str) -> str | None:
    """Best-effort key resolution when auth is disabled.

    Extracts the API key from headers and looks it up in the DB.
    Returns the key_id if found, None otherwise.  Never raises.
    """
    try:
        from vllm_metrics_proxy.auth import _extract_key_from_headers, get_api_key, mask_key
        hdrs = dict(request.headers)
        raw_key = _extract_key_from_headers(hdrs)
        if not raw_key:
            # Log which auth-related headers are present (values masked) for debugging
            auth_hdr = hdrs.get("authorization", "(none)")
            xkey_hdr = hdrs.get("x-api-key", "(none)")
            logger.warning(
                "KEY_RESOLVE [%s] no key extracted  auth=%s  x-api-key=%s  path=%s  all_headers=%s",
                id(request), auth_hdr[:20], xkey_hdr[:20], request.url.path,
                list(hdrs.keys()),
            )
            return None
        key_row = await get_api_key(db_path, raw_key)
        if key_row:
            logger.info("KEY_RESOLVE [%s] resolved raw_key=%s → name=%s",
                        id(request), mask_key(raw_key), key_row.get("name", ""))
            return key_row["id"]
        logger.warning("KEY_RESOLVE [%s] key not found in DB  raw_key=%s", id(request), mask_key(raw_key))
        return None
    except Exception as exc:
        logger.error("KEY_RESOLVE [%s] exception: %s", id(request), exc, exc_info=True)
        return None

# ---- Active request tracking ----

_active_requests: dict[str, dict] = {}
_cancel_flags: set[str] = set()
_live_resources: dict[str, httpx.AsyncClient | httpx.Response] = {}  # request_id -> closeable resource for force-cancel


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
    _live_resources.pop(request_id, None)


def cancel_active_request(request_id: str) -> bool:
    """Cancel an active request by closing its upstream resource."""
    if request_id not in _active_requests:
        return False
    _cancel_flags.add(request_id)
    # Force-close the httpx resource (Response for streaming, AsyncClient for non-streaming)
    resource = _live_resources.pop(request_id, None)
    if resource:
        try:
            resource.close()
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

    # When auth is disabled, still try to extract and resolve the key
    # from headers for metrics enrichment (non-blocking).
    if key_id == "__no_auth__":
        key_id = await _resolve_key_from_headers(request, db_path)

    body = await request.body()
    content_type = request.headers.get("content-type", "application/json")

    # Initialize before try/except to avoid UnboundLocalError on GET requests
    payload = None
    stream = False
    model = None

    if body:
        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                stream = payload.get("stream", False)
                model = payload.get("model")
        except (json.JSONDecodeError, AttributeError):
            pass

    # Resolve key name early for logging and active request display
    api_key_name = ""
    if key_id:
        key_row = await _get_api_key(db_path, key_id)
        if key_row:
            api_key_name = key_row.get("name", "")

    # Log request parameters — especially model params for chat endpoints
    log_params = {"model": model, "stream": stream}
    if api_key_name:
        log_params["key"] = api_key_name
    if isinstance(payload, dict):
        for key in ("temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens",
                     "frequency_penalty", "presence_penalty", "seed",
                     "stop", "tools", "tool_choice", "response_format",
                     "messages", "prompt"):
            val = payload.get(key)
            if val is not None:
                # Truncate messages/prompt to avoid log spam
                if key in ("messages", "prompt") and isinstance(val, (list, str)):
                    first_item = val[0] if isinstance(val, list) and val else val
                    truncated = str(first_item)[:200]
                    log_params[key] = f"...[{len(val)} items/chars] first={truncated!r}..."
                else:
                    log_params[key] = val
    logger.info("PROXY [%s] %s %s %s", request_id[:8], request.method, request.url.path, log_params)

    # Normalize upstream path for multi-client compatibility
    upstream = settings.vllm_upstream.rstrip("/")
    normalized_path = _normalize_upstream_path(request.url.path)
    logger.info("PATH NORMALIZE [%s] %s -> %s", request_id[:8], request.url.path, normalized_path)
    upstream_url = f"{upstream}{normalized_path}"

    # Transform Ollama-style request body to OpenAI format
    if normalized_path != request.url.path and isinstance(payload, dict):
        body, payload, content_type = _transform_body_for_upstream(
            request.url.path, normalized_path, payload, body
        )

    headers = dict(request.headers)
    headers.pop("host", None)
    # Let httpx compute Content-Length from the actual body (the body may have
    # been re-serialized by _transform_body_for_upstream, making the original
    # header stale — e.g. /v1/v1/messages → /v1/messages path normalisation).
    headers.pop("content-length", None)

    # Inject stream_options to get usage data in streaming chunks
    if stream and isinstance(payload, dict):
        stream_opts = payload.get("stream_options")
        if not stream_opts or not stream_opts.get("include_usage"):
            payload["stream_options"] = {"include_usage": True}
            body = json.dumps(payload).encode()
            content_type = "application/json"
            headers["content-length"] = str(len(body))
    
    # Track if this is an Anthropic-style request for response transformation
    is_anthropic = _is_anthropic_request(request.url.path)

    # Snapshot Prometheus counters before the request
    await snapshot_counters(settings.vllm_upstream)

    # Register active request (api_key_name already resolved above)
    register_active_request(request_id, model, stream, api_key_name=api_key_name)

    try:
        if stream:
            # Streaming: reader task handles unregister in its own finally
            return _handle_streaming(
                request.method, upstream_url, headers, content_type, body,
                request_id, start_time, model, db_path, key_id,
                original_path=request.url.path,
                wall_clock_timeout=settings.request_timeout_seconds,
                idle_timeout=settings.stream_idle_timeout,
            )
        else:
            # Non-streaming: register client for cancel, ensure unregister always runs
            try:
                client = httpx.AsyncClient(timeout=httpx.Timeout(settings.request_timeout_seconds, connect=10.0))
                _live_resources[request_id] = client
                async with client:
                    return await _handle_non_streaming(
                        client, request.method, upstream_url, headers, content_type, body,
                        request_id, start_time, model, db_path, key_id,
                        original_path=request.url.path,
                    )
            finally:
                _live_resources.pop(request_id, None)
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
    original_path: str = "",
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
    
    # Transform OpenAI response back to Anthropic format if needed
    if _is_anthropic_request(original_path):
        data = _transform_openai_to_anthropic(data)
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
    original_path: str = "",
    wall_clock_timeout: float = 300.0,
    idle_timeout: float = 30.0,
) -> StreamingResponse:
    is_anthropic = _is_anthropic_request(original_path)
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
        loop_triggered = False
        wall_clock_timed_out = False
        last_reasoning_only = False
        # Loop detection sliding window
        loop_window: list[str] = []

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(idle_timeout, connect=10.0)) as client:
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
                    _live_resources[request_id] = resp

                    async for line in resp.aiter_lines():
                        # Check cancellation flag (set by cancel or stream close)
                        if request_id in _cancel_flags:
                            was_cancelled = True
                            queue.put_nowait(
                                json.dumps({"error": {"message": "request cancelled by operator", "type": "server_error", "code": 499}})
                            )
                            queue.put_nowait("[DONE]")
                            break

                        # Wall-clock timeout: total output duration exceeded
                        if time.monotonic() - start_time > wall_clock_timeout:
                            wall_clock_timed_out = True
                            logger.error(
                                "STREAM TIMEOUT [%s] model=%s elapsed=%.1fs wall_clock_limit=%.1fs — force terminated",
                                request_id[:8], model, time.monotonic() - start_time, wall_clock_timeout,
                            )
                            # 只要推理过程中产生过 reasoning tokens，就用温和提示
                            # 让 agent 能接着前文继续，而不是硬中断
                            if reasoning_tokens > 0:
                                queue.put_nowait(
                                    json.dumps({
                                        "choices": [{
                                            "delta": {"content": f"\n\n|| 思考超过{wall_clock_timeout}s，当前处理中断。稍后请继续。"},
                                            "finish_reason": "length"
                                        }]
                                    })
                                )
                            else:
                                queue.put_nowait(
                                    json.dumps({"error": {"message": f"|| 请求超时（已达 {wall_clock_timeout}s），已终止。", "type": "server_error", "code": 504}})
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
                            # Track whether the last chunk was reasoning-only
                            # (has reasoning but no content)
                            has_content = bool(delta.get("content"))
                            has_reasoning = bool(delta.get("reasoning"))
                            last_reasoning_only = has_reasoning and not has_content

                        # --- Loop detection (always runs; config switch only controls stream termination) ---
                        if not loop_triggered:
                            # Collect chunk text content into sliding window
                            for choice in choices:
                                delta = choice.get("delta") or {}
                                content = delta.get("content") or delta.get("reasoning") or ""
                                if content:
                                    loop_window.append(content)
                                    if len(loop_window) > settings.loop_window_size:
                                        loop_window = loop_window[-settings.loop_window_size:]

                            # Run the ordered, DB-backed loop rules (tail_match,
                            # chunk_repeat, punct_spam, + any custom ones added via
                            # the maintenance page).  Each rule gates itself with
                            # its own thresholds; the first enabled hit wins.
                            loop_result, loop_reason = loop_rules.evaluate_loop_rules(loop_window)
                            if loop_result:
                                loop_triggered = True
                                # Write to dedicated loop debug log file
                                try:
                                    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
                                    os.makedirs(log_dir, exist_ok=True)
                                    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                                    loop_log_path = os.path.join(log_dir, f"loop_debug_{ts}_{request_id[:8]}.json")
                                    debug_entry = {
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "request_id": request_id,
                                        "model": model,
                                        "reason": loop_reason,
                                        "window_size": len(loop_window),
                                        "window": loop_window,
                                        "last_10_chunks": loop_window[-10:],
                                    }
                                    with open(loop_log_path, "w", encoding="utf-8") as f:
                                        json.dump(debug_entry, f, ensure_ascii=False, indent=2)
                                    logger.error(
                                        "LOOP DETECTED [%s] model=%s reason=%s window_size=%d "
                                        "debug_log=%s auto_terminate=%s",
                                        request_id[:8],
                                        model,
                                        loop_reason,
                                        len(loop_window),
                                        loop_log_path,
                                        settings.loop_detection_enabled,
                                    )
                                except Exception as log_exc:
                                    logger.error("LOOP DETECTED [%s] model=%s reason=%s log_write_failed=%s",
                                                 request_id[:8], model, loop_reason, log_exc)

                                # Only terminate the stream if loop_detection_enabled is True
                                if settings.loop_detection_enabled:
                                    queue.put_nowait(
                                        json.dumps({"error": {"message": "model output loop detected, stream terminated", "type": "server_error", "code": 499}})
                                    )
                                    queue.put_nowait("[DONE]")
                                    break

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
                elif wall_clock_timed_out:
                    record["status"] = "timeout"
                    if reasoning_tokens > 0:
                        record["error_message"] = f"stream exceeded wall-clock timeout of {wall_clock_timeout}s (reasoning interrupted)"
                    else:
                        record["error_message"] = f"stream exceeded wall-clock timeout of {wall_clock_timeout}s"
                elif loop_triggered:
                    record["status"] = "loop_detected"
                    record["error_message"] = "model output loop detected, stream terminated"

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
        
        # State for Anthropic response transformation
        anthro_first = True
        anthro_usage: dict | None = None
        
        try:
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    if finished.is_set() or reader_task.done():
                        break
                    continue
                
                if item is None:
                    break
                
                if item == "[DONE]":
                    # For Anthropic: emit final message_delta with usage
                    if is_anthropic:
                        msg_delta = {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {
                                "input_tokens": (anthro_usage or {}).get("prompt_tokens", 0),
                                "output_tokens": (anthro_usage or {}).get("completion_tokens", 0),
                            },
                        }
                        yield f"data: {json.dumps(msg_delta)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                
                if item.startswith("{"):
                    # JSON error from upstream
                    yield f"data: {item}\n\n"
                    break
                
                # Parse SSE line
                if not item.startswith("data: "):
                    yield f"{item}\n\n"
                    continue
                
                data_str = item[6:]
                if data_str.strip() == "[DONE]":
                    if is_anthropic:
                        msg_delta = {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {
                                "input_tokens": (anthro_usage or {}).get("prompt_tokens", 0),
                                "output_tokens": (anthro_usage or {}).get("completion_tokens", 0),
                            },
                        }
                        yield f"data: {json.dumps(msg_delta)}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    yield f"{item}\n\n"
                    continue
                
                # Extract usage from chunk if present
                if chunk.get("usage"):
                    anthro_usage = chunk["usage"]
                
                # Transform for Anthropic clients
                if is_anthropic:
                    if anthro_first:
                        # Emit message_start
                        msg_start = {
                            "type": "message_start",
                            "message": {
                                "id": chunk.get("id", ""),
                                "type": "message",
                                "role": "assistant",
                                "model": chunk.get("model", ""),
                                "content": [],
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0},
                            },
                        }
                        yield f"data: {json.dumps(msg_start)}\n\n"
                        
                        # Emit content_block_start
                        cb_start = {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
                        yield f"data: {json.dumps(cb_start)}\n\n"
                        anthro_first = False
                    
                    # Emit content_block_delta for text content
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or ""
                    reasoning = delta.get("reasoning") or ""
                    if text:
                        cb_delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
                        yield f"data: {json.dumps(cb_delta)}\n\n"
                    elif reasoning:
                        cb_delta = {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": reasoning}}
                        yield f"data: {json.dumps(cb_delta)}\n\n"
                    
                    # Handle tool_calls in delta
                    tool_calls = delta.get("tool_calls") or []
                    for tc in tool_calls:
                        func = tc.get("function") or {}
                        yield f"data: {json.dumps({'type': 'content_block_start', 'index': 1, 'content_block': {'type': 'tool_use', 'id': tc.get('id', ''), 'name': func.get('name', ''), 'input': {}}})}\n\n"
                        if func.get("arguments"):
                            yield f"data: {json.dumps({'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'input_json_delta', 'partial_json': func['arguments']}})}\n\n"
                    
                    # Check if this is the last chunk (has finish_reason)
                    finish = choice.get("finish_reason")
                    if finish:
                        # Emit content_block_stop
                        yield f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        # Emit message_delta with usage
                        msg_delta = {
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {
                                "input_tokens": (anthro_usage or {}).get("prompt_tokens", 0),
                                "output_tokens": (anthro_usage or {}).get("completion_tokens", 0),
                            },
                        }
                        yield f"data: {json.dumps(msg_delta)}\n\n"
                        # Emit message_stop
                        yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
                        yield "data: [DONE]\n\n"
                        break  # exit loop only on final chunk
                    continue  # keep processing more chunks
                else:
                    # Normal SSE line — pass through
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
