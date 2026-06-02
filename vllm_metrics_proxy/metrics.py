from __future__ import annotations

import re


def compute_metrics(
    *,
    request_id: str,
    model: str | None,
    stream: bool,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    cached_tokens: int | None,
    reasoning_tokens: int | None,
    latency_ms: float,
    ttft_ms: float | None,
) -> dict:
    """Compute derived metrics from raw usage data."""
    cached_ratio = None
    if prompt_tokens and prompt_tokens > 0 and cached_tokens is not None:
        cached_ratio = cached_tokens / prompt_tokens

    prompt_speed = None
    if ttft_ms and ttft_ms > 0 and prompt_tokens:
        prompt_speed = prompt_tokens / (ttft_ms / 1000.0)

    completion_speed = None
    if stream and ttft_ms and latency_ms and completion_tokens:
        generate_time = (latency_ms - ttft_ms) / 1000.0
        if generate_time > 0:
            completion_speed = completion_tokens / generate_time

    return {
        "id": request_id,
        "model": model,
        "stream": stream,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "prompt_speed": round(prompt_speed, 1) if prompt_speed else None,
        "completion_speed": round(completion_speed, 1) if completion_speed else None,
        "cached_ratio": round(cached_ratio, 3) if cached_ratio is not None else None,
        "status": "success",
        "error_message": None,
    }


_SINCE_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)(h|d)$")


def parse_since(value: str) -> float | None:
    """Parse a 'since' string like '1h', '24h', '7d' into hours. Returns None for 'all'."""
    if value == "all":
        return None
    m = _SINCE_PATTERN.match(value)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2)
    return num if unit == "h" else num * 24.0
