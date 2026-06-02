"""Fetch and parse vLLM Prometheus /metrics endpoint."""

from __future__ import annotations

import logging
import re

import httpx

logger = logging.getLogger(__name__)

# Pattern to match a Prometheus metric line:
# metric_name{label1="val1",...} numeric_value
_METRIC_RE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?P<labels>\{[^}]*\})?\s+'
    r'(?P<value>[0-9eE.+-]+)'
)


def _parse_prometheus(text: str) -> dict[tuple[str, str], float]:
    """Parse Prometheus text format into {(metric_name, labels_str): value}."""
    result: dict[tuple[str, str], float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        labels = m.group("labels") or ""
        value = float(m.group("value"))
        result[(name, labels)] = value
    return result


def _get_first(parsed: dict[tuple[str, str], float], name: str) -> float | None:
    """Get the first (usually only) value for a metric name."""
    for (n, _labels), value in parsed.items():
        if n == name and not n.endswith("_created"):
            return value
    return None


def _get_by_label(parsed: dict[tuple[str, str], float], name: str, label_key: str, label_val: str) -> float | None:
    """Get a metric value filtered by a specific label."""
    for (n, labels), value in parsed.items():
        if n == name and not n.endswith("_created"):
            if f'{label_key}="{label_val}"' in labels:
                return value
    return None


async def _fetch_raw_metrics(upstream_url: str, timeout: float = 3.0) -> dict[tuple[str, str], float] | None:
    """Fetch and parse vLLM /metrics. Returns parsed dict or None on failure."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{upstream_url}/metrics")
            resp.raise_for_status()
        return _parse_prometheus(resp.text)
    except Exception:
        return None


# ---- Per-request Prometheus counter delta tracker ----
# Snapshots multiple counters in a single Prometheus fetch.

_prev_counters: dict[str, int] | None = None


async def snapshot_counters(upstream_url: str) -> dict[str, int] | None:
    """Snapshot Prometheus counters before a request.

    Tracks: spec_decode draft/accepted, prefix_cache queries/hits,
    prompt_tokens_cached, generation_tokens.
    """
    global _prev_counters
    parsed = await _fetch_raw_metrics(upstream_url)
    if not parsed:
        return None
    metrics: dict[str, int] = {
        "spec_draft": int(_get_first(parsed, "vllm:spec_decode_num_draft_tokens_total") or 0),
        "spec_accepted": int(_get_first(parsed, "vllm:spec_decode_num_accepted_tokens_total") or 0),
        "cache_queries": int(_get_first(parsed, "vllm:prefix_cache_queries_total") or 0),
        "cache_hits": int(_get_first(parsed, "vllm:prefix_cache_hits_total") or 0),
        "cached_tokens": int(_get_first(parsed, "vllm:prompt_tokens_cached_total") or 0),
        "generation_tokens": int(_get_first(parsed, "vllm:generation_tokens_total") or 0),
    }
    _prev_counters = metrics
    return metrics


async def measure_counter_deltas(upstream_url: str) -> dict[str, int] | None:
    """Compute counter deltas since last snapshot.

    Call snapshot_counters() before the request, then this after.
    Returns dict with deltas, or None if no snapshot or no change.
    """
    global _prev_counters
    if _prev_counters is None:
        return None
    parsed = await _fetch_raw_metrics(upstream_url)
    if not parsed:
        return None
    cur: dict[str, int] = {
        "spec_draft": int(_get_first(parsed, "vllm:spec_decode_num_draft_tokens_total") or 0),
        "spec_accepted": int(_get_first(parsed, "vllm:spec_decode_num_accepted_tokens_total") or 0),
        "cache_queries": int(_get_first(parsed, "vllm:prefix_cache_queries_total") or 0),
        "cache_hits": int(_get_first(parsed, "vllm:prefix_cache_hits_total") or 0),
        "cached_tokens": int(_get_first(parsed, "vllm:prompt_tokens_cached_total") or 0),
        "generation_tokens": int(_get_first(parsed, "vllm:generation_tokens_total") or 0),
    }
    prev = _prev_counters
    _prev_counters = cur

    deltas: dict[str, int] = {k: cur[k] - prev[k] for k in cur}
    # Only return if there was actual activity
    if deltas["cache_queries"] > 0 or deltas["spec_draft"] > 0 or deltas["generation_tokens"] > 0:
        return deltas
    return None


# ---- Engine stats for dashboard ----

async def fetch_engine_stats(upstream_url: str, timeout: float = 5.0) -> dict:
    """Fetch vLLM Prometheus /metrics and return structured engine stats."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{upstream_url}/metrics")
            resp.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to fetch vLLM metrics: %s", exc)
        return {"error": f"upstream unreachable: {exc}"}

    parsed = _parse_prometheus(resp.text)

    draft_tokens = _get_first(parsed, "vllm:spec_decode_num_draft_tokens_total") or 0
    accepted_tokens = _get_first(parsed, "vllm:spec_decode_num_accepted_tokens_total") or 0

    return {
        "spec_decode_draft_tokens": int(draft_tokens),
        "spec_decode_accepted_tokens": int(accepted_tokens),
        "spec_decode_accept_rate": round(accepted_tokens / draft_tokens, 3) if draft_tokens > 0 else None,
        "kv_cache_usage_pct": round(_get_first(parsed, "vllm:kv_cache_usage_perc") or 0, 3),
        "num_requests_running": int(_get_first(parsed, "vllm:num_requests_running") or 0),
        "num_requests_waiting": int(_get_first(parsed, "vllm:num_requests_waiting") or 0),
        "engine_awake": int(_get_by_label(parsed, "vllm:engine_sleep_state", "sleep_state", "awake") or 0),
        "total_prompt_tokens": int(_get_first(parsed, "vllm:prompt_tokens_total") or 0),
        "total_cached_tokens": int(_get_first(parsed, "vllm:prompt_tokens_cached_total") or 0),
        "total_generation_tokens": int(_get_first(parsed, "vllm:generation_tokens_total") or 0),
    }
