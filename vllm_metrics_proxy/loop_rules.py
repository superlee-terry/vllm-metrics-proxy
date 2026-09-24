"""Live, DB-backed loop-detection rules.

The streaming hot path in ``proxy.py`` used to call a fixed chain of hardcoded
strategies (``_detect_loop`` → ``_punctuation_spam``) whose thresholds came from
scalar settings.  This module generalises that into an *ordered list of rules*:

* Each rule is ``{rule_id, type, enabled, params}`` where ``type`` is one of the
  known detectors (see :data:`RULE_TYPES`) and ``params`` holds that detector's
  tunable thresholds.
* The list lives in a module-level mutable (``_live_rules``).  Mutating it at
  runtime is picked up immediately by the streaming loop — **no restart** — and
  is persisted to the ``loop_rules`` table in ``metrics.db`` so it survives
  restarts (reloaded in the app lifespan).
* The page can add / edit / delete / reorder / enable-disable rules, and reset
  to the factory defaults.

The *kill switch* (whether a detected loop actually terminates the stream) is a
single global ``settings.loop_detection_enabled`` and is **not** a rule: it
gates every rule at once.  Detection itself always runs, so disabling the kill
switch still logs to ``loop_debug_*.json`` for tuning.
"""

from __future__ import annotations

import copy
import json
import logging
import uuid
from typing import Any

from vllm_metrics_proxy import db as _db

logger = logging.getLogger(__name__)


# ---- Detector building blocks (moved from proxy.py) -----------------------

# Common sentence punctuation (CN + EN) a model can degenerate into emitting
# endlessly in a runaway reasoning loop ('!!!!!' / '......' / '、、' / '？？？').
# Deliberately EXCLUDES symbols that can legitimately form a long run:
#   dash/hyphen (- – —), box-drawing (─│┌└┐┘═║), markdown ( # * ), code symbols.
_COMMON_PUNCT = frozenset(
    "!.:"
    ";,"
    "！，．？；："
    "、。…"
)


def _is_json_array_null_pattern(chunk: str) -> bool:
    """Detect if a chunk is likely part of a JSON array/object containing null values."""
    import re
    stripped = chunk.strip()
    JSON_NULL_SIGNATURES = [
        "^null$", ",null$", "^null,", "null,", ",null,",
        "\\[null", "null\\]", "\\{null", "null\\}",
        "\\s+null\\s+", "null\\s*,", ",\\s*null",
    ]
    for sig in JSON_NULL_SIGNATURES:
        if re.search(sig, stripped):
            return True
    return False


def _is_structural_sequence(window: list[str]) -> bool:
    """Detect a window that is JSON null/bracket construction (not a true loop)."""
    from collections import Counter
    null_count = sum(1 for c in window if "null" in c)
    if null_count < len(window) * 0.3:
        return False
    has_brackets = any("[" in c or "{" in c for c in window)
    has_closure = any("]" in c or "}" in c for c in window)
    if not (has_brackets and has_closure):
        return False
    counts = Counter(window)
    top_5 = counts.most_common(5)
    null_related = sum(1 for chunk, _ in top_5 if "null" in chunk)
    return null_related >= 3


def _are_repetitions_dense(positions: list[int], window_len: int, density_threshold: float = 0.15) -> bool:
    """True if the positions of a repeated chunk are densely packed (a real loop)."""
    if len(positions) < 2:
        return True
    gaps = [positions[i] - positions[i - 1] for i in range(1, len(positions))]
    avg_gap = sum(gaps) / len(gaps)
    return avg_gap / window_len < density_threshold


def _punctuation_spam(window: list[str], min_chars: int, min_chunks: int,
                      punct: frozenset[str] = _COMMON_PUNCT) -> tuple[bool, str]:
    """Detect a degenerate pure-punctuation run at the tail of the window.

    Scans backward from the newest chunk, accumulating both the total stripped
    length and the chunk count of consecutive chunks made up entirely of the
    configured punctuation set (``punct``, default ``_COMMON_PUNCT``).  Triggers
    when EITHER the total punctuation length >= ``min_chars`` OR the number of
    consecutive pure-punctuation chunks >= ``min_chunks`` (one-token-per-chunk
    loops).
    """
    acc = 0
    chunks = 0
    for c in reversed(window):
        s = c.strip()
        if not s:
            break
        if not all(ch in punct for ch in s):
            break
        acc += len(s)
        chunks += 1
    if acc >= min_chars:
        return True, f"punct_spam: trailing {acc}-char pure-punctuation run"
    if chunks >= min_chunks:
        return True, f"punct_spam: {chunks} consecutive pure-punctuation chunks"
    return False, ""


# ---- Detector implementations (parameterised by a rule's `params`) --------

def _detect_tail_match(window: list[str], params: dict) -> tuple[bool, str]:
    """Last N non-empty chunks all identical → loop.

    params: min_match (N), min_len, min_distinct.
    Pre-gates (identical to the original combined detector): skip when the
    window is JSON-structure building or carries a null-serialization chunk,
    to avoid false positives on vLLM null responses.
    """
    min_match = int(params.get("min_match", 5))
    min_len = int(params.get("min_len", 5))
    min_distinct = int(params.get("min_distinct", 2))
    if len(window) < min_match:
        return False, ""
    if _is_structural_sequence(window):
        return False, ""
    NULL_JSON_PATTERNS = (",null", "null", ",null,", "null,", ",null,,", "null,,")
    if any(c.strip() in NULL_JSON_PATTERNS for c in window):
        return False, ""
    tail = window[-min_match:]
    if (all(t == tail[0] for t in tail)
            and tail[0].strip()
            and len(tail[0].strip()) >= min_len
            and len(set(tail[0].strip())) >= min_distinct
            and tail[0].strip() not in NULL_JSON_PATTERNS):
        return True, f"tail_match: last {min_match} chunks identical='{tail[0][:200]}'"
    return False, ""


def _detect_chunk_repeat(window: list[str], params: dict) -> tuple[bool, str]:
    """A substantial chunk repeated >= threshold times in the recent slice → loop.

    params: threshold, min_len, recent.  Pre-gates (identical to the original
    combined detector): JSON-structure and null-serialization windows are
    exempt; then density / predecessor analysis distinguishes a real loop from
    harmless enumeration.
    """
    threshold = int(params.get("threshold", 3))
    min_len = int(params.get("min_len", 10))
    recent = int(params.get("recent", 10))
    if _is_structural_sequence(window):
        return False, ""
    NULL_JSON_PATTERNS = (",null", "null", ",null,", "null,", ",null,,", "null,,")
    if any(c.strip() in NULL_JSON_PATTERNS for c in window):
        return False, ""
    from collections import Counter
    tail_slice = window[-recent:] if len(window) >= recent else window
    substantial = [c for c in tail_slice
                   if len(c) >= min_len
                   and c.strip()
                   and len(set(c.strip())) >= 2
                   and not _is_json_array_null_pattern(c)]
    if len(substantial) < threshold:
        return False, ""
    counts = Counter(substantial)
    most_common_chunk, most_common_count = counts.most_common(1)[0]
    if most_common_count >= threshold:
        positions = [i for i, c in enumerate(window) if c == most_common_chunk]
        if len(positions) > 1 and not (positions[-1] - positions[0] < len(positions)):
            if not _are_repetitions_dense(positions, len(window)):
                return False, ""
        predecessors = [window[i - 1] for i in positions if i > 0]
        if most_common_chunk in predecessors:
            return True, (f"chunk_repeat: chunk({most_common_count}x, len={len(most_common_chunk)})"
                          f"='{most_common_chunk[:200]}'")
        if predecessors and len(set(predecessors)) >= 2:
            return False, ""  # diverse predecessors → enumeration, not a loop
        return True, (f"chunk_repeat: chunk({most_common_count}x, len={len(most_common_chunk)})"
                      f"='{most_common_chunk[:200]}'")
    return False, ""


def _detect_punct_spam(window: list[str], params: dict) -> tuple[bool, str]:
    """Trailing pure-punctuation run (params: min_chars, min_chunks, chars).

    ``chars`` is the per-rule punctuation set (a string of chars); when absent
    or empty the built-in ``_COMMON_PUNCT`` is used.
    """
    min_chars = int(params.get("min_chars", 10))
    min_chunks = int(params.get("min_chunks", 6))
    raw_chars = params.get("chars")
    if isinstance(raw_chars, str) and raw_chars.strip():
        punct = frozenset(raw_chars)
    else:
        punct = _COMMON_PUNCT
    return _punctuation_spam(window, min_chars, min_chunks, punct)


# ---- Rule registry (drives the maintenance UI) ---------------------------

# type -> {label, desc, params: {param: {label, type, min, max, default}}}
# ``chars`` is a special param type: a single string whose characters form the
# matching set (rendered as a char-set editor in the UI).
DEFAULT_PUNCT_CHARS = "".join(sorted(_COMMON_PUNCT))

RULE_TYPES: dict[str, dict] = {
    "tail_match": {
        "label": "尾部全同",
        "desc": "最近 N 个 chunk 完全相同判定为循环（排除 null 序列化产物 / 单字符重复）",
        "params": {
            "min_match": {"label": "连续相同 chunk 数", "type": "int", "min": 2, "max": 20, "default": 5},
            "min_len": {"label": "chunk 最小长度", "type": "int", "min": 1, "max": 200, "default": 5},
            "min_distinct": {"label": "最少不同字符", "type": "int", "min": 1, "max": 10, "default": 2},
        },
    },
    "chunk_repeat": {
        "label": "整块重复",
        "desc": "某段实质性 chunk（≥min_len）在最近 N 个 chunk 内重复 ≥threshold 次（含 JSON 豁免 / 密度 / 枚举判断）",
        "params": {
            "threshold": {"label": "重复次数阈值", "type": "int", "min": 2, "max": 20, "default": 3},
            "min_len": {"label": "chunk 最小长度", "type": "int", "min": 2, "max": 200, "default": 10},
            "recent": {"label": "统计窗口(近 N 个 chunk)", "type": "int", "min": 3, "max": 50, "default": 10},
        },
    },
    "punct_spam": {
        "label": "标点 spam",
        "desc": "尾部纯标点串总长 ≥min_chars 或连续 ≥min_chunks 个纯标点 chunk 即掐断；「匹配字符」可增删哪些标点算 spam（默认排除 dash / 制表框 / markdown）",
        "params": {
            "min_chars": {"label": "纯标点总长度阈值", "type": "int", "min": 3, "max": 200, "default": 10},
            "min_chunks": {"label": "连续纯标点 chunk 数", "type": "int", "min": 2, "max": 100, "default": 6},
            "chars": {"label": "匹配字符(标点集)", "type": "chars", "min": 1, "max": 200, "default": DEFAULT_PUNCT_CHARS},
        },
    },
}

_DETECTORS = {
    "tail_match": _detect_tail_match,
    "chunk_repeat": _detect_chunk_repeat,
    "punct_spam": _detect_punct_spam,
}


# ---- Live rule list (module-level; mutated at runtime) -------------------

_live_rules: list[dict] = []


def _default_params(rule_type: str) -> dict:
    schema = RULE_TYPES.get(rule_type, {}).get("params", {})
    return {k: spec["default"] for k, spec in schema.items()}


def default_rules() -> list[dict]:
    """Factory defaults: the three built-in detectors, in their original order."""
    return [
        {"rule_id": "tail_match", "type": "tail_match", "enabled": True, "params": _default_params("tail_match")},
        {"rule_id": "chunk_repeat", "type": "chunk_repeat", "enabled": True, "params": _default_params("chunk_repeat")},
        {"rule_id": "punct_spam", "type": "punct_spam", "enabled": True, "params": _default_params("punct_spam")},
    ]


def get_live_rules() -> list[dict]:
    return list(_live_rules)


def seed_default_rules() -> None:
    """Populate the live list with factory defaults if it is empty (first run)."""
    if not _live_rules:
        _live_rules.extend(default_rules())


def evaluate_loop_rules(window: list[str]) -> tuple[bool, str]:
    """Run the enabled loop rules in order against the sliding window.

    Returns ``(is_loop, reason)`` of the **first** enabled rule that fires
    (rules are ordered by list position — earlier = higher priority), or
    ``(False, "")`` if none fire.

    There is intentionally **no** global window-size gate here: each rule gates
    itself with its own minimum (``min_match`` / ``recent`` / punct run length),
    so a tail-match can fire after just a handful of chunks while the sliding
    window keeps growing up to ``loop_window_size``.
    """
    for rule in _live_rules:
        if not rule.get("enabled"):
            continue
        rtype = rule.get("type")
        detector = _DETECTORS.get(rtype)
        if detector is None:
            continue
        try:
            hit, reason = detector(window, rule.get("params", {}) or {})
        except Exception as e:  # a broken rule must never take down the stream
            logger.warning("loop rule %s raised %s", rule.get("rule_id"), e)
            continue
        if hit:
            return True, reason
    return False, ""


# ---- CRUD helpers (live-apply + persist the whole list) ------------------

def _rules_to_text(rules: list[dict]) -> str:
    return json.dumps(rules, ensure_ascii=False)


def _rules_from_text(text: str) -> list[dict]:
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("loop_rules must be a JSON list")
    return data


def _validate_params(rule_type: Any, params: Any) -> dict:
    if rule_type not in RULE_TYPES:
        raise ValueError(f"未知规则类型: {rule_type!r}")
    schema = RULE_TYPES[rule_type]["params"]
    if not isinstance(params, dict):
        raise ValueError(f"{rule_type}: params 必须是对象")
    out: dict[str, Any] = {}
    for key, spec in schema.items():
        raw = params.get(key, spec["default"])
        t = spec["type"]
        if t == "int":
            try:
                val: Any = int(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{rule_type}.{key} 必须是整数")
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and val < lo:
                raise ValueError(f"{rule_type}.{key} 不能小于 {lo}")
            if hi is not None and val > hi:
                raise ValueError(f"{rule_type}.{key} 不能大于 {hi}")
            out[key] = val
        elif t == "chars":
            # A string whose distinct characters form the matching set.
            # Preserve input order, de-duplicate, drop whitespace.
            if not isinstance(raw, str):
                raise ValueError(f"{rule_type}.{key} 必须是字符串")
            seen: set[str] = set()
            chars = "".join(c for c in raw if not c.isspace() and not (c in seen or seen.add(c)))
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and len(chars) < lo:
                raise ValueError(f"{rule_type}.{key} 至少 {lo} 个不同字符")
            if hi is not None and len(chars) > hi:
                raise ValueError(f"{rule_type}.{key} 至多 {hi} 个不同字符")
            out[key] = chars
    return out


async def _persist_rules(db_path: str, rules: list[dict]) -> None:
    await _db.upsert_setting(db_path, "loop_rules", _rules_to_text(rules))


async def apply_rules(db_path: str, rules: list[dict]) -> list[dict]:
    """Replace the live rule list and persist the full list to the DB.

    ``rules`` must be the complete, already-ordered list (the UI sends the
    whole list after any add/edit/delete/reorder, so one write covers all).
    Returns the normalised list actually applied.
    """
    # Validate every rule's type + params (and coerce) before committing.
    normalised: list[dict] = []
    for r in rules:
        rtype = r.get("type")
        normalised.append({
            "rule_id": r.get("rule_id") or str(uuid.uuid4()),
            "type": rtype,
            "enabled": bool(r.get("enabled", True)),
            "params": _validate_params(rtype, r.get("params") or {}),
        })
    _live_rules[:] = normalised
    await _persist_rules(db_path, normalised)
    logger.info("loop rules applied (live+DB): %s",
                [(r["rule_id"], r["type"], r["enabled"]) for r in normalised])
    return normalised


def _fill_param_defaults(rules: list[dict]) -> list[dict]:
    """Back-fill any param missing from a stored rule with its schema default.

    Keeps old DB rows (saved before a param existed, e.g. ``chars``) renderable
    and behaves identically to a freshly-seeded rule.
    """
    for r in rules:
        rtype = r.get("type")
        schema = RULE_TYPES.get(rtype, {}).get("params", {}) if isinstance(rtype, str) else {}
        params = r.get("params")
        if not isinstance(params, dict):
            r["params"] = {}
            params = r["params"]
        for key, spec in schema.items():
            if key not in params or params[key] in (None, ""):
                params[key] = spec["default"]
    return rules


async def load_rules_from_db(db_path: str) -> None:
    """Load stored rules into the live list at startup; seed defaults if none."""
    raw = await _db.get_setting(db_path, "loop_rules")
    if raw is None:
        seed_default_rules()
        # Persist the seeded defaults so a later reset / reload has a baseline.
        await _persist_rules(db_path, _live_rules)
        return
    try:
        data = _rules_from_text(raw)
    except Exception as e:
        logger.warning("corrupt stored loop_rules (%s); re-seeding defaults", e)
        seed_default_rules()
        return
    _live_rules[:] = _fill_param_defaults(data)
    logger.info("loop rules loaded from DB: %s",
                [(r.get("rule_id"), r.get("type"), r.get("enabled")) for r in data])
