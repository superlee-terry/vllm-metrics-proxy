"""Runtime-configurable proxy settings (loop detection / timeouts).

Design
------
The streaming hot path in ``proxy.py`` reads the module-level ``settings``
singleton (``vllm_metrics_proxy.config.settings``) on every chunk — e.g.
``settings.loop_window_size``, ``settings.loop_punct_spam_min_chars``,
``settings.request_timeout_seconds``.  Because that object is a process-global,
mutating one of its attributes at runtime is picked up immediately by all
subsequent requests — **no restart required**.

This module wraps that:

* :data:`CONFIG_ITEMS` – the registry of user-editable keys (label, type,
  range, description) shown on the maintenance page.
* :func:`validate_payload` – coerce / range-check a request body.
* :func:`get_state` – current value + source for every key (DB override,
  env-var, or code default).
* :func:`apply_overrides` – validate, mutate the live settings object and
  persist to the ``settings`` table in ``metrics.db``.
* :func:`load_into_live` – called at startup to apply stored overrides.

Values are stored as JSON text in the ``settings`` table; a key absent from
the table means "use the code/env default".
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from pydantic_core import PydanticUndefined

from vllm_metrics_proxy.config import settings as live_settings
from vllm_metrics_proxy import db as _db

logger = logging.getLogger(__name__)


# ---- Configurable items registry ------------------------------------------

# type: bool | int | float
CONFIG_ITEMS: dict[str, dict] = {
    # Loop detection
    "loop_detection_enabled": {
        "label": "循环检测启用", "type": "bool", "min": None, "max": None,
        "desc": "True=检出循环后掐断流；False=仅记录 loop_debug 日志不掐断",
    },
    "loop_window_size": {
        "label": "滑动窗口大小", "type": "int", "min": 5, "max": 200,
        "desc": "保留最近 N 个非空 chunk 用于循环检测（各规则在此窗口内自行判断）",
    },
    # NOTE: the per-rule thresholds (tail_match / chunk_repeat / punct_spam)
    # are no longer global scalars — they live inside the ordered loop-rules
    # list (the "循环检测规则" section of this page, backed by the `loop_rules`
    # row in the settings table).
    # Timeouts
    "request_timeout_seconds": {
        "label": "总时长上限(秒)", "type": "float", "min": 30, "max": 3600,
        "desc": "单个请求墙钟总时长上限（流式=输出总时长，非流式=整个请求）",
    },
    "stream_idle_timeout": {
        "label": "空闲超时(秒)", "type": "float", "min": 5, "max": 300,
        "desc": "两个 chunk 之间的最大间隔，超过即判定上游卡死",
    },
}


def _item(key: str) -> dict | None:
    return CONFIG_ITEMS.get(key)


def validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce a config payload.

    Returns the cleaned payload (only known keys, correctly typed, in range).
    Raises :class:`ValueError` describing the first problem found.
    """
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")
    cleaned: dict[str, Any] = {}
    for key, raw in payload.items():
        item = _item(key)
        if item is None:
            raise ValueError(f"未知配置项: {key}")
        if raw is None:
            # None == "revert to default" — handled by caller via DELETE/absent
            continue
        t = item["type"]
        if t == "bool":
            if isinstance(raw, bool):
                val: Any = raw
            elif raw in ("true", "True", "1", "yes"):
                val = True
            elif raw in ("false", "False", "0", "no"):
                val = False
            else:
                raise ValueError(f"{key} 必须是布尔值")
        elif t == "int":
            try:
                val = int(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{key} 必须是整数")
            lo, hi = item["min"], item["max"]
            if lo is not None and val < lo:
                raise ValueError(f"{key} 不能小于 {lo}")
            if hi is not None and val > hi:
                raise ValueError(f"{key} 不能大于 {hi}")
        elif t == "float":
            try:
                val = float(raw)
            except (TypeError, ValueError):
                raise ValueError(f"{key} 必须是数值")
            lo, hi = item["min"], item["max"]
            if lo is not None and val < lo:
                raise ValueError(f"{key} 不能小于 {lo}")
            if hi is not None and val > hi:
                raise ValueError(f"{key} 不能大于 {hi}")
        else:
            raise ValueError(f"内部错误：未知类型 {t} ({key})")
        cleaned[key] = val
    return cleaned


def _current_value(key: str) -> Any:
    return getattr(live_settings, key)


async def get_state(db_path: str) -> dict:
    """Build the full config state for the maintenance page.

    Returns ``{"settings": {key: {label, type, min, max, desc, current,
    default, db, source}}}``.  ``current`` is the effective (live) value,
    ``default`` is the code default, ``db`` is the stored override (or None),
    and ``source`` is 数据库 / 环境变量 / 默认值.
    """
    stored = await _db.get_all_settings(db_path)
    out: dict[str, dict] = {}
    for key, item in CONFIG_ITEMS.items():
        current = _current_value(key)
        default = _default_value(key)
        db_raw = stored.get(key)
        out[key] = {
            "label": item["label"],
            "type": item["type"],
            "min": item["min"],
            "max": item["max"],
            "desc": item["desc"],
            "current": current,
            "default": default,
            "db": db_raw,
            "source": _source(key, current, db_raw),
        }
    return {"settings": out}


def _default_value(key: str) -> Any:
    """Code-default value for a key (from the pydantic field, env-agnostic)."""
    from vllm_metrics_proxy.config import Settings
    field = Settings.model_fields.get(key)
    if field is not None and field.default is not PydanticUndefined:
        return field.default
    return _current_value(key)


def _source(key: str, current: Any, db_raw: str | None) -> str:
    if db_raw is not None:
        try:
            if _value_from_text(db_raw) == current:
                return "数据库"
        except Exception:
            return "数据库"
    env_val = os.environ.get(key)
    if env_val is not None:
        try:
            if _value_from_text(env_val) != current:
                return "环境变量"
        except Exception:
            return "环境变量"
    return "默认值"


def apply_override_to_live(key: str, value: Any) -> None:
    """Mutate the process-global settings object so the change takes effect now."""
    try:
        setattr(live_settings, key, value)
    except Exception as e:  # pydantic may enforce types/validators
        raise ValueError(f"写入配置 {key} 失败: {e}") from e


def _value_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _value_from_text(text: str) -> Any:
    return json.loads(text)


def diff_summary(cleaned: dict[str, Any]) -> list[str]:
    """Human-readable summary of which keys changed by a payload."""
    out = []
    for k, v in cleaned.items():
        item = _item(k)
        if item:
            out.append(f"{item['label']} -> {v}")
    return out


# ---- Persistence (DB) + live-apply ---------------------------------------

async def persist_changes(db_path: str, cleaned: dict[str, Any]) -> None:
    """Persist validated values into the settings table AND apply them live."""
    for key, value in cleaned.items():
        await _db.upsert_setting(db_path, key, _value_text(value))
        apply_override_to_live(key, value)
    if cleaned:
        logger.info("config updated (live+DB): %s", diff_summary(cleaned))


async def revert_to_default(db_path: str, key: str) -> None:
    """Remove a DB override and reset the live value to the code default."""
    await _db.delete_setting(db_path, key)
    apply_override_to_live(key, _default_value(key))
    logger.info("config reverted to default: %s", key)


async def load_overrides_into_live(db_path: str) -> None:
    """At startup, apply any stored DB overrides to the live settings object.

    Called in the app lifespan so settings persisted by the maintenance page
    survive a service restart.
    """
    stored = await _db.get_all_settings(db_path)
    for key, raw in stored.items():
        if _item(key) is None:
            # Keys we no longer manage as scalars — notably ``loop_rules``,
            # which is a JSON list handled by ``loop_rules.load_rules_from_db``.
            continue  # ignore keys we no longer manage
        try:
            value = _value_from_text(raw)
        except Exception as e:
            logger.warning("skip corrupt stored setting %s=%r: %s", key, raw, e)
            continue
        apply_override_to_live(key, value)
    if stored:
        logger.info("config overrides loaded from DB: %s", sorted(stored))
