"""Fetch GPU stats via nvidia-smi with TTL cache."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time

logger = logging.getLogger(__name__)

_CACHE_TTL = 4.0  # seconds
_cached_result: list[dict] | None = None
_cache_time: float = 0.0


def _parse_gpu_csv(text: str) -> list[dict]:
    """Parse nvidia-smi CSV output into a list of GPU stat dicts.

    Expected columns: index, name, temperature.gpu, utilization.gpu, memory.used, memory.total
    """
    gpus: list[dict] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1].strip(),
                "temperature": int(parts[2]),
                "utilization": int(parts[3]),
                "memory_used": int(parts[4]),
                "memory_total": int(parts[5]),
            })
        except (ValueError, IndexError):
            continue
    return gpus


def _run_nvidia_smi() -> list[dict]:
    """Run nvidia-smi synchronously and return parsed results."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi returned code %d: %s", result.returncode, result.stderr.strip())
            return []
        return _parse_gpu_csv(result.stdout)
    except FileNotFoundError:
        logger.debug("nvidia-smi not found")
        return []
    except Exception as exc:
        logger.warning("nvidia-smi failed: %s", exc)
        return []


async def fetch_gpu_stats() -> list[dict]:
    """Fetch GPU stats with 4s TTL cache. Runs nvidia-smi in executor."""
    global _cached_result, _cache_time

    now = time.monotonic()
    if _cached_result is not None and (now - _cache_time) < _CACHE_TTL:
        return _cached_result

    loop = asyncio.get_running_loop()
    gpus = await loop.run_in_executor(None, _run_nvidia_smi)

    _cached_result = gpus
    _cache_time = now
    return gpus
