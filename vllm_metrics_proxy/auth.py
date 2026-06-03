from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone

import aiosqlite
from fastapi import HTTPException
from starlette.requests import Request


# ---- Key masking ----

def mask_key(key_id: str) -> str:
    """Mask a UUID: keep first 4 and last 4 chars, replace middle with ****."""
    if len(key_id) < 8:
        return key_id
    return key_id[:4] + "****" + key_id[-4:]


# ---- Expiry parsing ----

_EXPIRY_RE = re.compile(r"^(\d+)(h|d)$")


def parse_expires_in(expires_in: str | None) -> datetime | None:
    """Parse '30d', '1h', etc. into a datetime. None means never expire."""
    if expires_in is None:
        return None
    m = _EXPIRY_RE.match(expires_in)
    if not m:
        raise ValueError(
            f"Invalid expires_in format: {expires_in!r}. Use '30d', '1h', etc."
        )
    value = int(m.group(1))
    unit = m.group(2)
    now = datetime.now(timezone.utc)
    if unit == "h":
        return now + timedelta(hours=value)
    return now + timedelta(days=value)


# ---- DB CRUD ----

async def create_api_key(
    db_path: str, *, name: str, expires_in: str | None = None,
) -> dict:
    """Create a new API key. Returns the full key record (with plaintext id)."""
    key_id = str(uuid.uuid4())
    expired_at = parse_expires_in(expires_in)
    expired_at_str = expired_at.isoformat() if expired_at else None

    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO api_keys (id, name, expired_at) VALUES (?, ?, ?)",
            (key_id, name, expired_at_str),
        )
        await conn.commit()

    return {
        "id": key_id,
        "name": name,
        "expired_at": expired_at_str,
        "enabled": 1,
    }


async def get_api_key(db_path: str, key_id: str) -> dict | None:
    """Fetch a single API key by its id."""
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT * FROM api_keys WHERE id = ?", (key_id,)
        ) as cur:
            row = await cur.fetchone()
            if row is None:
                return None
            return dict(row)


async def list_api_keys(db_path: str) -> list[dict]:
    """List all API keys with masked ids."""
    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(
            "SELECT * FROM api_keys ORDER BY created_at DESC"
        ) as cur:
            rows = await cur.fetchall()
            return [{"masked_id": mask_key(dict(r)["id"]), **dict(r)} for r in rows]


async def delete_api_key(db_path: str, key_id: str) -> bool:
    """Delete an API key. Returns True if deleted, False if not found."""
    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(
            "DELETE FROM api_keys WHERE id = ?", (key_id,)
        )
        await conn.commit()
        return cursor.rowcount > 0


async def update_api_key(
    db_path: str, key_id: str,
    *,
    enabled: bool | None = None,
    expires_in: str | None = None,
) -> bool:
    """Update an API key's enabled status and/or expiry. Returns True if updated."""
    fields: list[str] = []
    params: list = []

    if enabled is not None:
        fields.append("enabled = ?")
        params.append(1 if enabled else 0)

    if expires_in is not None:
        expired_at = parse_expires_in(expires_in)
        fields.append("expired_at = ?")
        params.append(expired_at.isoformat() if expired_at else None)

    if not fields:
        return False

    params.append(key_id)
    async with aiosqlite.connect(db_path) as conn:
        cursor = await conn.execute(
            f"UPDATE api_keys SET {', '.join(fields)} WHERE id = ?",
            params,
        )
        await conn.commit()
        return cursor.rowcount > 0


# ---- Verification (FastAPI Depends) ----

_NO_AUTH_SENTINEL = "__no_auth__"


def _extract_key_from_headers(headers: dict) -> str | None:
    """Extract API key from Authorization: Bearer or X-API-Key header."""
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    x_key = headers.get("x-api-key")
    if x_key:
        return x_key.strip()
    return None


async def verify_api_key(request: Request) -> str:
    """FastAPI Depends dependency.

    Returns key_id on success, raises HTTPException on failure.
    When auth_enabled is False, returns a sentinel value (skip auth).
    """
    if not request.app.state.settings.auth_enabled:
        return _NO_AUTH_SENTINEL

    key_id = _extract_key_from_headers(dict(request.headers))
    if not key_id:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")

    db_path = request.app.state.db_path
    key = await get_api_key(db_path, key_id)
    if key is None:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")

    if not key["enabled"]:
        raise HTTPException(status_code=403, detail="API key has been disabled")

    if key["expired_at"] is not None:
        expired_at = datetime.fromisoformat(key["expired_at"])
        if datetime.now(timezone.utc) > expired_at:
            raise HTTPException(status_code=401, detail="API key has expired")

    return key_id
