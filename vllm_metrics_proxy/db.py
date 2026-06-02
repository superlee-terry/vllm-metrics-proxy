from __future__ import annotations

import aiosqlite

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    model           TEXT,
    stream          INTEGER NOT NULL DEFAULT 0,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    cached_tokens   INTEGER,
    reasoning_tokens INTEGER,
    latency_ms      REAL,
    ttft_ms         REAL,
    prompt_speed    REAL,
    completion_speed REAL,
    cached_ratio    REAL,
    status          TEXT NOT NULL DEFAULT 'success',
    error_message   TEXT
);

CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_requests_model ON requests(model);
CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status);
"""

INSERT_SQL = """
INSERT INTO requests (
    id, model, stream, prompt_tokens, completion_tokens,
    cached_tokens, reasoning_tokens, latency_ms, ttft_ms,
    prompt_speed, completion_speed, cached_ratio, status, error_message
) VALUES (
    :id, :model, :stream, :prompt_tokens, :completion_tokens,
    :cached_tokens, :reasoning_tokens, :latency_ms, :ttft_ms,
    :prompt_speed, :completion_speed, :cached_ratio, :status, :error_message
)
"""


async def init_db(db_path: str) -> None:
    async with aiosqlite.connect(db_path) as conn:
        await conn.executescript(SCHEMA)
        await conn.commit()


async def insert_request(db_path: str, record: dict) -> None:
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(INSERT_SQL, record)
        await conn.commit()


async def get_requests(
    db_path: str, *, limit: int = 50, offset: int = 0,
    since_hours: float | None = None, model: str | None = None,
) -> list[dict]:
    query = "SELECT * FROM requests WHERE 1=1"
    params: list = []

    if since_hours is not None:
        query += " AND created_at >= datetime('now', ?)"
        params.append(f"-{since_hours} hours")

    if model:
        query += " AND model = ?"
        params.append(model)

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(query, params) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def get_requests_count(
    db_path: str, *, since_hours: float | None = None,
) -> int:
    query = "SELECT COUNT(*) FROM requests WHERE 1=1"
    params: list = []

    if since_hours is not None:
        query += " AND created_at >= datetime('now', ?)"
        params.append(f"-{since_hours} hours")

    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute(query, params) as cur:
            row = await cur.fetchone()
            return row[0]


async def get_summary(
    db_path: str, *, since_hours: float | None = None,
) -> dict:
    query = """
    SELECT
        COUNT(*) as total_requests,
        AVG(ttft_ms) as avg_ttft_ms,
        AVG(completion_speed) as avg_tps,
        AVG(cached_ratio) as avg_cache_ratio,
        SUM(prompt_tokens) as total_prompt_tokens,
        SUM(completion_tokens) as total_completion_tokens
    FROM requests
    WHERE status = 'success' AND 1=1
    """
    params: list = []

    if since_hours is not None:
        query = query.replace("AND 1=1", "AND created_at >= datetime('now', ?)")
        params.append(f"-{since_hours} hours")

    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute(query, params) as cur:
            row = await cur.fetchone()
            return {
                "total_requests": row[0] or 0,
                "avg_ttft_ms": round(row[1], 1) if row[1] else None,
                "avg_tps": round(row[2], 1) if row[2] else None,
                "avg_cache_ratio": round(row[3], 3) if row[3] else None,
                "total_prompt_tokens": row[4] or 0,
                "total_completion_tokens": row[5] or 0,
            }


async def get_summary_by_model(
    db_path: str, *, since_hours: float | None = None,
) -> list[dict]:
    query = """
    SELECT
        model,
        COUNT(*) as count,
        AVG(ttft_ms) as avg_ttft_ms,
        AVG(completion_speed) as avg_tps,
        AVG(cached_ratio) as avg_cache_ratio
    FROM requests
    WHERE status = 'success' AND 1=1
    GROUP BY model
    ORDER BY count DESC
    """
    params: list = []

    if since_hours is not None:
        query = query.replace("AND 1=1", "AND created_at >= datetime('now', ?)")
        params.append(f"-{since_hours} hours")

    async with aiosqlite.connect(db_path) as conn:
        async with conn.execute(query, params) as cur:
            rows = await cur.fetchall()
            return [
                {
                    "model": r[0],
                    "count": r[1],
                    "avg_ttft_ms": round(r[2], 1) if r[2] else None,
                    "avg_tps": round(r[3], 1) if r[3] else None,
                    "avg_cache_ratio": round(r[4], 3) if r[4] else None,
                }
                for r in rows
            ]
