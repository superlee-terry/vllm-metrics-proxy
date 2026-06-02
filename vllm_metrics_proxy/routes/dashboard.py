from __future__ import annotations

from fastapi import APIRouter, Request
from starlette.responses import FileResponse

from vllm_metrics_proxy.db import get_requests, get_requests_count, get_summary, get_summary_by_model
from vllm_metrics_proxy.metrics import parse_since

router = APIRouter()


@router.get("/")
async def index():
    return FileResponse("static/index.html")


@router.get("/api/health")
async def health():
    return {"status": "ok"}


@router.get("/api/summary")
async def summary(request: Request, since: str = "1h"):
    db_path = request.app.state.db_path
    since_hours = parse_since(since)

    summary_data = await get_summary(db_path, since_hours=since_hours)
    by_model = await get_summary_by_model(db_path, since_hours=since_hours)

    return {
        "period": since,
        **summary_data,
        "by_model": by_model,
    }


@router.get("/api/requests")
async def requests_list(
    request: Request,
    since: str = "1h",
    limit: int = 50,
    offset: int = 0,
):
    db_path = request.app.state.db_path
    since_hours = parse_since(since)

    rows = await get_requests(db_path, limit=limit, offset=offset, since_hours=since_hours)
    total = await get_requests_count(db_path, since_hours=since_hours)

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "requests": rows,
    }
