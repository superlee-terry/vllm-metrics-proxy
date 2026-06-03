from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from starlette.responses import FileResponse, JSONResponse

from vllm_metrics_proxy.auth import (
    create_api_key, delete_api_key, list_api_keys, update_api_key,
    verify_admin_token,
)
from vllm_metrics_proxy.config import settings
from vllm_metrics_proxy.db import get_requests, get_requests_count, get_summary, get_summary_by_model
from vllm_metrics_proxy.metrics import parse_since
from vllm_metrics_proxy.gpu_stats import fetch_gpu_stats
from vllm_metrics_proxy.vllm_metrics import fetch_engine_stats
from vllm_metrics_proxy.proxy import (
    get_active_requests,
    cancel_active_request,
    register_active_request,
    unregister_active_request,
)

router = APIRouter()


@router.get("/")
async def index():
    return FileResponse("static/index.html")


@router.get("/admin")
async def admin():
    return FileResponse("static/admin.html")


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


@router.get("/api/gpu-stats")
async def gpu_stats():
    """GPU temperature and utilization from nvidia-smi."""
    gpus = await fetch_gpu_stats()
    return {"gpus": gpus}


@router.get("/api/engine-stats")
async def engine_stats():
    """Real-time vLLM engine stats from Prometheus /metrics."""
    return await fetch_engine_stats(settings.vllm_upstream)


@router.get("/api/active-requests")
async def active_requests():
    """List currently in-flight requests."""
    return {"requests": get_active_requests()}


@router.post("/api/active-requests/{request_id}/cancel")
async def cancel_request(request_id: str):
    """Cancel an active request by ID. Returns 404 if not found."""
    from starlette.responses import JSONResponse

    cancelled = cancel_active_request(request_id)
    if not cancelled:
        return JSONResponse(
            status_code=404,
            content={"status": "not_found", "message": f"request {request_id} not active"},
        )
    return {"status": "cancelled", "request_id": request_id}


# ---- API Key Management ----

@router.post("/api/keys")
async def create_key(request: Request, _admin: None = Depends(verify_admin_token)):
    body = await request.json()
    name = body.get("name")
    if not name:
        return JSONResponse(status_code=400, content={"detail": "name is required"})
    expires_in = body.get("expires_in")
    try:
        key = await create_api_key(request.app.state.db_path, name=name, expires_in=expires_in)
        return key
    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})


@router.get("/api/keys")
async def list_keys(request: Request):
    keys = await list_api_keys(request.app.state.db_path)
    safe_keys = []
    for k in keys:
        safe = {
            "id": k["id"],
            "masked_id": k["masked_id"],
            "name": k["name"],
            "expired_at": k["expired_at"],
            "created_at": k["created_at"],
            "enabled": k["enabled"],
        }
        safe_keys.append(safe)
    return {"keys": safe_keys}


@router.delete("/api/keys/{key_id}")
async def remove_key(request: Request, key_id: str, _admin: None = Depends(verify_admin_token)):
    deleted = await delete_api_key(request.app.state.db_path, key_id)
    if not deleted:
        return JSONResponse(status_code=404, content={"detail": "API key not found"})
    return {"status": "deleted", "key_id": key_id}


@router.patch("/api/keys/{key_id}")
async def patch_key(request: Request, key_id: str, _admin: None = Depends(verify_admin_token)):
    body = await request.json()
    enabled = body.get("enabled")
    expires_in = body.get("expires_in")
    if enabled is None and expires_in is None:
        return JSONResponse(status_code=400, content={"detail": "no fields to update"})
    try:
        updated = await update_api_key(
            request.app.state.db_path, key_id,
            enabled=enabled, expires_in=expires_in,
        )
    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})
    if not updated:
        return JSONResponse(status_code=404, content={"detail": "API key not found"})
    return {"status": "updated", "key_id": key_id}
