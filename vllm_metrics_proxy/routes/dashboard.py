from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import FileResponse, JSONResponse

from vllm_metrics_proxy.auth import (
    create_api_key, create_admin_token, delete_api_key, list_api_keys,
    update_api_key, verify_admin_token,
)
from vllm_metrics_proxy.config import settings
from vllm_metrics_proxy import loop_rules
from vllm_metrics_proxy.config_manager import (
    CONFIG_ITEMS, get_state, persist_changes, revert_to_default, validate_payload,
)
from vllm_metrics_proxy.db import get_requests, get_requests_count, get_key_names_by_ids, get_summary, get_summary_by_model
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
    return FileResponse("static/welcome.html")


@router.get("/dashboard")
async def dashboard():
    return FileResponse("static/dashboard.html")


@router.get("/admin")
async def admin():
    return FileResponse("static/admin.html")


@router.get("/config")
async def config_page():
    return FileResponse("static/config.html")


@router.post("/api/auth/verify")
async def verify_dashboard_password(request: Request):
    """Verify the dashboard/admin password.

    Returns 200 on match, 403 on mismatch, 503 if password not configured.
    """
    expected = request.app.state.settings.dashboard_password
    if not expected:
        return JSONResponse(
            status_code=503,
            content={"detail": "DASHBOARD_PASSWORD 未配置，请在环境变量中设置后重启服务"},
        )
    body = await request.json()
    provided = body.get("password", "")
    if provided != expected:
        raise HTTPException(status_code=403, detail="密码错误")
    token = create_admin_token(provided)
    return {"status": "ok", "token": token}


@router.get("/api/health")
async def health():
    return {"status": "ok"}


@router.get("/api/summary")
async def summary(request: Request, since: str = "1h", _admin: None = Depends(verify_admin_token)):
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
    _admin: None = Depends(verify_admin_token),
):
    db_path = request.app.state.db_path
    since_hours = parse_since(since)

    rows = await get_requests(db_path, limit=limit, offset=offset, since_hours=since_hours)
    total = await get_requests_count(db_path, since_hours=since_hours)

    # Enrich with key names
    key_ids = [r["api_key_id"] for r in rows if r.get("api_key_id")]
    key_names = await get_key_names_by_ids(db_path, key_ids) if key_ids else {}
    for r in rows:
        kid = r.get("api_key_id")
        r["api_key_name"] = key_names.get(kid, "") if kid else ""

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "requests": rows,
    }


@router.get("/api/gpu-stats")
async def gpu_stats(_admin: None = Depends(verify_admin_token)):
    """GPU temperature and utilization from nvidia-smi."""
    gpus = await fetch_gpu_stats()
    return {"gpus": gpus}


@router.get("/api/engine-stats")
async def engine_stats(_admin: None = Depends(verify_admin_token)):
    """Real-time vLLM engine stats from Prometheus /metrics."""
    return await fetch_engine_stats(settings.vllm_upstream)


@router.get("/api/active-requests")
async def active_requests(_admin: None = Depends(verify_admin_token)):
    """List currently in-flight requests."""
    return {"requests": get_active_requests()}


@router.post("/api/active-requests/{request_id}/cancel")
async def cancel_request(request_id: str, _admin: None = Depends(verify_admin_token)):
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
    name = body.get("name") or ""
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
    name = body.get("name")
    enabled = body.get("enabled")
    expires_in = body.get("expires_in")
    if name is None and enabled is None and expires_in is None:
        return JSONResponse(status_code=400, content={"detail": "no fields to update"})
    try:
        updated = await update_api_key(
            request.app.state.db_path, key_id,
            name=name, enabled=enabled, expires_in=expires_in,
        )
    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})
    if not updated:
        return JSONResponse(status_code=404, content={"detail": "API key not found"})
    return {"status": "updated", "key_id": key_id}


# ---- Runtime config (loop detection / timeouts) — live-editable ----

@router.get("/api/config")
async def get_config(request: Request, _admin: None = Depends(verify_admin_token)):
    """Current config state: effective value, code default, DB override, source."""
    return await get_state(request.app.state.db_path)


@router.put("/api/config")
async def put_config(request: Request, _admin: None = Depends(verify_admin_token)):
    """Update one or more config keys. Validates, persists to DB, applies live.

    Body: {"<key>": value, ...}.  A key set to ``null`` reverts it to the
    code default.  Changes take effect immediately (no restart).
    """
    body = await request.json()
    if not isinstance(body, dict) or not body:
        return JSONResponse(status_code=400, content={"detail": "empty or invalid body"})

    # Separate nulls (revert) from concrete values (set).
    reverts = [k for k, v in body.items() if v is None]
    sets = {k: v for k, v in body.items() if v is not None}

    try:
        cleaned = validate_payload(sets) if sets else {}
    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})

    db_path = request.app.state.db_path
    for key in reverts:
        if key not in CONFIG_ITEMS:
            return JSONResponse(status_code=422, content={"detail": f"未知配置项: {key}"})
        await revert_to_default(db_path, key)

    if cleaned:
        await persist_changes(db_path, cleaned)

    return {
        "status": "ok",
        "updated": sorted(cleaned),
        "reverted": reverts,
    }


# ---- Loop-detection rules — ordered, DB-backed list (live) --------------

@router.get("/api/loop-rules")
async def get_loop_rules(_admin: None = Depends(verify_admin_token)):
    """Current ordered rule list + the schema for each known rule type.

    ``rules`` is the live list in priority order (earlier = higher priority).
    ``types`` lets the page render param editors / validation per rule type.
    """
    return {
        "rules": loop_rules.get_live_rules(),
        "types": loop_rules.RULE_TYPES,
    }


@router.put("/api/loop-rules")
async def put_loop_rules(request: Request, _admin: None = Depends(verify_admin_token)):
    """Replace the whole ordered rule list (add / edit / delete / reorder).

    The page sends the complete list it has rendered, so a single PUT covers
    every mutation.  Each item: ``{"rule_id", "type", "enabled", "params"}``.
    Applied live (no restart) and persisted to the settings table.
    """
    body = await request.json()
    if not isinstance(body, list):
        return JSONResponse(status_code=400, content={"detail": "body must be a list of rules"})
    try:
        normalised = await loop_rules.apply_rules(request.app.state.db_path, body)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})
    return {
        "status": "ok",
        "count": len(normalised),
        "rules": normalised,
    }


@router.post("/api/loop-rules/reset")
async def reset_loop_rules(request: Request, _admin: None = Depends(verify_admin_token)):
    """Restore the factory-default rules (tail_match / chunk_repeat / punct_spam)."""
    try:
        normalised = await loop_rules.apply_rules(
            request.app.state.db_path, loop_rules.default_rules()
        )
    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})
    return {
        "status": "ok",
        "count": len(normalised),
        "rules": normalised,
    }
