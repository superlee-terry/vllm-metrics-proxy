from fastapi import APIRouter, Request
from starlette.responses import JSONResponse

from vllm_metrics_proxy.proxy import proxy_request

router = APIRouter()


@router.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def forward_v1(request: Request) -> JSONResponse:
    return await proxy_request(request)
