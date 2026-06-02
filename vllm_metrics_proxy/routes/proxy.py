from fastapi import APIRouter, Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from vllm_metrics_proxy.proxy import proxy_request

router = APIRouter()


@router.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"], response_model=None)
async def forward_v1(request: Request):
    return await proxy_request(request)


@router.get("/health", include_in_schema=False)
@router.api_route("/ping", methods=["GET", "POST"], include_in_schema=False)
@router.get("/version", include_in_schema=False)
@router.get("/openapi.json", include_in_schema=False)
async def forward_utils(request: Request) -> Response:
    """Forward utility endpoints to vLLM without metrics recording."""
    from vllm_metrics_proxy.config import settings
    import httpx

    body = await request.body()
    upstream = settings.vllm_upstream.rstrip("/")
    upstream_url = f"{upstream}{request.url.path}"

    headers = dict(request.headers)
    headers.pop("host", None)

    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        resp = await client.request(
            request.method, upstream_url, content=body, headers=headers,
            follow_redirects=True,
        )

    return Response(
        status_code=resp.status_code,
        content=resp.content,
        headers=dict(resp.headers),
    )
