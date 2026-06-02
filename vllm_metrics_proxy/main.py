from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from vllm_metrics_proxy.config import Settings, settings
from vllm_metrics_proxy.db import init_db
from vllm_metrics_proxy.routes.proxy import router as proxy_router
from vllm_metrics_proxy.routes.dashboard import router as dashboard_router


def create_app(settings_override: Settings | None = None, db_path: str | None = None) -> FastAPI:
    _settings = settings_override or settings
    _db_path = db_path or _settings.db_path

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_db(_db_path)
        yield

    app = FastAPI(title="vLLM Metrics Proxy", lifespan=lifespan)
    app.state.settings = _settings
    app.state.db_path = _db_path

    app.include_router(proxy_router)
    app.include_router(dashboard_router)

    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app
