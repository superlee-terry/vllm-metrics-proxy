from fastapi import APIRouter
from starlette.responses import FileResponse

router = APIRouter()


@router.get("/")
async def index():
    return FileResponse("static/index.html")


@router.get("/api/health")
async def health():
    return {"status": "ok"}
