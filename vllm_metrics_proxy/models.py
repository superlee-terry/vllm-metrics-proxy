from pydantic import BaseModel


class RequestRecord(BaseModel):
    id: str
    model: str | None = None
    stream: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    latency_ms: float | None = None
    ttft_ms: float | None = None
    prompt_speed: float | None = None
    completion_speed: float | None = None
    cached_ratio: float | None = None
    status: str = "success"
    error_message: str | None = None
