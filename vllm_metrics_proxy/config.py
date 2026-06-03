from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vllm_upstream: str = "http://localhost:8001"
    proxy_port: int = 8000
    db_path: str = "./metrics.db"
    log_level: str = "INFO"
    auth_enabled: bool = False

    model_config = {"env_prefix": ""}


settings = Settings()
