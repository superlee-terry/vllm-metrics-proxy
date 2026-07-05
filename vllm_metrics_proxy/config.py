from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    vllm_upstream: str = "http://localhost:11434"
    proxy_port: int = 8000
    db_path: str = "./metrics.db"
    log_level: str = "INFO"
    auth_enabled: bool = True
    dashboard_password: str = ""
    session_ttl_seconds: int = 86400       # admin token 有效期（秒），默认 24h

    # Loop detection config
    loop_detection_enabled: bool = True
    loop_window_size: int = 20            # sliding window of recent non-empty chunk contents
    loop_repeat_threshold: int = 3        # pattern repeated N times triggers cutoff
    loop_min_tail_match: int = 5          # last N chunks identical triggers cutoff

    # Timeout config
    request_timeout_seconds: float = 360.0  # wall-clock total duration cap for ALL requests
                                          # (streaming: total output time; non-streaming: total request time)
    stream_idle_timeout: float = 30.0      # max seconds between two consecutive chunks (stall detection)

    model_config = {"env_prefix": ""}


settings = Settings()
