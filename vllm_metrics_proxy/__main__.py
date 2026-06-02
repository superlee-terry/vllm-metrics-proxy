import logging
import uvicorn

from vllm_metrics_proxy.config import settings


def main():
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(levelname)s %(name)s %(message)s",
    )
    uvicorn.run(
        "vllm_metrics_proxy.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=settings.proxy_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
