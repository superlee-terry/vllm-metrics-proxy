# vLLM Metrics Proxy

Transparent reverse proxy for vLLM that records per-request metrics (latency, tokens, cache ratio) into SQLite and displays them in a Web UI dashboard.

## Quick Start

```bash
pip install -e .

# Point VLLM_UPSTREAM at your vLLM instance, start the proxy
VLLM_UPSTREAM=http://localhost:8001 vllm-metrics-proxy
```

Then send requests to `http://localhost:8000/v1/...` instead of `http://localhost:8001/v1/...`.

Open `http://localhost:8000/` for the metrics dashboard.

## Configuration

| Env Var | Default | Description |
|---|---|---|
| `VLLM_UPSTREAM` | `http://localhost:8001` | vLLM upstream URL |
| `PROXY_PORT` | `8000` | Proxy listen port |
| `DB_PATH` | `./metrics.db` | SQLite database path |
| `LOG_LEVEL` | `INFO` | Log level |

## Metrics Collected

- **Latency**: end-to-end latency (ms)
- **TTFT**: time to first content token (ms, streaming only)
- **Prompt speed**: prompt_tokens / ttft_seconds (tok/s)
- **Completion speed**: completion_tokens / generate_time (tok/s, streaming only)
- **Cache hit ratio**: cached_tokens / prompt_tokens
- **Token usage**: prompt_tokens, completion_tokens, cached_tokens, reasoning_tokens

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
