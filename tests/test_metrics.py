import pytest
from vllm_metrics_proxy.metrics import compute_metrics, parse_since


class TestComputeMetrics:
    def test_non_streaming_full_usage(self):
        record = compute_metrics(
            request_id="r1",
            model="qwen3.6-27b",
            stream=False,
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=80,
            reasoning_tokens=None,
            latency_ms=1000.0,
            ttft_ms=None,
        )
        assert record["id"] == "r1"
        assert record["model"] == "qwen3.6-27b"
        assert record["stream"] is False
        assert record["prompt_tokens"] == 100
        assert record["completion_tokens"] == 50
        assert record["cached_tokens"] == 80
        assert record["latency_ms"] == 1000.0
        assert record["ttft_ms"] is None
        assert record["completion_speed"] is None
        assert record["total_tps"] == 150.0
        assert record["cached_ratio"] == 0.8
        assert record["spec_draft_tokens"] is None
        assert record["spec_accepted_tokens"] is None

    def test_streaming_with_ttft(self):
        record = compute_metrics(
            request_id="r2",
            model="qwen3.6-27b",
            stream=True,
            prompt_tokens=200,
            completion_tokens=100,
            cached_tokens=150,
            reasoning_tokens=10,
            latency_ms=2000.0,
            ttft_ms=50.0,
            spec_draft_tokens=20,
            spec_accepted_tokens=12,
        )
        assert record["ttft_ms"] == 50.0
        assert record["prompt_speed"] == 1000.0  # (200-150) / 0.05s
        assert record["completion_speed"] == pytest.approx(
            100 / ((2000.0 - 50.0) / 1000), rel=0.01
        )
        assert record["total_tps"] == pytest.approx(300 / 2.0, rel=0.01)
        assert record["cached_ratio"] == 0.75
        assert record["spec_draft_tokens"] == 20
        assert record["spec_accepted_tokens"] == 12

    def test_zero_prompt_tokens_no_division_by_zero(self):
        record = compute_metrics(
            request_id="r3",
            model="test",
            stream=False,
            prompt_tokens=0,
            completion_tokens=0,
            cached_tokens=0,
            reasoning_tokens=None,
            latency_ms=100.0,
            ttft_ms=None,
        )
        assert record["cached_ratio"] is None  # avoid 0/0
        assert record["prompt_speed"] is None
        assert record["completion_speed"] is None

    def test_zero_ttft_no_speed(self):
        record = compute_metrics(
            request_id="r4",
            model="test",
            stream=True,
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=0,
            reasoning_tokens=None,
            latency_ms=1000.0,
            ttft_ms=0.0,
        )
        assert record["prompt_speed"] is None


class TestParseSince:
    def test_1m(self):
        assert parse_since("1m") == pytest.approx(1 / 60.0)

    def test_5m(self):
        assert parse_since("5m") == pytest.approx(5 / 60.0)

    def test_30m(self):
        assert parse_since("30m") == pytest.approx(0.5)

    def test_1h(self):
        assert parse_since("1h") == 1.0

    def test_24h(self):
        assert parse_since("24h") == 24.0

    def test_7d(self):
        assert parse_since("7d") == 168.0

    def test_all(self):
        assert parse_since("all") is None

    def test_invalid(self):
        assert parse_since("abc") is None
