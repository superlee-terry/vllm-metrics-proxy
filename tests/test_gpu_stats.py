"""Tests for GPU stats module."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from vllm_metrics_proxy.gpu_stats import _parse_gpu_csv, fetch_gpu_stats
from vllm_metrics_proxy.main import create_app
from vllm_metrics_proxy.db import init_db


class TestParseGpuCsv:
    """Test CSV parsing of nvidia-smi output."""

    def test_parse_two_gpus(self):
        csv_output = (
            "0, NVIDIA GeForce RTX 3090, 40, 0, 22555, 24576\n"
            "1, NVIDIA GeForce RTX 3090, 49, 0, 22555, 24576\n"
        )
        gpus = _parse_gpu_csv(csv_output)
        assert len(gpus) == 2
        assert gpus[0]["index"] == 0
        assert gpus[0]["name"] == "NVIDIA GeForce RTX 3090"
        assert gpus[0]["temperature"] == 40
        assert gpus[0]["utilization"] == 0
        assert gpus[0]["memory_used"] == 22555
        assert gpus[0]["memory_total"] == 24576

    def test_parse_single_gpu(self):
        csv_output = "0, NVIDIA A100-SXM4-80GB, 65, 87, 40960, 81920\n"
        gpus = _parse_gpu_csv(csv_output)
        assert len(gpus) == 1
        assert gpus[0]["temperature"] == 65
        assert gpus[0]["utilization"] == 87

    def test_parse_empty_string(self):
        gpus = _parse_gpu_csv("")
        assert gpus == []

    def test_parse_whitespace_only(self):
        gpus = _parse_gpu_csv("  \n  \n")
        assert gpus == []

    def test_parse_malformed_line_skipped(self):
        csv_output = (
            "0, NVIDIA GeForce RTX 3090, 40, 0, 22555, 24576\n"
            "bad line\n"
            "1, NVIDIA GeForce RTX 3090, 49, 0, 22555, 24576\n"
        )
        gpus = _parse_gpu_csv(csv_output)
        assert len(gpus) == 2

    def test_parse_insufficient_columns_skipped(self):
        csv_output = "0, GPU, 40\n"
        gpus = _parse_gpu_csv(csv_output)
        assert gpus == []


class TestFetchGpuStats:
    """Test fetch_gpu_stats with mocked subprocess."""

    SAMPLE_CSV = (
        "0, NVIDIA GeForce RTX 3090, 40, 0, 22555, 24576\n"
        "1, NVIDIA GeForce RTX 3090, 49, 0, 22555, 24576\n"
    )

    @pytest.mark.asyncio
    async def test_returns_parsed_gpus(self):
        with patch("vllm_metrics_proxy.gpu_stats.subprocess") as mock_sub:
            mock_sub.run.return_value.returncode = 0
            mock_sub.run.return_value.stdout = self.SAMPLE_CSV

            gpus = await fetch_gpu_stats()
            assert len(gpus) == 2
            assert gpus[0]["temperature"] == 40
            assert gpus[1]["temperature"] == 49

    @pytest.mark.asyncio
    async def test_returns_empty_on_nvidia_smi_not_found(self):
        import vllm_metrics_proxy.gpu_stats as gpu_mod
        gpu_mod._cached_result = None  # clear cache

        with patch("vllm_metrics_proxy.gpu_stats.subprocess") as mock_sub:
            mock_sub.run.side_effect = FileNotFoundError("nvidia-smi not found")

            gpus = await fetch_gpu_stats()
            assert gpus == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_nonzero_exit(self):
        import vllm_metrics_proxy.gpu_stats as gpu_mod
        gpu_mod._cached_result = None  # clear cache

        with patch("vllm_metrics_proxy.gpu_stats.subprocess") as mock_sub:
            mock_sub.run.return_value.returncode = 1
            mock_sub.run.return_value.stderr = "error"

            gpus = await fetch_gpu_stats()
            assert gpus == []


@pytest.mark.asyncio
async def test_gpu_stats_endpoint(tmp_path):
    """Test /api/gpu-stats returns expected shape."""
    db_path = str(tmp_path / "test_gpu.db")
    await init_db(db_path)
    app = create_app(db_path=db_path)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/gpu-stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "gpus" in data
    assert isinstance(data["gpus"], list)
