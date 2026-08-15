import json

import pytest

from scripts.benchmark import build_document, percentile, summarize, write_document


def test_percentiles_use_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.50) == 2.5
    assert percentile(values, 0.95) == pytest.approx(3.85)


def test_summary_calculates_error_rate_and_required_metrics() -> None:
    outcomes = [
        {"success": True, "latency_ms": 10, "cache_hit": True},
        {
            "success": False,
            "latency_ms": 20,
            "cache_hit": None,
            "error_category": "HTTP503",
        },
    ]

    result = summarize(
        scenario="cached",
        concurrency=2,
        outcomes=outcomes,
        total_elapsed_seconds=0.5,
    )

    assert result["total_requests"] == 2
    assert result["successful_requests"] == 1
    assert result["failed_requests"] == 1
    assert result["error_rate"] == 0.5
    assert result["requests_per_second"] == 4
    assert result["p50_latency_ms"] == 15


def test_result_serialization_contains_required_metadata_and_fields(
    tmp_path,
) -> None:
    result = summarize(
        scenario="uncached",
        concurrency=1,
        outcomes=[{"success": True, "latency_ms": 12, "cache_hit": False}],
        total_elapsed_seconds=0.012,
    )
    document = build_document(
        base_url="http://127.0.0.1:8000",
        request_count=1,
        concurrency_levels=[1],
        scenarios=["uncached"],
        warmup_requests=1,
        simulated_provider_latency_ms=50,
        results=[result],
    )
    output = tmp_path / "result.json"

    write_document(document, output)
    serialized = json.loads(output.read_text(encoding="utf-8"))

    assert serialized["metadata"]["simulated_provider_latency_ms"] == 50
    assert serialized["metadata"]["uvicorn_workers"] == 1
    assert serialized["results"][0]["p95_latency_ms"] == 12
    assert serialized["results"][0]["error_rate"] == 0
