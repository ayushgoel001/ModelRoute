import asyncio
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.exc import OperationalError

from app.api.schemas import ChatCompletionResponse
from app.db.models import RequestMetric
from app.exceptions import MetricsUnavailableError
from app.providers.base import ProviderMetadata
from app.services.metrics import MetricsService
from tests.conftest import FakeMetricsService, request


class RecordingSession:
    def __init__(self, records, *, fail_commit=False, connection_refused=False) -> None:
        self.records = records
        self.fail_commit = fail_commit
        self.connection_refused = connection_refused

    async def __aenter__(self):
        if self.connection_refused:
            raise ConnectionRefusedError("database unavailable")
        return self

    async def __aexit__(self, *args):
        return None

    def add(self, metric) -> None:
        self.records.append(metric)

    async def commit(self) -> None:
        if self.fail_commit:
            raise OperationalError("insert", {}, RuntimeError("database down"))


def response(*, cache_hit=False) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        request_id=uuid4(),
        provider="gemini",
        model="gemini-test",
        content="generated private content",
        input_tokens=100,
        output_tokens=20,
        latency_ms=42.5,
        cache_hit=cache_hit,
        fallback_used=True,
    )


def metadata(*, input_price=0.75, output_price=3.75) -> ProviderMetadata:
    return ProviderMetadata(
        name="gemini",
        model="gemini-test",
        available=True,
        input_cost_per_million=input_price,
        output_cost_per_million=output_price,
    )


def test_successful_metric_records_all_request_fields_without_private_data() -> None:
    records = []
    service = MetricsService(lambda: RecordingSession(records))
    completion = response()

    stored = asyncio.run(
        service.record_success(
            completion, routing_strategy="cheapest", provider=metadata()
        )
    )

    assert stored is True
    metric = records[0]
    assert metric.request_id == completion.request_id
    assert (metric.provider, metric.model) == ("gemini", "gemini-test")
    assert metric.routing_strategy == "cheapest"
    assert metric.status == "success"
    assert metric.latency_ms == 42.5
    assert (metric.input_tokens, metric.output_tokens) == (100, 20)
    assert metric.cache_hit is False
    assert metric.fallback_used is True
    assert metric.error_category is None
    assert "prompt" not in RequestMetric.__table__.columns
    assert "content" not in RequestMetric.__table__.columns
    assert not hasattr(metric, "prompt")
    assert not hasattr(metric, "content")


@pytest.mark.parametrize(
    ("provider", "expected"),
    [
        (metadata(), Decimal("0.000150")),
        (metadata(input_price=2, output_price=10), Decimal("0.000400")),
    ],
)
def test_cost_uses_selected_provider_pricing(provider, expected) -> None:
    assert MetricsService.estimate_cost(
        input_tokens=100,
        output_tokens=20,
        provider=provider,
        cache_hit=False,
    ) == expected


def test_cache_hit_cost_is_zero_even_when_tokens_are_present() -> None:
    assert MetricsService.estimate_cost(
        input_tokens=100,
        output_tokens=20,
        provider=metadata(),
        cache_hit=True,
    ) == 0


def test_failure_metric_stores_normalized_error_and_no_provider_usage() -> None:
    records = []
    service = MetricsService(lambda: RecordingSession(records))

    asyncio.run(
        service.record_failure(
            request_id=uuid4(),
            routing_strategy="fixed",
            latency_ms=12.0,
            error_category="ProviderTimeoutError",
        )
    )

    metric = records[0]
    assert metric.status == "error"
    assert metric.error_category == "ProviderTimeoutError"
    assert metric.provider is None and metric.model is None
    assert metric.estimated_cost_usd == 0


def test_insert_failure_is_fail_open() -> None:
    service = MetricsService(lambda: RecordingSession([], fail_commit=True))

    stored = asyncio.run(
        service.record_success(response(), routing_strategy="fixed", provider=metadata())
    )

    assert stored is False


def test_refused_database_connection_is_fail_open() -> None:
    service = MetricsService(
        lambda: RecordingSession([], connection_refused=True)
    )

    stored = asyncio.run(
        service.record_success(response(), routing_strategy="fixed", provider=metadata())
    )

    assert stored is False


class MappingResult:
    def __init__(self, rows) -> None:
        self.rows = rows

    def mappings(self):
        return self

    def one(self):
        return self.rows[0]

    def all(self):
        return self.rows


class QuerySession:
    def __init__(self, results) -> None:
        self.results = list(results)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def execute(self, statement):
        del statement
        return self.results.pop(0)


def aggregate_row(**overrides):
    row = {
        "total_requests": 4,
        "successful_requests": 3,
        "cache_hits": 1,
        "fallback_count": 1,
        "input_tokens": 300,
        "output_tokens": 80,
        "estimated_cost_usd": Decimal("0.0123"),
        "average_latency_ms": 50.0,
        "p50_latency_ms": 40.0,
        "p95_latency_ms": 95.0,
    }
    row.update(overrides)
    return row


def test_summary_rates_totals_percentiles_and_provider_breakdown() -> None:
    provider_row = {
        "provider": "gemini",
        "model": "gemini-test",
        "request_count": 3,
        "success_count": 3,
        "average_latency_ms": 45.0,
        "input_tokens": 300,
        "output_tokens": 80,
        "estimated_cost_usd": Decimal("0.0123"),
    }
    service = MetricsService(
        lambda: QuerySession(
            [MappingResult([aggregate_row()]), MappingResult([provider_row])]
        )
    )

    summary = asyncio.run(service.summary())

    assert summary["total_requests"] == 4
    assert summary["successful_requests"] == 3
    assert summary["success_rate"] == 0.75
    assert summary["cache_hit_rate"] == 0.25
    assert summary["fallback_count"] == 1
    assert (summary["input_tokens"], summary["output_tokens"]) == (300, 80)
    assert summary["estimated_cost_usd"] == 0.0123
    assert (summary["p50_latency_ms"], summary["p95_latency_ms"]) == (40, 95)
    assert summary["providers"][0]["request_count"] == 3


def test_empty_summary_has_finite_zero_values() -> None:
    empty = aggregate_row(
        total_requests=0,
        successful_requests=0,
        cache_hits=0,
        fallback_count=0,
        input_tokens=None,
        output_tokens=None,
        estimated_cost_usd=None,
        average_latency_ms=None,
        p50_latency_ms=None,
        p95_latency_ms=None,
    )
    service = MetricsService(
        lambda: QuerySession([MappingResult([empty]), MappingResult([])])
    )

    summary = asyncio.run(service.summary())

    assert summary["total_requests"] == 0
    assert summary["success_rate"] == 0
    assert summary["cache_hit_rate"] == 0
    assert summary["estimated_cost_usd"] == 0
    assert summary["p50_latency_ms"] == 0
    assert summary["providers"] == []


class UnavailableMetrics(FakeMetricsService):
    async def summary(self):
        raise MetricsUnavailableError("database unavailable")

    async def recent(self, limit=20):
        del limit
        raise MetricsUnavailableError("database unavailable")


def test_metrics_api_database_failure_returns_503(app_factory) -> None:
    application = app_factory(metrics_service=UnavailableMetrics())

    response_value = request(application, "GET", "/v1/metrics/summary")

    assert response_value.status_code == 503
    assert response_value.json() == {"detail": "Metrics database unavailable"}


class DashboardMetrics(FakeMetricsService):
    async def summary(self):
        return {
            **aggregate_row(),
            "success_rate": 0.75,
            "cache_hit_rate": 0.25,
            "providers": [],
        }

    async def recent(self, limit=20):
        del limit
        return []


def test_dashboard_renders_summary_without_external_assets(app_factory) -> None:
    application = app_factory(metrics_service=DashboardMetrics())

    response_value = request(application, "GET", "/dashboard")

    assert response_value.status_code == 200
    assert "ModelRoute" in response_value.text
    assert 'id="total-requests"' in response_value.text
    assert 'id="p50-latency"' in response_value.text
    assert 'id="p95-latency"' in response_value.text
    assert 'id="recent-rows"' in response_value.text
    assert 'id="cache-hit-segment"' in response_value.text
    assert 'id="latency-chart"' in response_value.text
    assert 'const summaryEndpoint = "/v1/metrics/summary"' in response_value.text
    assert 'const recentEndpoint = "/v1/metrics/recent?limit=20"' in response_value.text
    assert "const refreshIntervalMs = 15000" in response_value.text
    assert "Request ID" in response_value.text
    assert "Metrics up to date" in response_value.text
    assert "Refresh unavailable — showing last data" in response_value.text
    assert "Raw prompts and generated content are not persisted" in response_value.text
    assert "https://" not in response_value.text


class DashboardMetricsWithPrivateExtras(DashboardMetrics):
    async def recent(self, limit=20):
        del limit
        return [
            {
                "timestamp": "2026-08-24T12:00:00Z",
                "status": "success",
                "provider": "mock",
                "model": "mock-model",
                "routing_strategy": "fixed",
                "latency_ms": 4.2,
                "cache_hit": False,
                "fallback_used": False,
                "input_tokens": 3,
                "output_tokens": 2,
                "estimated_cost_usd": 0,
                "request_id": "safe-request-id",
                "prompt": "private prompt must never render",
                "content": "private generated content must never render",
            }
        ]


def test_dashboard_ignores_prompt_and_content_even_if_metrics_mapping_has_extras(
    app_factory,
) -> None:
    application = app_factory(metrics_service=DashboardMetricsWithPrivateExtras())

    response_value = request(application, "GET", "/dashboard")

    assert response_value.status_code == 200
    assert "safe-request-id" in response_value.text
    assert "private prompt must never render" not in response_value.text
    assert "private generated content must never render" not in response_value.text
