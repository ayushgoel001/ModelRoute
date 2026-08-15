from app.exceptions import ProviderRequestError, ProviderTimeoutError
from tests.conftest import (
    FakeMetricsService,
    FakeProvider,
    FakeRedis,
    provider_result,
    request,
)


def test_api_rate_limit_router_cache_miss_provider_response(
    app_factory, valid_payload
) -> None:
    redis = FakeRedis()
    provider = FakeProvider("openai")
    metrics = FakeMetricsService()
    application = app_factory(
        [provider],
        redis_client=redis,
        default_provider="openai",
        metrics_service=metrics,
    )

    response = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    )

    assert response.status_code == 200
    assert response.json()["cache_hit"] is False
    assert provider.calls == 1
    assert len(redis.eval_keys) == 1
    assert len(redis.values) == 1
    assert len(metrics.records) == 1
    assert metrics.records[0]["status"] == "success"
    assert metrics.records[0]["provider"] == "openai"
    assert metrics.records[0]["cache_hit"] is False


def test_api_rate_limit_router_cache_hit_skips_provider(
    app_factory, valid_payload
) -> None:
    redis = FakeRedis()
    provider = FakeProvider("openai")
    metrics = FakeMetricsService()
    application = app_factory(
        [provider],
        redis_client=redis,
        default_provider="openai",
        metrics_service=metrics,
    )

    first = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    )
    second = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    )

    assert first.json()["cache_hit"] is False
    assert second.json()["cache_hit"] is True
    assert provider.calls == 1
    assert len(metrics.records) == 2
    assert metrics.records[1]["cache_hit"] is True
    assert metrics.records[1]["estimated_cost_usd"] == 0


def test_api_primary_retry_then_fallback_response(
    app_factory, valid_payload
) -> None:
    primary = FakeProvider(
        "openai",
        effects=[ProviderTimeoutError("openai"), ProviderTimeoutError("openai")],
    )
    fallback = FakeProvider("gemini", effects=[provider_result("gemini")])
    metrics = FakeMetricsService()
    application = app_factory(
        [primary, fallback],
        default_provider="openai",
        max_retries=1,
        metrics_service=metrics,
    )

    response = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "gemini"
    assert response.json()["fallback_used"] is True
    assert (primary.calls, fallback.calls) == (2, 1)
    assert metrics.records[0]["provider"] == "gemini"
    assert metrics.records[0]["fallback_used"] is True


def test_api_metrics_write_failure_does_not_break_success(
    app_factory, valid_payload
) -> None:
    metrics = FakeMetricsService()
    metrics.fail_writes = True
    application = app_factory(
        [FakeProvider("openai")],
        default_provider="openai",
        metrics_service=metrics,
    )

    response = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    )

    assert response.status_code == 200
    assert metrics.records == []


def test_api_all_providers_fail_records_normalized_failure(
    app_factory, valid_payload
) -> None:
    metrics = FakeMetricsService()
    provider = FakeProvider(
        "openai", effects=[ProviderRequestError("openai")]
    )
    application = app_factory(
        [provider],
        default_provider="openai",
        max_retries=0,
        metrics_service=metrics,
    )

    response = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    )

    assert response.status_code == 503
    assert len(metrics.records) == 1
    assert metrics.records[0]["status"] == "error"
    assert metrics.records[0]["provider"] is None
    assert metrics.records[0]["error_category"] == "ProviderRequestError"


def test_api_rate_limiter_unavailable_stops_before_router(
    app_factory, valid_payload
) -> None:
    redis = FakeRedis()
    redis.fail_eval = True
    provider = FakeProvider("openai")
    application = app_factory(
        [provider], redis_client=redis, default_provider="openai"
    )

    response = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    )

    assert response.status_code == 503
    assert provider.calls == 0
