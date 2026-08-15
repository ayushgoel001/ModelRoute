from app.exceptions import ProviderTimeoutError
from tests.conftest import FakeProvider, FakeRedis, provider_result, request


def test_api_rate_limit_router_cache_miss_provider_response(
    app_factory, valid_payload
) -> None:
    redis = FakeRedis()
    provider = FakeProvider("openai")
    application = app_factory(
        [provider], redis_client=redis, default_provider="openai"
    )

    response = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    )

    assert response.status_code == 200
    assert response.json()["cache_hit"] is False
    assert provider.calls == 1
    assert len(redis.eval_keys) == 1
    assert len(redis.values) == 1


def test_api_rate_limit_router_cache_hit_skips_provider(
    app_factory, valid_payload
) -> None:
    redis = FakeRedis()
    provider = FakeProvider("openai")
    application = app_factory(
        [provider], redis_client=redis, default_provider="openai"
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


def test_api_primary_retry_then_fallback_response(
    app_factory, valid_payload
) -> None:
    primary = FakeProvider(
        "openai",
        effects=[ProviderTimeoutError("openai"), ProviderTimeoutError("openai")],
    )
    fallback = FakeProvider("gemini", effects=[provider_result("gemini")])
    application = app_factory(
        [primary, fallback], default_provider="openai", max_retries=1
    )

    response = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "gemini"
    assert response.json()["fallback_used"] is True
    assert (primary.calls, fallback.calls) == (2, 1)


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
