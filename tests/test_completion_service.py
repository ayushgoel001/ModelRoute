import asyncio

import pytest

from app.api.schemas import ChatCompletionRequest
from app.exceptions import (
    AllProvidersFailedError,
    ProviderAuthenticationError,
    ProviderRequestError,
    ProviderTimeoutError,
)
from app.providers.base import GenerationRequest
from app.services.cache import ResponseCache
from app.services.completion_service import CompletionService
from app.services.latency import LatencyTracker
from app.services.router import ProviderRouter
from tests.conftest import FakeProvider, FakeRedis, provider_result

REQUEST = ChatCompletionRequest(prompt="hello", strategy="fixed", max_tokens=20)


def service(
    providers,
    redis,
    *,
    max_retries=1,
    sleep=asyncio.sleep,
):
    tracker = LatencyTracker(initial_latency_ms=500)
    provider_router = ProviderRouter(
        providers,
        default_provider=providers[0].metadata.name,
        latency_tracker=tracker,
        allow_mock=True,
    )
    return CompletionService(
        router=provider_router,
        cache=ResponseCache(redis, ttl_seconds=300),
        latency_tracker=tracker,
        max_retries=max_retries,
        retry_delay_seconds=0.01,
        sleep=sleep,
    )


def complete(completion_service):
    return asyncio.run(completion_service.complete(REQUEST))


def test_transient_failure_then_successful_retry() -> None:
    provider = FakeProvider(
        "openai",
        effects=[ProviderTimeoutError("openai"), provider_result("openai")],
    )

    response = complete(service([provider], FakeRedis(), sleep=lambda _: _done()))

    assert response.provider == "openai"
    assert provider.calls == 2
    assert response.fallback_used is False


async def _done() -> None:
    return None


def test_persistent_transient_failure_has_bounded_attempts() -> None:
    provider = FakeProvider(
        "openai",
        effects=[ProviderTimeoutError("openai"), ProviderTimeoutError("openai")],
    )

    with pytest.raises(AllProvidersFailedError):
        complete(service([provider], FakeRedis(), sleep=lambda _: _done()))
    assert provider.calls == 2


@pytest.mark.parametrize(
    "error",
    [ProviderAuthenticationError("openai"), ProviderRequestError("openai")],
)
def test_non_retryable_provider_errors_are_not_retried(error) -> None:
    provider = FakeProvider("openai", effects=[error])

    with pytest.raises(AllProvidersFailedError):
        complete(service([provider], FakeRedis()))
    assert provider.calls == 1


def test_primary_success_does_not_call_secondary() -> None:
    primary = FakeProvider("openai")
    secondary = FakeProvider("gemini")

    response = complete(service([primary, secondary], FakeRedis()))

    assert (primary.calls, secondary.calls) == (1, 0)
    assert response.fallback_used is False


def test_primary_failure_then_secondary_success_uses_fallback() -> None:
    primary = FakeProvider("openai", effects=[ProviderRequestError("openai")])
    secondary = FakeProvider("gemini")

    response = complete(service([primary, secondary], FakeRedis()))

    assert (primary.calls, secondary.calls) == (1, 1)
    assert response.provider == "gemini"
    assert response.fallback_used is True


def test_primary_retry_success_avoids_fallback() -> None:
    primary = FakeProvider(
        "openai",
        effects=[ProviderTimeoutError("openai"), provider_result("openai")],
    )
    secondary = FakeProvider("gemini")

    response = complete(
        service([primary, secondary], FakeRedis(), sleep=lambda _: _done())
    )

    assert (primary.calls, secondary.calls) == (2, 0)
    assert response.fallback_used is False


def test_all_candidates_fail_cleanly() -> None:
    primary = FakeProvider("openai", effects=[ProviderRequestError("openai")])
    secondary = FakeProvider("gemini", effects=[ProviderRequestError("gemini")])

    with pytest.raises(AllProvidersFailedError):
        complete(service([primary, secondary], FakeRedis()))


def test_fallback_candidate_uses_its_own_cached_response() -> None:
    redis = FakeRedis()
    primary = FakeProvider("openai", effects=[ProviderRequestError("openai")])
    secondary = FakeProvider("gemini")
    completion_service = service([primary, secondary], redis)
    generation_request = GenerationRequest("hello", 0.2, 20)
    secondary_key = ResponseCache.build_key(secondary, generation_request)
    asyncio.run(
        completion_service.cache.set(secondary_key, provider_result("gemini"))
    )

    response = complete(completion_service)

    assert (primary.calls, secondary.calls) == (1, 0)
    assert response.provider == "gemini"
    assert response.cache_hit is True
    assert response.fallback_used is True


def test_primary_and_fallback_cache_keys_never_cross_contaminate() -> None:
    request = GenerationRequest("same", 0.2, 20)
    primary = FakeProvider("openai")
    secondary = FakeProvider("gemini")

    assert ResponseCache.build_key(primary, request) != ResponseCache.build_key(
        secondary, request
    )


def test_cache_hit_avoids_provider_and_uses_fresh_request_id() -> None:
    redis = FakeRedis()
    provider = FakeProvider("openai")
    completion_service = service([provider], redis)

    first = complete(completion_service)
    second = complete(completion_service)

    assert provider.calls == 1
    assert first.cache_hit is False
    assert second.cache_hit is True
    assert first.request_id != second.request_id
    assert second.latency_ms >= 0


def test_cache_get_failure_fails_open() -> None:
    redis = FakeRedis()
    redis.fail_get = True
    provider = FakeProvider("openai")

    response = complete(service([provider], redis))

    assert response.provider == "openai"
    assert provider.calls == 1


def test_cache_set_failure_fails_open() -> None:
    redis = FakeRedis()
    redis.fail_set = True
    provider = FakeProvider("openai")

    response = complete(service([provider], redis))

    assert response.provider == "openai"
    assert response.cache_hit is False


def test_provider_error_is_not_cached() -> None:
    redis = FakeRedis()
    provider = FakeProvider("openai", effects=[ProviderRequestError("openai")])

    with pytest.raises(AllProvidersFailedError):
        complete(service([provider], redis))
    assert redis.values == {}


def test_retry_delay_is_injected_without_real_sleep() -> None:
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    provider = FakeProvider(
        "openai",
        effects=[ProviderTimeoutError("openai"), provider_result("openai")],
    )

    complete(service([provider], FakeRedis(), sleep=record_delay))

    assert delays == [0.01]
