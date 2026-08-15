import asyncio
from types import SimpleNamespace

from app.providers.base import GenerationRequest, ProviderResult
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider
from app.services.cache import ResponseCache
from tests.conftest import FakeProvider, FakeRedis


def test_cache_miss_then_hit_with_configured_ttl() -> None:
    redis = FakeRedis()
    cache = ResponseCache(redis, ttl_seconds=321)
    result = ProviderResult("answer", "openai", "model-a", 2, 3)

    assert asyncio.run(cache.get("key")) is None
    asyncio.run(cache.set("key", result))

    assert asyncio.run(cache.get("key")) == result
    assert redis.expirations["key"] == 321


def test_cache_key_never_contains_raw_prompt() -> None:
    request = GenerationRequest("private prompt text", 0.2, 20)
    key = ResponseCache.build_key(FakeProvider("openai"), request)

    assert key.startswith("modelroute:cache:")
    assert "private prompt text" not in key
    assert len(key.removeprefix("modelroute:cache:")) == 64


def test_response_affecting_fields_change_cache_key() -> None:
    provider = FakeProvider("openai", model="model-a")
    baseline = GenerationRequest("prompt-a", 0.2, 20)

    assert ResponseCache.build_key(provider, baseline) != ResponseCache.build_key(
        provider, GenerationRequest("prompt-b", 0.2, 20)
    )
    assert ResponseCache.build_key(provider, baseline) != ResponseCache.build_key(
        provider, GenerationRequest("prompt-a", 0.2, 21)
    )
    assert ResponseCache.build_key(provider, baseline) != ResponseCache.build_key(
        FakeProvider("gemini", model="model-a"), baseline
    )
    assert ResponseCache.build_key(provider, baseline) != ResponseCache.build_key(
        FakeProvider("openai", model="model-b"), baseline
    )


def test_effective_temperature_changes_key_only_when_provider_uses_it() -> None:
    request_a = GenerationRequest("same", 0.1, 20)
    request_b = GenerationRequest("same", 1.9, 20)
    sampling_provider = FakeProvider("sampling", include_temperature=True)

    assert ResponseCache.build_key(
        sampling_provider, request_a
    ) != ResponseCache.build_key(sampling_provider, request_b)


def test_ignored_gemini_temperature_does_not_change_cache_key() -> None:
    gemini = GeminiProvider(
        api_key=None,
        model="gemini-3.7-flash",
        timeout_seconds=1,
        input_cost_per_million=0.75,
        output_cost_per_million=3.75,
        client=SimpleNamespace(),
        available=True,
    )
    cold = GenerationRequest("same", 0.0, 20)
    hot = GenerationRequest("same", 2.0, 20)

    assert ResponseCache.build_key(gemini, cold) == ResponseCache.build_key(gemini, hot)


def test_reasoning_and_thinking_defaults_are_part_of_cache_identity() -> None:
    request = GenerationRequest("same", 0.2, 20)
    openai = OpenAIProvider(
        api_key=None,
        model="gpt-5.6-luna",
        timeout_seconds=1,
        input_cost_per_million=0.2,
        output_cost_per_million=1.2,
        client=SimpleNamespace(),
        available=True,
    )
    gemini = GeminiProvider(
        api_key=None,
        model="gemini-3.7-flash",
        timeout_seconds=1,
        input_cost_per_million=0.75,
        output_cost_per_million=3.75,
        client=SimpleNamespace(),
        available=True,
    )
    max_tokens_only_openai = FakeProvider("openai", model="gpt-5.6-luna")
    max_tokens_only_gemini = FakeProvider("gemini", model="gemini-3.7-flash")

    assert ResponseCache.build_key(openai, request) != ResponseCache.build_key(
        max_tokens_only_openai, request
    )
    assert ResponseCache.build_key(gemini, request) != ResponseCache.build_key(
        max_tokens_only_gemini, request
    )
