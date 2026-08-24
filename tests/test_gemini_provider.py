import asyncio
from types import SimpleNamespace

import httpx
import pytest
from google.genai._gaos.lib import compat_errors as gemini_errors

from app.exceptions import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.base import GenerationRequest
from app.providers.gemini_provider import GeminiProvider

REQUEST = GenerationRequest(prompt="hello", temperature=1.7, max_tokens=12)


class FakeInteractions:
    def __init__(self, effect) -> None:
        self.effect = effect
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.effect, Exception):
            raise self.effect
        if callable(self.effect):
            return await self.effect()
        return self.effect


def provider(effect, *, timeout: float = 1.0):
    interactions = FakeInteractions(effect)
    client = SimpleNamespace(
        aio=SimpleNamespace(interactions=interactions),
    )
    adapter = GeminiProvider(
        api_key=None,
        model="gemini-3.7-flash",
        timeout_seconds=timeout,
        input_cost_per_million=0.75,
        output_cost_per_million=3.75,
        client=client,
        available=True,
    )
    return adapter, interactions


def status_error(error_type, status_code: int):
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com")
    response = httpx.Response(status_code, request=request)
    return error_type("safe test error", response=response, body={})


def rate_limit_error(body: object):
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com")
    response = httpx.Response(429, request=request)
    return gemini_errors.RateLimitError(
        "raw provider text must remain private",
        response=response,
        body=body,
    )


def test_gemini_success_normalizes_interaction_and_usage() -> None:
    response = SimpleNamespace(
        output_text="normalized",
        usage=SimpleNamespace(total_input_tokens=4, total_output_tokens=7),
    )
    adapter, calls = provider(response)

    result = asyncio.run(adapter.generate(REQUEST))

    assert (result.provider, result.model) == ("gemini", "gemini-3.7-flash")
    assert (result.content, result.input_tokens, result.output_tokens) == (
        "normalized",
        4,
        7,
    )
    assert calls.kwargs["model"] == "gemini-3.7-flash"
    assert calls.kwargs["input"] == "hello"
    assert calls.kwargs["generation_config"] == {
        "max_output_tokens": 12,
        "thinking_level": "low",
    }
    assert "temperature" not in calls.kwargs["generation_config"]
    assert "top_p" not in calls.kwargs["generation_config"]
    assert "top_k" not in calls.kwargs["generation_config"]


def test_gemini_effective_parameters_match_low_thinking_request() -> None:
    adapter, _ = provider(SimpleNamespace(output_text="", usage=None))

    assert adapter.effective_parameters(REQUEST) == {
        "max_output_tokens": 12,
        "thinking_level": "low",
    }


def test_gemini_timeout_is_normalized_and_bounded() -> None:
    async def slow_response():
        await asyncio.sleep(0.05)

    adapter, _ = provider(slow_response, timeout=0.001)

    with pytest.raises(ProviderTimeoutError) as raised:
        asyncio.run(adapter.generate(REQUEST))

    assert raised.value.timeout_source == "outer_asyncio"
    assert raised.value.retryable is True


def test_gemini_sdk_http_timeout_source_is_normalized() -> None:
    sdk_timeout = gemini_errors.APITimeoutError(
        request=httpx.Request(
            "POST", "https://generativelanguage.googleapis.com"
        )
    )
    adapter, _ = provider(sdk_timeout)

    with pytest.raises(ProviderTimeoutError) as raised:
        asyncio.run(adapter.generate(REQUEST))

    assert raised.value.timeout_source == "sdk_http"
    assert raised.value.retryable is True


@pytest.mark.parametrize(
    ("body", "expected_source"),
    [
        (
            {
                "error": {
                    "code": "quota_exceeded",
                    "message": "raw quota text",
                }
            },
            "daily_quota",
        ),
        (
            {
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "raw rate-limit text",
                }
            },
            "short_term",
        ),
        (
            {
                "error": {
                    "code": "private_unrecognized_provider_code",
                    "message": "raw unknown text",
                }
            },
            "unknown_429",
        ),
        ({"error": {"message": "missing code"}}, "unknown_429"),
        ({"error": "malformed error body"}, "unknown_429"),
        (None, "unknown_429"),
    ],
)
def test_gemini_rate_limit_source_is_allowlisted(
    body: object,
    expected_source: str,
) -> None:
    adapter, _ = provider(rate_limit_error(body))

    with pytest.raises(ProviderRateLimitError) as raised:
        asyncio.run(adapter.generate(REQUEST))

    assert raised.value.rate_limit_source == expected_source
    assert raised.value.retryable is True


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (status_error(gemini_errors.RateLimitError, 429), ProviderRateLimitError),
        (
            status_error(gemini_errors.InternalServerError, 500),
            ProviderUnavailableError,
        ),
        (
            status_error(gemini_errors.AuthenticationError, 401),
            ProviderAuthenticationError,
        ),
        (status_error(gemini_errors.BadRequestError, 400), ProviderRequestError),
    ],
)
def test_gemini_sdk_errors_are_normalized(sdk_error, expected) -> None:
    adapter, _ = provider(sdk_error)

    with pytest.raises(expected):
        asyncio.run(adapter.generate(REQUEST))


def test_gemini_without_key_is_unavailable_but_constructible() -> None:
    adapter = GeminiProvider(
        api_key=None,
        model="gemini-3.7-flash",
        timeout_seconds=1,
        input_cost_per_million=0.75,
        output_cost_per_million=3.75,
    )

    assert adapter.metadata.available is False
    with pytest.raises(ProviderUnavailableError):
        asyncio.run(adapter.generate(REQUEST))


def test_gemini_temperature_does_not_change_effective_parameters() -> None:
    adapter, _ = provider(SimpleNamespace(output_text="", usage=None))
    colder = GenerationRequest(prompt="same", temperature=0.0, max_tokens=20)
    hotter = GenerationRequest(prompt="same", temperature=2.0, max_tokens=20)

    assert adapter.effective_parameters(colder) == adapter.effective_parameters(hotter)
