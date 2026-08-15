import asyncio
from types import SimpleNamespace

import httpx2
import openai
import pytest

from app.exceptions import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.base import GenerationRequest
from app.providers.openai_provider import OpenAIProvider

REQUEST = GenerationRequest(prompt="hello", temperature=1.7, max_tokens=12)


class FakeResponses:
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
    responses = FakeResponses(effect)
    client = SimpleNamespace(responses=responses)
    adapter = OpenAIProvider(
        api_key=None,
        model="gpt-5.6-luna",
        timeout_seconds=timeout,
        input_cost_per_million=0.2,
        output_cost_per_million=1.2,
        client=client,
        available=True,
    )
    return adapter, responses


def status_error(error_type, status_code: int):
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx2.Response(status_code, request=request)
    return error_type("safe test error", response=response, body={})


def test_openai_success_normalizes_response_and_usage() -> None:
    response = SimpleNamespace(
        output_text="normalized",
        usage=SimpleNamespace(input_tokens=4, output_tokens=7),
    )
    adapter, calls = provider(response)

    result = asyncio.run(adapter.generate(REQUEST))

    assert (result.provider, result.model) == ("openai", "gpt-5.6-luna")
    assert (result.content, result.input_tokens, result.output_tokens) == (
        "normalized",
        4,
        7,
    )
    assert calls.kwargs == {
        "model": "gpt-5.6-luna",
        "input": "hello",
        "max_output_tokens": 12,
        "reasoning": {"effort": "low"},
    }
    assert "temperature" not in calls.kwargs


def test_openai_effective_parameters_match_low_reasoning_request() -> None:
    adapter, _ = provider(SimpleNamespace(output_text="", usage=None))

    assert adapter.effective_parameters(REQUEST) == {
        "max_output_tokens": 12,
        "reasoning": {"effort": "low"},
    }


def test_openai_timeout_is_normalized_and_bounded() -> None:
    async def slow_response():
        await asyncio.sleep(0.05)

    adapter, _ = provider(slow_response, timeout=0.001)

    with pytest.raises(ProviderTimeoutError):
        asyncio.run(adapter.generate(REQUEST))


@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (status_error(openai.RateLimitError, 429), ProviderRateLimitError),
        (status_error(openai.InternalServerError, 500), ProviderUnavailableError),
        (status_error(openai.AuthenticationError, 401), ProviderAuthenticationError),
        (status_error(openai.BadRequestError, 400), ProviderRequestError),
    ],
)
def test_openai_sdk_errors_are_normalized(sdk_error, expected) -> None:
    adapter, _ = provider(sdk_error)

    with pytest.raises(expected):
        asyncio.run(adapter.generate(REQUEST))


def test_openai_without_key_is_unavailable_but_constructible() -> None:
    adapter = OpenAIProvider(
        api_key=None,
        model="gpt-5.6-luna",
        timeout_seconds=1,
        input_cost_per_million=0.2,
        output_cost_per_million=1.2,
    )

    assert adapter.metadata.available is False
    with pytest.raises(ProviderUnavailableError):
        asyncio.run(adapter.generate(REQUEST))
