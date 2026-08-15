import asyncio

from app.providers.base import ProviderResult
from app.providers.mock_provider import MockProvider


def generate(provider: MockProvider) -> ProviderResult:
    return asyncio.run(
        provider.generate(
            prompt="A predictable prompt",
            temperature=0.2,
            max_tokens=100,
        )
    )


def test_mock_provider_generates_normalized_result() -> None:
    result = generate(MockProvider())

    assert result.content == "Mock response: A predictable prompt"
    assert result.provider == "mock"
    assert result.model == "mock-model"
    assert result.input_tokens > 0
    assert result.output_tokens > 0


def test_mock_provider_is_deterministic_for_same_input() -> None:
    provider = MockProvider()

    assert generate(provider) == generate(provider)
