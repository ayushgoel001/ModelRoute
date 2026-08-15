import asyncio

from app.providers.base import GenerationRequest, ProviderResult
from app.providers.mock_provider import MockProvider


REQUEST = GenerationRequest(
    prompt="A predictable prompt",
    temperature=0.2,
    max_tokens=100,
)


def generate(provider: MockProvider) -> ProviderResult:
    return asyncio.run(provider.generate(REQUEST))


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


def test_mock_provider_has_routing_metadata() -> None:
    metadata = MockProvider().metadata

    assert metadata.available is True
    assert metadata.identity == "mock:mock-model"
