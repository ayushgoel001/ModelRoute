import asyncio

from app.config import Settings
from app.main import _build_providers
from app.providers.gemini_provider import GeminiProvider
from app.providers.openai_provider import OpenAIProvider


def test_gemini_introductory_pricing_defaults(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_INPUT_COST_PER_MILLION", raising=False)
    monkeypatch.delenv("GEMINI_OUTPUT_COST_PER_MILLION", raising=False)
    settings = Settings(_env_file=None)

    assert settings.gemini_input_cost_per_million == 0.75
    assert settings.gemini_output_cost_per_million == 3.75


def test_gemini_pricing_remains_environment_configurable(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_INPUT_COST_PER_MILLION", "1.25")
    monkeypatch.setenv("GEMINI_OUTPUT_COST_PER_MILLION", "6.25")

    settings = Settings(_env_file=None)

    assert settings.gemini_input_cost_per_million == 1.25
    assert settings.gemini_output_cost_per_million == 6.25


def test_database_url_has_safe_local_postgresql_default(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.database_url.startswith("postgresql+asyncpg://")


def test_mock_latency_defaults_to_zero_and_is_environment_configurable(
    monkeypatch,
) -> None:
    monkeypatch.delenv("MOCK_PROVIDER_LATENCY_MS", raising=False)
    assert Settings(_env_file=None).mock_provider_latency_ms == 0

    monkeypatch.setenv("MOCK_PROVIDER_LATENCY_MS", "50")
    assert Settings(_env_file=None).mock_provider_latency_ms == 50


def test_real_provider_registry_uses_normal_environment_configuration() -> None:
    settings = Settings(
        _env_file=None,
        openai_api_key="test-openai-key",
        openai_model="test-openai-model",
        gemini_api_key="test-gemini-key",
        gemini_model="test-gemini-model",
    )

    providers = _build_providers(settings)
    openai = next(provider for provider in providers if provider.metadata.name == "openai")
    gemini = next(provider for provider in providers if provider.metadata.name == "gemini")

    assert isinstance(openai, OpenAIProvider)
    assert isinstance(gemini, GeminiProvider)
    assert openai.metadata.model == "test-openai-model"
    assert gemini.metadata.model == "test-gemini-model"
    assert openai.metadata.available is True
    assert gemini.metadata.available is True

    async def close_providers() -> None:
        await asyncio.gather(*(provider.close() for provider in providers))

    asyncio.run(close_providers())
