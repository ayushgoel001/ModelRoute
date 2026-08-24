import pytest
from pydantic import ValidationError

from app.config import PUBLIC_GEMINI_DEMO_MODEL, Settings


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


def test_public_gemini_demo_has_safe_disabled_defaults(monkeypatch) -> None:
    monkeypatch.delenv("PUBLIC_GEMINI_DEMO_ENABLED", raising=False)
    settings = Settings(_env_file=None)

    assert settings.public_gemini_demo_enabled is False
    assert settings.gemini_model == PUBLIC_GEMINI_DEMO_MODEL
    assert settings.public_gemini_demo_max_output_tokens == 256
    assert settings.public_gemini_demo_max_prompt_chars == 2_000
    assert settings.public_gemini_demo_client_limit == 2
    assert settings.public_gemini_demo_global_limit == 15


def test_public_gemini_output_cap_cannot_be_configured_above_256() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, public_gemini_demo_max_output_tokens=257)
