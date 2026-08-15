from app.config import Settings


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
