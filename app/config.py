from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PUBLIC_GEMINI_DEMO_MODEL = "gemini-3.7-flash"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    modelroute_app_name: str = "ModelRoute"
    modelroute_app_env: str = "development"
    modelroute_mock_model: str = "mock-model"
    mock_provider_latency_ms: float = Field(default=0.0, ge=0)

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-luna"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = PUBLIC_GEMINI_DEMO_MODEL

    public_gemini_demo_enabled: bool = False
    public_gemini_demo_max_output_tokens: int = Field(default=256, gt=0, le=256)
    public_gemini_demo_max_prompt_chars: int = Field(default=2_000, gt=0, le=10_000)
    public_gemini_demo_client_limit: int = Field(default=2, gt=0)
    public_gemini_demo_client_window_seconds: int = Field(default=3_600, gt=0)
    public_gemini_demo_global_limit: int = Field(default=15, gt=0)
    public_gemini_demo_global_window_seconds: int = Field(default=86_400, gt=0)

    default_provider: Literal["openai", "gemini", "mock"] = "openai"
    provider_timeout_seconds: float = Field(default=20.0, gt=0)
    provider_max_retries: int = Field(default=1, ge=0, le=5)
    provider_retry_delay_seconds: float = Field(default=0.25, ge=0)

    redis_url: str = "redis://localhost:6379/0"
    database_url: str = (
        "postgresql+asyncpg://modelroute:modelroute@localhost:5432/modelroute"
    )
    cache_ttl_seconds: int = Field(default=300, gt=0)
    rate_limit_capacity: int = Field(default=20, gt=0)
    rate_limit_refill_rate: float = Field(default=1.0, gt=0)

    openai_input_cost_per_million: float = Field(default=0.20, ge=0)
    openai_output_cost_per_million: float = Field(default=1.20, ge=0)
    gemini_input_cost_per_million: float = Field(default=0.75, ge=0)
    gemini_output_cost_per_million: float = Field(default=3.75, ge=0)

    initial_provider_latency_ms: float = Field(default=1_000.0, gt=0)
    latency_ewma_alpha: float = Field(default=0.3, gt=0, le=1)

    @property
    def app_name(self) -> str:
        return self.modelroute_app_name

    @property
    def app_env(self) -> str:
        return self.modelroute_app_env

    @property
    def mock_model(self) -> str:
        return self.modelroute_mock_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
