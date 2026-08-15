from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.6-luna"
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.7-flash"

    default_provider: Literal["openai", "gemini", "mock"] = "openai"
    provider_timeout_seconds: float = Field(default=20.0, gt=0)
    provider_max_retries: int = Field(default=1, ge=0, le=5)
    provider_retry_delay_seconds: float = Field(default=0.25, ge=0)

    redis_url: str = "redis://localhost:6379/0"
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
