import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = "ModelRoute"
    app_env: str = "development"
    mock_model: str = "mock-model"


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("MODELROUTE_APP_NAME", "ModelRoute"),
        app_env=os.getenv("MODELROUTE_APP_ENV", "development"),
        mock_model=os.getenv("MODELROUTE_MOCK_MODEL", "mock-model"),
    )
