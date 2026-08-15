from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.providers.mock_provider import MockProvider
from app.services.completion_service import CompletionService


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    application = FastAPI(title=resolved_settings.app_name)
    application.state.completion_service = CompletionService(
        provider=MockProvider(model=resolved_settings.mock_model)
    )
    application.include_router(router)
    return application


app = create_app()
