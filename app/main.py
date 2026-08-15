from collections.abc import Iterable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from redis import asyncio as redis_async

from app.api.routes import router
from app.config import Settings, get_settings
from app.providers.base import BaseProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.mock_provider import MockProvider
from app.providers.openai_provider import OpenAIProvider
from app.services.cache import ResponseCache
from app.services.completion_service import CompletionService
from app.services.latency import LatencyTracker
from app.services.rate_limiter import TokenBucketRateLimiter
from app.services.router import ProviderRouter


def _secret_value(secret: Any | None) -> str | None:
    return secret.get_secret_value() if secret is not None else None


def _build_providers(settings: Settings) -> list[BaseProvider]:
    return [
        OpenAIProvider(
            api_key=_secret_value(settings.openai_api_key),
            model=settings.openai_model,
            timeout_seconds=settings.provider_timeout_seconds,
            input_cost_per_million=settings.openai_input_cost_per_million,
            output_cost_per_million=settings.openai_output_cost_per_million,
        ),
        GeminiProvider(
            api_key=_secret_value(settings.gemini_api_key),
            model=settings.gemini_model,
            timeout_seconds=settings.provider_timeout_seconds,
            input_cost_per_million=settings.gemini_input_cost_per_million,
            output_cost_per_million=settings.gemini_output_cost_per_million,
        ),
        MockProvider(model=settings.mock_model),
    ]


def create_app(
    settings: Settings | None = None,
    *,
    redis_client: Any | None = None,
    providers: Iterable[BaseProvider] | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    owns_redis_client = redis_client is None
    shared_redis = redis_client or redis_async.from_url(
        resolved_settings.redis_url,
        decode_responses=True,
    )
    configured_providers = list(providers or _build_providers(resolved_settings))
    latency_tracker = LatencyTracker(
        alpha=resolved_settings.latency_ewma_alpha,
        initial_latency_ms=resolved_settings.initial_provider_latency_ms,
    )
    provider_router = ProviderRouter(
        configured_providers,
        default_provider=resolved_settings.default_provider,
        latency_tracker=latency_tracker,
        allow_mock=resolved_settings.default_provider == "mock",
    )
    response_cache = ResponseCache(
        shared_redis,
        ttl_seconds=resolved_settings.cache_ttl_seconds,
    )
    rate_limiter = TokenBucketRateLimiter(
        shared_redis,
        capacity=resolved_settings.rate_limit_capacity,
        refill_rate=resolved_settings.rate_limit_refill_rate,
    )
    completion_service = CompletionService(
        router=provider_router,
        cache=response_cache,
        latency_tracker=latency_tracker,
        max_retries=resolved_settings.provider_max_retries,
        retry_delay_seconds=resolved_settings.provider_retry_delay_seconds,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        if owns_redis_client:
            await shared_redis.aclose()
        for provider in configured_providers:
            await provider.close()

    application = FastAPI(title=resolved_settings.app_name, lifespan=lifespan)
    application.state.settings = resolved_settings
    application.state.redis_client = shared_redis
    application.state.providers = configured_providers
    application.state.latency_tracker = latency_tracker
    application.state.completion_service = completion_service
    application.state.rate_limiter = rate_limiter
    application.include_router(router)
    return application


app = create_app()
