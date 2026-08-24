import asyncio
import logging
from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import UUID, uuid4

from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.exceptions import (
    AllProvidersFailedError,
    NoEligibleProviderError,
    ProviderError,
    PublicGeminiDemoError,
    PublicGeminiDemoRequestError,
    PublicGeminiDemoUnavailableError,
    PublicGeminiQuotaUnavailableError,
)
from app.providers.base import (
    BaseProvider,
    GenerationRequest,
    ProviderMetadata,
    ProviderResult,
)
from app.services.cache import ResponseCache
from app.services.gemini_demo_quota import PublicGeminiDemoQuota
from app.services.latency import LatencyTracker
from app.services.metrics import MetricsService
from app.services.router import ProviderRouter

logger = logging.getLogger(__name__)


class CompletionService:
    def __init__(
        self,
        *,
        router: ProviderRouter,
        cache: ResponseCache,
        latency_tracker: LatencyTracker,
        max_retries: int,
        retry_delay_seconds: float,
        metrics: MetricsService | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        public_gemini_demo_enabled: bool = False,
        public_gemini_demo_model: str = "gemini-3.7-flash",
        public_gemini_max_output_tokens: int = 256,
        public_gemini_max_prompt_chars: int = 2_000,
        public_gemini_quota: PublicGeminiDemoQuota | None = None,
    ) -> None:
        self.router = router
        self.cache = cache
        self.latency_tracker = latency_tracker
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.metrics = metrics
        self.sleep = sleep
        self.public_gemini_demo_enabled = public_gemini_demo_enabled
        self.public_gemini_demo_model = public_gemini_demo_model
        self.public_gemini_max_output_tokens = public_gemini_max_output_tokens
        self.public_gemini_max_prompt_chars = public_gemini_max_prompt_chars
        self.public_gemini_quota = public_gemini_quota

    async def complete(
        self,
        request: ChatCompletionRequest,
        *,
        started_at: float | None = None,
        client_identity: str | None = None,
    ) -> ChatCompletionResponse:
        gateway_started_at = started_at if started_at is not None else perf_counter()
        request_id = uuid4()
        generation_request = GenerationRequest(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        try:
            preferred_provider = (
                request.preferred_provider.value
                if request.preferred_provider is not None
                else None
            )
            candidates = self.router.route(
                request.strategy.value,
                generation_request,
                preferred_provider=preferred_provider,
            )
        except NoEligibleProviderError:
            await self._record_failure(
                request_id=request_id,
                request=request,
                started_at=gateway_started_at,
                error_category="NoEligibleProviderError",
            )
            raise

        last_error_category = "AllProvidersFailedError"

        for candidate_index, provider in enumerate(candidates):
            try:
                provider_request = self._provider_request(provider, generation_request)
            except PublicGeminiDemoError as exc:
                await self._record_failure(
                    request_id=request_id,
                    request=request,
                    started_at=gateway_started_at,
                    error_category=type(exc).__name__,
                )
                raise

            cache_key = self.cache.build_key(provider, provider_request)
            cached = await self.cache.get(cache_key)
            if cached is not None:
                response = self._response(
                    cached,
                    request_id=request_id,
                    started_at=gateway_started_at,
                    cache_hit=True,
                    fallback_used=candidate_index > 0,
                )
                await self._record_success(response, request, provider.metadata)
                return response

            if self._is_public_gemini(provider):
                if self.public_gemini_quota is None:
                    exc = PublicGeminiQuotaUnavailableError(
                        "Live Gemini demo quota is unavailable"
                    )
                    await self._record_failure(
                        request_id=request_id,
                        request=request,
                        started_at=gateway_started_at,
                        error_category=type(exc).__name__,
                    )
                    raise exc
                try:
                    await self.public_gemini_quota.consume(
                        client_identity or "unidentified"
                    )
                except PublicGeminiDemoError as exc:
                    await self._record_failure(
                        request_id=request_id,
                        request=request,
                        started_at=gateway_started_at,
                        error_category=type(exc).__name__,
                    )
                    raise

            for attempt in range(self.max_retries + 1):
                attempt_started_at = perf_counter()
                try:
                    result = await provider.generate(provider_request)
                except ProviderError as exc:
                    last_error_category = type(exc).__name__
                    logger.warning(
                        "Provider %s failed; retryable=%s attempt=%s",
                        provider.metadata.name,
                        exc.retryable,
                        attempt + 1,
                    )
                    if exc.retryable and attempt < self.max_retries:
                        if self.retry_delay_seconds:
                            await self.sleep(self.retry_delay_seconds)
                        continue
                    break

                provider_latency_ms = (perf_counter() - attempt_started_at) * 1_000
                self.latency_tracker.record(
                    provider.metadata.identity, provider_latency_ms
                )
                await self.cache.set(cache_key, result)
                response = self._response(
                    result,
                    request_id=request_id,
                    started_at=gateway_started_at,
                    cache_hit=False,
                    fallback_used=candidate_index > 0,
                )
                await self._record_success(response, request, provider.metadata)
                return response

        await self._record_failure(
            request_id=request_id,
            request=request,
            started_at=gateway_started_at,
            error_category=last_error_category,
        )
        raise AllProvidersFailedError("All eligible providers failed")

    def _is_public_gemini(self, provider: BaseProvider) -> bool:
        return (
            self.public_gemini_demo_enabled
            and provider.metadata.name == "gemini"
        )

    def _provider_request(
        self,
        provider: BaseProvider,
        request: GenerationRequest,
    ) -> GenerationRequest:
        if not self._is_public_gemini(provider):
            return request
        if provider.metadata.model != self.public_gemini_demo_model:
            raise PublicGeminiDemoUnavailableError(
                "Live Gemini demo model is not configured"
            )
        if len(request.prompt) > self.public_gemini_max_prompt_chars:
            raise PublicGeminiDemoRequestError(
                "Live Gemini demo prompt exceeds the public limit"
            )
        return GenerationRequest(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=min(
                request.max_tokens,
                self.public_gemini_max_output_tokens,
            ),
        )

    async def _record_success(
        self,
        response: ChatCompletionResponse,
        request: ChatCompletionRequest,
        provider_metadata: ProviderMetadata,
    ) -> None:
        if self.metrics is not None:
            await self.metrics.record_success(
                response,
                routing_strategy=request.strategy.value,
                provider=provider_metadata,
            )

    async def _record_failure(
        self,
        *,
        request_id: UUID,
        request: ChatCompletionRequest,
        started_at: float,
        error_category: str,
    ) -> None:
        if self.metrics is not None:
            await self.metrics.record_failure(
                request_id=request_id,
                routing_strategy=request.strategy.value,
                latency_ms=(perf_counter() - started_at) * 1_000,
                error_category=error_category,
            )

    @staticmethod
    def _response(
        result: ProviderResult,
        *,
        request_id: UUID,
        started_at: float,
        cache_hit: bool,
        fallback_used: bool,
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            request_id=request_id,
            provider=result.provider,
            model=result.model,
            content=result.content,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=(perf_counter() - started_at) * 1_000,
            cache_hit=cache_hit,
            fallback_used=fallback_used,
        )
