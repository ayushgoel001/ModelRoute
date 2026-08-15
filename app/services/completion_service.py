import asyncio
import logging
from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.exceptions import AllProvidersFailedError, ProviderError
from app.providers.base import GenerationRequest, ProviderResult
from app.services.cache import ResponseCache
from app.services.latency import LatencyTracker
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
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.router = router
        self.cache = cache
        self.latency_tracker = latency_tracker
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.sleep = sleep

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        started_at = perf_counter()
        generation_request = GenerationRequest(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        candidates = self.router.route(request.strategy.value, generation_request)

        for candidate_index, provider in enumerate(candidates):
            cache_key = self.cache.build_key(provider, generation_request)
            cached = await self.cache.get(cache_key)
            if cached is not None:
                return self._response(
                    cached,
                    started_at=started_at,
                    cache_hit=True,
                    fallback_used=candidate_index > 0,
                )

            for attempt in range(self.max_retries + 1):
                attempt_started_at = perf_counter()
                try:
                    result = await provider.generate(generation_request)
                except ProviderError as exc:
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
                return self._response(
                    result,
                    started_at=started_at,
                    cache_hit=False,
                    fallback_used=candidate_index > 0,
                )

        raise AllProvidersFailedError("All eligible providers failed")

    @staticmethod
    def _response(
        result: ProviderResult,
        *,
        started_at: float,
        cache_hit: bool,
        fallback_used: bool,
    ) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            request_id=uuid4(),
            provider=result.provider,
            model=result.model,
            content=result.content,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=(perf_counter() - started_at) * 1_000,
            cache_hit=cache_hit,
            fallback_used=fallback_used,
        )
