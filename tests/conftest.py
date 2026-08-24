import asyncio
from collections.abc import Iterable
from typing import Any

import httpx
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from app.config import Settings
from app.main import create_app
from app.providers.base import (
    BaseProvider,
    GenerationRequest,
    ProviderMetadata,
    ProviderResult,
)
from app.services.metrics import MetricsService


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}
        self.buckets: dict[str, tuple[float, float]] = {}
        self.quota_counts: dict[str, int] = {}
        self.quota_eval_calls: list[tuple[str, str]] = []
        self.quota_lock = asyncio.Lock()
        self.now = 1_000.0
        self.fail_get = False
        self.fail_set = False
        self.fail_eval = False
        self.eval_keys: list[str] = []

    async def get(self, key: str) -> str | None:
        if self.fail_get:
            raise RedisConnectionError("cache unavailable")
        return self.values.get(key)

    async def set(self, key: str, value: str, *, ex: int) -> None:
        if self.fail_set:
            raise RedisConnectionError("cache unavailable")
        self.values[key] = value
        self.expirations[key] = ex

    async def eval(
        self,
        script: str,
        number_of_keys: int,
        *args: object,
    ) -> list[object]:
        del script
        if self.fail_eval:
            raise RedisConnectionError("redis unavailable")
        if number_of_keys == 2:
            (
                client_key,
                global_key,
                client_limit,
                global_limit,
                client_ttl,
                global_ttl,
            ) = args
            assert isinstance(client_key, str)
            assert isinstance(global_key, str)
            async with self.quota_lock:
                self.quota_eval_calls.append((client_key, global_key))
                client_count = self.quota_counts.get(client_key, 0)
                global_count = self.quota_counts.get(global_key, 0)
                if global_count >= int(global_limit):
                    return [0, 2]
                if client_count >= int(client_limit):
                    return [0, 1]
                self.quota_counts[client_key] = client_count + 1
                self.quota_counts[global_key] = global_count + 1
                self.expirations[client_key] = int(client_ttl)
                self.expirations[global_key] = int(global_ttl)
                return [1, 0]

        assert number_of_keys == 1
        key, capacity, refill_rate, ttl = args
        assert isinstance(key, str)
        self.eval_keys.append(key)
        tokens, previous = self.buckets.get(key, (float(capacity), self.now))
        tokens = min(
            float(capacity),
            tokens + (self.now - previous) * float(refill_rate),
        )
        allowed = tokens >= 1
        if allowed:
            tokens -= 1
        self.buckets[key] = (tokens, self.now)
        self.expirations[key] = ttl
        return [int(allowed), str(tokens)]

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeProvider(BaseProvider):
    def __init__(
        self,
        name: str,
        *,
        model: str | None = None,
        available: bool = True,
        input_cost: float = 1.0,
        output_cost: float = 1.0,
        effects: Iterable[ProviderResult | Exception] = (),
        include_temperature: bool = False,
    ) -> None:
        self.metadata = ProviderMetadata(
            name=name,
            model=model or f"{name}-model",
            available=available,
            input_cost_per_million=input_cost,
            output_cost_per_million=output_cost,
        )
        self.effects = list(effects)
        self.include_temperature = include_temperature
        self.calls = 0
        self.requests: list[GenerationRequest] = []

    def effective_parameters(self, request: GenerationRequest) -> dict[str, object]:
        parameters: dict[str, object] = {"max_output_tokens": request.max_tokens}
        if self.include_temperature:
            parameters["temperature"] = request.temperature
        return parameters

    async def generate(self, request: GenerationRequest) -> ProviderResult:
        self.calls += 1
        self.requests.append(request)
        if self.effects:
            effect = self.effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect
        return ProviderResult(
            content=f"{self.metadata.name} response",
            provider=self.metadata.name,
            model=self.metadata.model,
            input_tokens=3,
            output_tokens=2,
        )


class FakeMetricsService:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.fail_writes = False

    async def record_success(self, response, *, routing_strategy, provider) -> bool:
        if self.fail_writes:
            return False
        self.records.append(
            {
                "request_id": response.request_id,
                "provider": response.provider,
                "model": response.model,
                "routing_strategy": routing_strategy,
                "status": "success",
                "latency_ms": response.latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "estimated_cost_usd": MetricsService.estimate_cost(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    provider=provider,
                    cache_hit=response.cache_hit,
                ),
                "cache_hit": response.cache_hit,
                "fallback_used": response.fallback_used,
                "error_category": None,
            }
        )
        return True

    async def record_failure(
        self, *, request_id, routing_strategy, latency_ms, error_category
    ) -> bool:
        if self.fail_writes:
            return False
        self.records.append(
            {
                "request_id": request_id,
                "provider": None,
                "model": None,
                "routing_strategy": routing_strategy,
                "status": "error",
                "latency_ms": latency_ms,
                "input_tokens": 0,
                "output_tokens": 0,
                "estimated_cost_usd": 0,
                "cache_hit": False,
                "fallback_used": False,
                "error_category": error_category,
            }
        )
        return True


def provider_result(name: str, model: str | None = None) -> ProviderResult:
    return ProviderResult(
        content=f"{name} response",
        provider=name,
        model=model or f"{name}-model",
        input_tokens=3,
        output_tokens=2,
    )


def request(
    app: Any,
    method: str,
    path: str,
    *,
    json: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, json=json, headers=headers)

    return asyncio.run(send())


@pytest.fixture
def app_factory():
    def build(
        providers: Iterable[BaseProvider] | None = None,
        *,
        redis_client: FakeRedis | None = None,
        default_provider: str = "mock",
        max_retries: int = 1,
        capacity: int = 20,
        refill_rate: float = 1.0,
        metrics_service: Any | None = None,
        public_gemini_demo_enabled: bool = False,
        public_gemini_demo_max_output_tokens: int = 256,
        public_gemini_demo_max_prompt_chars: int = 2_000,
        public_gemini_demo_client_limit: int = 2,
        public_gemini_demo_global_limit: int = 15,
    ):
        settings = Settings(
            _env_file=None,
            default_provider=default_provider,
            provider_max_retries=max_retries,
            provider_retry_delay_seconds=0,
            rate_limit_capacity=capacity,
            rate_limit_refill_rate=refill_rate,
            public_gemini_demo_enabled=public_gemini_demo_enabled,
            public_gemini_demo_max_output_tokens=(
                public_gemini_demo_max_output_tokens
            ),
            public_gemini_demo_max_prompt_chars=public_gemini_demo_max_prompt_chars,
            public_gemini_demo_client_limit=public_gemini_demo_client_limit,
            public_gemini_demo_global_limit=public_gemini_demo_global_limit,
        )
        return create_app(
            settings,
            redis_client=redis_client or FakeRedis(),
            providers=providers or [FakeProvider("mock")],
            metrics_service=metrics_service or FakeMetricsService(),
        )

    return build


@pytest.fixture
def valid_payload() -> dict[str, object]:
    return {
        "prompt": "Explain binary search",
        "strategy": "fixed",
        "temperature": 0.2,
        "max_tokens": 512,
    }
