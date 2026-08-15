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


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}
        self.buckets: dict[str, tuple[float, float]] = {}
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
        key: str,
        capacity: int,
        refill_rate: float,
        ttl: int,
    ) -> list[object]:
        del script, number_of_keys
        if self.fail_eval:
            raise RedisConnectionError("rate limiter unavailable")
        self.eval_keys.append(key)
        tokens, previous = self.buckets.get(key, (float(capacity), self.now))
        tokens = min(float(capacity), tokens + (self.now - previous) * refill_rate)
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
    ):
        settings = Settings(
            default_provider=default_provider,
            provider_max_retries=max_retries,
            provider_retry_delay_seconds=0,
            rate_limit_capacity=capacity,
            rate_limit_refill_rate=refill_rate,
        )
        return create_app(
            settings,
            redis_client=redis_client or FakeRedis(),
            providers=providers or [FakeProvider("mock")],
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
