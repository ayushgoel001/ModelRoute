import hashlib
import json
import logging
from typing import Any

from redis.exceptions import RedisError

from app.providers.base import BaseProvider, GenerationRequest, ProviderResult

logger = logging.getLogger(__name__)


class ResponseCache:
    def __init__(self, redis_client: Any, *, ttl_seconds: int) -> None:
        self.redis = redis_client
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def build_key(provider: BaseProvider, request: GenerationRequest) -> str:
        identity = {
            "provider": provider.metadata.name,
            "model": provider.metadata.model,
            "prompt": request.prompt,
            "parameters": provider.effective_parameters(request),
        }
        canonical = json.dumps(
            identity,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"modelroute:cache:{digest}"

    async def get(self, key: str) -> ProviderResult | None:
        try:
            cached = await self.redis.get(key)
            if cached is None:
                return None
            data = json.loads(cached)
            return ProviderResult(
                content=data["content"],
                provider=data["provider"],
                model=data["model"],
                input_tokens=data["input_tokens"],
                output_tokens=data["output_tokens"],
            )
        except (RedisError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            logger.warning("Response cache read failed; continuing without cache")
            return None

    async def set(self, key: str, result: ProviderResult) -> None:
        payload = json.dumps(
            {
                "content": result.content,
                "provider": result.provider,
                "model": result.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        try:
            await self.redis.set(key, payload, ex=self.ttl_seconds)
        except RedisError:
            logger.warning("Response cache write failed; continuing without cache")
