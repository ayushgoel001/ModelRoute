import hashlib
import logging
from typing import Any

from redis.exceptions import RedisError

from app.exceptions import (
    PublicGeminiQuotaExceededError,
    PublicGeminiQuotaUnavailableError,
)

logger = logging.getLogger(__name__)

GEMINI_DEMO_QUOTA_SCRIPT = """
local client_count = tonumber(redis.call('GET', KEYS[1]) or '0')
local global_count = tonumber(redis.call('GET', KEYS[2]) or '0')
local client_limit = tonumber(ARGV[1])
local global_limit = tonumber(ARGV[2])
local client_ttl = tonumber(ARGV[3])
local global_ttl = tonumber(ARGV[4])

if global_count >= global_limit then
  return {0, 2}
end
if client_count >= client_limit then
  return {0, 1}
end

if client_count == 0 then
  redis.call('SET', KEYS[1], 1, 'EX', client_ttl)
else
  redis.call('INCR', KEYS[1])
end
if global_count == 0 then
  redis.call('SET', KEYS[2], 1, 'EX', global_ttl)
else
  redis.call('INCR', KEYS[2])
end

return {1, 0}
"""


class PublicGeminiDemoQuota:
    def __init__(
        self,
        redis_client: Any,
        *,
        client_limit: int,
        client_window_seconds: int,
        global_limit: int,
        global_window_seconds: int,
    ) -> None:
        self.redis = redis_client
        self.client_limit = client_limit
        self.client_window_seconds = client_window_seconds
        self.global_limit = global_limit
        self.global_window_seconds = global_window_seconds

    @staticmethod
    def client_key(client_identity: str) -> str:
        digest = hashlib.sha256(client_identity.encode("utf-8")).hexdigest()
        return f"modelroute:public-gemini:client:{digest}"

    @staticmethod
    def global_key() -> str:
        return "modelroute:public-gemini:global"

    async def consume(self, client_identity: str) -> None:
        try:
            result = await self.redis.eval(
                GEMINI_DEMO_QUOTA_SCRIPT,
                2,
                self.client_key(client_identity),
                self.global_key(),
                self.client_limit,
                self.global_limit,
                self.client_window_seconds,
                self.global_window_seconds,
            )
            allowed = bool(int(result[0]))
            denial_scope = int(result[1])
        except (RedisError, IndexError, TypeError, ValueError) as exc:
            logger.error("Public Gemini demo quota Redis operation failed")
            raise PublicGeminiQuotaUnavailableError(
                "Live Gemini demo quota is unavailable"
            ) from exc

        if not allowed:
            scope = "global" if denial_scope == 2 else "client"
            raise PublicGeminiQuotaExceededError(scope)
