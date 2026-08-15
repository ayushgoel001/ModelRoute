import hashlib
import logging
import math
from typing import Any

from redis.exceptions import RedisError

from app.exceptions import RateLimiterUnavailableError

logger = logging.getLogger(__name__)

TOKEN_BUCKET_SCRIPT = """
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local redis_time = redis.call('TIME')
local now = tonumber(redis_time[1]) + tonumber(redis_time[2]) / 1000000
local state = redis.call('HMGET', KEYS[1], 'tokens', 'timestamp')
local tokens = tonumber(state[1]) or capacity
local previous = tonumber(state[2]) or now
if now < previous then previous = now end
tokens = math.min(capacity, tokens + (now - previous) * refill_rate)
local allowed = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'timestamp', now)
redis.call('EXPIRE', KEYS[1], ttl)
return {allowed, tostring(tokens)}
"""


class TokenBucketRateLimiter:
    def __init__(
        self,
        redis_client: Any,
        *,
        capacity: int,
        refill_rate: float,
    ) -> None:
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.bucket_ttl_seconds = max(60, math.ceil(2 * capacity / refill_rate))

    @staticmethod
    def bucket_key(client_identity: str) -> str:
        digest = hashlib.sha256(client_identity.encode("utf-8")).hexdigest()
        return f"modelroute:ratelimit:{digest}"

    async def allow(self, client_identity: str) -> bool:
        key = self.bucket_key(client_identity)
        try:
            result = await self.redis.eval(
                TOKEN_BUCKET_SCRIPT,
                1,
                key,
                self.capacity,
                self.refill_rate,
                self.bucket_ttl_seconds,
            )
            return bool(int(result[0]))
        except (RedisError, IndexError, TypeError, ValueError) as exc:
            logger.error("Rate-limit Redis operation failed")
            raise RateLimiterUnavailableError(
                "Shared rate-limit enforcement is unavailable"
            ) from exc
