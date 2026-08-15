import asyncio

import pytest

from app.exceptions import RateLimiterUnavailableError
from app.services.rate_limiter import TOKEN_BUCKET_SCRIPT, TokenBucketRateLimiter
from tests.conftest import FakeRedis, request


def test_requests_within_capacity_are_allowed() -> None:
    limiter = TokenBucketRateLimiter(FakeRedis(), capacity=2, refill_rate=1)

    assert asyncio.run(limiter.allow("client:a")) is True
    assert asyncio.run(limiter.allow("client:a")) is True


def test_burst_exhaustion_returns_false() -> None:
    limiter = TokenBucketRateLimiter(FakeRedis(), capacity=2, refill_rate=1)

    assert [asyncio.run(limiter.allow("client:a")) for _ in range(3)] == [
        True,
        True,
        False,
    ]


def test_continuous_refill_restores_tokens() -> None:
    redis = FakeRedis()
    limiter = TokenBucketRateLimiter(redis, capacity=1, refill_rate=0.5)
    assert asyncio.run(limiter.allow("client:a")) is True
    assert asyncio.run(limiter.allow("client:a")) is False

    redis.advance(2)

    assert asyncio.run(limiter.allow("client:a")) is True


def test_different_client_ids_have_independent_buckets() -> None:
    limiter = TokenBucketRateLimiter(FakeRedis(), capacity=1, refill_rate=1)

    assert asyncio.run(limiter.allow("client:a")) is True
    assert asyncio.run(limiter.allow("client:a")) is False
    assert asyncio.run(limiter.allow("client:b")) is True


def test_client_identity_is_sha256_namespaced() -> None:
    unsafe = "../../raw user value with spaces"
    key = TokenBucketRateLimiter.bucket_key(unsafe)

    assert key.startswith("modelroute:ratelimit:")
    assert unsafe not in key
    assert len(key.removeprefix("modelroute:ratelimit:")) == 64


def test_redis_failure_fails_closed() -> None:
    redis = FakeRedis()
    redis.fail_eval = True
    limiter = TokenBucketRateLimiter(redis, capacity=1, refill_rate=1)

    with pytest.raises(RateLimiterUnavailableError):
        asyncio.run(limiter.allow("client:a"))


def test_bucket_has_abandonment_ttl() -> None:
    redis = FakeRedis()
    limiter = TokenBucketRateLimiter(redis, capacity=20, refill_rate=1)

    asyncio.run(limiter.allow("client:a"))
    key = limiter.bucket_key("client:a")

    assert redis.expirations[key] == limiter.bucket_ttl_seconds
    assert limiter.bucket_ttl_seconds >= 60


def test_lua_script_performs_one_atomic_state_transition() -> None:
    assert "HMGET" in TOKEN_BUCKET_SCRIPT
    assert "HSET" in TOKEN_BUCKET_SCRIPT
    assert "EXPIRE" in TOKEN_BUCKET_SCRIPT
    assert "TIME" in TOKEN_BUCKET_SCRIPT


def test_capacity_exhaustion_returns_http_429(
    app_factory, valid_payload
) -> None:
    application = app_factory(capacity=1, refill_rate=0.0001)

    first = request(
        application,
        "POST",
        "/v1/chat/completions",
        json=valid_payload,
        headers={"X-Client-ID": "same"},
    )
    second = request(
        application,
        "POST",
        "/v1/chat/completions",
        json=valid_payload,
        headers={"X-Client-ID": "same"},
    )

    assert first.status_code == 200
    assert second.status_code == 429


def test_x_client_id_selects_independent_http_buckets(
    app_factory, valid_payload
) -> None:
    application = app_factory(capacity=1, refill_rate=0.0001)

    first = request(
        application,
        "POST",
        "/v1/chat/completions",
        json=valid_payload,
        headers={"X-Client-ID": "A"},
    )
    second = request(
        application,
        "POST",
        "/v1/chat/completions",
        json=valid_payload,
        headers={"X-Client-ID": "B"},
    )

    assert first.status_code == second.status_code == 200


def test_missing_client_id_falls_back_to_client_host(
    app_factory, valid_payload
) -> None:
    redis = FakeRedis()
    application = app_factory(redis_client=redis)

    response = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    )

    assert response.status_code == 200
    assert redis.eval_keys == [TokenBucketRateLimiter.bucket_key("ip:127.0.0.1")]


def test_rate_limit_redis_failure_returns_http_503(
    app_factory, valid_payload
) -> None:
    redis = FakeRedis()
    redis.fail_eval = True
    application = app_factory(redis_client=redis)

    response = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Rate limiter unavailable"}
