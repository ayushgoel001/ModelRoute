import asyncio

from app.config import PUBLIC_GEMINI_DEMO_MODEL, Settings
from app.exceptions import (
    ProviderRequestError,
    ProviderTimeoutError,
    PublicGeminiQuotaExceededError,
)
from app.main import create_app
from app.services.gemini_demo_quota import (
    GEMINI_DEMO_QUOTA_SCRIPT,
    PublicGeminiDemoQuota,
)
from tests.conftest import (
    FakeMetricsService,
    FakeProvider,
    FakeRedis,
    request,
)


def gemini_payload(prompt: str = "public Gemini verification") -> dict[str, object]:
    return {
        "prompt": prompt,
        "strategy": "fixed",
        "temperature": 0.2,
        "max_tokens": 128,
        "preferred_provider": "gemini",
    }


def demo_providers(*, gemini_effects=()):
    return [
        FakeProvider(
            "gemini",
            model=PUBLIC_GEMINI_DEMO_MODEL,
            effects=gemini_effects,
        ),
        FakeProvider("mock"),
    ]


def test_feature_disabled_rejects_gemini_but_mock_remains_available(
    app_factory,
) -> None:
    providers = demo_providers()
    application = app_factory(providers, public_gemini_demo_enabled=False)

    gemini_response = request(
        application,
        "POST",
        "/v1/chat/completions",
        json=gemini_payload(),
    )
    mock_response = request(
        application,
        "POST",
        "/v1/chat/completions",
        json={**gemini_payload("mock still works"), "preferred_provider": "mock"},
    )

    assert gemini_response.status_code == 403
    assert gemini_response.json() == {
        "detail": "Live Gemini demo is disabled. MockProvider remains available."
    }
    assert mock_response.status_code == 200
    assert mock_response.json()["provider"] == "mock"
    assert providers[0].calls == 0


def test_gemini_cache_hit_does_not_consume_another_quota_unit(app_factory) -> None:
    redis = FakeRedis()
    providers = demo_providers()
    application = app_factory(
        providers,
        redis_client=redis,
        public_gemini_demo_enabled=True,
    )
    headers = {"X-Client-ID": "cache-client"}

    first = request(
        application,
        "POST",
        "/v1/chat/completions",
        json=gemini_payload(),
        headers=headers,
    )
    second = request(
        application,
        "POST",
        "/v1/chat/completions",
        json=gemini_payload(),
        headers=headers,
    )

    assert first.status_code == second.status_code == 200
    assert first.json()["cache_hit"] is False
    assert second.json()["cache_hit"] is True
    assert first.json()["request_id"] != second.json()["request_id"]
    assert providers[0].calls == 1
    assert len(redis.quota_eval_calls) == 1


def test_one_cache_miss_quota_unit_covers_the_existing_bounded_retry(
    app_factory,
) -> None:
    redis = FakeRedis()
    providers = demo_providers(
        gemini_effects=[ProviderTimeoutError("gemini")]
    )
    application = app_factory(
        providers,
        redis_client=redis,
        public_gemini_demo_enabled=True,
        max_retries=1,
    )

    response = request(
        application,
        "POST",
        "/v1/chat/completions",
        json=gemini_payload("retry accounting"),
        headers={"X-Client-ID": "retry-client"},
    )

    assert response.status_code == 200
    assert providers[0].calls == 2
    assert len(redis.quota_eval_calls) == 1


def test_unpinned_direct_api_gemini_candidate_is_still_guarded(app_factory) -> None:
    redis = FakeRedis()
    providers = demo_providers()
    application = app_factory(
        providers,
        redis_client=redis,
        public_gemini_demo_enabled=True,
    )
    payload = {
        **gemini_payload("direct fastest request"),
        "strategy": "fastest",
        "max_tokens": 8192,
    }
    del payload["preferred_provider"]

    response = request(
        application,
        "POST",
        "/v1/chat/completions",
        json=payload,
        headers={"X-Client-ID": "direct-client"},
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "gemini"
    assert providers[0].requests[0].max_tokens == 256
    assert len(redis.quota_eval_calls) == 1


def test_feature_flag_does_not_disable_normal_non_public_gemini_routing(
    app_factory,
) -> None:
    redis = FakeRedis()
    gemini = FakeProvider("gemini", model=PUBLIC_GEMINI_DEMO_MODEL)
    application = app_factory(
        [gemini],
        redis_client=redis,
        default_provider="gemini",
        public_gemini_demo_enabled=False,
    )

    response = request(
        application,
        "POST",
        "/v1/chat/completions",
        json={
            "prompt": "normal private deployment behavior",
            "strategy": "fixed",
            "temperature": 0.2,
            "max_tokens": 512,
        },
    )

    assert response.status_code == 200
    assert response.json()["provider"] == "gemini"
    assert gemini.requests[0].max_tokens == 512
    assert redis.quota_eval_calls == []


def test_per_client_quota_rejection_is_terminal(app_factory) -> None:
    redis = FakeRedis()
    providers = demo_providers()
    application = app_factory(
        providers,
        redis_client=redis,
        public_gemini_demo_enabled=True,
        public_gemini_demo_client_limit=2,
    )
    headers = {"X-Client-ID": "same-client"}

    responses = [
        request(
            application,
            "POST",
            "/v1/chat/completions",
            json=gemini_payload(f"unique prompt {index}"),
            headers=headers,
        )
        for index in range(3)
    ]

    assert [response.status_code for response in responses] == [200, 200, 429]
    assert responses[-1].json() == {
        "detail": "Live Gemini demo limit reached. MockProvider remains available."
    }
    assert providers[0].calls == 2
    assert providers[1].calls == 0


def test_rotating_client_ids_cannot_bypass_global_quota_and_mock_still_works(
    app_factory,
) -> None:
    redis = FakeRedis()
    providers = demo_providers()
    application = app_factory(
        providers,
        redis_client=redis,
        public_gemini_demo_enabled=True,
        public_gemini_demo_global_limit=2,
    )

    responses = [
        request(
            application,
            "POST",
            "/v1/chat/completions",
            json=gemini_payload(f"rotated prompt {index}"),
            headers={"X-Client-ID": f"rotated-{index}"},
        )
        for index in range(3)
    ]
    mock_response = request(
        application,
        "POST",
        "/v1/chat/completions",
        json={**gemini_payload("mock after quota"), "preferred_provider": "mock"},
        headers={"X-Client-ID": "rotated-2"},
    )

    assert [response.status_code for response in responses] == [200, 200, 429]
    assert providers[0].calls == 2
    assert providers[1].calls == 1
    assert mock_response.status_code == 200
    assert mock_response.json()["provider"] == "mock"


def test_public_gemini_constraints_are_enforced_server_side(app_factory) -> None:
    redis = FakeRedis()
    providers = demo_providers()
    application = app_factory(
        providers,
        redis_client=redis,
        public_gemini_demo_enabled=True,
        public_gemini_demo_max_prompt_chars=20,
    )
    oversized_tokens = {**gemini_payload("short prompt"), "max_tokens": 8192}

    accepted = request(
        application,
        "POST",
        "/v1/chat/completions",
        json=oversized_tokens,
        headers={"X-Client-ID": "constraints"},
    )
    rejected = request(
        application,
        "POST",
        "/v1/chat/completions",
        json=gemini_payload("x" * 21),
        headers={"X-Client-ID": "constraints"},
    )

    assert accepted.status_code == 200
    assert providers[0].requests[0].max_tokens == 256
    assert rejected.status_code == 422
    assert providers[0].calls == 1
    assert len(redis.quota_eval_calls) == 1


def test_public_provider_exception_is_sanitized_and_does_not_fallback(
    app_factory,
) -> None:
    secret = "provider-secret-that-must-not-leak"
    providers = demo_providers(
        gemini_effects=[ProviderRequestError("gemini", secret)]
    )
    application = app_factory(providers, public_gemini_demo_enabled=True)

    response = request(
        application,
        "POST",
        "/v1/chat/completions",
        json=gemini_payload(),
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "No provider could fulfill the request"}
    assert secret not in response.text
    assert providers[1].calls == 0


def test_quota_lua_and_concurrent_checks_enforce_both_limits() -> None:
    async def exercise() -> tuple[int, int]:
        client_redis = FakeRedis()
        client_quota = PublicGeminiDemoQuota(
            client_redis,
            client_limit=2,
            client_window_seconds=3_600,
            global_limit=15,
            global_window_seconds=86_400,
        )

        async def allowed(quota, identity: str) -> bool:
            try:
                await quota.consume(identity)
                return True
            except PublicGeminiQuotaExceededError:
                return False

        client_results = await asyncio.gather(
            *(allowed(client_quota, "one-client") for _ in range(20))
        )

        global_redis = FakeRedis()
        global_quota = PublicGeminiDemoQuota(
            global_redis,
            client_limit=2,
            client_window_seconds=3_600,
            global_limit=15,
            global_window_seconds=86_400,
        )
        global_results = await asyncio.gather(
            *(allowed(global_quota, f"client-{index}") for index in range(30))
        )
        return sum(client_results), sum(global_results)

    client_allowed, global_allowed = asyncio.run(exercise())

    assert (client_allowed, global_allowed) == (2, 15)
    assert "redis.call('GET', KEYS[1])" in GEMINI_DEMO_QUOTA_SCRIPT
    assert "redis.call('GET', KEYS[2])" in GEMINI_DEMO_QUOTA_SCRIPT
    assert "redis.call('INCR', KEYS[1])" in GEMINI_DEMO_QUOTA_SCRIPT
    assert "redis.call('INCR', KEYS[2])" in GEMINI_DEMO_QUOTA_SCRIPT


def test_homepage_never_renders_configured_api_secret() -> None:
    secret = "test-gemini-secret-never-rendered"
    settings = Settings(
        _env_file=None,
        default_provider="mock",
        gemini_api_key=secret,
        public_gemini_demo_enabled=True,
    )
    application = create_app(
        settings,
        redis_client=FakeRedis(),
        providers=demo_providers(),
        metrics_service=FakeMetricsService(),
    )

    response = request(application, "GET", "/")

    assert response.status_code == 200
    assert secret not in response.text
