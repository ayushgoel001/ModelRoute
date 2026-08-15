from uuid import UUID

import pytest

from tests.conftest import FakeProvider, request


def test_valid_completion_returns_normalized_response(
    app_factory, valid_payload
) -> None:
    response = request(
        app_factory(), "POST", "/v1/chat/completions", json=valid_payload
    )

    assert response.status_code == 200
    body = response.json()
    UUID(body["request_id"])
    assert body["provider"] == "mock"
    assert body["model"] == "mock-model"
    assert body["content"] == "mock response"
    assert body["input_tokens"] > 0
    assert body["output_tokens"] > 0
    assert body["latency_ms"] >= 0
    assert body["cache_hit"] is False
    assert body["fallback_used"] is False


def test_prompt_is_required(app_factory, valid_payload) -> None:
    del valid_payload["prompt"]

    response = request(
        app_factory(), "POST", "/v1/chat/completions", json=valid_payload
    )
    assert response.status_code == 422


@pytest.mark.parametrize("prompt", ["", "   ", "\t\n"])
def test_blank_prompt_is_rejected(app_factory, valid_payload, prompt: str) -> None:
    valid_payload["prompt"] = prompt

    response = request(
        app_factory(), "POST", "/v1/chat/completions", json=valid_payload
    )
    assert response.status_code == 422


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_invalid_temperature_is_rejected(
    app_factory, valid_payload, temperature: float
) -> None:
    valid_payload["temperature"] = temperature

    response = request(
        app_factory(), "POST", "/v1/chat/completions", json=valid_payload
    )
    assert response.status_code == 422


@pytest.mark.parametrize("max_tokens", [0, -1, 8193])
def test_invalid_max_tokens_is_rejected(
    app_factory, valid_payload, max_tokens: int
) -> None:
    valid_payload["max_tokens"] = max_tokens

    response = request(
        app_factory(), "POST", "/v1/chat/completions", json=valid_payload
    )
    assert response.status_code == 422


def test_invalid_strategy_is_rejected(app_factory, valid_payload) -> None:
    valid_payload["strategy"] = "random"

    response = request(
        app_factory(), "POST", "/v1/chat/completions", json=valid_payload
    )
    assert response.status_code == 422


@pytest.mark.parametrize("strategy", ["fixed", "cheapest", "fastest"])
def test_all_strategies_can_use_explicit_mock_provider(
    app_factory, valid_payload, strategy: str
) -> None:
    valid_payload["strategy"] = strategy

    response = request(
        app_factory(), "POST", "/v1/chat/completions", json=valid_payload
    )
    assert response.status_code == 200
    assert response.json()["provider"] == "mock"


def test_requests_receive_distinct_request_ids(app_factory, valid_payload) -> None:
    application = app_factory()
    first = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    ).json()
    second = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    ).json()

    assert first["request_id"] != second["request_id"]
    assert second["cache_hit"] is True


def test_no_eligible_real_provider_returns_503(app_factory, valid_payload) -> None:
    unavailable = FakeProvider("openai", available=False)
    application = app_factory([unavailable], default_provider="openai")

    response = request(
        application, "POST", "/v1/chat/completions", json=valid_payload
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "No provider could fulfill the request"}
