import asyncio
from uuid import UUID

import httpx
import pytest

from app.main import app


def post(payload: dict[str, object]) -> httpx.Response:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post("/v1/chat/completions", json=payload)

    return asyncio.run(send_request())


def valid_payload() -> dict[str, object]:
    return {
        "prompt": "Explain binary search",
        "strategy": "fixed",
        "temperature": 0.2,
        "max_tokens": 512,
    }


def test_valid_completion_returns_normalized_response() -> None:
    response = post(valid_payload())

    assert response.status_code == 200
    body = response.json()
    UUID(body["request_id"])
    assert body["provider"] == "mock"
    assert body["model"] == "mock-model"
    assert body["content"] == "Mock response: Explain binary search"
    assert body["input_tokens"] > 0
    assert body["output_tokens"] > 0
    assert body["latency_ms"] >= 0


def test_prompt_is_required() -> None:
    payload = valid_payload()
    del payload["prompt"]

    assert post(payload).status_code == 422


@pytest.mark.parametrize("prompt", ["", "   ", "\t\n"])
def test_blank_prompt_is_rejected(prompt: str) -> None:
    payload = valid_payload() | {"prompt": prompt}

    assert post(payload).status_code == 422


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_invalid_temperature_is_rejected(temperature: float) -> None:
    payload = valid_payload() | {"temperature": temperature}

    assert post(payload).status_code == 422


@pytest.mark.parametrize("max_tokens", [0, -1, 8193])
def test_invalid_max_tokens_is_rejected(max_tokens: int) -> None:
    payload = valid_payload() | {"max_tokens": max_tokens}

    assert post(payload).status_code == 422


def test_invalid_strategy_is_rejected() -> None:
    payload = valid_payload() | {"strategy": "random"}

    assert post(payload).status_code == 422


@pytest.mark.parametrize("strategy", ["fixed", "cheapest", "fastest"])
def test_all_phase_one_strategies_use_mock_provider(strategy: str) -> None:
    payload = valid_payload() | {"strategy": strategy}

    response = post(payload)

    assert response.status_code == 200
    assert response.json()["provider"] == "mock"


def test_requests_receive_distinct_request_ids() -> None:
    first = post(valid_payload()).json()
    second = post(valid_payload()).json()

    assert first["request_id"] != second["request_id"]
