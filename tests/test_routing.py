import pytest

from app.exceptions import NoEligibleProviderError
from app.providers.base import GenerationRequest
from app.services.latency import LatencyTracker
from app.services.router import ProviderRouter
from tests.conftest import FakeProvider

REQUEST = GenerationRequest(prompt="small prompt", temperature=0.2, max_tokens=100)


def names(providers) -> list[str]:
    return [provider.metadata.name for provider in providers]


def router(providers, *, default="openai", allow_mock=False):
    return ProviderRouter(
        providers,
        default_provider=default,
        latency_tracker=LatencyTracker(initial_latency_ms=500),
        allow_mock=allow_mock,
    )


def test_fixed_places_available_preferred_provider_first() -> None:
    candidates = router([FakeProvider("gemini"), FakeProvider("openai")]).route(
        "fixed", REQUEST
    )

    assert names(candidates) == ["openai", "gemini"]


def test_fixed_skips_unavailable_preferred_provider() -> None:
    candidates = router(
        [FakeProvider("openai", available=False), FakeProvider("gemini")]
    ).route("fixed", REQUEST)

    assert names(candidates) == ["gemini"]


def test_cheapest_uses_injected_price_metadata() -> None:
    expensive = FakeProvider("openai", input_cost=10, output_cost=20)
    cheap = FakeProvider("gemini", input_cost=1, output_cost=2)

    candidates = router([expensive, cheap]).route("cheapest", REQUEST)

    assert names(candidates) == ["gemini", "openai"]


def test_fastest_uses_latency_history() -> None:
    openai = FakeProvider("openai")
    gemini = FakeProvider("gemini")
    tracker = LatencyTracker(alpha=0.5, initial_latency_ms=500)
    tracker.record(openai.metadata.identity, 300)
    tracker.record(gemini.metadata.identity, 100)
    provider_router = ProviderRouter(
        [openai, gemini],
        default_provider="openai",
        latency_tracker=tracker,
    )

    assert names(provider_router.route("fastest", REQUEST)) == ["gemini", "openai"]


def test_fastest_without_history_has_deterministic_tie_breaker() -> None:
    candidates = router([FakeProvider("openai"), FakeProvider("gemini")]).route(
        "fastest", REQUEST
    )

    assert names(candidates) == ["gemini", "openai"]


def test_unavailable_providers_are_excluded_for_every_strategy() -> None:
    providers = [FakeProvider("openai", available=False), FakeProvider("gemini")]

    for strategy in ("fixed", "cheapest", "fastest"):
        assert names(router(providers).route(strategy, REQUEST)) == ["gemini"]


def test_mock_is_not_an_implicit_production_fallback() -> None:
    provider_router = router(
        [FakeProvider("openai", available=False), FakeProvider("mock")]
    )

    with pytest.raises(NoEligibleProviderError):
        provider_router.route("fixed", REQUEST)


def test_mock_can_be_explicitly_enabled() -> None:
    candidates = router(
        [FakeProvider("mock")], default="mock", allow_mock=True
    ).route("fixed", REQUEST)

    assert names(candidates) == ["mock"]


def test_latency_tracker_uses_ewma() -> None:
    tracker = LatencyTracker(alpha=0.25, initial_latency_ms=900)

    assert tracker.get("openai:model") == 900
    tracker.record("openai:model", 100)
    tracker.record("openai:model", 300)
    assert tracker.get("openai:model") == pytest.approx(150)
