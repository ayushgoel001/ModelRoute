from collections.abc import Iterable

from app.exceptions import NoEligibleProviderError
from app.providers.base import BaseProvider, GenerationRequest
from app.providers.mock_provider import approximate_token_count
from app.services.latency import LatencyTracker


class ProviderRouter:
    def __init__(
        self,
        providers: Iterable[BaseProvider],
        *,
        default_provider: str,
        latency_tracker: LatencyTracker,
        allow_mock: bool = False,
    ) -> None:
        self.providers = list(providers)
        self.default_provider = default_provider
        self.latency_tracker = latency_tracker
        self.allow_mock = allow_mock

    def route(
        self,
        strategy: str,
        request: GenerationRequest,
        *,
        preferred_provider: str | None = None,
    ) -> list[BaseProvider]:
        eligible = [
            provider
            for provider in self.providers
            if provider.metadata.available
            and (provider.metadata.name != "mock" or self.allow_mock)
            and (
                preferred_provider is None
                or provider.metadata.name == preferred_provider
            )
        ]
        if not eligible:
            raise NoEligibleProviderError("No eligible provider is configured")

        if strategy == "fixed":
            return sorted(
                eligible,
                key=lambda provider: provider.metadata.name != self.default_provider,
            )
        if strategy == "cheapest":
            return sorted(
                eligible,
                key=lambda provider: (
                    self._estimated_cost(provider, request),
                    provider.metadata.identity,
                ),
            )
        if strategy == "fastest":
            return sorted(
                eligible,
                key=lambda provider: (
                    self.latency_tracker.get(provider.metadata.identity),
                    provider.metadata.identity,
                ),
            )
        raise ValueError(f"Unsupported routing strategy: {strategy}")

    @staticmethod
    def _estimated_cost(provider: BaseProvider, request: GenerationRequest) -> float:
        estimated_input_tokens = approximate_token_count(request.prompt)
        metadata = provider.metadata
        return (
            estimated_input_tokens * metadata.input_cost_per_million
            + request.max_tokens * metadata.output_cost_per_million
        ) / 1_000_000
