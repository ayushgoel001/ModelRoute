class LatencyTracker:
    def __init__(self, *, alpha: float = 0.3, initial_latency_ms: float = 1_000.0) -> None:
        self.alpha = alpha
        self.initial_latency_ms = initial_latency_ms
        self._ewma_ms: dict[str, float] = {}

    def get(self, provider_identity: str) -> float:
        return self._ewma_ms.get(provider_identity, self.initial_latency_ms)

    def record(self, provider_identity: str, latency_ms: float) -> None:
        previous = self._ewma_ms.get(provider_identity)
        if previous is None:
            self._ewma_ms[provider_identity] = latency_ms
        else:
            self._ewma_ms[provider_identity] = (
                self.alpha * latency_ms + (1 - self.alpha) * previous
            )
