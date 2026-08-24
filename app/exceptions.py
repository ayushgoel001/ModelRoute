class GatewayError(Exception):
    """Base class for expected gateway failures safe to normalize at the API edge."""


class NoEligibleProviderError(GatewayError):
    pass


class AllProvidersFailedError(GatewayError):
    pass


class RateLimiterUnavailableError(GatewayError):
    pass


class MetricsUnavailableError(GatewayError):
    pass


class PublicGeminiDemoError(GatewayError):
    pass


class PublicGeminiDemoUnavailableError(PublicGeminiDemoError):
    pass


class PublicGeminiDemoRequestError(PublicGeminiDemoError):
    pass


class PublicGeminiQuotaExceededError(PublicGeminiDemoError):
    def __init__(self, scope: str) -> None:
        super().__init__("Live Gemini demo quota exhausted")
        self.scope = scope


class PublicGeminiQuotaUnavailableError(PublicGeminiDemoError):
    pass


class ProviderError(GatewayError):
    retryable = False

    def __init__(self, provider: str, message: str = "Provider request failed") -> None:
        super().__init__(message)
        self.provider = provider


class ProviderTimeoutError(ProviderError):
    retryable = True

    _SAFE_TIMEOUT_SOURCES = frozenset({"outer_asyncio", "sdk_http"})

    def __init__(
        self,
        provider: str,
        message: str = "Provider request failed",
        *,
        timeout_source: str | None = None,
    ) -> None:
        super().__init__(provider, message)
        self.timeout_source = (
            timeout_source
            if timeout_source in self._SAFE_TIMEOUT_SOURCES
            else None
        )


class ProviderRateLimitError(ProviderError):
    retryable = True


class ProviderUnavailableError(ProviderError):
    retryable = True


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderRequestError(ProviderError):
    pass
