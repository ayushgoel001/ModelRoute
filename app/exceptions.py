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

    _SAFE_RATE_LIMIT_SOURCES = frozenset(
        {"daily_quota", "short_term", "unknown_429"}
    )

    def __init__(
        self,
        provider: str,
        message: str = "Provider request failed",
        *,
        rate_limit_source: str | None = None,
    ) -> None:
        super().__init__(provider, message)
        self.rate_limit_source = (
            rate_limit_source
            if rate_limit_source in self._SAFE_RATE_LIMIT_SOURCES
            else None
        )


class ProviderUnavailableError(ProviderError):
    retryable = True


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderRequestError(ProviderError):
    pass
