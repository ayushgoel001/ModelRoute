class GatewayError(Exception):
    """Base class for expected gateway failures safe to normalize at the API edge."""


class NoEligibleProviderError(GatewayError):
    pass


class AllProvidersFailedError(GatewayError):
    pass


class RateLimiterUnavailableError(GatewayError):
    pass


class ProviderError(GatewayError):
    retryable = False

    def __init__(self, provider: str, message: str = "Provider request failed") -> None:
        super().__init__(message)
        self.provider = provider


class ProviderTimeoutError(ProviderError):
    retryable = True


class ProviderRateLimitError(ProviderError):
    retryable = True


class ProviderUnavailableError(ProviderError):
    retryable = True


class ProviderAuthenticationError(ProviderError):
    pass


class ProviderRequestError(ProviderError):
    pass
