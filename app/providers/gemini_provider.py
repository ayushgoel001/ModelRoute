import asyncio
from typing import Any, Mapping

from google import genai
# google-genai 2.18.1 Interactions registers this compatibility error hook;
# its emitted exceptions are not subclasses of the public google.genai.errors API.
from google.genai._gaos.lib import compat_errors as gemini_errors
from pydantic import JsonValue

from app.exceptions import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.base import (
    BaseProvider,
    GenerationRequest,
    ProviderMetadata,
    ProviderResult,
)

_RATE_LIMIT_SOURCES = {
    "quota_exceeded": "daily_quota",
    "rate_limit_exceeded": "short_term",
}


def _rate_limit_source(error: Any) -> str:
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return "unknown_429"
    error_body = body.get("error")
    if not isinstance(error_body, dict):
        return "unknown_429"
    code = error_body.get("code")
    if not isinstance(code, str):
        return "unknown_429"
    return _RATE_LIMIT_SOURCES.get(code, "unknown_429")


class GeminiProvider(BaseProvider):
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        timeout_seconds: float,
        input_cost_per_million: float,
        output_cost_per_million: float,
        client: Any | None = None,
        available: bool | None = None,
    ) -> None:
        is_available = bool(api_key) if available is None else available
        self.metadata = ProviderMetadata(
            name="gemini",
            model=model,
            available=is_available,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
        )
        self.timeout_seconds = timeout_seconds
        self._owns_client = client is None and bool(api_key)
        self._client = client
        if self._client is None and api_key:
            self._client = genai.Client(
                api_key=api_key,
                http_options=genai.types.HttpOptions(
                    timeout=int(timeout_seconds * 1_000),
                    retry_options=genai.types.HttpRetryOptions(attempts=1),
                ),
            )

    def effective_parameters(
        self, request: GenerationRequest
    ) -> Mapping[str, JsonValue]:
        # Gemini 3.7 Flash omits deprecated temperature/top-p/top-k controls.
        return {
            "max_output_tokens": request.max_tokens,
            "thinking_level": "low",
        }

    async def generate(self, request: GenerationRequest) -> ProviderResult:
        if self._client is None:
            raise ProviderUnavailableError("gemini", "Gemini is not configured")

        generation_config = dict(self.effective_parameters(request))
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self._client.aio.interactions.create(
                    model=self.metadata.model,
                    input=request.prompt,
                    generation_config=generation_config,
                    timeout=self.timeout_seconds,
                )
        except gemini_errors.APITimeoutError as exc:
            raise ProviderTimeoutError(
                "gemini", timeout_source="sdk_http"
            ) from exc
        except TimeoutError as exc:
            raise ProviderTimeoutError(
                "gemini", timeout_source="outer_asyncio"
            ) from exc
        except gemini_errors.RateLimitError as exc:
            raise ProviderRateLimitError(
                "gemini",
                rate_limit_source=_rate_limit_source(exc),
            ) from exc
        except (
            gemini_errors.AuthenticationError,
            gemini_errors.PermissionDeniedError,
        ) as exc:
            raise ProviderAuthenticationError("gemini") from exc
        except (
            gemini_errors.BadRequestError,
            gemini_errors.UnprocessableEntityError,
        ) as exc:
            raise ProviderRequestError("gemini") from exc
        except (
            gemini_errors.InternalServerError,
            gemini_errors.APIConnectionError,
        ) as exc:
            raise ProviderUnavailableError("gemini") from exc
        except gemini_errors.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ProviderUnavailableError("gemini") from exc
            raise ProviderRequestError("gemini") from exc

        usage = getattr(response, "usage", None)
        return ProviderResult(
            content=getattr(response, "output_text", "") or "",
            provider=self.metadata.name,
            model=self.metadata.model,
            input_tokens=getattr(usage, "total_input_tokens", 0) or 0,
            output_tokens=getattr(usage, "total_output_tokens", 0) or 0,
        )

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aio.aclose()
