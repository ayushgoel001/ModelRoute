import asyncio
from typing import Any, Mapping

import openai
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


class OpenAIProvider(BaseProvider):
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
            name="openai",
            model=model,
            available=is_available,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
        )
        self.timeout_seconds = timeout_seconds
        self._owns_client = client is None and bool(api_key)
        self._client = client
        if self._client is None and api_key:
            self._client = openai.AsyncOpenAI(
                api_key=api_key,
                timeout=timeout_seconds,
                max_retries=0,
            )

    def effective_parameters(
        self, request: GenerationRequest
    ) -> Mapping[str, JsonValue]:
        # GPT-5.6 reasoning configurations do not accept sampling temperature.
        return {
            "max_output_tokens": request.max_tokens,
            "reasoning": {"effort": "low"},
        }

    async def generate(self, request: GenerationRequest) -> ProviderResult:
        if self._client is None:
            raise ProviderUnavailableError("openai", "OpenAI is not configured")

        parameters = dict(self.effective_parameters(request))
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await self._client.responses.create(
                    model=self.metadata.model,
                    input=request.prompt,
                    **parameters,
                )
        except (TimeoutError, openai.APITimeoutError) as exc:
            raise ProviderTimeoutError("openai") from exc
        except openai.RateLimitError as exc:
            raise ProviderRateLimitError("openai") from exc
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise ProviderAuthenticationError("openai") from exc
        except openai.BadRequestError as exc:
            raise ProviderRequestError("openai") from exc
        except (openai.InternalServerError, openai.APIConnectionError) as exc:
            raise ProviderUnavailableError("openai") from exc
        except openai.APIStatusError as exc:
            if exc.status_code >= 500:
                raise ProviderUnavailableError("openai") from exc
            raise ProviderRequestError("openai") from exc

        usage = getattr(response, "usage", None)
        return ProviderResult(
            content=getattr(response, "output_text", "") or "",
            provider=self.metadata.name,
            model=self.metadata.model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.close()
