import re
from typing import Mapping

from pydantic import JsonValue

from app.providers.base import (
    BaseProvider,
    GenerationRequest,
    ProviderMetadata,
    ProviderResult,
)

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def approximate_token_count(text: str) -> int:
    """Count words and punctuation as a deterministic tokenizer-free estimate."""
    return len(TOKEN_PATTERN.findall(text))


class MockProvider(BaseProvider):
    def __init__(
        self,
        model: str = "mock-model",
        *,
        available: bool = True,
        input_cost_per_million: float = 0.0,
        output_cost_per_million: float = 0.0,
    ) -> None:
        self.metadata = ProviderMetadata(
            name="mock",
            model=model,
            available=available,
            input_cost_per_million=input_cost_per_million,
            output_cost_per_million=output_cost_per_million,
        )

    def effective_parameters(
        self, request: GenerationRequest
    ) -> Mapping[str, JsonValue]:
        del request
        return {}

    async def generate(self, request: GenerationRequest) -> ProviderResult:
        normalized_prompt = request.prompt.strip()
        content = f"Mock response: {normalized_prompt}"
        return ProviderResult(
            content=content,
            provider=self.metadata.name,
            model=self.metadata.model,
            input_tokens=approximate_token_count(normalized_prompt),
            output_tokens=approximate_token_count(content),
        )
