import re

from app.providers.base import BaseProvider, ProviderResult

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def approximate_token_count(text: str) -> int:
    """Count words and punctuation as a deterministic tokenizer-free estimate."""
    return len(TOKEN_PATTERN.findall(text))


class MockProvider(BaseProvider):
    name = "mock"

    def __init__(self, model: str = "mock-model") -> None:
        self.model = model

    async def generate(
        self,
        *,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        del temperature, max_tokens
        normalized_prompt = prompt.strip()
        content = f"Mock response: {normalized_prompt}"
        return ProviderResult(
            content=content,
            provider=self.name,
            model=self.model,
            input_tokens=approximate_token_count(normalized_prompt),
            output_tokens=approximate_token_count(content),
        )
