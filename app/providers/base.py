from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderResult:
    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


class BaseProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        *,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResult:
        """Generate text and return a provider-independent result."""
