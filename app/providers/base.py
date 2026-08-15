from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

from pydantic import JsonValue


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    temperature: float
    max_tokens: int


@dataclass(frozen=True, slots=True)
class ProviderMetadata:
    name: str
    model: str
    available: bool
    input_cost_per_million: float
    output_cost_per_million: float

    @property
    def identity(self) -> str:
        return f"{self.name}:{self.model}"


@dataclass(frozen=True, slots=True)
class ProviderResult:
    content: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


class BaseProvider(ABC):
    metadata: ProviderMetadata

    @abstractmethod
    def effective_parameters(
        self, request: GenerationRequest
    ) -> Mapping[str, JsonValue]:
        """Return only parameters that actually affect this provider request."""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> ProviderResult:
        """Generate text and return a provider-independent result."""

    async def close(self) -> None:
        return None
