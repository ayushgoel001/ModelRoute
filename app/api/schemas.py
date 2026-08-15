from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator


class Strategy(StrEnum):
    FIXED = "fixed"
    CHEAPEST = "cheapest"
    FASTEST = "fastest"


NonBlankPrompt = Annotated[str, StringConstraints(min_length=1)]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: NonBlankPrompt
    strategy: Strategy = Strategy.FIXED
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)

    @field_validator("prompt")
    @classmethod
    def reject_blank_prompt(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value.strip()


class ChatCompletionResponse(BaseModel):
    request_id: UUID
    provider: str
    model: str
    content: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)


class HealthResponse(BaseModel):
    status: Literal["ok"]
