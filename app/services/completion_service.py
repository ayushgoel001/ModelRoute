from time import perf_counter
from uuid import uuid4

from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse
from app.providers.base import BaseProvider


class CompletionService:
    def __init__(self, provider: BaseProvider) -> None:
        self.provider = provider

    async def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        started_at = perf_counter()
        result = await self.provider.generate(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        latency_ms = (perf_counter() - started_at) * 1_000

        return ChatCompletionResponse(
            request_id=uuid4(),
            provider=result.provider,
            model=result.model,
            content=result.content,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            latency_ms=latency_ms,
        )
