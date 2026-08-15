from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse, HealthResponse
from app.services.completion_service import CompletionService

router = APIRouter()


def get_completion_service(request: Request) -> CompletionService:
    return request.app.state.completion_service


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_completion(
    completion_request: ChatCompletionRequest,
    service: Annotated[CompletionService, Depends(get_completion_service)],
) -> ChatCompletionResponse:
    try:
        return await service.complete(completion_request)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Provider request failed",
        ) from exc
