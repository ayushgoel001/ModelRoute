from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.schemas import ChatCompletionRequest, ChatCompletionResponse, HealthResponse
from app.exceptions import (
    AllProvidersFailedError,
    NoEligibleProviderError,
    RateLimiterUnavailableError,
)
from app.services.completion_service import CompletionService
from app.services.rate_limiter import TokenBucketRateLimiter

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


def get_completion_service(request: Request) -> CompletionService:
    return request.app.state.completion_service


def get_rate_limiter(request: Request) -> TokenBucketRateLimiter:
    return request.app.state.rate_limiter


def client_identity(request: Request) -> str:
    supplied_id = request.headers.get("X-Client-ID")
    if supplied_id and supplied_id.strip():
        return f"client:{supplied_id.strip().casefold()}"
    host = request.client.host if request.client else "unknown"
    return f"ip:{host.casefold()}"


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def homepage(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_completion(
    completion_request: ChatCompletionRequest,
    request: Request,
    service: Annotated[CompletionService, Depends(get_completion_service)],
    rate_limiter: Annotated[TokenBucketRateLimiter, Depends(get_rate_limiter)],
) -> ChatCompletionResponse:
    started_at = perf_counter()
    try:
        allowed = await rate_limiter.allow(client_identity(request))
    except RateLimiterUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Rate limiter unavailable",
        ) from exc
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

    try:
        return await service.complete(completion_request, started_at=started_at)
    except (NoEligibleProviderError, AllProvidersFailedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No provider could fulfill the request",
        ) from exc
