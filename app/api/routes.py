from pathlib import Path
from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    PreferredProvider,
)
from app.config import PUBLIC_GEMINI_DEMO_MODEL
from app.exceptions import (
    AllProvidersFailedError,
    NoEligibleProviderError,
    PublicGeminiDemoRequestError,
    PublicGeminiDemoUnavailableError,
    PublicGeminiQuotaExceededError,
    PublicGeminiQuotaUnavailableError,
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


def public_gemini_available(request: Request) -> bool:
    settings = request.app.state.settings
    if not settings.public_gemini_demo_enabled:
        return False
    return any(
        provider.metadata.name == "gemini"
        and provider.metadata.model == PUBLIC_GEMINI_DEMO_MODEL
        and provider.metadata.available
        for provider in request.app.state.providers
    )


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def homepage(request: Request) -> HTMLResponse:
    settings = request.app.state.settings
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "gemini_demo_available": public_gemini_available(request),
            "gemini_demo_max_output_tokens": (
                settings.public_gemini_demo_max_output_tokens
            ),
            "gemini_demo_max_prompt_chars": (
                settings.public_gemini_demo_max_prompt_chars
            ),
        },
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
    identity = client_identity(request)
    try:
        allowed = await rate_limiter.allow(identity)
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

    if completion_request.preferred_provider == PreferredProvider.GEMINI:
        if not request.app.state.settings.public_gemini_demo_enabled:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Live Gemini demo is disabled. MockProvider remains available.",
            )
        if not public_gemini_available(request):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Live Gemini demo is unavailable. MockProvider remains available.",
            )

    try:
        return await service.complete(
            completion_request,
            started_at=started_at,
            client_identity=identity,
        )
    except PublicGeminiQuotaExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Live Gemini demo limit reached. MockProvider remains available.",
        ) from exc
    except PublicGeminiDemoRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                "Live Gemini demo prompt exceeds the public size limit. "
                "MockProvider remains available."
            ),
        ) from exc
    except (
        PublicGeminiDemoUnavailableError,
        PublicGeminiQuotaUnavailableError,
    ) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Live Gemini demo is unavailable. MockProvider remains available.",
        ) from exc
    except (NoEligibleProviderError, AllProvidersFailedError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No provider could fulfill the request",
        ) from exc
