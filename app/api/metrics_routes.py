from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.exceptions import MetricsUnavailableError
from app.services.metrics import MetricsService

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")


def get_metrics_service(request: Request) -> MetricsService:
    return request.app.state.metrics_service


def unavailable(exc: MetricsUnavailableError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Metrics database unavailable",
    )


@router.get("/v1/metrics/summary")
async def metrics_summary(
    metrics: Annotated[MetricsService, Depends(get_metrics_service)],
) -> dict:
    try:
        return await metrics.summary()
    except MetricsUnavailableError as exc:
        raise unavailable(exc) from exc


@router.get("/v1/metrics/recent")
async def recent_metrics(
    metrics: Annotated[MetricsService, Depends(get_metrics_service)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[dict]:
    try:
        return await metrics.recent(limit)
    except MetricsUnavailableError as exc:
        raise unavailable(exc) from exc


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    metrics: Annotated[MetricsService, Depends(get_metrics_service)],
) -> HTMLResponse:
    try:
        summary = await metrics.summary()
        recent = await metrics.recent(20)
    except MetricsUnavailableError as exc:
        raise unavailable(exc) from exc
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"summary": summary, "recent": recent},
    )
