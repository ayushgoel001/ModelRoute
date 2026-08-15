import logging
from collections.abc import Callable
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import ChatCompletionResponse
from app.db.models import RequestMetric
from app.exceptions import MetricsUnavailableError
from app.providers.base import ProviderMetadata

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AsyncSession]


class MetricsService:
    def __init__(self, session_factory: SessionFactory) -> None:
        self.session_factory = session_factory

    @staticmethod
    def estimate_cost(
        *,
        input_tokens: int,
        output_tokens: int,
        provider: ProviderMetadata,
        cache_hit: bool,
    ) -> Decimal:
        if cache_hit:
            return Decimal("0")
        input_price = Decimal(str(provider.input_cost_per_million))
        output_price = Decimal(str(provider.output_cost_per_million))
        return (
            Decimal(input_tokens) * input_price
            + Decimal(output_tokens) * output_price
        ) / Decimal(1_000_000)

    async def record_success(
        self,
        response: ChatCompletionResponse,
        *,
        routing_strategy: str,
        provider: ProviderMetadata,
    ) -> bool:
        metric = RequestMetric(
            request_id=response.request_id,
            provider=response.provider,
            model=response.model,
            routing_strategy=routing_strategy,
            status="success",
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            estimated_cost_usd=self.estimate_cost(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                provider=provider,
                cache_hit=response.cache_hit,
            ),
            cache_hit=response.cache_hit,
            fallback_used=response.fallback_used,
            error_category=None,
        )
        return await self._record(metric)

    async def record_failure(
        self,
        *,
        request_id: UUID,
        routing_strategy: str,
        latency_ms: float,
        error_category: str,
    ) -> bool:
        metric = RequestMetric(
            request_id=request_id,
            provider=None,
            model=None,
            routing_strategy=routing_strategy,
            status="error",
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            estimated_cost_usd=Decimal("0"),
            cache_hit=False,
            fallback_used=False,
            error_category=error_category,
        )
        return await self._record(metric)

    async def _record(self, metric: RequestMetric) -> bool:
        try:
            async with self.session_factory() as session:
                session.add(metric)
                await session.commit()
            return True
        except (SQLAlchemyError, OSError) as exc:
            logger.error(
                "Request metric insert failed (%s); completion result is unaffected",
                type(exc).__name__,
            )
            return False

    async def summary(self) -> dict[str, Any]:
        aggregate_statement = select(
            func.count(RequestMetric.id).label("total_requests"),
            func.count(RequestMetric.id)
            .filter(RequestMetric.status == "success")
            .label("successful_requests"),
            func.count(RequestMetric.id)
            .filter(RequestMetric.cache_hit.is_(True))
            .label("cache_hits"),
            func.count(RequestMetric.id)
            .filter(RequestMetric.fallback_used.is_(True))
            .label("fallback_count"),
            func.sum(RequestMetric.input_tokens).label("input_tokens"),
            func.sum(RequestMetric.output_tokens).label("output_tokens"),
            func.sum(RequestMetric.estimated_cost_usd).label("estimated_cost_usd"),
            func.avg(RequestMetric.latency_ms).label("average_latency_ms"),
            func.percentile_cont(0.5)
            .within_group(RequestMetric.latency_ms)
            .label("p50_latency_ms"),
            func.percentile_cont(0.95)
            .within_group(RequestMetric.latency_ms)
            .label("p95_latency_ms"),
        )
        provider_statement = (
            select(
                RequestMetric.provider,
                RequestMetric.model,
                func.count(RequestMetric.id).label("request_count"),
                func.count(RequestMetric.id)
                .filter(RequestMetric.status == "success")
                .label("success_count"),
                func.avg(RequestMetric.latency_ms).label("average_latency_ms"),
                func.sum(RequestMetric.input_tokens).label("input_tokens"),
                func.sum(RequestMetric.output_tokens).label("output_tokens"),
                func.sum(RequestMetric.estimated_cost_usd).label(
                    "estimated_cost_usd"
                ),
            )
            .where(RequestMetric.provider.is_not(None))
            .group_by(RequestMetric.provider, RequestMetric.model)
            .order_by(RequestMetric.provider, RequestMetric.model)
        )
        try:
            async with self.session_factory() as session:
                aggregate = (await session.execute(aggregate_statement)).mappings().one()
                providers = (
                    (await session.execute(provider_statement)).mappings().all()
                )
        except (SQLAlchemyError, OSError) as exc:
            raise MetricsUnavailableError("Metrics database unavailable") from exc

        total = int(aggregate["total_requests"] or 0)
        successful = int(aggregate["successful_requests"] or 0)
        cache_hits = int(aggregate["cache_hits"] or 0)
        return {
            "total_requests": total,
            "successful_requests": successful,
            "success_rate": successful / total if total else 0.0,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / total if total else 0.0,
            "fallback_count": int(aggregate["fallback_count"] or 0),
            "input_tokens": int(aggregate["input_tokens"] or 0),
            "output_tokens": int(aggregate["output_tokens"] or 0),
            "estimated_cost_usd": float(aggregate["estimated_cost_usd"] or 0),
            "average_latency_ms": float(aggregate["average_latency_ms"] or 0),
            "p50_latency_ms": float(aggregate["p50_latency_ms"] or 0),
            "p95_latency_ms": float(aggregate["p95_latency_ms"] or 0),
            "providers": [self._provider_row(row) for row in providers],
        }

    async def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        statement = (
            select(RequestMetric)
            .order_by(RequestMetric.timestamp.desc())
            .limit(min(max(limit, 1), 100))
        )
        try:
            async with self.session_factory() as session:
                metrics = (await session.execute(statement)).scalars().all()
        except (SQLAlchemyError, OSError) as exc:
            raise MetricsUnavailableError("Metrics database unavailable") from exc
        return [
            {
                "request_id": metric.request_id,
                "timestamp": metric.timestamp,
                "provider": metric.provider,
                "model": metric.model,
                "routing_strategy": metric.routing_strategy,
                "status": metric.status,
                "latency_ms": metric.latency_ms,
                "input_tokens": metric.input_tokens,
                "output_tokens": metric.output_tokens,
                "estimated_cost_usd": float(metric.estimated_cost_usd),
                "cache_hit": metric.cache_hit,
                "fallback_used": metric.fallback_used,
                "error_category": metric.error_category,
            }
            for metric in metrics
        ]

    @staticmethod
    def _provider_row(row: Any) -> dict[str, Any]:
        return {
            "provider": row["provider"],
            "model": row["model"],
            "request_count": int(row["request_count"] or 0),
            "success_count": int(row["success_count"] or 0),
            "average_latency_ms": float(row["average_latency_ms"] or 0),
            "input_tokens": int(row["input_tokens"] or 0),
            "output_tokens": int(row["output_tokens"] or 0),
            "estimated_cost_usd": float(row["estimated_cost_usd"] or 0),
        }
