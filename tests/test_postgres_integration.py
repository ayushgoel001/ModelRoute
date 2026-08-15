import asyncio
import os
from uuid import uuid4

import pytest
from sqlalchemy import delete, text

from app.api.schemas import ChatCompletionResponse
from app.db.database import Database
from app.db.models import RequestMetric
from app.providers.base import ProviderMetadata
from app.services.metrics import MetricsService

DATABASE_URL = os.getenv("MODELROUTE_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="set MODELROUTE_TEST_DATABASE_URL for real PostgreSQL verification",
)


def completion(
    provider: str,
    model: str,
    *,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    cache_hit: bool = False,
    fallback_used: bool = False,
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        request_id=uuid4(),
        provider=provider,
        model=model,
        content="must never be persisted",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        fallback_used=fallback_used,
    )


def provider(name: str, model: str, price: float) -> ProviderMetadata:
    return ProviderMetadata(
        name=name,
        model=model,
        available=True,
        input_cost_per_million=price,
        output_cost_per_million=price * 2,
    )


def test_real_postgresql_schema_persistence_percentiles_and_aggregates() -> None:
    async def verify() -> None:
        database = Database(DATABASE_URL)
        try:
            await database.initialize()
            async with database.session_factory() as session:
                await session.execute(delete(RequestMetric))
                await session.commit()

            metrics = MetricsService(database.session_factory)
            await metrics.record_success(
                completion(
                    "openai",
                    "openai-test",
                    latency_ms=10,
                    input_tokens=100,
                    output_tokens=20,
                ),
                routing_strategy="fixed",
                provider=provider("openai", "openai-test", 1),
            )
            await metrics.record_success(
                completion(
                    "gemini",
                    "gemini-test",
                    latency_ms=20,
                    input_tokens=80,
                    output_tokens=10,
                    cache_hit=True,
                ),
                routing_strategy="cheapest",
                provider=provider("gemini", "gemini-test", 2),
            )
            await metrics.record_success(
                completion(
                    "gemini",
                    "gemini-test",
                    latency_ms=30,
                    input_tokens=50,
                    output_tokens=5,
                    fallback_used=True,
                ),
                routing_strategy="fastest",
                provider=provider("gemini", "gemini-test", 2),
            )
            await metrics.record_failure(
                request_id=uuid4(),
                routing_strategy="fixed",
                latency_ms=40,
                error_category="ProviderTimeoutError",
            )

            summary = await metrics.summary()
            assert summary["total_requests"] == 4
            assert summary["successful_requests"] == 3
            assert summary["success_rate"] == 0.75
            assert summary["cache_hits"] == 1
            assert summary["cache_hit_rate"] == 0.25
            assert summary["fallback_count"] == 1
            assert (summary["input_tokens"], summary["output_tokens"]) == (230, 35)
            assert summary["p50_latency_ms"] == pytest.approx(25)
            assert summary["p95_latency_ms"] == pytest.approx(38.5)
            assert len(summary["providers"]) == 2
            recent = await metrics.recent()
            assert recent[0]["timestamp"].tzinfo is not None
            assert recent[0]["timestamp"].utcoffset().total_seconds() == 0

            async with database.session_factory() as new_session:
                persisted = await new_session.scalar(
                    text("SELECT count(*) FROM request_metrics")
                )
                columns = (
                    await new_session.execute(
                        text(
                            "SELECT column_name FROM information_schema.columns "
                            "WHERE table_name = 'request_metrics'"
                        )
                    )
                ).scalars().all()
            assert persisted == 4
            assert "prompt" not in columns
            assert "content" not in columns
        finally:
            await database.close()

    asyncio.run(verify())
