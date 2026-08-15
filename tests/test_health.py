import asyncio

import httpx

from app.main import app


def get(path: str) -> httpx.Response:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(path)

    return asyncio.run(send_request())


def test_health_returns_ok() -> None:
    response = get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
