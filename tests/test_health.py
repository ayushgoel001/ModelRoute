from tests.conftest import request


def test_health_returns_ok(app_factory) -> None:
    response = request(app_factory(), "GET", "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
