from tests.conftest import request


def test_homepage_exposes_interactive_demo_and_project_navigation(app_factory) -> None:
    response = request(app_factory(), "GET", "/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "ModelRoute" in response.text
    assert 'href="/dashboard"' in response.text
    assert 'href="/docs"' in response.text
    assert 'href="/health"' in response.text
    assert 'const endpoint = "/v1/chat/completions"' in response.text
    assert '"X-Client-ID": clientId' in response.text
    assert "exact-response caching" in response.text
