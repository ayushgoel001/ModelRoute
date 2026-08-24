from app.config import PUBLIC_GEMINI_DEMO_MODEL
from tests.conftest import FakeProvider, request


def test_homepage_exposes_interactive_demo_and_project_navigation(app_factory) -> None:
    response = request(app_factory(), "GET", "/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "ModelRoute" in response.text
    assert 'href="/dashboard"' in response.text
    assert 'href="/docs"' in response.text
    assert 'href="/health"' in response.text
    assert 'href="https://github.com/ayushgoel001/ModelRoute"' in response.text
    assert 'target="_blank" rel="noopener noreferrer"' in response.text
    assert 'const endpoint = "/v1/chat/completions"' in response.text
    assert '"X-Client-ID": clientId' in response.text
    assert "preferred_provider: providerMode" in response.text
    assert "REPEAT EXACT REQUEST" in response.text
    assert "COPY RESPONSE" in response.text
    assert "REQUEST PIPELINE" in response.text.upper()
    assert "Different wording intentionally produces a different cache key" in response.text
    assert 'const geminiDemoAvailable = false;' in response.text
    assert 'id="gemini-mode"' in response.text
    assert "disabled aria-disabled" in response.text


def test_homepage_gemini_control_reflects_server_configuration(app_factory) -> None:
    application = app_factory(
        [
            FakeProvider("mock"),
            FakeProvider("gemini", model=PUBLIC_GEMINI_DEMO_MODEL),
        ],
        public_gemini_demo_enabled=True,
    )

    response = request(application, "GET", "/")

    assert response.status_code == 200
    assert 'const geminiDemoAvailable = true;' in response.text
    gemini_button = response.text.split('id="gemini-mode"', maxsplit=1)[1].split(
        ">", maxsplit=1
    )[0]
    assert "disabled" not in gemini_button
