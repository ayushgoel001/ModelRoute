from tests.conftest import request


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
    assert '"X-Client-ID"' not in response.text
    assert "preferred_provider" not in response.text
    assert "Repeat exact request" in response.text
    assert "Copy response" in response.text
    assert "Request pipeline" in response.text
    assert "Different wording intentionally creates a new cache key" in response.text
    assert "MockProvider demo" in response.text
    assert "Gemini" not in response.text
    assert "OpenAI" not in response.text


def test_homepage_preserves_guided_request_controls_and_cache_flow(app_factory) -> None:
    response = request(app_factory(), "GET", "/")

    assert response.status_code == 200
    assert "One API for routing, protecting, and observing LLM traffic." in response.text
    assert "Configure request" in response.text
    assert "Send request" in response.text
    assert "Inspect result" in response.text
    assert "Repeat for cache" in response.text
    assert "MockProvider demo" in response.text
    assert "Mock only" in response.text
    assert 'id="prompt"' in response.text
    assert 'id="strategy"' in response.text
    assert 'id="temperature"' in response.text
    assert 'id="max-tokens"' in response.text
    assert 'event.ctrlKey || event.metaKey' in response.text
    assert 'id="cache-badge"' in response.text
    assert "Want to see exact-response caching?" in response.text
    assert "Served from Redis exact cache" in response.text
    assert "do not submit confidential, personal, or sensitive information" in response.text


def test_homepage_clears_stale_result_before_fetch_and_keeps_it_clear_on_error(
    app_factory,
) -> None:
    response = request(app_factory(), "GET", "/")
    html = response.text
    reset_function = html.split(
        "function resetResponseForNewRequest()", maxsplit=1
    )[1].split("function renderRequestError", maxsplit=1)[0]
    error_function = html.split("function renderRequestError", maxsplit=1)[1].split(
        "function showNonRequestError", maxsplit=1
    )[0]
    send_function = html.split("async function sendRequest", maxsplit=1)[1].split(
        "async function copyValue", maxsplit=1
    )[0]

    assert "clearResponseFields()" in reset_function
    assert 'result.classList.add("hidden")' in reset_function
    assert 'cacheBadge.classList.add("hidden")' in reset_function
    assert send_function.index("resetResponseForNewRequest()") < send_function.index(
        "fetch(endpoint"
    )
    assert "clearResponseFields()" in error_function
    assert 'result.classList.add("hidden")' in error_function
    assert 'cacheBadge.classList.add("hidden")' in error_function
    assert "renderRequestError(errorMessage(response.status, data))" in send_function
