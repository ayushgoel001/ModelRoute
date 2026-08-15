# ModelRoute

ModelRoute is an LLM gateway and observability platform being built in focused phases. Phase 1 provides a unified FastAPI completion endpoint backed by a deterministic, network-free mock provider.

## Phase 1 functionality

- `GET /health`
- `POST /v1/chat/completions`
- Validated prompts, strategies, temperatures, and output-token limits
- An asynchronous provider abstraction and deterministic `MockProvider`
- Normalized responses with request IDs, token estimates, and latency
- Automated API and provider tests

The modular-monolith request flow is:

```text
Client -> FastAPI route -> CompletionService -> MockProvider -> normalized response
```

## Install

Python 3.11 or newer is required. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

If PowerShell execution policy prevents activation, the environment's Python can be invoked directly as `.\.venv\Scripts\python.exe`.

## Run

```powershell
uvicorn app.main:app --reload
```

The API documentation is available at `http://127.0.0.1:8000/docs`.

## Test

```powershell
python -m pytest
```

## Example request

```powershell
$body = @{
    prompt = "Explain binary search"
    strategy = "fixed"
    temperature = 0.2
    max_tokens = 512
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/v1/chat/completions" `
    -ContentType "application/json" `
    -Body $body
```

## Not implemented yet

Phase 1 intentionally excludes real LLM providers, routing policies, timeouts, retries, fallback, Redis caching and rate limiting, PostgreSQL observability, dashboards, benchmarks, authentication, streaming, and Docker. These belong to later phases.
