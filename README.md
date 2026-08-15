# ModelRoute

ModelRoute is a modular FastAPI LLM gateway. Phase 2 implements the core request lifecycle: provider routing, bounded resilience, exact Redis caching, and shared token-bucket rate limiting.

## Request lifecycle

```text
Client
  -> FastAPI validation
  -> Redis token bucket
  -> ordered provider routing
  -> candidate-specific exact cache lookup
  -> provider timeout/retry
  -> fallback candidate when needed
  -> cache successful result
  -> normalized response
```

The gateway exposes:

- `GET /health`
- `POST /v1/chat/completions`
- Mock, OpenAI, and Gemini provider adapters
- Fixed, cheapest-estimate, and fastest-recent-latency routing
- Per-attempt timeouts, one configurable retry bound, and provider fallback
- Normalized provider errors and usage metadata
- Redis exact-response caching with SHA-256 identities and TTL
- Redis token-bucket rate limiting with an atomic Lua state transition

## Provider adapters and capabilities

OpenAI uses the official async Python SDK and the Responses API. Gemini uses the official `google-genai` async Interactions API. Provider SDK response objects stay inside their adapters.

ModelRoute exposes a normalized request, while each adapter sends only parameters supported by its provider/model. The current GPT-5.6 Luna adapter uses low reasoning effort and omits `temperature`. The Gemini 3.7 Flash adapter uses low thinking, sends `max_output_tokens`, and deliberately omits deprecated sampling controls such as `temperature`, `top_p`, and `top_k`. Cache keys use these effective provider parameters, so reasoning/thinking settings are represented while ignored inputs do not fragment the cache.

A provider is unavailable when its API key is absent. Mock is used only when explicitly selected with `DEFAULT_PROVIDER=mock`; it is never an implicit production fallback.

## Routing

- `fixed` places the available `DEFAULT_PROVIDER` first, followed by other eligible real providers.
- `cheapest` orders candidates using a deterministic input-token approximation and configured input/output prices. This is a pre-generation estimate, not actual billing.
- `fastest` orders candidates using a process-local EWMA of successful call latency. Providers without history use a deterministic initial latency.

Pricing is configuration metadata and can become stale. The OpenAI defaults reflect the project-specified July 2026 Luna estimate. As of 2026-08-15, Gemini 3.7 Flash Standard uses introductory pricing of $0.75 per 1M input tokens and $3.75 per 1M output tokens. These promotional rates expire December 31, 2026; starting January 1, 2027, the configured defaults should be updated to $1.50 input and $7.50 output per 1M tokens. All rates remain environment-configurable, and tests inject controlled prices rather than depending on commercial pricing.

## Resilience

Each external attempt has `PROVIDER_TIMEOUT_SECONDS`. `PROVIDER_MAX_RETRIES=1` means one initial call plus at most one retry. Only timeouts, rate limits, connection failures, and temporary server failures are retried. Authentication and invalid-request failures are not retried. If a candidate remains unsuccessful, ModelRoute proceeds to the next routed candidate.

Clients receive a clean `503` when no candidate can fulfill a request. Raw SDK exceptions and credentials are not returned.

The project pins `google-genai==2.18.1`. Its asynchronous Interactions API installs a compatibility error hook whose precise timeout, authentication, rate-limit, request, and server exception classes are separate from the public `google.genai.errors` hierarchy. That compatibility import is isolated inside `GeminiProvider` and covered by adapter tests.

## Redis behavior

Cache identities include provider, model, prompt, and effective generation parameters, serialized canonically and hashed with SHA-256. Raw prompts never appear in Redis keys. Cached data contains only the normalized provider result; every gateway request receives a fresh `request_id`, current latency, and `cache_hit=true`.

Cache Redis errors fail open because caching is an optimization. Rate-limit Redis errors fail closed with HTTP `503` because shared enforcement cannot be trusted.

The rate limiter uses `X-Client-ID` when supplied, otherwise the request client address. Identities are normalized and SHA-256 hashed before use in Redis keys. Each accepted request consumes one token, tokens refill continuously, and exhausted buckets return HTTP `429`. The Lua script atomically refills, consumes, persists state, and applies an abandonment TTL.

## Install

Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

If PowerShell activation is restricted, invoke `.\.venv\Scripts\python.exe` directly.

## Configure

Copy `.env.example` to `.env`, then set provider keys locally:

```dotenv
OPENAI_API_KEY=your-openai-key
GEMINI_API_KEY=your-gemini-key
```

Never commit `.env`. To develop without paid providers, use `DEFAULT_PROVIDER=mock`. Redis is still required because rate limiting fails closed by design.

Important settings include:

```dotenv
DEFAULT_PROVIDER=openai
PROVIDER_TIMEOUT_SECONDS=20
PROVIDER_MAX_RETRIES=1
PROVIDER_RETRY_DELAY_SECONDS=0.25
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SECONDS=300
RATE_LIMIT_CAPACITY=20
RATE_LIMIT_REFILL_RATE=1.0
```

## Run

Start Redis separately, then run:

```powershell
uvicorn app.main:app --reload
```

API documentation is available at `http://127.0.0.1:8000/docs`.

Example request:

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
    -Headers @{ "X-Client-ID" = "local-example" } `
    -ContentType "application/json" `
    -Body $body
```

## Test

Ordinary tests mock provider and Redis boundaries and make no paid or external calls:

```powershell
python -m pytest
```

Live provider smoke checks and real Redis/Lua verification are separate manual verification steps.

## Current limitations

- PostgreSQL metrics and persistent observability are not implemented.
- The metrics dashboard is not implemented.
- The final Dockerfile and Docker Compose environment are not implemented.
- Benchmarking and load generation are not implemented.
- Fastest-routing latency history is process-local and needs shared aggregation when horizontally scaled.
- Configured pricing metadata can become stale and must be maintained.
- Pre-generation routing cost is only an estimate.
- Streaming, authentication, semantic caching, and circuit breakers are outside Phase 2.
