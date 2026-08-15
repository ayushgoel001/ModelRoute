from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Any
from uuid import uuid4

import httpx


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for values in milliseconds."""
    if not values:
        return 0.0
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(
    *,
    scenario: str,
    concurrency: int,
    outcomes: list[dict[str, Any]],
    total_elapsed_seconds: float,
) -> dict[str, Any]:
    latencies = [float(outcome["latency_ms"]) for outcome in outcomes]
    successes = sum(bool(outcome["success"]) for outcome in outcomes)
    total = len(outcomes)
    failed = total - successes
    error_categories = Counter(
        outcome["error_category"]
        for outcome in outcomes
        if outcome.get("error_category")
    )
    return {
        "scenario": scenario,
        "concurrency": concurrency,
        "total_requests": total,
        "successful_requests": successes,
        "failed_requests": failed,
        "error_rate": failed / total if total else 0.0,
        "total_elapsed_seconds": total_elapsed_seconds,
        "requests_per_second": total / total_elapsed_seconds
        if total_elapsed_seconds
        else 0.0,
        "average_latency_ms": sum(latencies) / total if total else 0.0,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "unexpected_cache_hits": sum(
            outcome.get("cache_hit") is True for outcome in outcomes
        )
        if scenario == "uncached"
        else 0,
        "unexpected_cache_misses": sum(
            outcome.get("cache_hit") is False for outcome in outcomes
        )
        if scenario == "cached"
        else 0,
        "error_categories": dict(sorted(error_categories.items())),
    }


def _command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip()


def environment_metadata() -> dict[str, Any]:
    docker_kernel = _command_version(
        ["docker", "info", "--format", "{{.KernelVersion}}"]
    )
    return {
        "python_version": platform.python_version(),
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "docker_version": _command_version(["docker", "--version"]),
        "docker_compose_version": _command_version(
            ["docker", "compose", "version"]
        ),
        "docker_engine": _command_version(
            ["docker", "info", "--format", "{{.OperatingSystem}} ({{.OSType}})"]
        ),
        "docker_kernel": docker_kernel,
        "container_runtime": "Docker Desktop with WSL2 Linux engine"
        if "WSL2" in docker_kernel
        else "Docker engine",
    }


def build_document(
    *,
    base_url: str,
    request_count: int,
    concurrency_levels: list[int],
    scenarios: list[str],
    warmup_requests: int,
    simulated_provider_latency_ms: float,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": base_url,
            "requests_per_scenario": request_count,
            "concurrency_levels": concurrency_levels,
            "scenarios": scenarios,
            "warmup_requests": warmup_requests,
            "simulated_provider_latency_ms": simulated_provider_latency_ms,
            "uvicorn_workers": 1,
            "rate_limiter": "enabled and configured as non-binding",
            "postgresql_metrics": "enabled",
            "environment": environment_metadata(),
        },
        "results": results,
    }


def write_document(document: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _payload(prompt: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "strategy": "fixed",
        "temperature": 0.2,
        "max_tokens": 64,
    }


async def _request(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    client_id: str,
) -> dict[str, Any]:
    started_at = perf_counter()
    try:
        response = await client.post(
            "/v1/chat/completions",
            json=payload,
            headers={"X-Client-ID": client_id},
        )
        latency_ms = (perf_counter() - started_at) * 1_000
        cache_hit = None
        if response.is_success:
            try:
                cache_hit = response.json().get("cache_hit")
            except ValueError:
                return {
                    "success": False,
                    "latency_ms": latency_ms,
                    "cache_hit": None,
                    "error_category": "InvalidJSONResponse",
                }
        return {
            "success": response.is_success,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit,
            "error_category": None
            if response.is_success
            else f"HTTP{response.status_code}",
        }
    except httpx.HTTPError as exc:
        return {
            "success": False,
            "latency_ms": (perf_counter() - started_at) * 1_000,
            "cache_hit": None,
            "error_category": type(exc).__name__,
        }


async def run_case(
    *,
    base_url: str,
    request_count: int,
    concurrency: int,
    scenario: str,
    warmup_requests: int,
) -> dict[str, Any]:
    run_id = uuid4().hex
    client_id = f"benchmark-{run_id}"
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    async with httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(30),
        limits=limits,
    ) as client:
        if scenario == "cached":
            measured_payloads = [
                _payload(f"ModelRoute benchmark cached {run_id}")
            ] * request_count
            warmup_payloads = [measured_payloads[0]] * max(warmup_requests, 1)
        else:
            measured_payloads = [
                _payload(f"ModelRoute benchmark uncached {run_id} request {index}")
                for index in range(request_count)
            ]
            warmup_payloads = [
                _payload(f"ModelRoute benchmark warmup {run_id} request {index}")
                for index in range(warmup_requests)
            ]

        for payload in warmup_payloads:
            outcome = await _request(client, payload, client_id)
            if not outcome["success"]:
                raise RuntimeError(
                    f"benchmark warm-up failed: {outcome['error_category']}"
                )

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for payload in measured_payloads:
            queue.put_nowait(payload)
        outcomes: list[dict[str, Any]] = []

        async def worker() -> None:
            while True:
                try:
                    payload = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                outcomes.append(await _request(client, payload, client_id))
                queue.task_done()

        started_at = perf_counter()
        await asyncio.gather(
            *(worker() for _ in range(min(concurrency, request_count)))
        )
        elapsed = perf_counter() - started_at

    return summarize(
        scenario=scenario,
        concurrency=concurrency,
        outcomes=outcomes,
        total_elapsed_seconds=elapsed,
    )


def parse_concurrency(value: str) -> list[int]:
    levels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not levels or any(level < 1 for level in levels):
        raise argparse.ArgumentTypeError("concurrency must contain positive integers")
    return levels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark client-observed ModelRoute request latency."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=parse_concurrency, default=[1, 10, 25, 50])
    parser.add_argument(
        "--scenario", choices=["uncached", "cached", "all"], default="all"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--simulated-provider-latency-ms", type=float, default=50)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/final.json")
    )
    args = parser.parse_args()
    if args.requests < 1:
        parser.error("--requests must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    return args


async def async_main() -> int:
    args = parse_args()
    scenarios = ["uncached", "cached"] if args.scenario == "all" else [args.scenario]
    results: list[dict[str, Any]] = []
    for concurrency in args.concurrency:
        for scenario in scenarios:
            print(
                f"Running {scenario}: requests={args.requests} "
                f"concurrency={concurrency}",
                flush=True,
            )
            result = await run_case(
                base_url=args.base_url.rstrip("/"),
                request_count=args.requests,
                concurrency=concurrency,
                scenario=scenario,
                warmup_requests=args.warmup,
            )
            results.append(result)
            print(
                f"  {result['requests_per_second']:.2f} req/s, "
                f"p50={result['p50_latency_ms']:.2f} ms, "
                f"p95={result['p95_latency_ms']:.2f} ms, "
                f"errors={result['failed_requests']}",
                flush=True,
            )

    document = build_document(
        base_url=args.base_url.rstrip("/"),
        request_count=args.requests,
        concurrency_levels=args.concurrency,
        scenarios=scenarios,
        warmup_requests=args.warmup,
        simulated_provider_latency_ms=args.simulated_provider_latency_ms,
        results=results,
    )
    write_document(document, args.output)
    invalid = any(
        result["failed_requests"]
        or result["unexpected_cache_hits"]
        or result["unexpected_cache_misses"]
        for result in results
    )
    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
