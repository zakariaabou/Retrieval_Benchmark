from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


async def worker(
    client: httpx.AsyncClient, url: str, duration: float, latencies: list[float]
) -> None:
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        started = time.perf_counter()
        response = await client.post(
            f"{url}/search", json={"query": "virtual environments", "top_k": 10}
        )
        response.raise_for_status()
        latencies.append((time.perf_counter() - started) * 1000)


async def main(url: str, concurrency: int, duration: float) -> None:
    latencies: list[float] = []
    async with httpx.AsyncClient(timeout=60) as client:
        await asyncio.gather(
            *(worker(client, url, duration, latencies) for _ in range(concurrency))
        )
    ordered = sorted(latencies)
    percentile = lambda p: ordered[min(len(ordered) - 1, int(len(ordered) * p))]  # noqa: E731
    print(
        {
            "requests": len(ordered),
            "qps": len(ordered) / duration,
            "p50_ms": statistics.median(ordered),
            "p95_ms": percentile(0.95),
            "p99_ms": percentile(0.99),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, choices=[1, 4, 8], default=1)
    parser.add_argument("--duration", type=float, default=60)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.concurrency, args.duration))
