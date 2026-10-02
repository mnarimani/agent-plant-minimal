"""Prove that a slow plant-model path does not stall /health.

  AGENTPLANT_DEMO=1 uvicorn minimal_api.app:app --port 8765 &
  python tests/test_event_loop_nonblocking.py
"""

from __future__ import annotations

import concurrent.futures
import time
import urllib.request

BASE = "http://127.0.0.1:8765"


def _get(path: str, timeout: float = 5.0) -> tuple[int, str]:
    req = urllib.request.Request(BASE + path)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8", errors="replace")


def test_slow_probe_does_not_block_health():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        slow_f = pool.submit(_get, "/api/v1/plant-model/_probe/slow?seconds=2", 10.0)
        time.sleep(0.15)
        t0 = time.perf_counter()
        status, body = _get("/api/v1/health", timeout=1.5)
        elapsed = time.perf_counter() - t0
        assert status == 200, body
        assert elapsed < 1.0, f"/health took {elapsed:.2f}s while slow probe in flight"
        slow_status, _ = slow_f.result(timeout=12)
        assert slow_status == 200


if __name__ == "__main__":
    test_slow_probe_does_not_block_health()
    print("OK: /health stayed responsive during slow probe")
