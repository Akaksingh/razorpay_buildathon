"""Load test the real HTTP gateway: throughput and tail latency under concurrency.

The in-process benchmark (04) measures the pipeline; this measures the *service*: a
uvicorn process is started, warmed, and hit by N concurrent clients for a fixed
duration at each concurrency level. Server-side compute time is read from the
X-Chakra-Latency-Ms header so queueing is separated from work. A share of requests
commit (graph observer + propensity ledger on the async worker) so the background
path is exercised too.

Clients are threads, one keep-alive connection each, like N independent checkout
pods. An asyncio httpx client was tried first and produced second-long tails on
Windows loopback at 16 connections while a threaded client against the same server
measured p99 under 100 ms -- the collapse was in the load generator, not the gateway.
Measure with the kind of client production will use.

The gateway runs one worker: the scoring pipeline is synchronous CPU work executed in
FastAPI's threadpool, so throughput is bounded by the GIL-serialised share of ~2.6 ms
per order and extra concurrency turns into queueing. Scale out, not up.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx
import numpy as np
import pandas as pd

from chakrashield.config import DATA_DIR, LATENCY_BUDGET_MS, REPORT_DIR


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else 0.0


def start_server(port: int) -> subprocess.Popen:
    env = {**os.environ, "PYTHONUTF8": "1"}
    log = os.environ.get("CHAKRA_LOAD_LOG")        # capture the gateway's stdout/stderr for post-mortems
    sink = open(log, "w", encoding="utf-8") if log else subprocess.DEVNULL
    proc = subprocess.Popen([sys.executable, "scripts/serve.py", "--port", str(port)], env=env,
                            stdout=sink, stderr=subprocess.STDOUT)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2.0)
            if r.status_code == 200 and r.json().get("ok"):
                return proc
        except Exception:
            pass
        time.sleep(0.5)
    stop_server(proc)
    raise RuntimeError("gateway did not come up")


def stop_server(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        pass


def request_bodies(n: int) -> list[dict]:
    from chakrashield.serving.app import build_request_from_row

    orders = pd.read_pickle(DATA_DIR / "orders.pkl")
    cod = orders[orders.payment_method == "COD"].tail(20_000)
    rows = cod.sample(n=n, random_state=3).to_dict("records")
    return [build_request_from_row(r).model_dump(exclude_none=True) for r in rows]


def run_level(base: str, bodies: list[dict], concurrency: int, duration: float, commit_share: float, explain: str = "auto") -> dict:
    lat, server_ms, errors = [], [], Counter()
    attempts = 0
    lock = threading.Lock()
    every = max(1, int(round(1 / commit_share))) if commit_share > 0 else 0
    ready, go = threading.Barrier(concurrency + 1), threading.Barrier(concurrency + 1)
    t_first = [None]
    clock = [0.0]

    def worker(wid: int) -> None:
        nonlocal attempts
        i = wid
        # verify=False skips building an SSL context per client (~100 ms each on Windows; 64 of them
        # would eat the window); the clock starts only after every client is constructed.
        with httpx.Client(base_url=base, timeout=60.0, verify=False) as client:
            ready.wait()
            go.wait()
            t_start = clock[0]
            stop = t_start + duration
            while time.perf_counter() < stop:
                body = bodies[i % len(bodies)]
                commit = every and (i % every == 0)
                t0 = time.perf_counter()
                try:
                    r = client.post(f"/v1/risk/evaluate?commit={'true' if commit else 'false'}&explain={explain}", json=body)
                    dt = (time.perf_counter() - t0) * 1e3
                    with lock:
                        attempts += 1
                        if r.status_code == 200:
                            lat.append(dt)
                            h = r.headers.get("x-chakra-latency-ms")
                            if h:
                                server_ms.append(float(h))
                            if t_first[0] is None:
                                t_first[0] = time.perf_counter() - t_start
                        else:
                            errors[f"HTTP {r.status_code}"] += 1
                except Exception as exc:
                    with lock:
                        attempts += 1
                        errors[type(exc).__name__] += 1
                i += concurrency

    threads = [threading.Thread(target=worker, args=(w,), daemon=True) for w in range(concurrency)]
    for t in threads:
        t.start()
    ready.wait()                        # every client has its connection object
    clock[0] = t_start = time.perf_counter()
    go.wait()                           # start the clock together
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t_start
    return {
        "concurrency": concurrency, "duration_s": duration, "elapsed_s": elapsed, "requests": len(lat), "attempts": attempts,
        "errors": dict(errors), "first_completion_s": t_first[0],
        "rps": len(lat) / elapsed,
        "latency_ms": {"p50": pct(lat, 50), "p95": pct(lat, 95), "p99": pct(lat, 99), "max": max(lat) if lat else 0.0,
                       "mean": statistics.mean(lat) if lat else 0.0},
        "server_compute_ms_p50": pct(server_ms, 50), "server_compute_ms_p99": pct(server_ms, 99),
        "queueing_ms_p50": pct(lat, 50) - pct(server_ms, 50),
        "budget_breach_share": float(np.mean([x > LATENCY_BUDGET_MS for x in lat])) if lat else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--concurrency", default="1,4,16,64")
    ap.add_argument("--commit-share", type=float, default=0.2)
    ap.add_argument("--bodies", type=int, default=2000)
    ap.add_argument("--explain", default="auto", choices=["auto", "always", "never"])
    a = ap.parse_args()
    levels = [int(x) for x in a.concurrency.split(",")]
    bodies = request_bodies(a.bodies)
    print(f"[load] {len(bodies)} request bodies | {a.duration:.0f}s per level | commit share {a.commit_share:.0%} | explain={a.explain} | threaded keep-alive clients")
    proc = start_server(a.port)
    base = f"http://127.0.0.1:{a.port}"
    try:
        run_level(base, bodies, 4, 3.0, a.commit_share, a.explain)          # warm-up
        results = []
        for c in levels:
            r = run_level(base, bodies, c, a.duration, a.commit_share, a.explain)
            results.append(r)
            L = r["latency_ms"]
            print(f"[c={c:3d}] {r['rps']:7.1f} req/s | p50 {L['p50']:7.2f} p95 {L['p95']:7.2f} p99 {L['p99']:7.2f} ms | "
                  f"server compute p50 {r['server_compute_ms_p50']:.2f} ms | queueing p50 {r['queueing_ms_p50']:.2f} ms | "
                  f"errors {sum(r['errors'].values())}{(' ' + json.dumps(r['errors'])) if r['errors'] else ''} | > budget {r['budget_breach_share']:.1%}"
                  + (f" | first completion {r['first_completion_s']:.2f}s" if r["first_completion_s"] is not None else " | NO COMPLETIONS"))
        if proc.poll() is not None:
            raise RuntimeError(f"gateway process exited with code {proc.returncode} during the run")
        h = httpx.get(f"{base}/v1/ledger/stats", timeout=30.0).json()
        out = {"levels": results, "budget_ms": LATENCY_BUDGET_MS, "workers": 1, "explain": a.explain, "client": "threads, one keep-alive connection each",
               "ledger_after": {k: h.get(k) for k in ("decisions", "control_cohort", "outcomes")},
               "note": "single uvicorn worker; the pipeline runs in FastAPI's threadpool, so throughput is bounded by the "
                       "GIL-serialised share of the ~2.6 ms per order and extra concurrency becomes queueing"}
        (REPORT_DIR / "load_test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[done] {REPORT_DIR / 'load_test.json'} | ledger decisions {h.get('decisions')} control {h.get('control_cohort')}")
    finally:
        stop_server(proc)


if __name__ == "__main__":
    main()
