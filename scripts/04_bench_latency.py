"""Latency benchmark: in-process pipeline and end-to-end HTTP.

Reports p50 / p95 / p99 for each stage so a regression is attributable.
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from chakrashield.config import DATA_DIR, REPORT_DIR, LATENCY_BUDGET_MS
from chakrashield.serving.app import app, state, build_request_from_row, evaluate_pipeline


def pct(xs, q):
    return float(np.percentile(xs, q))


def main(n: int = 3000) -> None:
    from fastapi.testclient import TestClient

    with TestClient(app) as client:          # triggers lifespan: loads models, store, graph
        orders = pd.read_pickle(DATA_DIR / "orders.pkl")
        cod = orders[orders.payment_method == "COD"].tail(20_000)
        rows = cod.sample(n=n, random_state=1).to_dict("records")
        reqs = [build_request_from_row(r) for r in rows]

        stage: dict[str, list[float]] = {}
        total: list[float] = []
        for req in reqs:
            t0 = time.perf_counter()
            resp = evaluate_pipeline(req, commit=False)
            total.append((time.perf_counter() - t0) * 1e3)
            for k, v in resp["latency_ms"].items():
                if k in ("budget_ms", "within_budget"):   # not stage timings
                    continue
                stage.setdefault(k, []).append(float(v))

        http: list[float] = []
        for req in reqs[:1000]:
            body = req.model_dump()
            t0 = time.perf_counter()
            r = client.post("/v1/risk/evaluate", json=body)
            http.append((time.perf_counter() - t0) * 1e3)
            assert r.status_code == 200, r.text

    out = {
        "n_inprocess": n, "n_http": 1000, "budget_ms": LATENCY_BUDGET_MS, "scorer_backend": state.scorer.backend,
        "store_backend": state.store.backend,
        "inprocess_ms": {"p50": pct(total, 50), "p95": pct(total, 95), "p99": pct(total, 99), "max": max(total), "mean": statistics.mean(total)},
        "http_ms": {"p50": pct(http, 50), "p95": pct(http, 95), "p99": pct(http, 99), "max": max(http)},
        "stages_ms_p50": {k: pct(v, 50) for k, v in stage.items()},
        "stages_ms_p99": {k: pct(v, 99) for k, v in stage.items()},
        "budget_breaches_inprocess": int(sum(1 for t in total if t > LATENCY_BUDGET_MS)),
    }
    (REPORT_DIR / "latency.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
