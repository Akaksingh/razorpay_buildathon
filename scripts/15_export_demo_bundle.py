"""Export a static demo bundle: real gateway responses, curated, for an offline/artifact frontend.

Every number in the bundle comes from an actual call into the running gateway (via
``fastapi.testclient.TestClient``, i.e. the real ASGI app, not a hand-rolled shortcut), so the
static console built on top of it shows exactly what the live server would have said for that
order. What it cannot show: free-form edits to an order's own fields (address, GMV, channel...)
re-scored by the model, since that needs the trained booster and the live feature store. The
bundle instead spans every *policy* knob the console lets a viewer change without re-scoring --
merchant and friction budget -- by calling ``/v1/risk/evaluate`` once per (order, merchant,
budget) combination and letting the static frontend look up the exact response instead of
recomputing anything.

    python scripts/15_export_demo_bundle.py --out artifacts/demo_bundle.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from chakrashield.config import DATA_DIR, REPORT_DIR
from chakrashield.schemas import RiskRequest

N_PER_COHORT = 14
FRICTION_BUDGETS = [None, 0.3]          # None = merchant/config default (no request-level budget)
COHORTS = ("legit", "impulse", "ring")


def budget_key(fb) -> str:
    return "default" if fb is None else f"b{int(round(fb * 100))}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/demo_bundle.json")
    a = ap.parse_args()
    t0 = time.time()

    from fastapi.testclient import TestClient
    from chakrashield.serving.app import app

    bundle: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "orders": [], "responses": {},
                    "friction_budgets": [budget_key(b) for b in FRICTION_BUDGETS]}

    with TestClient(app) as client:
        health = client.get("/healthz").json()
        merchants = client.get("/v1/merchants").json()
        report = client.get("/v1/report").json()
        bundle["health"] = health
        bundle["merchants"] = merchants
        bundle["report"] = report
        print(f"[health] scorer={health['scorer_backend']} model={health['model_version']} "
              f"merchants={[m['id'] for m in merchants['merchants']]}")

        scenarios = client.get("/v1/scenarios").json()
        merchant_ids = [m["id"] for m in merchants["merchants"]]

        # ---- pick base orders: every named scenario + a curated pool of real historical orders ----
        orders_df = pd.read_pickle(DATA_DIR / "orders.pkl")
        cod = orders_df[orders_df.payment_method == "COD"].tail(15_000)
        base: list[dict] = []
        for i, sc in enumerate(scenarios):
            # scenario dicts omit any field that just takes its pydantic default (e.g. weight_grams); run
            # them through the same model the server validates against so the embedded req always carries
            # the value that was actually scored, not a partial hand-written dict.
            req = json.loads(RiskRequest(**sc["req"]).model_dump_json())
            base.append({"key": f"scenario_{i}", "kind": "scenario", "name": sc["name"], "tag": sc["tag"],
                        "cohort": None, "truth": None, "req": req})
        for cohort in COHORTS:
            pool = cod[cod.cohort == cohort]
            rows = pool.sample(n=min(N_PER_COHORT, len(pool)), random_state=7).to_dict("records")
            for r in rows:
                req = {"order_id": r["order_id"], "customer_phone": str(r["customer_phone"]), "delivery_pin": str(r["delivery_pin"]),
                       "shipping_address": r["shipping_address"], "cart_gmv": float(r["cart_gmv"]), "items_count": int(r["items_count"]),
                       "weight_grams": float(r["weight_grams"]), "device_fingerprint_hash": r["device_fingerprint"],
                       "payment_method": r["payment_method"],
                       "payment_switch_from": (None if pd.isna(r.get("payment_switch_from")) else r.get("payment_switch_from")),
                       "acquisition_channel": r["acquisition_channel"], "coupon_applied": bool(r["coupon_applied"]),
                       "checkout_seconds": float(r["checkout_seconds"]), "hour_of_day": int(r["hour_of_day"]),
                       "is_new_customer": bool(r["is_new_customer"]), "merchant_margin": float(r["merchant_margin"]), "cac": float(r["cac"])}
                base.append({"key": f"{cohort}_{r['order_id']}", "kind": "historical", "name": None, "tag": None,
                            "cohort": cohort, "truth": "RTO" if r["rto"] else "delivered", "req": req})
        bundle["orders"] = base   # keep req: the static order-summary panel displays the real address/GMV/etc.
        print(f"[orders] {len(base)} base orders ({len(scenarios)} scenarios + {len(base) - len(scenarios)} historical)")

        # ---- one real /v1/risk/evaluate call per (order, merchant, budget); explain=always so every ----
        # ---- response carries reason codes, and the "auto" toggle is simulated client-side by hiding ----
        # ---- them when the decision is ALLOW_COD, which is exactly what explain=auto does server-side. ----
        n_calls = 0
        for b in base:
            for mid in merchant_ids:
                for fb in FRICTION_BUDGETS:
                    body = {**b["req"], "merchant_id": mid}
                    if fb is not None:
                        body["friction_budget"] = fb
                    r = client.post("/v1/risk/evaluate?commit=false&explain=always", json=body)
                    if r.status_code != 200:
                        print(f"  [warn] {b['key']} {mid} {fb}: {r.status_code} {r.text[:200]}")
                        continue
                    bundle["responses"][f"{b['key']}|{mid}|{budget_key(fb)}"] = r.json()
                    n_calls += 1
        print(f"[evaluate] {n_calls} real responses cached ({time.time() - t0:.0f}s so far)")

        # ---- ring visualizer: top rings + every ring's subgraph ----
        rings = client.get("/v1/graph/rings?top=40").json()
        bundle["rings"] = rings
        bundle["subgraphs"] = {}
        for ring in rings["rings"]:
            sg = client.get(f"/v1/graph/subgraph?seed={ring['ring_id']}&max_nodes=140").json()
            bundle["subgraphs"][ring["ring_id"]] = sg
        print(f"[graph] {len(rings['rings'])} rings, {len(bundle['subgraphs'])} subgraphs cached")

        # ---- CE3.0: candidates + a real compiled packet (JSON + printable HTML) for each ----
        cands = client.get("/v1/dispute/candidates?n=40").json()
        bundle["dispute_candidates"] = cands
        bundle["dispute_packets"] = {}
        for c in cands:
            tid = c["transaction_id"]
            packet = client.post("/v1/dispute/ce3-compile", json={"transaction_id": tid}).json()
            html = client.get(f"/v1/dispute/packet/{tid}.html").text
            bundle["dispute_packets"][tid] = {"packet": packet, "html": html}
        print(f"[dispute] {len(cands)} candidates, {sum(1 for v in bundle['dispute_packets'].values() if v['packet']['eligible'])} eligible")

        # ---- model health snapshot: drift monitor + ledger + learner, after the calls above have ----
        # ---- populated the rolling drift window (evaluate_pipeline records it regardless of commit) ----
        bundle["drift_snapshot"] = client.get("/v1/monitor/drift").json()
        bundle["ledger_snapshot"] = client.get("/v1/ledger/stats").json()
        bundle["behaviour_snapshot"] = client.get("/v1/behaviour").json()
        print(f"[health] drift status={bundle['drift_snapshot']['status']} ledger decisions={bundle['ledger_snapshot'].get('decisions')} "
              f"learner observations={bundle['behaviour_snapshot'].get('observations')}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(bundle, separators=(",", ":"), ensure_ascii=False)
    out.write_text(text, encoding="utf-8")
    print(f"[done] {out} — {len(text) / 1e6:.2f} MB, {time.time() - t0:.0f}s total")


if __name__ == "__main__":
    main()
