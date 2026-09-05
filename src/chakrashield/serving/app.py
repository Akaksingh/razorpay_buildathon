"""FastAPI Risk Gateway: synchronous intervention API + async graph observer.

Request path (budget 25 ms, typical ~3 ms on the memory store):
    hydrate (store reads)  ->  ONNX score  ->  isotonic + conformal  ->  resolver  ->  reason codes
The graph is *never* touched on the request path; the background worker
ingests each evaluated order into the syndicate graph and republishes
ring statistics to the store, where the next request picks them up.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..config import CONFORMAL_ALPHA, DATA_DIR, ECONOMICS, LATENCY_BUDGET_MS, MODEL_DIR, REPORT_DIR
from ..dispute.ce3 import TransactionLedger, compile_ce3
from ..features.vectorizer import hydrate
from ..features.velocity import record_order, record_outcome
from ..graph.syndicate import SyndicateGraph
from ..policy.economics import TransactionContext
from ..policy.reason_codes import reason_codes
from ..policy.resolver import DynamicRiskResolver
from ..runtime.scorer import Scorer
from ..schemas import DisputeRequest, DisputeResponse, RiskRequest, RiskResponse
from ..store.feature_store import MemoryStore, get_store

STATIC = Path(__file__).parent / "static"


@dataclass
class State:
    scorer: Scorer | None = None
    store: Any = None
    graph: SyndicateGraph | None = None
    ledger: TransactionLedger | None = None
    clock_ts: float = 0.0
    report: dict = field(default_factory=dict)
    latency: dict = field(default_factory=dict)
    world: dict = field(default_factory=dict)
    queue: asyncio.Queue | None = None
    worker_task: asyncio.Task | None = None
    ingested: int = 0
    outcomes: dict[str, dict] = field(default_factory=dict)   # order_id -> hashes for outcome recording
    started_at: float = 0.0


state = State()


# --------------------------------------------------------------------------- lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.time()
    state.store = get_store()
    snap = DATA_DIR / "store.pkl"
    if isinstance(state.store, MemoryStore) and snap.exists():
        state.clock_ts = state.store.load(snap)
    elif (DATA_DIR / "world.json").exists():
        state.clock_ts = json.loads((DATA_DIR / "world.json").read_text(encoding="utf-8")).get("clock_ts", time.time())
    else:
        state.clock_ts = time.time()
    state.clock_ts = float(os.environ.get("CHAKRA_NOW_TS", state.clock_ts))

    gpath = DATA_DIR / "graph.pkl"
    state.graph = SyndicateGraph.load(gpath, store=state.store) if gpath.exists() else SyndicateGraph(store=state.store)
    if (DATA_DIR / "orders.pkl").exists():
        state.ledger = TransactionLedger.from_frame(pd.read_pickle(DATA_DIR / "orders.pkl"))
    else:
        state.ledger = TransactionLedger()
    state.scorer = Scorer(MODEL_DIR).load()
    for name, attr in (("evaluation.json", "report"), ("latency.json", "latency")):
        p = REPORT_DIR / name
        if p.exists():
            setattr(state, attr, json.loads(p.read_text(encoding="utf-8")))
    if (DATA_DIR / "world.json").exists():
        state.world = json.loads((DATA_DIR / "world.json").read_text(encoding="utf-8"))

    state.queue = asyncio.Queue(maxsize=10_000)
    state.worker_task = asyncio.create_task(_graph_worker())
    state.started_at = time.time()
    # Warm the request path (pydantic validators, first ONNX run on real data, SHAP) so the
    # first live checkout does not pay ~100 ms of lazy initialisation.
    for sc in SCENARIOS[:2]:
        evaluate_pipeline(RiskRequest(**sc["req"]), commit=False)
    print(f"[chakrashield] ready in {time.time() - t:.1f}s | scorer={state.scorer.backend} store={state.store.backend} "
          f"graph={state.graph.stats()} ledger={len(state.ledger)}")
    yield
    state.worker_task.cancel()


async def _graph_worker() -> None:
    """Async Syndicate Graph Observer. Drains evaluated orders off the request path."""
    while True:
        job = await state.queue.get()
        try:
            kind = job.pop("kind")
            if kind == "order":
                record_order(state.store, phone_h=job["hashes"]["phone"], device_h=job["hashes"]["device"],
                             addr_h=job["hashes"]["addr"], pin=job["pin"], ts=job["ts"], gmv=job["gmv"])
                state.graph.ingest(job["order_id"], {k: v for k, v in job["hashes"].items() if v}, gmv=job["gmv"])
                state.outcomes[job["order_id"]] = {"hashes": job["hashes"], "pin": job["pin"]}
                state.ingested += 1
            elif kind == "outcome":
                h = job["hashes"]
                record_outcome(state.store, phone_h=h["phone"], device_h=h["device"], addr_h=h["addr"], pin=job["pin"], rto=job["rto"])
                state.graph.outcome(job["order_id"], job["rto"])
        except Exception as exc:  # never let the observer die
            print(f"[graph-worker] error: {exc!r}")
        finally:
            state.queue.task_done()


app = FastAPI(title="ChakraShield Risk Gateway", version="1.0.0", lifespan=lifespan,
              description="In-line dynamic checkout intervenor & subgraph abuse sentinel")


# --------------------------------------------------------------------------- core pipeline
def evaluate_pipeline(req: RiskRequest, commit: bool = True) -> dict:
    t0 = time.perf_counter()
    hyd = hydrate(req, state.store, now_ts=state.clock_ts)
    t1 = time.perf_counter()
    sc = state.scorer.score(hyd.vector, explain=True)
    t2 = time.perf_counter()
    is_new = req.is_new_customer if req.is_new_customer is not None else hyd.velocity.phone_is_new
    ctx = TransactionContext(
        gmv=req.cart_gmv, merchant_margin=req.merchant_margin if req.merchant_margin is not None else ECONOMICS.default_margin,
        cac=req.cac if req.cac is not None else ECONOMICS.default_cac, p_loss=sc.p_loss, is_new_customer=bool(is_new),
        weight_grams=req.weight_grams, addr_defect=hyd.address.defect_score, logistics_loss=req.logistics_loss,
        holding_cost=req.holding_cost,
    )
    dec = DynamicRiskResolver.resolve_action(ctx, sc.conformal_set)
    codes = reason_codes(sc.contribs, hyd.features)
    t3 = time.perf_counter()
    total = (t3 - t0) * 1e3
    latency = {**{f"hydrate.{k}": round(v, 3) for k, v in hyd.timings_ms.items()},
               **{f"score.{k}": round(v, 3) for k, v in sc.timings_ms.items()},
               "hydrate_total": round((t1 - t0) * 1e3, 3), "score_total": round((t2 - t1) * 1e3, 3),
               "resolve_explain": round((t3 - t2) * 1e3, 3), "total": round(total, 3),
               "budget_ms": LATENCY_BUDGET_MS, "within_budget": total <= LATENCY_BUDGET_MS}
    if commit and state.queue is not None:
        oid = req.order_id or f"live_{int(time.time() * 1000)}"
        try:
            state.queue.put_nowait({"kind": "order", "order_id": oid, "hashes": hyd.hashes, "pin": req.delivery_pin,
                                    "ts": state.clock_ts, "gmv": req.cart_gmv})
        except asyncio.QueueFull:
            pass
    else:
        oid = req.order_id
    conformal = {"alpha": CONFORMAL_ALPHA, "prediction_set": sc.conformal_set, "certainty": dec.certainty,
                 "nonconformity": sc.nonconformity, "quantiles": {"q0": state.scorer.conformal.q0, "q1": state.scorer.conformal.q1}}
    return {
        "order_id": oid, "decision": dec.action, "action_label": dec.ux["label"], "friction_level": dec.ux["friction"],
        "p_loss": round(sc.p_loss, 4), "p_raw": round(sc.p_raw, 4), "tau_star": round(ctx.tau_star, 4), "tau_soft": round(ctx.tau_soft, 4),
        "conformal": conformal, "expected_costs": {k: round(v, 2) for k, v in dec.expected_costs.items()},
        "expected_saving_vs_allow": round(dec.expected_saving_vs_allow, 2), "admissible_actions": dec.admissible,
        "reason_codes": codes, "economics": ctx.as_dict(), "graph": hyd.graph, "address": hyd.address.as_dict(),
        "velocity": hyd.velocity.as_dict(), "features": hyd.features, "hashes": hyd.hashes,
        "rationale": dec.rationale, "latency_ms": latency, "model_version": state.scorer.version,
        "scorer_backend": state.scorer.backend,
    }


def _opt(v):
    """pandas hands back NaN for missing strings; the API contract says None."""
    return None if v is None or (isinstance(v, float) and v != v) else v


def build_request_from_row(r: dict) -> RiskRequest:
    """Turn a historical order row into a live request (bench + demo pickers)."""
    return RiskRequest(
        order_id=r["order_id"], customer_phone=str(r["customer_phone"]), customer_email=_opt(r.get("customer_email")),
        delivery_pin=str(r["delivery_pin"]), shipping_address=r["shipping_address"], cart_gmv=float(r["cart_gmv"]),
        items_count=int(r["items_count"]), weight_grams=float(r["weight_grams"]), device_fingerprint_hash=r["device_fingerprint"],
        ip_hash=_opt(r.get("ip")), vpa=_opt(r.get("vpa")), payment_method=r["payment_method"], payment_switch_from=_opt(r.get("payment_switch_from")),
        acquisition_channel=r["acquisition_channel"], coupon_applied=bool(r["coupon_applied"]),
        checkout_seconds=float(r["checkout_seconds"]), hour_of_day=int(r["hour_of_day"]),
        merchant_margin=float(r.get("merchant_margin", ECONOMICS.default_margin)), cac=float(r.get("cac", ECONOMICS.default_cac)),
    )


# --------------------------------------------------------------------------- routes
@app.post("/v1/risk/evaluate", response_model=RiskResponse, response_model_exclude_none=True)
async def risk_evaluate(req: RiskRequest, commit: bool = Query(True, description="Push to the async graph observer")):
    if state.scorer is None:
        raise HTTPException(503, "scorer not loaded")
    out = evaluate_pipeline(req, commit=commit)
    resp = JSONResponse(content=out)
    resp.headers["X-Chakra-Latency-Ms"] = str(out["latency_ms"]["total"])
    resp.headers["X-Chakra-Model"] = out["model_version"]
    return resp


@app.post("/v1/risk/outcome/{order_id}")
async def risk_outcome(order_id: str, rto: bool):
    """Delivery outcome callback from the 3PL / OMS. Closes the learning loop."""
    meta = state.outcomes.get(order_id)
    if meta is None:
        raise HTTPException(404, "order not seen by the gateway")
    await state.queue.put({"kind": "outcome", "order_id": order_id, "hashes": meta["hashes"], "pin": meta["pin"], "rto": rto})
    return {"order_id": order_id, "rto": rto, "queued": True}


@app.post("/v1/dispute/ce3-compile", response_model=DisputeResponse)
async def ce3_compile(req: DisputeRequest):
    return compile_ce3(state.ledger, req.transaction_id, req.dispute_reason_code, req.dispute_date)


@app.get("/v1/dispute/candidates")
async def ce3_candidates(n: int = 8):
    ids = state.ledger.sample_disputable(n)
    out = []
    for i in ids:
        t = state.ledger.get(i)
        out.append({"transaction_id": i, "amount_inr": t.amount, "date": pd.Timestamp(t.ts, unit="s").strftime("%Y-%m-%d"),
                    "prior_card_txns": len(state.ledger.by_card(t.card_token)) - 1})
    return out


@app.get("/v1/graph/rings")
async def graph_rings(top: int = 25):
    return {"stats": state.graph.stats(), "rings": state.graph.rings(top=top)}


@app.get("/v1/graph/subgraph")
async def graph_subgraph(seed: str, max_nodes: int = 120):
    return state.graph.subgraph(seed, max_nodes=max_nodes)


def _refresh_reports() -> None:
    """Reports are produced by offline scripts that may run after the gateway started."""
    for name, attr in (("evaluation.json", "report"), ("latency.json", "latency")):
        p = REPORT_DIR / name
        if p.exists() and (not getattr(state, attr) or p.stat().st_mtime > state.started_at):
            try:
                setattr(state, attr, json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass  # being written right now; serve the previous copy


@app.get("/v1/report")
async def report():
    _refresh_reports()
    return {"evaluation": state.report, "latency": state.latency, "world": state.world, "alpha": CONFORMAL_ALPHA,
            "economics": ECONOMICS.to_dict(), "model_version": state.scorer.version if state.scorer else None}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi.responses import Response

    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><circle cx="16" cy="16" r="13" fill="none" '
           'stroke="#2a78d6" stroke-width="4"/><circle cx="16" cy="16" r="6" fill="none" stroke="#eb6834" stroke-width="4"/></svg>')
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/v1/scenarios")
async def scenarios():
    return SCENARIOS


@app.get("/v1/orders/sample")
async def orders_sample(n: int = 6, cohort: str | None = None):
    """Real historical COD orders (from the tail of history) for the picker."""
    df = pd.read_pickle(DATA_DIR / "orders.pkl")
    df = df[df.payment_method == "COD"].tail(15_000)
    if cohort:
        df = df[df.cohort == cohort]
    rows = df.sample(n=min(n, len(df)), random_state=int(time.time()) % 10_000).to_dict("records")
    return [{k: (None if pd.isna(v) else v) for k, v in r.items() if k not in ("latent_p_rto",)} for r in rows]


@app.get("/healthz")
async def healthz():
    return {
        "ok": state.scorer is not None, "scorer_backend": state.scorer.backend if state.scorer else None,
        "model_version": state.scorer.version if state.scorer else None, "store_backend": state.store.backend,
        "graph": state.graph.stats() if state.graph else None, "ledger_txns": len(state.ledger) if state.ledger else 0,
        "live_ingested": state.ingested, "queue_depth": state.queue.qsize() if state.queue else 0,
        "clock": pd.Timestamp(state.clock_ts, unit="s").isoformat(), "uptime_s": round(time.time() - state.started_at, 1),
        "latency_budget_ms": LATENCY_BUDGET_MS, "alpha": CONFORMAL_ALPHA,
    }


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


# --------------------------------------------------------------------------- demo scenarios
SCENARIOS = [
    {"name": "Metro, complete address, returning buyer", "tag": "frictionless",
     "req": {"customer_phone": "9876501234", "delivery_pin": "560034", "shipping_address": "Flat 1203, Prestige Lakeside, Koramangala 5th Block, Bengaluru 560034",
             "cart_gmv": 1499, "items_count": 1, "device_fingerprint_hash": "fp_demo_returning_01", "payment_method": "COD",
             "acquisition_channel": "ORGANIC", "checkout_seconds": 95, "hour_of_day": 14, "is_new_customer": False, "merchant_margin": 0.18, "cac": 120}},
    {"name": "Tier-4 PIN, landmark-only address, Meta ad, high basket", "tag": "step-up",
     "req": {"customer_phone": "7012349876", "delivery_pin": "845401", "shipping_address": "Near Hanuman Temple, Ward 4",
             "cart_gmv": 2899, "items_count": 2, "device_fingerprint_hash": "fp_demo_new_02", "payment_method": "COD",
             "acquisition_channel": "META_ADS", "checkout_seconds": 40, "hour_of_day": 23, "is_new_customer": True, "merchant_margin": 0.18, "cac": 540}},
    {"name": "Card failed → switched to COD, coupon, affiliate, 1 am (intent risk, deliverable address)", "tag": "step-up",
     "req": {"customer_phone": "8899001122", "delivery_pin": "226010", "shipping_address": "H.No 14, Vikas Khand 2, Gomti Nagar, near Bus Stand, Lucknow 226010",
             "cart_gmv": 3499, "items_count": 3, "device_fingerprint_hash": "fp_demo_switch_03", "payment_method": "COD",
             "payment_switch_from": "CARD_FAILED", "acquisition_channel": "AFFILIATE", "coupon_applied": True,
             "checkout_seconds": 25, "hour_of_day": 1, "is_new_customer": True, "merchant_margin": 0.18, "cac": 470}},
    {"name": "High-CAC influencer buyer, good address, Tier-2 (CAC-insult guard)", "tag": "allow-flagged",
     "req": {"customer_phone": "9988776655", "delivery_pin": "302017", "shipping_address": "H.No 12-45, Malviya Nagar, near Petrol Pump, Jaipur 302017",
             "cart_gmv": 1899, "items_count": 1, "device_fingerprint_hash": "fp_demo_infl_04", "payment_method": "COD",
             "acquisition_channel": "INFLUENCER", "checkout_seconds": 120, "hour_of_day": 19, "is_new_customer": True, "merchant_margin": 0.22, "cac": 800}},
    {"name": "Junk address, Tier-3, late night", "tag": "prepaid",
     "req": {"customer_phone": "6001122334", "delivery_pin": "813001", "shipping_address": "asdf asdf",
             "cart_gmv": 4999, "items_count": 4, "device_fingerprint_hash": "fp_demo_junk_05", "payment_method": "COD",
             "acquisition_channel": "META_ADS", "checkout_seconds": 12, "hour_of_day": 2, "is_new_customer": True, "merchant_margin": 0.18, "cac": 420}},
]
