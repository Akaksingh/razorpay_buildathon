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

from ..config import (BEHAVIOUR_PATH, CONFORMAL_ALPHA, DATA_DIR, ECONOMICS, EXPLORATION_EPSILON, LATENCY_BUDGET_MS,
                      LEDGER_PATH, MODEL_DIR, REPORT_DIR)
from ..dispute.ce3 import TransactionLedger, compile_ce3
from ..features.vectorizer import hydrate
from ..features.velocity import record_order, record_outcome
from ..graph.syndicate import SyndicateGraph
from ..learning.exploration import control_draw, propensity
from ..learning.ledger import DecisionLedger
from ..learning.response import BehaviourLearner, segment_key
from ..monitoring.drift import ConformalDriftMonitor, DriftBaseline
from ..policy.economics import TransactionContext
from ..policy.reason_codes import reason_codes
from ..policy.resolver import ACTION_UX, ALLOW, PREPAID, STEP_UP, DynamicRiskResolver
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
    drift: ConformalDriftMonitor | None = None
    extras: dict = field(default_factory=dict)          # graph_guard / domain_shift / behaviour / feedback_loop reports
    extras_mtime: dict = field(default_factory=dict)
    decisions: DecisionLedger | None = None             # propensity-logged decision ledger
    learner: BehaviourLearner | None = None             # per-segment buyer-response estimates


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

    conf = state.scorer.conformal
    state.drift = ConformalDriftMonitor(state.store, DriftBaseline.from_reports(state.report, q0=conf.q0, q1=conf.q1))
    state.decisions = DecisionLedger(LEDGER_PATH)
    state.learner = BehaviourLearner(prior=ECONOMICS).load(BEHAVIOUR_PATH)

    state.queue = asyncio.Queue(maxsize=10_000)
    state.worker_task = asyncio.create_task(_graph_worker())
    state.started_at = time.time()
    # Warm the request path (pydantic validators, first ONNX run on real data, SHAP) so the
    # first live checkout does not pay ~100 ms of lazy initialisation.
    for sc in SCENARIOS[:2]:
        evaluate_pipeline(RiskRequest(**sc["req"]), commit=False, explain="always")
    print(f"[chakrashield] ready in {time.time() - t:.1f}s | scorer={state.scorer.backend} store={state.store.backend} "
          f"graph={state.graph.stats()} ledger={len(state.ledger)}")
    yield
    state.worker_task.cancel()
    if state.learner is not None and state.learner.observations:
        state.learner.snapshot(BEHAVIOUR_PATH)


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
                state.outcomes[job["order_id"]] = {"hashes": job["hashes"], "pin": job["pin"],
                                                   **{k: job.get(k) for k in ("served_action", "policy_action", "p_loss", "segment")}}
                state.ingested += 1
            elif kind == "log":
                state.decisions.log_decision(job["rec"])
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
def _shadow_price_for(req: RiskRequest) -> tuple[float, str]:
    """lambda_f for this request: explicit price > budget mapped through the validation frontier > config."""
    if req.friction_shadow_price is not None:
        return float(req.friction_shadow_price), "request"
    if req.friction_budget is not None:
        _refresh_reports()
        fr = (state.report or {}).get("friction_frontier_valid") or []
        if not fr:
            return 0.0, "no_frontier"
        ok = [r for r in fr if r["friction_share"] <= req.friction_budget]
        lam = (min(ok, key=lambda r: r["lambda"]) if ok else max(fr, key=lambda r: r["lambda"]))["lambda"]
        return float(lam), "frontier"
    return float(ECONOMICS.friction_shadow_price), "config"


def evaluate_pipeline(req: RiskRequest, commit: bool = True, explain: str = "auto") -> dict:
    t0 = time.perf_counter()
    hyd = hydrate(req, state.store, now_ts=state.clock_ts)
    t1 = time.perf_counter()
    sc = state.scorer.score(hyd.vector, explain=(explain == "always"))
    t2 = time.perf_counter()
    is_new = req.is_new_customer if req.is_new_customer is not None else hyd.velocity.phone_is_new
    lam, lam_source = _shadow_price_for(req)
    # buyer-response parameters: learned per segment where there is data, the configured prior otherwise
    seg = segment_key(req.acquisition_channel, int(hyd.features["pin_tier"]), req.cart_gmv)
    econ, est = state.learner.economics_for(seg, ECONOMICS) if state.learner is not None else (ECONOMICS, None)
    ctx = TransactionContext(
        gmv=req.cart_gmv, merchant_margin=req.merchant_margin if req.merchant_margin is not None else ECONOMICS.default_margin,
        cac=req.cac if req.cac is not None else ECONOMICS.default_cac, p_loss=sc.p_loss, is_new_customer=bool(is_new),
        weight_grams=req.weight_grams, addr_defect=hyd.address.defect_score, logistics_loss=req.logistics_loss,
        holding_cost=req.holding_cost, friction_shadow_price=lam, econ=econ,
    )
    dec = DynamicRiskResolver.resolve_action(ctx, sc.conformal_set)
    if state.drift is not None:
        try:
            state.drift.record(dec.certainty, sc.p_loss)       # one HINCRBY; label-free drift signal
        except Exception:
            pass
    # epsilon-greedy shadow control band: a deterministic slice of *flagged* live orders ships
    # frictionless anyway, tagged and propensity-logged, so retraining is not survival-biased
    oid = req.order_id or (f"live_{int(time.time() * 1000)}" if commit else None)
    eps = EXPLORATION_EPSILON if commit else 0.0
    served, is_control, draw = dec.action, False, None
    if eps > 0 and dec.action != ALLOW:
        is_control, draw = control_draw(f"{oid}|{hyd.hashes['phone']}", eps)
        if is_control:
            served = ALLOW
    prop = propensity(dec.action, served, eps)
    ux = ACTION_UX[served]
    rationale = (f"CONTROL COHORT (ε={eps:.0%}): resolver chose {dec.action}; served frictionless COD to earn an "
                 f"unbiased delivery label for retraining. " + dec.rationale) if is_control else dec.rationale
    # TreeSHAP is ~2/3 of the request cost. In "auto" mode it runs only when the decision applies
    # friction: an ALLOW needs no defence, a step-up or a block must carry reason codes.
    contribs, bias, shap_ms = sc.contribs, sc.bias, sc.timings_ms.get("treeshap", 0.0)
    if contribs is None and explain == "auto" and dec.action != ALLOW:
        contribs, bias, shap_ms = state.scorer.explain(hyd.vector)
    codes = reason_codes(contribs, hyd.features) if contribs is not None else []
    t3 = time.perf_counter()
    total = (t3 - t0) * 1e3
    latency = {**{f"hydrate.{k}": round(v, 3) for k, v in hyd.timings_ms.items()},
               **{f"score.{k}": round(v, 3) for k, v in sc.timings_ms.items()},
               "score.treeshap": round(shap_ms, 3),
               "hydrate_total": round((t1 - t0) * 1e3, 3), "score_total": round((t2 - t1) * 1e3, 3),
               "resolve_explain": round((t3 - t2) * 1e3, 3), "total": round(total, 3),
               "budget_ms": LATENCY_BUDGET_MS, "within_budget": total <= LATENCY_BUDGET_MS}
    if commit and state.queue is not None:
        try:
            state.queue.put_nowait({"kind": "order", "order_id": oid, "hashes": hyd.hashes, "pin": req.delivery_pin,
                                    "ts": state.clock_ts, "gmv": req.cart_gmv, "served_action": served,
                                    "policy_action": dec.action, "p_loss": round(sc.p_loss, 5), "segment": seg})
            state.queue.put_nowait({"kind": "log", "rec": {
                "ts": time.time(), "order_id": oid, "merchant_id": req.merchant_id, "segment": seg, "p_loss": round(sc.p_loss, 5),
                "p_raw": round(sc.p_raw, 5), "conformal_set": sc.conformal_set, "certainty": dec.certainty,
                "policy_action": dec.action, "served_action": served, "is_control_cohort": is_control,
                "propensity": prop, "epsilon": eps, "draw": draw, "tau_star": round(ctx.tau_star, 5),
                "tau_soft": round(ctx.tau_soft, 5), "shadow_price": lam, "model_version": state.scorer.version,
                "gmv": req.cart_gmv, "merchant_margin": ctx.merchant_margin, "cac": ctx.cac,
                "weight_grams": req.weight_grams, "is_new_customer": bool(is_new), "addr_defect": hyd.address.defect_score,
                "phone_h": hyd.hashes["phone"], "features": [round(float(v), 6) for v in hyd.vector[0]]}})
        except asyncio.QueueFull:
            pass
    conformal = {"alpha": CONFORMAL_ALPHA, "prediction_set": sc.conformal_set, "certainty": dec.certainty,
                 "nonconformity": sc.nonconformity, "quantiles": {"q0": state.scorer.conformal.q0, "q1": state.scorer.conformal.q1}}
    return {
        "order_id": oid, "decision": served, "policy_action": dec.action, "action_label": ux["label"], "friction_level": ux["friction"],
        "exploration": {"is_control_cohort": is_control, "epsilon": eps, "propensity": prop, "policy_action": dec.action},
        "behaviour": {**(est.as_dict() if est else {}), "segment": seg, "applied": bool(est and est.source != "prior")},
        "p_loss": round(sc.p_loss, 4), "p_raw": round(sc.p_raw, 4), "tau_star": round(ctx.tau_star, 4), "tau_soft": round(ctx.tau_soft, 4),
        "conformal": conformal, "expected_costs": {k: round(v, 2) for k, v in dec.expected_costs.items()},
        "expected_saving_vs_allow": round(dec.expected_saving_vs_allow, 2), "admissible_actions": dec.admissible,
        "reason_codes": codes, "economics": ctx.as_dict(), "graph": hyd.graph, "address": hyd.address.as_dict(),
        "velocity": hyd.velocity.as_dict(), "features": hyd.features, "hashes": hyd.hashes,
        "rationale": rationale, "latency_ms": latency, "model_version": state.scorer.version,
        "scorer_backend": state.scorer.backend, "explained": contribs is not None, "explain_mode": explain,
        "elasticity": ctx.elasticity(),
        "friction": {"shadow_price": round(lam, 2), "source": lam_source, "budget": req.friction_budget,
                     "budget_changed_action": dec.budget_changed_action},
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
async def risk_evaluate(req: RiskRequest, commit: bool = Query(True, description="Push to the async graph observer"),
                        explain: str = Query("auto", pattern="^(auto|always|never)$",
                                             description="auto: TreeSHAP only when the decision applies friction (production default); always; never")):
    if state.scorer is None:
        raise HTTPException(503, "scorer not loaded")
    out = evaluate_pipeline(req, commit=commit, explain=explain)
    resp = JSONResponse(content=out)
    resp.headers["X-Chakra-Latency-Ms"] = str(out["latency_ms"]["total"])
    resp.headers["X-Chakra-Model"] = out["model_version"]
    return resp


@app.post("/v1/risk/outcome/{order_id}")
async def risk_outcome(order_id: str, rto: bool,
                       stepup_result: str | None = Query(None, pattern="^(paid|abandoned)$", description="if a deposit was asked"),
                       prepaid_result: str | None = Query(None, pattern="^(paid|abandoned)$", description="if prepaid was mandated")):
    """Delivery / intervention outcome callback from the 3PL and the checkout. Closes both learning loops:
    the entity velocity + graph statistics, and the per-segment buyer-response estimates."""
    meta = state.outcomes.get(order_id)
    if meta is None:
        raise HTTPException(404, "order not seen by the gateway")
    served, seg, p = meta.get("served_action"), meta.get("segment"), meta.get("p_loss")
    learned = None
    if state.learner is not None and seg and p is not None:
        if served == STEP_UP and stepup_result:
            state.learner.observe_stepup(seg, float(p), abandoned=(stepup_result == "abandoned"),
                                         rto=(bool(rto) if stepup_result == "paid" else None))
            learned = "stepup"
        elif served == PREPAID and prepaid_result:
            state.learner.observe_prepaid(seg, float(p), abandoned=(prepaid_result == "abandoned"))
            learned = "prepaid"
        if learned and state.learner.observations % 200 == 0:
            state.learner.snapshot(BEHAVIOUR_PATH)
    shipped = stepup_result != "abandoned" and prepaid_result != "abandoned"
    if shipped:   # an abandoned order never shipped: no delivery outcome to fold into velocity / graph stats
        await state.queue.put({"kind": "outcome", "order_id": order_id, "hashes": meta["hashes"], "pin": meta["pin"], "rto": rto})
    if state.decisions is not None:
        state.decisions.log_outcome(order_id, rto, {"served_action": served, "policy_action": meta.get("policy_action"),
                                                    "segment": seg, "stepup_result": stepup_result, "prepaid_result": prepaid_result})
    return {"order_id": order_id, "rto": rto, "shipped": shipped, "learned": learned, "segment": seg}


@app.get("/v1/behaviour")
async def behaviour():
    """Learned buyer response per segment (delta_s, delta_bad, rho, delta_p) with the prior and global levels."""
    if state.learner is None:
        raise HTTPException(503, "learner not initialised")
    return {**state.learner.summary(), "snapshot": str(BEHAVIOUR_PATH)}


@app.get("/v1/ledger/stats")
async def ledger_stats():
    """Propensity ledger counters: decisions, control-cohort pass-throughs, outcomes, served-action mix."""
    if state.queue is not None:
        await state.queue.join()                  # drain the async writer so counters are current
    return {**(state.decisions.stats() if state.decisions else {}), "epsilon": EXPLORATION_EPSILON}


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
    return {"stats": state.graph.stats(), "rings": state.graph.rings(top=top), "shared_entities": state.graph.shared_entities(top=10)}


@app.get("/v1/monitor/drift")
async def monitor_drift():
    """Label-free drift status from conformal set dynamics over the last hour of 5-minute windows."""
    if state.drift is None:
        raise HTTPException(503, "monitor not initialised")
    return state.drift.snapshot()


@app.get("/v1/graph/subgraph")
async def graph_subgraph(seed: str, max_nodes: int = 120):
    return state.graph.subgraph(seed, max_nodes=max_nodes)


EXTRA_REPORTS = ("graph_guard", "domain_shift", "behaviour", "feedback_loop")


def _refresh_reports() -> None:
    """Reports are produced by offline scripts that may run after the gateway started."""
    for name, attr in (("evaluation.json", "report"), ("latency.json", "latency")):
        p = REPORT_DIR / name
        if p.exists() and (not getattr(state, attr) or p.stat().st_mtime > state.started_at):
            try:
                setattr(state, attr, json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass  # being written right now; serve the previous copy
    for name in EXTRA_REPORTS:
        p = REPORT_DIR / f"{name}.json"
        if p.exists() and (name not in state.extras or p.stat().st_mtime > state.extras_mtime.get(name, 0)):
            try:
                state.extras[name] = json.loads(p.read_text(encoding="utf-8"))
                state.extras_mtime[name] = p.stat().st_mtime
            except json.JSONDecodeError:
                pass


@app.get("/v1/report")
async def report():
    _refresh_reports()
    return {"evaluation": state.report, "latency": state.latency, "world": state.world, "alpha": CONFORMAL_ALPHA,
            "economics": ECONOMICS.to_dict(), "model_version": state.scorer.version if state.scorer else None,
            "extra": state.extras}


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
        "drift_status": state.drift.snapshot()["status"] if state.drift else None,
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
    {"name": "Tier-4 PIN, thin address (house no. + landmark), Meta ad, evening — ambiguous, soft step-up", "tag": "step-up",
     "req": {"customer_phone": "7012349876", "delivery_pin": "845401", "shipping_address": "H.No 7, Ward 4, near Hanuman Temple",
             "cart_gmv": 1799, "items_count": 2, "device_fingerprint_hash": "fp_demo_new_02", "payment_method": "COD",
             "acquisition_channel": "META_ADS", "checkout_seconds": 40, "hour_of_day": 20, "is_new_customer": True, "merchant_margin": 0.18, "cac": 540}},
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
    {"name": "Tier-4 PIN, landmark-only address, Meta ad, high basket, 11 pm — certified RTO, address-driven", "tag": "prepaid-address",
     "req": {"customer_phone": "7012349876", "delivery_pin": "845401", "shipping_address": "Near Hanuman Temple, Ward 4",
             "cart_gmv": 2899, "items_count": 2, "device_fingerprint_hash": "fp_demo_new_02", "payment_method": "COD",
             "acquisition_channel": "META_ADS", "checkout_seconds": 40, "hour_of_day": 23, "is_new_customer": True, "merchant_margin": 0.18, "cac": 540}},
]
