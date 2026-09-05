"""Learning buyer response online: constant prior vs learned per segment vs oracle.

The synthetic world is given a *hidden* truth for how buyers respond to each
intervention, varying by segment (channel group x PIN tier x basket band):
paid-social buyers abandon a deposit prompt far more often than organic ones,
Tier-4 buyers more than metro ones, and so on. The resolver is run three ways
over the last two chronological splits (the conformal-calibration and test
eras, ~7,700 COD orders):

    prior     the configured constants (delta_s = 11%, delta_bad = 65%, rho = 10%, delta_p = 38%) everywhere
    learned   BehaviourLearner updated from the step-up / prepaid outcomes it caused,
              arriving after a 7-day delay, per segment with two-level shrinkage
    oracle    the true segment parameters (unattainable upper bound)

P&L is the exact expectation under the *true* behaviour for the action each
resolver chose. The learner also writes its snapshot so the gateway starts
from what it learned.
"""
from __future__ import annotations

import heapq
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from chakrashield.config import BEHAVIOUR_PATH, DATA_DIR, ECONOMICS, MODEL_DIR, REPORT_DIR
from chakrashield.features.vectorizer import FEATURE_NAMES
from chakrashield.learning.response import BehaviourLearner, segment_key
from chakrashield.policy.economics import TransactionContext
from chakrashield.policy.resolver import PREPAID, STEP_UP, DynamicRiskResolver
from chakrashield.policy.simulation import BehaviourSim, order_pnl
from chakrashield.runtime.scorer import Scorer

DELAY = 7 * 86400.0
CHANNEL_FACTOR = {"ORGANIC": 0.6, "SEARCH": 0.9, "SOCIAL": 1.5}
TIER_ADD = {1: 0.0, 2: 0.01, 3: 0.04, 4: 0.08}
BAND_ADD = {"low": -0.02, "mid": 0.0, "high": 0.04}


def truth(segment: str, base_mult: float = 1.0) -> dict:
    """Hidden buyer response for a segment; the resolver never sees this.

    base_mult scales the world's abandonment relative to the configured prior: 1.0 means the
    prior is right on average (only the segment structure is unknown); 2.2 means the prior is
    wrong by more than 2x, the realistic case for a merchant who has never measured it.
    """
    ch, tier, band = segment.split("|")
    t = int(tier[1:])
    ds = min(0.6, max(0.02, ECONOMICS.stepup_abandon_rate * base_mult * CHANNEL_FACTOR[ch] + TIER_ADD[t] + BAND_ADD[band]))
    db = 0.65 - (0.07 if ch == "SOCIAL" else 0.0) + (0.06 if t >= 3 else 0.0)
    rho = 0.10 + (0.05 if t >= 3 else 0.0)
    dp = min(0.8, max(0.1, ECONOMICS.prepaid_abandon_rate * base_mult * CHANNEL_FACTOR[ch] + TIER_ADD[t]))
    return {"delta_s": ds, "delta_bad": db, "rho": rho, "delta_p": dp}


def run(st: pd.DataFrame, p_all: np.ndarray, sets, y: np.ndarray, base_mult: float, label: str) -> tuple[dict, BehaviourLearner]:
    rng = random.Random(11)
    learner = BehaviourLearner(prior=ECONOMICS)
    pending: list = []
    pnl = {"prior": 0.0, "learned": 0.0, "oracle": 0.0}
    actions = {k: {"ALLOW_COD": 0, "STEP_UP_DEPOSIT": 0, "FORCE_PREPAID": 0} for k in pnl}
    seg_count: dict[str, int] = {}
    trajectory: list[dict] = []
    print(f"[{label}] {len(st):,} COD orders from the conformal + test eras | resolver prior delta_s {ECONOMICS.stepup_abandon_rate:.0%} | world x{base_mult}")

    for i, row in enumerate(st.itertuples(index=False)):
        ts = float(row.ts)
        while pending and pending[0][0] <= ts:                       # delayed outcomes arrive
            _, kind, seg_, p_, ab, rt, ax = heapq.heappop(pending)
            (learner.observe_stepup(seg_, p_, ab, rt, addr_attr=ax) if kind == "stepup" else learner.observe_prepaid(seg_, p_, ab))
        seg = segment_key(row.acquisition_channel, int(row.pin_tier), float(row.cart_gmv))
        seg_count[seg] = seg_count.get(seg, 0) + 1
        tr = truth(seg, base_mult)
        p = float(p_all[i])
        base = dict(gmv=float(row.cart_gmv), merchant_margin=float(row.merchant_margin), cac=float(row.cac), p_loss=p,
                    is_new_customer=bool(row.is_new_customer), weight_grams=float(row.weight_grams), addr_defect=float(row.addr_defect_score))
        ctx_prior = TransactionContext(**base, econ=ECONOMICS)
        econ_l, _ = learner.economics_for(seg, ECONOMICS)
        ctx_learned = TransactionContext(**base, econ=econ_l)
        ctx_oracle = TransactionContext(**base, econ=replace(ECONOMICS, stepup_abandon_rate=tr["delta_s"], stepup_rto_residual=tr["rho"],
                                                              prepaid_abandon_rate=tr["delta_p"]))
        sim_true = BehaviourSim(stepup_good_abandon=tr["delta_s"], stepup_bad_abandon=tr["delta_bad"], stepup_rto_residual=tr["rho"],
                                prepaid_good_abandon=tr["delta_p"])
        chosen = {}
        for k, cx in (("prior", ctx_prior), ("learned", ctx_learned), ("oracle", ctx_oracle)):
            a = DynamicRiskResolver.resolve_action(cx, sets[i]).action
            chosen[k] = a
            actions[k][a] += 1
            pnl[k] += order_pnl(a, int(y[i]), ctx_prior, sim_true)
        # what the learner's own action lets it observe, drawn from the hidden truth
        a = chosen["learned"]
        bad = y[i] == 1
        if a == STEP_UP:
            abandoned = rng.random() < (tr["delta_bad"] if bad else tr["delta_s"])
            ax = ctx_prior.address_attribution
            rho_eff = tr["rho"] + (1 - tr["rho"]) * ax
            rto = None if abandoned else (bad and rng.random() < rho_eff)
            heapq.heappush(pending, (ts + DELAY, "stepup", seg, p, abandoned, rto, ax))
        elif a == PREPAID:
            abandoned = rng.random() < (0.85 if bad else tr["delta_p"])
            heapq.heappush(pending, (ts + DELAY, "prepaid", seg, p, abandoned, None, 0.0))
        if (i + 1) % 250 == 0:
            top = sorted(seg_count, key=seg_count.get, reverse=True)[:4]
            trajectory.append({"orders": i + 1, "pnl": dict(pnl), "estimates": {
                s: {"delta_s": learner.estimate(s).delta_s, "truth": truth(s, base_mult)["delta_s"], "source": learner.estimate(s).source} for s in top}})

    rows = []
    for s, n in sorted(seg_count.items(), key=lambda kv: -kv[1]):
        e, t = learner.estimate(s), truth(s, base_mult)
        rows.append({"segment": s, "orders": n, **e.as_dict(), "true_delta_s": round(t["delta_s"], 4), "true_delta_bad": round(t["delta_bad"], 4),
                     "true_rho": round(t["rho"], 4), "true_delta_p": round(t["delta_p"], 4),
                     "abs_err_delta_s": round(abs(e.delta_s - t["delta_s"]), 4),
                     "prior_err_delta_s": round(abs(ECONOMICS.stepup_abandon_rate - t["delta_s"]), 4)})
    learned_rows = [r for r in rows if r["source"] == "segment"]
    summary = {
        "label": label, "base_mult": base_mult, "orders": int(len(st)), "pnl": pnl, "actions": actions,
        "learned_minus_prior": pnl["learned"] - pnl["prior"], "oracle_minus_prior": pnl["oracle"] - pnl["prior"],
        "recovered_share": (pnl["learned"] - pnl["prior"]) / (pnl["oracle"] - pnl["prior"]) if pnl["oracle"] != pnl["prior"] else None,
        "segments_learned": len(learned_rows), "segments_seen": len(rows), "observations": learner.observations,
        "mean_abs_err_delta_s_learned": float(np.mean([r["abs_err_delta_s"] for r in learned_rows])) if learned_rows else None,
        "mean_abs_err_delta_s_prior": float(np.mean([r["prior_err_delta_s"] for r in learned_rows])) if learned_rows else None,
    }
    rec = summary["recovered_share"]
    print(f"[{label}] P&L prior {pnl['prior']:,.0f} | learned {pnl['learned']:,.0f} | oracle {pnl['oracle']:,.0f} | recovered "
          f"{'n/a' if rec is None else f'{rec:.0%}'} of the gap | {summary['segments_learned']} segments learned from {learner.observations:,} observations "
          f"| mean |delta_s err| prior {summary['mean_abs_err_delta_s_prior']:.3f} -> learned {summary['mean_abs_err_delta_s_learned']:.3f}")
    for r in rows[:4]:
        print(f"    {r['segment']:18s} n={r['orders']:4d} delta_s true {r['true_delta_s']:.3f} learned {r['delta_s']:.3f} ({r['source']}) | "
              f"delta_bad true {r['true_delta_bad']:.2f} learned {r['delta_bad']:.2f} | rho true {r['true_rho']:.2f} learned {r['rho']:.2f}")
    return {"summary": summary, "segments": rows, "trajectory": trajectory}, learner


def main() -> None:
    full = pd.read_pickle(DATA_DIR / "features.pkl")
    cod = full[full.payment_method == "COD"].sort_values("ts").reset_index(drop=True)
    S = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))
    lo, hi = S["conf"][0], S["test"][1]
    st = cod.iloc[lo:hi].reset_index(drop=True)
    scorer = Scorer(MODEL_DIR).load()
    p_all = scorer.score_batch(st[list(FEATURE_NAMES)].to_numpy(dtype=np.float32))
    sets = scorer.conformal.predict_set(p_all)
    y = st["rto"].to_numpy(dtype=int)
    scenarios = {"prior_right_on_average": 1.0, "prior_wrong_by_2x": 2.2}
    out, learners = {}, {}
    for label, mult in scenarios.items():
        out[label], learners[label] = run(st, p_all, sets, y, mult, label)
    (REPORT_DIR / "behaviour.json").write_text(json.dumps({"scenarios": out, "headline": "prior_wrong_by_2x",
                                                            "truth_model": "delta_s = prior x base_mult x channel factor + tier + basket band"}, indent=2), encoding="utf-8")
    learners["prior_right_on_average"].snapshot(BEHAVIOUR_PATH)     # the gateway starts from the world it will actually see
    print(f"[done] {REPORT_DIR / 'behaviour.json'} and {BEHAVIOUR_PATH}")


if __name__ == "__main__":
    main()
