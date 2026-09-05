"""Survivorship bias in the retraining loop, and the control band that fixes it.

Once the engine is live, only orders served frictionless COD earn a delivery
label. A model retrained on those survivors -- on a rolling window, as
production retraining jobs do -- stops seeing the high-risk boundary it is
supposed to police. This script replays that loop on the synthetic world,
where the simulator knows every label, and compares three retraining regimes
cycle by cycle:

    naive    engine without a control band; retrain on survivors, unweighted
    ipw      engine with an epsilon control band; retrain on survivors plus the
             control cohort, each order weighted 1 / propensity (Horvitz-Thompson)
    oracle   retrain on every order's true label (unattainable upper bound)

Every cycle's model is scored on the *next* cycle before that cycle's labels
exist, on all orders (the simulator can), so the numbers are out-of-sample:
AUC, ECE, the calibration gap on the high-risk boundary (true latent RTO
propensity >= 0.5), and the resolver's net P&L. The exploration cost of the
band is reported in rupees.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from chakrashield.config import CONFORMAL_ALPHA, DATA_DIR, ECONOMICS, MODEL_DIR, REPORT_DIR
from chakrashield.features.vectorizer import FEATURE_NAMES
from chakrashield.learning.exploration import control_draw, ipw_weight, propensity
from chakrashield.models.calibration import expected_calibration_error
from chakrashield.models.conformal import ConformalCalibrator
from chakrashield.models.cost_sensitive_booster import PARAMS
from chakrashield.policy.economics import TransactionContext
from chakrashield.policy.resolver import ALLOW, DynamicRiskResolver
from chakrashield.policy.simulation import BehaviourSim, order_pnl, simulate

SWEEP = [(0.02, 50.0), (0.02, 20.0), (0.05, 20.0), (0.10, 10.0)]   # (epsilon, IPW weight cap): band sizing
WARM_FRAC = 0.40          # pre-engine era: everything shipped, every label observed
N_CYCLES = 4              # retraining cycles over the remaining 60%
WINDOW_FRAC = 0.30        # rolling training window (share of all orders), as production jobs use
BOUNDARY = 0.5            # "high-risk boundary": true latent RTO propensity at or above this
META = ["cart_gmv", "merchant_margin", "cac", "is_new_customer", "weight_grams", "addr_defect_score"]


class Model:
    """Booster + weighted isotonic + conformal, trained on (X, y, w)."""

    def __init__(self, X, y, w, rounds: int):
        n_cal = max(200, int(0.2 * len(y)))
        Xtr, ytr, wtr = X[:-n_cal], y[:-n_cal], w[:-n_cal]
        Xca, yca, wca = X[-n_cal:], y[-n_cal:], w[-n_cal:]
        self.booster = lgb.train({**PARAMS, "verbose": -1}, lgb.Dataset(Xtr, label=ytr, weight=wtr), num_boost_round=rounds)
        raw = self.booster.predict(Xca)
        self.iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(raw, yca, sample_weight=wca)
        p = self.predict(Xca)
        self.conf = ConformalCalibrator.fit(p, yca, CONFORMAL_ALPHA)

    def predict(self, X):
        return np.clip(self.iso.predict(self.booster.predict(X)), 1e-3, 1 - 1e-3)


def contexts(meta: pd.DataFrame, p: np.ndarray):
    return [TransactionContext(gmv=float(g), merchant_margin=float(m), cac=float(c), p_loss=float(pp), is_new_customer=bool(nw),
                               weight_grams=float(wt), addr_defect=float(ad), econ=ECONOMICS)
            for g, m, c, nw, wt, ad, pp in zip(meta.cart_gmv, meta.merchant_margin, meta.cac, meta.is_new_customer,
                                               meta.weight_grams, meta.addr_defect_score, p)]


def run_loop(X, y, latent, meta, oids, rounds: int, eps: float, cap: float) -> dict:
    n = len(y)
    sim = BehaviourSim()
    warm_end = int(WARM_FRAC * n)
    edges = np.linspace(warm_end, n, N_CYCLES + 1).astype(int)
    window = int(WINDOW_FRAC * n)

    # labelled pool per regime: index -> (label, weight); the warm era is fully observed for everyone
    pools = {k: {i: (int(y[i]), 1.0) for i in range(warm_end)} for k in ("naive", "ipw", "oracle")}
    models = {k: None for k in pools}
    cycles, exploration_cost = [], 0.0
    print(f"[setup] eps {eps:.0%} cap {cap:.0f} | {n:,} COD orders | warm {warm_end:,} | {N_CYCLES} cycles of {edges[1] - edges[0]:,} | window {window:,} | rounds {rounds}")

    for c in range(N_CYCLES):
        lo, hi = edges[c], edges[c + 1]
        idx = np.arange(lo, hi)
        row = {"cycle": c + 1, "orders": int(hi - lo), "variants": {}}
        for k in pools:
            # (re)train on the rolling window of what this regime has observed
            obs = sorted(i for i in pools[k] if i >= lo - window)
            Xk = X[obs]
            yk = np.array([pools[k][i][0] for i in obs])
            wk = np.array([pools[k][i][1] for i in obs])
            models[k] = Model(Xk, yk, wk, rounds)
            m = models[k]
            # score the coming cycle out-of-sample, on every order (the simulator knows all labels)
            p = m.predict(X[idx])
            sets = m.conf.predict_set(p)
            ctxs = contexts(meta.iloc[idx].reset_index(drop=True), p)
            policy = np.array([DynamicRiskResolver.resolve_action(cx, s).action for cx, s in zip(ctxs, sets)], dtype=object)
            served = policy.copy()
            if k == "ipw":
                for j, i in enumerate(idx):
                    if policy[j] != ALLOW and control_draw(str(oids[i]), eps)[0]:
                        served[j] = ALLOW
                        exploration_cost += order_pnl(policy[j], int(y[i]), ctxs[j], sim) - order_pnl(ALLOW, int(y[i]), ctxs[j], sim)
            hb = latent[idx] >= BOUNDARY
            res = simulate(served, y[idx], ctxs, sim)
            row["variants"][k] = {
                "auc": float(roc_auc_score(y[idx], p)), "ece": expected_calibration_error(p, y[idx]),
                "boundary_n": int(hb.sum()), "boundary_pred": float(p[hb].mean()), "boundary_actual": float(y[idx][hb].mean()),
                "boundary_gap": float(y[idx][hb].mean() - p[hb].mean()), "boundary_ece": expected_calibration_error(p[hb], y[idx][hb], bins=8),
                "pnl": res["pnl_total"], "friction_share": 1.0 - res["action_share"][ALLOW],
                "training_rows": int(len(obs)), "training_rto_rate": float(np.average(yk, weights=wk)),
                "n_control": int(((served == ALLOW) & (policy != ALLOW)).sum()) if k == "ipw" else 0,
            }
            # what this regime gets to observe from the cycle it just served
            for j, i in enumerate(idx):
                if k == "oracle":
                    pools[k][i] = (int(y[i]), 1.0)
                elif served[j] == ALLOW:
                    prop = propensity(policy[j], served[j], eps if k == "ipw" else 0.0)
                    pools[k][i] = (int(y[i]), ipw_weight(ALLOW, prop, cap=cap) if k == "ipw" else 1.0)
        cycles.append(row)
        v = row["variants"]
        print(f"[cycle {c + 1}] " + " | ".join(
            f"{k}: AUC {v[k]['auc']:.3f} ECE {v[k]['ece']:.3f} boundary gap {v[k]['boundary_gap']:+.3f} P&L {v[k]['pnl']:,.0f} (train RTO {v[k]['training_rto_rate']:.1%})"
            for k in ("naive", "ipw", "oracle")))

    last = cycles[-1]["variants"]
    total = {k: sum(cy["variants"][k]["pnl"] for cy in cycles) for k in last}
    worst = {k: {"max_abs_boundary_gap": max(abs(cy["variants"][k]["boundary_gap"]) for cy in cycles),
                 "max_ece": max(cy["variants"][k]["ece"] for cy in cycles),
                 "min_auc": min(cy["variants"][k]["auc"] for cy in cycles)} for k in last}
    summary = {
        "final_cycle": {k: {m: last[k][m] for m in ("auc", "ece", "boundary_gap", "boundary_ece", "pnl", "training_rto_rate")} for k in last},
        "worst_cycle": worst, "cumulative_pnl": total, "ipw_minus_naive_cumulative": total["ipw"] - total["naive"],
        "oracle_minus_naive_cumulative": total["oracle"] - total["naive"],
        "gap_recovered_share": (total["ipw"] - total["naive"]) / (total["oracle"] - total["naive"]) if total["oracle"] != total["naive"] else None,
        "exploration_cost_rupees": exploration_cost, "control_orders": int(sum(cy["variants"]["ipw"]["n_control"] for cy in cycles)),
    }
    print(f"[summary eps {eps:.0%} cap {cap:.0f}] cumulative P&L naive {total['naive']:,.0f} | ipw {total['ipw']:,.0f} | oracle {total['oracle']:,.0f} "
          f"| band cost ₹{exploration_cost:,.0f} on {summary['control_orders']} control orders | worst boundary gap naive {worst['naive']['max_abs_boundary_gap']:.3f} "
          f"ipw {worst['ipw']['max_abs_boundary_gap']:.3f} | worst ECE naive {worst['naive']['max_ece']:.3f} ipw {worst['ipw']['max_ece']:.3f}")
    return {"epsilon": eps, "cap": cap, "cycles": cycles, "summary": summary}


def main() -> None:
    full = pd.read_pickle(DATA_DIR / "features.pkl")
    cod = full[full.payment_method == "COD"].sort_values("ts").reset_index(drop=True)
    X = cod[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    y = cod["rto"].to_numpy(dtype=int)
    latent = cod["latent_p_rto"].to_numpy(dtype=float)
    meta = cod[META].reset_index(drop=True)
    oids = cod["order_id"].to_numpy()
    rounds = int(json.loads((MODEL_DIR / "training_summary.json").read_text(encoding="utf-8"))["chakra"]["best_iter"])
    runs = [run_loop(X, y, latent, meta, oids, rounds, eps, cap) for eps, cap in SWEEP]
    # headline: the *cheapest* band whose IPW regime never loses calibration (worst-cycle ECE <= 0.10) or
    # ranking (worst-cycle AUC >= 0.75); a bigger band buys little and costs real orders
    ok = [r for r in runs if r["summary"]["worst_cycle"]["ipw"]["max_ece"] <= 0.10 and r["summary"]["worst_cycle"]["ipw"]["min_auc"] >= 0.75]
    headline = min(ok, key=lambda r: r["summary"]["exploration_cost_rupees"]) if ok else \
        min(runs, key=lambda r: r["summary"]["worst_cycle"]["ipw"]["max_ece"])
    report = {"warm_frac": WARM_FRAC, "n_cycles": N_CYCLES, "window_frac": WINDOW_FRAC, "boundary": BOUNDARY,
              "sweep": [{"epsilon": r["epsilon"], "cap": r["cap"], **r["summary"]} for r in runs],
              "headline_epsilon": headline["epsilon"], "headline_cap": headline["cap"], "headline": headline}
    (REPORT_DIR / "feedback_loop.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[headline] eps {headline['epsilon']:.0%} cap {headline['cap']:.0f}")
    print(f"[done] {REPORT_DIR / 'feedback_loop.json'}")


if __name__ == "__main__":
    main()
