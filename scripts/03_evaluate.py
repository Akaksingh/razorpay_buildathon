"""Policy evaluation on the chronological test split: rupees, not F1.

Policies compared on identical orders with identical true outcomes:

    ALLOW_ALL            no risk engine (the merchant's status quo)
    BASE@0.5             unweighted booster, block if p > 0.5 (the "accuracy" model)
    BASE@F1              unweighted booster, F1-optimal global threshold
    BASE@GLOBAL_COST     unweighted booster, single global threshold tuned for P&L on valid
    BASE@TAU*(x)         unweighted booster, instance-dependent tau*(x), hard block
    CHAKRA@TAU*(x)       cost-sensitive booster, tau*(x), hard block
    CHAKRA_FULL          cost-sensitive + conformal set + 3-action expected-cost resolver
    ORACLE               perfect foresight (upper bound)

Plus a sensitivity sweep over the buyer-behaviour parameters the resolver
does NOT know, so the ranking is shown to be robust to mis-specification.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):          # Windows consoles default to cp1252; we print ₹ and Δ
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import lightgbm as lgb
from sklearn.metrics import precision_recall_curve

from chakrashield.config import DATA_DIR, MODEL_DIR, REPORT_DIR, ECONOMICS
from chakrashield.features.vectorizer import FEATURE_NAMES
from chakrashield.models.calibration import IsotonicKnots
from chakrashield.models.conformal import ConformalCalibrator
from chakrashield.policy.economics import TransactionContext
from chakrashield.policy.resolver import ALLOW, PREPAID, STEP_UP, DynamicRiskResolver
from chakrashield.policy.simulation import BehaviourSim, order_pnl, simulate

META_COLS = ["cart_gmv", "merchant_margin", "cac", "is_new_customer", "weight_grams", "addr_defect_score"]


def load_model(tag: str):
    b = lgb.Booster(model_file=str(MODEL_DIR / f"{tag}.txt"))
    iso = IsotonicKnots.load(MODEL_DIR / f"{tag}.isotonic.json")
    conf = ConformalCalibrator.load(MODEL_DIR / f"{tag}.conformal.json")
    return b, iso, conf


def contexts(meta: pd.DataFrame, p: np.ndarray) -> list[TransactionContext]:
    return [TransactionContext(gmv=float(g), merchant_margin=float(m), cac=float(c), p_loss=float(pp),
                               is_new_customer=bool(nw), weight_grams=float(w), addr_defect=float(ad), econ=ECONOMICS)
            for g, m, c, nw, w, ad, pp in zip(meta.cart_gmv, meta.merchant_margin, meta.cac, meta.is_new_customer,
                                             meta.weight_grams, meta.addr_defect_score, p)]


def main() -> None:
    full = pd.read_pickle(DATA_DIR / "features.pkl")
    cod = full[full.payment_method == "COD"].sort_values("ts").reset_index(drop=True)
    S = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))
    X = cod[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    y = cod["rto"].to_numpy(dtype=int)
    te = slice(*S["test"])
    va = slice(*S["valid"])
    meta_te, meta_va = cod[META_COLS].iloc[te].reset_index(drop=True), cod[META_COLS].iloc[va].reset_index(drop=True)
    Xte, yte, Xva, yva = X[te], y[te], X[va], y[va]

    bb, biso, bconf = load_model("baseline_rto")
    cb, ciso, cconf = load_model("chakra_rto")
    pb_te = np.clip(biso.apply(bb.predict(Xte)), 1e-3, 1 - 1e-3)
    pc_te = np.clip(ciso.apply(cb.predict(Xte)), 1e-3, 1 - 1e-3)
    pb_va = np.clip(biso.apply(bb.predict(Xva)), 1e-3, 1 - 1e-3)

    ctx_b = contexts(meta_te, pb_te)
    ctx_c = contexts(meta_te, pc_te)
    sim = BehaviourSim()

    # --- global thresholds tuned on VALID (never on test) ------------------
    pr, rc, th = precision_recall_curve(yva, pb_va)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-9)
    thr_f1 = float(th[int(np.argmax(f1))])
    ctx_va = contexts(meta_va, pb_va)
    best_t, best_pnl = 0.5, -np.inf
    for t in np.linspace(0.05, 0.95, 91):
        a = np.where(pb_va > t, PREPAID, ALLOW)
        v = simulate(a, yva, ctx_va, sim)["pnl_total"]
        if v > best_pnl:
            best_t, best_pnl = float(t), v

    # --- policies on TEST ---------------------------------------------------
    tau_b = np.array([c.tau_star for c in ctx_b])
    tau_c = np.array([c.tau_star for c in ctx_c])
    policies = {
        "ALLOW_ALL": np.full(len(yte), ALLOW, dtype=object),
        "BASE@0.5": np.where(pb_te > 0.5, PREPAID, ALLOW),
        "BASE@F1": np.where(pb_te > thr_f1, PREPAID, ALLOW),
        "BASE@GLOBAL_COST": np.where(pb_te > best_t, PREPAID, ALLOW),
        "BASE@TAU*(x)": np.where(pb_te > tau_b, PREPAID, ALLOW),
        "CHAKRA@TAU*(x)": np.where(pc_te > tau_c, PREPAID, ALLOW),
    }
    csets = cconf.predict_set(pc_te)
    full_actions, certainty = [], []
    for c, s in zip(ctx_c, csets):
        d = DynamicRiskResolver.resolve_action(c, s)
        full_actions.append(d.action)
        certainty.append(d.certainty)
    policies["CHAKRA_FULL"] = np.array(full_actions, dtype=object)
    # oracle: perfect label knowledge, best action per order
    oracle = []
    for c, yy in zip(ctx_c, yte):
        if yy == 0:
            oracle.append(ALLOW)
        else:
            oracle.append(max((STEP_UP, PREPAID), key=lambda a: order_pnl(a, 1, c, sim)))
    policies["ORACLE"] = np.array(oracle, dtype=object)

    results = {name: simulate(acts, yte, ctx_c if name.startswith("CHAKRA") or name == "ORACLE" else ctx_b, sim)
               for name, acts in policies.items()}
    base_pnl = results["ALLOW_ALL"]["pnl_total"]
    for name, r in results.items():
        r["delta_vs_allow_all"] = r["pnl_total"] - base_pnl
        r["delta_vs_base_05"] = r["pnl_total"] - results["BASE@0.5"]["pnl_total"]
        r["uplift_pct_vs_allow_all"] = 100 * (r["pnl_total"] - base_pnl) / abs(base_pnl) if base_pnl else 0.0
    print(f"{'policy':18s} {'P&L (₹)':>14s} {'Δ vs ALLOW':>12s} {'Δ vs BASE@0.5':>14s} {'allow':>6s} {'stepup':>7s} {'prepaid':>8s} {'good lost':>10s} {'RTO shipped':>12s}")
    for name, r in results.items():
        a = r["actions"]
        print(f"{name:18s} {r['pnl_total']:14,.0f} {r['delta_vs_allow_all']:12,.0f} {r['delta_vs_base_05']:14,.0f} "
              f"{a[ALLOW]:6d} {a[STEP_UP]:7d} {a[PREPAID]:8d} {r['good_customers_lost_expected']:10.0f} {r['rto_shipped_expected']:12.0f}")

    # --- certainty breakdown for the full policy -----------------------------
    cert = pd.Series(certainty)
    cert_tab = {k: {"n": int((cert == k).sum()), "rto_rate": float(yte[(cert == k).to_numpy()].mean()) if (cert == k).any() else 0.0}
                for k in ["CERTIFIED_LOW", "AMBIGUOUS", "CERTIFIED_HIGH", "NOVEL"]}

    # --- sensitivity sweep over behaviour the resolver does not know ---------
    sweep = []
    for sga in (0.06, 0.11, 0.18, 0.25, 0.35):
        for sba in (0.40, 0.65, 0.85):
            s2 = BehaviourSim(stepup_good_abandon=sga, stepup_bad_abandon=sba)
            row = {"stepup_good_abandon": sga, "stepup_bad_abandon": sba}
            for name in ("ALLOW_ALL", "BASE@0.5", "BASE@GLOBAL_COST", "BASE@TAU*(x)", "CHAKRA@TAU*(x)", "CHAKRA_FULL"):
                ctxs = ctx_c if name.startswith("CHAKRA") else ctx_b
                row[name] = simulate(policies[name], yte, ctxs, s2)["pnl_total"]
            row["full_beats_best_binary"] = row["CHAKRA_FULL"] > max(row["BASE@GLOBAL_COST"], row["BASE@TAU*(x)"], row["CHAKRA@TAU*(x)"])
            sweep.append(row)
    wins = sum(1 for r in sweep if r["full_beats_best_binary"])
    print(f"[sensitivity] CHAKRA_FULL beats best binary policy in {wins}/{len(sweep)} behaviour scenarios")

    # --- per-decile calibration & tau distribution for the console -----------
    deciles = pd.qcut(pc_te, 10, labels=False, duplicates="drop")
    calib_curve = [{"decile": int(d), "p_mean": float(pc_te[deciles == d].mean()), "rto_rate": float(yte[deciles == d].mean()),
                    "n": int((deciles == d).sum())} for d in sorted(set(deciles))]
    tau_hist = np.histogram(tau_c, bins=np.linspace(0, 1, 21))
    p_hist = np.histogram(pc_te, bins=np.linspace(0, 1, 21))

    report = {
        "test_orders": int(len(yte)), "test_rto_rate": float(yte.mean()), "test_gmv": float(meta_te.cart_gmv.sum()),
        "thresholds": {"f1_optimal": thr_f1, "global_cost_optimal": best_t},
        "behaviour_sim": sim.as_dict(), "policies": results, "certainty": cert_tab, "sensitivity": sweep,
        "sensitivity_wins": [wins, len(sweep)], "calibration_curve": calib_curve,
        "tau_star_hist": {"edges": tau_hist[1].tolist(), "counts": tau_hist[0].tolist()},
        "p_loss_hist": {"edges": p_hist[1].tolist(), "counts": p_hist[0].tolist()},
        "conformal_test": cconf.evaluate(pc_te, yte),
        "model_metrics": json.loads((MODEL_DIR / "training_summary.json").read_text(encoding="utf-8")),
    }
    (REPORT_DIR / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] report -> {REPORT_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()
