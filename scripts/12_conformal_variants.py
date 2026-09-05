"""Conformal alpha and its conditioning, treated as business knobs rather than statistics.

The served engine runs a class-conditional split-conformal layer at alpha = 0.10 and lets the set
gate the resolver: {0} forces ALLOW, {1} forbids it, {0,1} hands the order to the expected-cost
argmin. alpha therefore does two things at once -- it is the coverage promise a merchant is given,
and it decides how many orders the rupee argmin is allowed to touch. This script sweeps both:

    alpha        in {0.05, 0.10, 0.15, 0.20, 0.30}
    conditioning in {marginal (one quantile), class (served), class x PIN tier}

Every variant is fitted on the conf split with the served scorer's calibrated probabilities and
evaluated on the chronological test split: coverage per class, coverage per class inside each PIN
tier, the singleton / ambiguous / empty mix, and -- because the gate changes decisions -- the full
resolver's net P&L on identical orders with identical true outcomes. The alpha a merchant would
pick is chosen on VALID (the same split 02 uses for model selection); the test argmax is reported
alongside so the two can be compared, never substituted.

Two references frame the sweep: alpha -> 0 turns every set into {0,1} and leaves the ungated
argmin; the served configuration is the (class, 0.10) cell.
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

from chakrashield.config import CONFORMAL_ALPHA, DATA_DIR, ECONOMICS, MODEL_DIR, REPORT_DIR
from chakrashield.features.vectorizer import FEATURE_NAMES
from chakrashield.models.conformal import (ConformalCalibrator, GroupConformalCalibrator, MarginalConformalCalibrator,
                                           coverage_by_group, set_metrics)
from chakrashield.policy.economics import TransactionContext
from chakrashield.policy.resolver import ALLOW, PREPAID, STEP_UP, DynamicRiskResolver
from chakrashield.policy.simulation import BehaviourSim, simulate
from chakrashield.runtime.scorer import Scorer

META_COLS = ["cart_gmv", "merchant_margin", "cac", "is_new_customer", "weight_grams", "addr_defect_score"]
ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.30)
VARIANTS = ("marginal", "class", "class_x_tier")
SERVED = ("class", CONFORMAL_ALPHA)
#: a (tier, class) cell counts as under-covered when it misses 1 - alpha by more than this
UNDERCOVER_TOL = 0.02


def contexts(meta: pd.DataFrame, p: np.ndarray) -> list[TransactionContext]:
    return [TransactionContext(gmv=float(g), merchant_margin=float(m), cac=float(c), p_loss=float(pp),
                               is_new_customer=bool(nw), weight_grams=float(w), addr_defect=float(ad), econ=ECONOMICS)
            for g, m, c, nw, w, ad, pp in zip(meta.cart_gmv, meta.merchant_margin, meta.cac, meta.is_new_customer,
                                             meta.weight_grams, meta.addr_defect_score, p)]


def fit_variant(name: str, p: np.ndarray, y: np.ndarray, tiers: np.ndarray, alpha: float):
    if name == "marginal":
        return MarginalConformalCalibrator.fit(p, y, alpha)
    if name == "class":
        return ConformalCalibrator.fit(p, y, alpha)
    return GroupConformalCalibrator.fit(p, y, tiers, alpha)


def sets_of(cal, p: np.ndarray, tiers: np.ndarray) -> list[list[int]]:
    return cal.predict_set(p, tiers) if isinstance(cal, GroupConformalCalibrator) else cal.predict_set(p)


def resolver_pnl(sets: list[list[int]], y: np.ndarray, ctxs: list[TransactionContext], sim: BehaviourSim) -> dict:
    acts = np.array([DynamicRiskResolver.resolve_action(c, s).action for c, s in zip(ctxs, sets)], dtype=object)
    r = simulate(acts, y, ctxs, sim)
    return {"pnl_total": r["pnl_total"], "actions": r["actions"], "friction_share": 1.0 - r["action_share"][ALLOW],
            "good_lost": r["good_customers_lost_expected"], "rto_shipped": r["rto_shipped_expected"]}


def undercovered_cells(by_tier: dict[int, dict], alpha: float) -> list[str]:
    out = []
    for tier, m in by_tier.items():
        for c in (0, 1):
            if m[f"n{c}"] and m[f"coverage_class{c}"] < 1 - alpha - UNDERCOVER_TOL:
                out.append(f"tier{tier}/class{c}")
    return out


def main() -> None:
    full = pd.read_pickle(DATA_DIR / "features.pkl")
    cod = full[full.payment_method == "COD"].sort_values("ts").reset_index(drop=True)
    S = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))
    X = cod[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    y = cod["rto"].to_numpy(dtype=int)
    tiers = cod["pin_tier"].to_numpy(dtype=int)
    scorer = Scorer(MODEL_DIR).load()
    sim = BehaviourSim()

    split = {}
    for nm in ("valid", "conf", "test"):
        sl = slice(*S[nm])
        p = scorer.score_batch(X[sl])
        meta = cod[META_COLS].iloc[sl].reset_index(drop=True)
        split[nm] = {"p": p, "y": y[sl], "tiers": tiers[sl], "ctx": contexts(meta, p)}
    conf, test, valid = split["conf"], split["test"], split["valid"]
    print(f"[data] conf {len(conf['y'])} orders (RTO {conf['y'].mean():.1%}) | test {len(test['y'])} (RTO {test['y'].mean():.1%}) | "
          f"test tiers {dict(zip(*np.unique(test['tiers'], return_counts=True)))}")

    ungated = resolver_pnl([[0, 1]] * len(test["y"]), test["y"], test["ctx"], sim)
    allow_all = simulate(np.full(len(test["y"]), ALLOW, dtype=object), test["y"], test["ctx"], sim)["pnl_total"]
    print(f"[reference] ALLOW_ALL ₹{allow_all:,.0f} | ungated argmin (every set {{0,1}}) ₹{ungated['pnl_total']:,.0f}")

    rows = []
    for alpha in ALPHAS:
        for name in VARIANTS:
            cal = fit_variant(name, conf["p"], conf["y"], conf["tiers"], alpha)
            sets_te = sets_of(cal, test["p"], test["tiers"])
            sets_va = sets_of(cal, valid["p"], valid["tiers"])
            m = set_metrics(sets_te, test["y"])
            by_tier = coverage_by_group(sets_te, test["y"], test["tiers"])
            pnl_te = resolver_pnl(sets_te, test["y"], test["ctx"], sim)
            pnl_va = resolver_pnl(sets_va, valid["y"], valid["ctx"], sim)
            cert = pd.Series([len(s) for s in sets_te])
            rows.append({
                "variant": name, "alpha": alpha, "served": (name, alpha) == SERVED,
                **m, "coverage_by_tier": by_tier, "undercovered_cells": undercovered_cells(by_tier, alpha),
                "min_tier_coverage_class1": min(v["coverage_class1"] for v in by_tier.values()),
                "min_tier_coverage_class0": min(v["coverage_class0"] for v in by_tier.values()),
                "rto_rate_in_set": {"certified_low": float(test["y"][[s == [0] for s in sets_te]].mean()) if (cert == 1).any() else None,
                                    "ambiguous": float(test["y"][[s == [0, 1] for s in sets_te]].mean()) if (cert == 2).any() else None,
                                    "certified_high": float(test["y"][[s == [1] for s in sets_te]].mean()) if any(s == [1] for s in sets_te) else None},
                "test": pnl_te, "valid_pnl": pnl_va["pnl_total"],
                "quantiles": ({"q": cal.q} if name == "marginal" else {"q0": cal.q0, "q1": cal.q1}),
            })

    print(f"{'variant':13s} {'α':>5s} {'cov0':>6s} {'cov1':>6s} {'min tier cov1':>13s} {'single':>7s} {'ambig':>6s} {'empty':>6s} "
          f"{'allow/stepup/prepaid':>21s} {'friction':>8s} {'good lost':>9s} {'RTO ship':>8s} {'VALID P&L':>11s} {'TEST P&L':>11s}")
    for r in rows:
        a = r["test"]["actions"]
        print(f"{r['variant']:13s} {r['alpha']:5.2f} {r['coverage_class0']:6.3f} {r['coverage_class1']:6.3f} {r['min_tier_coverage_class1']:13.3f} "
              f"{r['frac_singleton']:7.1%} {r['frac_ambiguous']:6.1%} {r['frac_empty']:6.1%} "
              f"{a[ALLOW]:6d}/{a[STEP_UP]:5d}/{a[PREPAID]:7d} {r['test']['friction_share']:8.1%} {r['test']['good_lost']:9.0f} "
              f"{r['test']['rto_shipped']:8.0f} {r['valid_pnl']:11,.0f} {r['test']['pnl_total']:11,.0f}{'  <- served' if r['served'] else ''}")

    # --- which alpha, chosen how ---------------------------------------------------------------
    served_row = next(r for r in rows if r["served"])
    best_by_variant = {}
    for name in VARIANTS:
        sub = [r for r in rows if r["variant"] == name]
        by_valid = max(sub, key=lambda r: r["valid_pnl"])
        by_test = max(sub, key=lambda r: r["test"]["pnl_total"])
        best_by_variant[name] = {"alpha_chosen_on_valid": by_valid["alpha"], "test_pnl_at_valid_choice": by_valid["test"]["pnl_total"],
                                 "alpha_argmax_on_test": by_test["alpha"], "test_pnl_argmax": by_test["test"]["pnl_total"]}
    overall_valid = max(rows, key=lambda r: r["valid_pnl"])
    overall_test = max(rows, key=lambda r: r["test"]["pnl_total"])
    print(f"[choice] served (class, α={SERVED[1]:.2f}) test ₹{served_row['test']['pnl_total']:,.0f} | chosen on VALID: "
          f"({overall_valid['variant']}, α={overall_valid['alpha']:.2f}) test ₹{overall_valid['test']['pnl_total']:,.0f} | "
          f"test argmax: ({overall_test['variant']}, α={overall_test['alpha']:.2f}) ₹{overall_test['test']['pnl_total']:,.0f}")
    for name, b in best_by_variant.items():
        print(f"         {name:13s} valid picks α={b['alpha_chosen_on_valid']:.2f} (test ₹{b['test_pnl_at_valid_choice']:,.0f}); "
              f"test argmax α={b['alpha_argmax_on_test']:.2f} (₹{b['test_pnl_argmax']:,.0f})")

    # --- does tier conditioning repair per-tier under-coverage? -------------------------------
    print(f"{'α':>5s}  {'class-conditional: tier cov1 (1..4)':>40s}  {'class x tier: tier cov1 (1..4)':>36s}  under-covered cells (class -> class x tier)")
    tier_fix = []
    for alpha in ALPHAS:
        rc = next(r for r in rows if r["variant"] == "class" and r["alpha"] == alpha)
        rt = next(r for r in rows if r["variant"] == "class_x_tier" and r["alpha"] == alpha)
        cov_c = [rc["coverage_by_tier"][t]["coverage_class1"] for t in sorted(rc["coverage_by_tier"])]
        cov_t = [rt["coverage_by_tier"][t]["coverage_class1"] for t in sorted(rt["coverage_by_tier"])]
        tier_fix.append({"alpha": alpha, "class_tier_cov1": cov_c, "class_x_tier_tier_cov1": cov_t,
                         "class_undercovered": rc["undercovered_cells"], "class_x_tier_undercovered": rt["undercovered_cells"],
                         "ambiguous_class": rc["frac_ambiguous"], "ambiguous_class_x_tier": rt["frac_ambiguous"],
                         "pnl_delta_tier_minus_class": rt["test"]["pnl_total"] - rc["test"]["pnl_total"]})
        print(f"{alpha:5.2f}  {'  '.join(f'{v:.3f}' for v in cov_c):>40s}  {'  '.join(f'{v:.3f}' for v in cov_t):>36s}  "
              f"{rc['undercovered_cells'] or '-'} -> {rt['undercovered_cells'] or '-'}")

    report = {
        "alphas": list(ALPHAS), "variants": list(VARIANTS), "served": {"variant": SERVED[0], "alpha": SERVED[1]},
        "undercover_tolerance": UNDERCOVER_TOL, "behaviour_sim": sim.as_dict(),
        "test_orders": int(len(test["y"])), "test_rto_rate": float(test["y"].mean()),
        "test_tier_counts": {int(k): int(v) for k, v in zip(*np.unique(test["tiers"], return_counts=True))},
        "conf_orders": int(len(conf["y"])),
        "conf_tier_cells": {int(t): {"n0": int(((conf["tiers"] == t) & (conf["y"] == 0)).sum()), "n1": int(((conf["tiers"] == t) & (conf["y"] == 1)).sum())}
                            for t in sorted(set(conf["tiers"].tolist()))},
        "references": {"allow_all_pnl": allow_all, "ungated_argmin": ungated},
        "rows": rows, "best_by_variant": best_by_variant,
        "chosen_on_valid": {"variant": overall_valid["variant"], "alpha": overall_valid["alpha"], "test_pnl": overall_valid["test"]["pnl_total"]},
        "argmax_on_test": {"variant": overall_test["variant"], "alpha": overall_test["alpha"], "test_pnl": overall_test["test"]["pnl_total"]},
        "served_test_pnl": served_row["test"]["pnl_total"], "tier_conditioning": tier_fix,
    }
    (REPORT_DIR / "conformal_variants.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] report -> {REPORT_DIR / 'conformal_variants.json'}")


if __name__ == "__main__":
    main()
