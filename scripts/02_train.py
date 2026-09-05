"""Train tempered cost-sensitive candidates, select one by resolver P&L on VALID, calibrate, conformalise, export ONNX.

Candidates differ only in the weight temperature gamma:

    w_i = (C_FN(x_i) if y_i = 1 else C_FP(x_i)) ** gamma      gamma = 0 -> unweighted, 1 -> full rupee weights

Early stopping uses each candidate's cost-weighted validation log-loss. Model *selection* uses the metric
that matters -- net merchant P&L of the full three-action resolver on the validation split (never test):

    baseline_rto := the gamma = 0 candidate (the "accuracy" model, kept for comparison)
    chakra_rto   := the selected candidate (served by the gateway, exported to ONNX)

Splits are chronological (fraud models evaluated on shuffled data lie):
    train 60% | valid 10% (early stop + selection) | calib 10% (isotonic) | conf 10% (conformal) | test 10%
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

from chakrashield.config import CONFORMAL_ALPHA, DATA_DIR, MODEL_DIR, ECONOMICS
from chakrashield.features.vectorizer import FEATURE_NAMES
from chakrashield.models.calibration import IsotonicKnots, expected_calibration_error
from chakrashield.models.conformal import ConformalCalibrator
from chakrashield.models.cost_sensitive_booster import instance_weights, save_booster, train_booster
from chakrashield.models.onnx_export import export_onnx
from chakrashield.policy.economics import TransactionContext
from chakrashield.policy.resolver import DynamicRiskResolver
from chakrashield.policy.simulation import BehaviourSim, simulate

META_COLS = ["cart_gmv", "merchant_margin", "cac", "is_new_customer", "weight_grams", "addr_defect_score"]
GAMMAS = (0.0, 0.5, 1.0)
ARTIFACT_SUFFIXES = (".txt", ".txt.json", ".isotonic.json", ".conformal.json")


def chrono_splits(n: int, fracs=(0.6, 0.1, 0.1, 0.1, 0.1)) -> dict[str, np.ndarray]:
    edges = np.cumsum([0] + [int(round(f * n)) for f in fracs])
    edges[-1] = n
    names = ["train", "valid", "calib", "conf", "test"]
    return {nm: np.arange(edges[i], edges[i + 1]) for i, nm in enumerate(names)}


def contexts(meta: pd.DataFrame, p: np.ndarray) -> list[TransactionContext]:
    return [TransactionContext(gmv=float(g), merchant_margin=float(m), cac=float(c), p_loss=float(pp),
                               is_new_customer=bool(nw), weight_grams=float(w), addr_defect=float(ad), econ=ECONOMICS)
            for g, m, c, nw, w, ad, pp in zip(meta.cart_gmv, meta.merchant_margin, meta.cac, meta.is_new_customer,
                                             meta.weight_grams, meta.addr_defect_score, p)]


def resolver_pnl(booster, iso, conf, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame) -> float:
    """Net merchant P&L of the full resolver (conformal gating + 3-action argmin) on one split."""
    p = np.clip(iso.apply(booster.predict(X)), 1e-3, 1 - 1e-3)
    ctxs = contexts(meta, p)
    acts = np.array([DynamicRiskResolver.resolve_action(c, s).action for c, s in zip(ctxs, conf.predict_set(p))], dtype=object)
    return float(simulate(acts, y, ctxs, BehaviourSim())["pnl_total"])


def fit_candidate(tag: str, gamma: float, X, y, meta, S) -> dict:
    w_tr = instance_weights(meta.iloc[S["train"]], y[S["train"]], gamma=gamma) if gamma > 0 else None
    w_va = instance_weights(meta.iloc[S["valid"]], y[S["valid"]], gamma=gamma) if gamma > 0 else None
    t = time.time()
    res = train_booster(X[S["train"]], y[S["train"]], X[S["valid"]], y[S["valid"]], list(FEATURE_NAMES), w_tr, w_va)

    txt = MODEL_DIR / f"{tag}.txt"
    save_booster(res, txt, list(FEATURE_NAMES), extra={"gamma": gamma})
    booster = lgb.Booster(model_file=str(txt))     # reload the truncated model: this is what serves

    iso = IsotonicKnots.fit(booster.predict(X[S["calib"]]), y[S["calib"]])
    iso.save(MODEL_DIR / f"{tag}.isotonic.json")
    conf = ConformalCalibrator.fit(iso.apply(booster.predict(X[S["conf"]])), y[S["conf"]], CONFORMAL_ALPHA)
    conf.save(MODEL_DIR / f"{tag}.conformal.json")

    valid_pnl = resolver_pnl(booster, iso, conf, X[S["valid"]], y[S["valid"]], meta.iloc[S["valid"]].reset_index(drop=True))

    p_te_raw = booster.predict(X[S["test"]])
    p_te = np.clip(iso.apply(p_te_raw), 1e-3, 1 - 1e-3)
    yt = y[S["test"]]
    metrics = {
        "gamma": gamma, "auc": float(roc_auc_score(yt, p_te)), "pr_auc": float(average_precision_score(yt, p_te)),
        "logloss": float(log_loss(yt, p_te)), "ece_raw": expected_calibration_error(p_te_raw, yt),
        "ece_calibrated": expected_calibration_error(p_te, yt), "brier": float(np.mean((p_te - yt) ** 2)),
        "conformal": conf.evaluate(p_te, yt), "best_iter": res.best_iter, "weighted": gamma > 0,
        "valid_weighted_logloss": res.valid_metric, "valid_pnl_full": valid_pnl,
    }
    imp = booster.feature_importance(importance_type="gain")
    metrics["feature_importance"] = {n: float(v) for n, v in sorted(zip(FEATURE_NAMES, imp), key=lambda kv: -kv[1])}
    print(f"[{tag}] gamma={gamma:.1f} {res.best_iter:4d} trees in {time.time() - t:.1f}s | valid wlogloss {res.valid_metric:.4f} | "
          f"test AUC {metrics['auc']:.4f} PR-AUC {metrics['pr_auc']:.4f} ECE {metrics['ece_raw']:.4f}->{metrics['ece_calibrated']:.4f} | "
          f"VALID resolver P&L {valid_pnl:,.0f}")

    version = hashlib.sha256(txt.read_bytes()).hexdigest()[:12]
    meta_json = json.loads((MODEL_DIR / f"{tag}.txt.json").read_text(encoding="utf-8"))
    meta_json.update({"version": version, "alpha": CONFORMAL_ALPHA,
                      "test_metrics": {k: v for k, v in metrics.items() if k != "feature_importance"}})
    (MODEL_DIR / f"{tag}.txt.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")
    return {"booster": booster, "iso": iso, "conf": conf, "metrics": metrics, "version": version}


def main() -> None:
    full = pd.read_pickle(DATA_DIR / "features.pkl")
    cod = full[full.payment_method == "COD"].sort_values("ts").reset_index(drop=True)
    X = cod[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    y = cod["rto"].to_numpy(dtype=int)
    meta = cod[META_COLS].reset_index(drop=True)
    S = chrono_splits(len(cod))
    print(f"[data] {len(cod):,} COD orders; RTO {y.mean():.1%}; splits " + ", ".join(f"{k}={len(v)}" for k, v in S.items()))

    cands: dict[str, dict] = {}
    for g in GAMMAS:
        tag = f"cand_g{int(round(g * 10)):02d}"
        cands[tag] = fit_candidate(tag, g, X, y, meta, S)

    best_tag = max(cands, key=lambda t: cands[t]["metrics"]["valid_pnl_full"])
    base_tag = next(t for t in cands if cands[t]["metrics"]["gamma"] == 0.0)
    for src, dst in ((base_tag, "baseline_rto"), (best_tag, "chakra_rto")):
        for suf in ARTIFACT_SUFFIXES:
            shutil.copyfile(MODEL_DIR / f"{src}{suf}", MODEL_DIR / f"{dst}{suf}")
    print(f"[select] served chakra_rto := {best_tag} (gamma={cands[best_tag]['metrics']['gamma']}) by VALID resolver P&L; "
          f"baseline_rto := {base_tag}")

    sample = X[S["test"]][:2000]
    rep = export_onnx(cands[best_tag]["booster"], len(FEATURE_NAMES), MODEL_DIR / "chakra_rto.onnx", sample=sample)
    print(f"[onnx] {rep['bytes'] / 1024:.0f} KB, parity max|diff| {rep['parity_max_abs_diff']:.2e}")

    (DATA_DIR / "splits.json").write_text(json.dumps({k: [int(v[0]), int(v[-1]) + 1] for k, v in S.items()}), encoding="utf-8")
    cand_rows = [{"tag": t, "gamma": c["metrics"]["gamma"], "best_iter": c["metrics"]["best_iter"], "auc": c["metrics"]["auc"],
                  "pr_auc": c["metrics"]["pr_auc"], "ece_calibrated": c["metrics"]["ece_calibrated"],
                  "valid_weighted_logloss": c["metrics"]["valid_weighted_logloss"], "valid_pnl_full": c["metrics"]["valid_pnl_full"],
                  "selected": t == best_tag, "version": c["version"]} for t, c in cands.items()]
    summary = {"baseline": cands[base_tag]["metrics"], "chakra": cands[best_tag]["metrics"], "candidates": cand_rows,
               "selected_tag": best_tag, "selected_gamma": cands[best_tag]["metrics"]["gamma"],
               "selection_metric": "net merchant P&L of the full resolver on the VALID split",
               "onnx": rep, "alpha": CONFORMAL_ALPHA, "economics": ECONOMICS.to_dict(), "n_cod": int(len(cod)), "rto_rate": float(y.mean())}
    (MODEL_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] models in {MODEL_DIR}")


if __name__ == "__main__":
    main()
