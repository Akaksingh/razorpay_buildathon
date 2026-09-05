"""Train baseline + cost-sensitive boosters, calibrate, conformalise, export ONNX.

Splits are *chronological* (fraud models evaluated on shuffled data lie):
    train 60% | valid 10% (early stop) | calib 10% (isotonic) | conf 10% (conformal) | test 10%
"""
from __future__ import annotations

import hashlib
import json
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

META_COLS = ["cart_gmv", "merchant_margin", "cac", "is_new_customer", "weight_grams"]


def chrono_splits(n: int, fracs=(0.6, 0.1, 0.1, 0.1, 0.1)) -> dict[str, np.ndarray]:
    edges = np.cumsum([0] + [int(round(f * n)) for f in fracs])
    edges[-1] = n
    names = ["train", "valid", "calib", "conf", "test"]
    return {nm: np.arange(edges[i], edges[i + 1]) for i, nm in enumerate(names)}


def fit_pipeline(tag: str, X, y, meta, S, weighted: bool) -> dict:
    w_tr = instance_weights(meta.iloc[S["train"]], y[S["train"]]) if weighted else None
    w_va = instance_weights(meta.iloc[S["valid"]], y[S["valid"]]) if weighted else None
    t = time.time()
    res = train_booster(X[S["train"]], y[S["train"]], X[S["valid"]], y[S["valid"]], list(FEATURE_NAMES), w_tr, w_va)
    print(f"[{tag}] {res.best_iter} trees in {time.time() - t:.1f}s; valid logloss {res.valid_metric:.4f}")

    txt = MODEL_DIR / f"{tag}.txt"
    save_booster(res, txt, list(FEATURE_NAMES))
    booster = lgb.Booster(model_file=str(txt))     # reload the truncated model: this is what serves

    p_cal_raw = booster.predict(X[S["calib"]])
    iso = IsotonicKnots.fit(p_cal_raw, y[S["calib"]])
    iso.save(MODEL_DIR / f"{tag}.isotonic.json")

    p_conf = iso.apply(booster.predict(X[S["conf"]]))
    conf = ConformalCalibrator.fit(p_conf, y[S["conf"]], CONFORMAL_ALPHA)
    conf.save(MODEL_DIR / f"{tag}.conformal.json")

    p_te_raw = booster.predict(X[S["test"]])
    p_te = np.clip(iso.apply(p_te_raw), 1e-3, 1 - 1e-3)
    yt = y[S["test"]]
    metrics = {
        "auc": float(roc_auc_score(yt, p_te)), "pr_auc": float(average_precision_score(yt, p_te)),
        "logloss": float(log_loss(yt, p_te)), "ece_raw": expected_calibration_error(p_te_raw, yt),
        "ece_calibrated": expected_calibration_error(p_te, yt), "brier": float(np.mean((p_te - yt) ** 2)),
        "conformal": conf.evaluate(p_te, yt), "best_iter": res.best_iter, "weighted": weighted,
    }
    imp = booster.feature_importance(importance_type="gain")
    metrics["feature_importance"] = {n: float(v) for n, v in sorted(zip(FEATURE_NAMES, imp), key=lambda kv: -kv[1])}
    print(f"[{tag}] test AUC {metrics['auc']:.4f} PR-AUC {metrics['pr_auc']:.4f} ECE raw {metrics['ece_raw']:.4f} "
          f"-> cal {metrics['ece_calibrated']:.4f}; conformal cov0 {conf.empirical_cov0:.3f} cov1 {conf.empirical_cov1:.3f}")

    version = hashlib.sha256(txt.read_bytes()).hexdigest()[:12]
    meta_json = json.loads((MODEL_DIR / f"{tag}.txt.json").read_text(encoding="utf-8"))
    meta_json.update({"version": version, "alpha": CONFORMAL_ALPHA, "test_metrics": {k: v for k, v in metrics.items() if k != "feature_importance"}})
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

    base = fit_pipeline("baseline_rto", X, y, meta, S, weighted=False)
    cs = fit_pipeline("chakra_rto", X, y, meta, S, weighted=True)

    sample = X[S["test"]][:2000]
    rep = export_onnx(cs["booster"], len(FEATURE_NAMES), MODEL_DIR / "chakra_rto.onnx", sample=sample)
    print(f"[onnx] {rep['bytes'] / 1024:.0f} KB, parity max|diff| {rep['parity_max_abs_diff']:.2e}")

    (DATA_DIR / "splits.json").write_text(json.dumps({k: [int(v[0]), int(v[-1]) + 1] for k, v in S.items()}), encoding="utf-8")
    summary = {"baseline": base["metrics"], "chakra": cs["metrics"], "onnx": rep, "alpha": CONFORMAL_ALPHA,
               "economics": ECONOMICS.to_dict(), "n_cod": int(len(cod)), "rto_rate": float(y.mean())}
    (MODEL_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] models in {MODEL_DIR}")


if __name__ == "__main__":
    main()
