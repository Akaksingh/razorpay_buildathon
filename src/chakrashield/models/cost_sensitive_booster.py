"""Cost-sensitive gradient boosting for COD RTO.

Instance-dependent cost-sensitive learning: each training example is
weighted by the rupee cost of getting *it* wrong.

    y = 1 (RTO)      -> w_1(x) = C_FN(x) = L_logistics + lambda * V
    y = 0 (delivered)-> w_0(x) = C_FP(x) = M * V + kappa * CAC

Minimising weighted log-loss under these weights is the empirical
risk minimiser for the merchant's expected loss, not for accuracy.
The booster therefore spends its capacity on the expensive boundary:
high-GMV paid-acquisition orders where a false block burns CAC, and
heavy parcels to Tier-4 PINs where a false allow burns two shipping legs.

Weighting distorts probability scale, so the trained scorer is
isotonically recalibrated on a disjoint split before it meets tau*(x).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..config import ECONOMICS, Economics
from ..policy.economics import TransactionContext

PARAMS = {
    "objective": "binary",
    "learning_rate": 0.04,
    "num_leaves": 31,
    "min_data_in_leaf": 60,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 2.0,
    "max_bin": 255,
    "verbose": -1,
    "seed": 7,
    "deterministic": True,
    "num_threads": 4,
    "force_row_wise": True,
}
MAX_ROUNDS = 1200
EARLY_STOP = 80


def instance_weights(meta: pd.DataFrame, y: np.ndarray, econ: Economics = ECONOMICS) -> np.ndarray:
    """w_i = C_FN(x_i) if y_i = 1 else C_FP(x_i), normalised to mean 1 for optimiser stability."""
    w = np.empty(len(y), dtype=float)
    for i, (gmv, m, cac, new, wt) in enumerate(zip(
            meta["cart_gmv"].to_numpy(), meta["merchant_margin"].to_numpy(), meta["cac"].to_numpy(),
            meta["is_new_customer"].to_numpy(), meta["weight_grams"].to_numpy())):
        ctx = TransactionContext(gmv=float(gmv), merchant_margin=float(m), cac=float(cac), p_loss=0.0,
                                 is_new_customer=bool(new), weight_grams=float(wt), econ=econ)
        w[i] = ctx.cost_fn if y[i] == 1 else ctx.cost_fp
    return w / w.mean()


@dataclass
class TrainResult:
    booster: lgb.Booster
    best_iter: int
    train_rows: int
    valid_metric: float
    weighted: bool


def train_booster(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray,
                  feature_names: list[str], w_tr: np.ndarray | None = None, w_va: np.ndarray | None = None,
                  params: dict | None = None) -> TrainResult:
    p = dict(PARAMS)
    if params:
        p.update(params)
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, feature_name=feature_names, free_raw_data=False)
    dva = lgb.Dataset(X_va, label=y_va, weight=w_va, feature_name=feature_names, reference=dtr, free_raw_data=False)
    evals: dict = {}
    booster = lgb.train(
        p, dtr, num_boost_round=MAX_ROUNDS, valid_sets=[dva], valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.record_evaluation(evals)],
    )
    metric = float(evals["valid"]["binary_logloss"][booster.best_iteration - 1])
    return TrainResult(booster=booster, best_iter=booster.best_iteration, train_rows=len(y_tr),
                       valid_metric=metric, weighted=w_tr is not None)


def save_booster(res: TrainResult, path: Path, feature_names: list[str], extra: dict | None = None) -> None:
    res.booster.save_model(str(path), num_iteration=res.best_iter)
    meta = {"best_iter": res.best_iter, "train_rows": res.train_rows, "valid_logloss": res.valid_metric,
            "weighted": res.weighted, "feature_names": feature_names, "params": PARAMS, **(extra or {})}
    with open(str(path) + ".json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
