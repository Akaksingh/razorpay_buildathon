"""GraphSAGE ring embeddings: is learned message passing worth more than the union-find rules?

The syndicate guard flags a phone with hand-written ratios over union-find
aggregates. This experiment trains a two-layer mean-aggregation GraphSAGE
(src/chakrashield/graph/embeddings.py, plain torch) on the same entity graph
and asks three questions, each answered on phones the model never saw:

    1. Ring detection. Train on the first 60 % of orders by time (ring truth =
       cohort == 'ring', which in production is the confirmed-fraud list), then
       score phones whose FIRST order falls in the last 20 %, on the graph as it
       stands at the end. Compared with the union-find is_ring flag built from
       the same stream, and with a logistic regression on the same node features
       so the value of message passing is isolated from the value of the features.
    2. Future RTO. Same protocol with the label "this new phone's orders came
       back": the graph's RTO-rate features are frozen at the window's start, so
       no outcome the model is asked to predict is visible to it.
    3. Stacking. The GNN phone scores become one (then two) extra columns of the
       served feature frame and LightGBM is retrained with 02_train.py's PARAMS
       on its chronological splits. Training-era orders carry cross-fitted
       (out-of-fold) scores so the booster cannot copy the ring labels.

Caveat that applies throughout: a snapshot's *structure* is end-of-era, not
point-in-time -- a phone's later co-occurrences are in the graph when it is
scored. Outcomes are point-in-time by construction. The serving path would
need per-entity asynchronous scoring published to the feature store, like the
ring stats; this script does not implement that, it decides whether it should.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, precision_recall_curve, roc_auc_score

from chakrashield.config import DATA_DIR, MODEL_DIR, REPORT_DIR
from chakrashield.data.replay import entity_hashes
from chakrashield.features.vectorizer import FEATURE_NAMES
from chakrashield.graph.embeddings import (
    N_FEATURES, EntityGraph, TrainConfig, build_entity_graph, fit_sage, hash_orders, out_of_fold_scores, standardiser,
)
from chakrashield.graph.syndicate import SyndicateGraph
from chakrashield.models.cost_sensitive_booster import train_booster

chrono_splits = importlib.import_module("02_train").chrono_splits

WINDOW = 0.20            # the test era: last 20 % of COD orders; the training-era RTO window mirrors it (40-60 %)
FOLDS = 5                # cross-fitting folds for the stacking column
CONFIG = TrainConfig()


# ----------------------------------------------------------------- helpers
def union_find_flags(orders: pd.DataFrame, outcome_cutoff: float) -> dict[str, bool]:
    """Guarded union-find over the whole order stream, outcomes only up to the cutoff -> phone hash -> is_ring."""
    import heapq

    g = SyndicateGraph(guard=True)
    recs = orders.to_dict("records")
    events = [(float(r["ts"]), 0, i) for i, r in enumerate(recs)]
    events += [(float(r["outcome_ts"]), 1, i) for i, r in enumerate(recs) if float(r["outcome_ts"]) <= outcome_cutoff]
    heapq.heapify(events)
    hashes: dict[str, str] = {}
    while events:
        _, kind, i = heapq.heappop(events)
        r = recs[i]
        h = entity_hashes(r)
        if kind == 0:
            g.ingest(r["order_id"], {k: v for k, v in h.items() if v}, gmv=float(r["cart_gmv"]))
            hashes[r["customer_phone"]] = h["phone"]
        else:
            g.outcome(r["order_id"], bool(r["rto"]))
    return {ph: bool(g.lookup("phone", h).is_ring) for ph, h in hashes.items()}


def binary_prf(y: np.ndarray, flag: np.ndarray) -> dict:
    tp, fp, fn = int((flag & y).sum()), int((flag & ~y).sum()), int((~flag & y).sum())
    return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "tp": tp, "fp": fp, "fn": fn,
            "flagged": int(flag.sum())}


def precision_at_recall(y: np.ndarray, s: np.ndarray, recall: float) -> tuple[float, float]:
    """Best precision reachable at >= the given recall, and the threshold that gets it."""
    p, r, thr = precision_recall_curve(y, s)
    ok = np.flatnonzero(r[:-1] >= recall)
    if len(ok) == 0:
        return 0.0, float("inf")
    i = ok[np.argmax(p[:-1][ok])]
    return float(p[i]), float(thr[i])


def recall_at_precision(y: np.ndarray, s: np.ndarray, precision: float) -> float:
    p, r, _ = precision_recall_curve(y, s)
    ok = p[:-1] >= precision
    return float(r[:-1][ok].max()) if ok.any() else 0.0


def ranking(y: np.ndarray, s: np.ndarray) -> dict:
    return {"auc": float(roc_auc_score(y, s)), "pr_auc": float(average_precision_score(y, s))}


def phone_labels(g: EntityGraph, phones: pd.DataFrame, mask: np.ndarray, col: str) -> np.ndarray:
    """Node-aligned label vector (NaN outside the selected phones) from a per-phone table."""
    labels = np.full(g.n_nodes, np.nan)
    idx = g.phone_nodes(phones.loc[mask, "phone_h"])
    ok = idx >= 0
    labels[idx[ok]] = phones.loc[mask, col].to_numpy(dtype=float)[ok]
    return labels


def window_rto(orders: pd.DataFrame, lo: float, hi: float) -> pd.Series:
    """Per phone first seen in [lo, hi): 1 if at least half of its orders before hi came back."""
    sub = orders[orders.ts < hi]
    first = sub.groupby("customer_phone").ts.min()
    new = first[(first >= lo) & (first < hi)].index
    rate = sub[sub.customer_phone.isin(new)].groupby("customer_phone").rto.mean()
    return (rate >= 0.5).astype(float)


def fit_lr(g: EntityGraph, labels: np.ndarray) -> LogisticRegression:
    mean, std = standardiser(g)
    m = ~np.isnan(labels)
    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit((g.features[m] - mean) / std, labels[m])
    lr.mean_, lr.std_ = mean, std
    return lr


def lr_score(lr: LogisticRegression, g: EntityGraph) -> np.ndarray:
    return lr.predict_proba((g.features - lr.mean_) / lr.std_)[:, 1]


def stack(cod: pd.DataFrame, S: dict, extra: list[str]) -> dict:
    names = list(FEATURE_NAMES) + extra
    X = cod[names].to_numpy(dtype=np.float32)
    y = cod["rto"].to_numpy(dtype=int)
    t = time.time()
    res = train_booster(X[S["train"]], y[S["train"]], X[S["valid"]], y[S["valid"]], names)
    p = res.booster.predict(X[S["test"]], num_iteration=res.best_iter)
    yt = y[S["test"]]
    imp = dict(zip(names, res.booster.feature_importance(importance_type="gain")))
    rank = sorted(names, key=lambda n: -imp[n])
    return {"features": len(names), "best_iter": res.best_iter, "auc": float(roc_auc_score(yt, p)),
            "pr_auc": float(average_precision_score(yt, p)), "logloss": float(log_loss(yt, p)),
            "extra_gain_rank": {c: rank.index(c) + 1 for c in extra},
            "extra_gain_share": {c: float(imp[c] / sum(imp.values())) for c in extra}, "seconds": round(time.time() - t, 1)}


# -------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    orders = pd.read_pickle(DATA_DIR / "orders.pkl").sort_values("ts").reset_index(drop=True)
    full = pd.read_pickle(DATA_DIR / "features.pkl")
    cod = full[full.payment_method == "COD"].sort_values("ts").reset_index(drop=True)
    S = chrono_splits(len(cod))
    cod_ts = cod.ts.to_numpy()
    T60, T80 = float(cod_ts[S["valid"][0]]), float(cod_ts[S["conf"][0]])
    T40 = float(cod_ts[int(round(0.4 * len(cod)))])
    day = lambda t: round((t - float(orders.ts.min())) / 86400, 1)
    print(f"[eras] train < day {day(T60)} (RTO window from day {day(T40)}) | mid < day {day(T80)} | test = last {WINDOW:.0%}")

    hashed = hash_orders(orders)
    phones = orders.groupby("customer_phone").agg(first_ts=("ts", "min"), n_orders=("order_id", "size"),
                                                  is_ring=("cohort", lambda s: float((s == "ring").any())),
                                                  resident=("shared_addr_id", lambda s: bool(s.notna().any())))
    phones["phone_h"] = hashed.groupby("customer_phone").phone.first().reindex(phones.index)
    rto_train = window_rto(orders, T40, T60)
    rto_test = window_rto(orders, T80, np.inf)
    phones["rto_train"] = rto_train.reindex(phones.index)
    phones["rto_test"] = rto_test.reindex(phones.index)
    in_train = (phones.first_ts < T60).to_numpy()
    in_test = (phones.first_ts >= T80).to_numpy()
    print(f"[phones] {len(phones):,} total | training era {in_train.sum():,} ({int(phones.is_ring[in_train].sum())} ring) "
          f"| RTO window {int(phones.rto_train.notna().sum()):,} ({phones.rto_train.mean():.1%} RTO) "
          f"| test era {in_test.sum():,} ({int(phones.is_ring[in_test].sum())} ring, {phones.rto_test[in_test].mean():.1%} RTO)")

    t = time.time()
    g_train = build_entity_graph(hashed, order_cutoff=T60, outcome_cutoff=T40)
    g_mid = build_entity_graph(hashed, order_cutoff=T80, outcome_cutoff=T60)
    g_test = build_entity_graph(hashed, order_cutoff=None, outcome_cutoff=T80)
    graphs = {"train": g_train, "mid": g_mid, "test": g_test}
    print(f"[graph] " + " | ".join(f"{k}: {g.n_nodes:,} nodes, {g.n_edges:,} edges" for k, g in graphs.items())
          + f" | {time.time() - t:.1f}s")

    # ---- targets on the training snapshot
    y_ring = phone_labels(g_train, phones, in_train, "is_ring")
    y_rto = phone_labels(g_train, phones, phones.rto_train.notna().to_numpy(), "rto_train")
    fits = {}
    for name, labels in (("ring", y_ring), ("rto", y_rto)):
        t = time.time()
        sc = fit_sage(g_train, labels, CONFIG)
        fits[name] = {"scorer": sc, "seconds": round(time.time() - t, 1), "epochs_run": len(sc.history), "best_epoch": sc.best_epoch,
                      "best_valid_loss": sc.history[sc.best_epoch]["valid_loss"], "labelled": int((~np.isnan(labels)).sum()),
                      "positive_rate": float(np.nanmean(labels))}
        print(f"[sage:{name}] {fits[name]['labelled']:,} labelled nodes, {fits[name]['positive_rate']:.1%} positive | "
              f"best epoch {sc.best_epoch} of {len(sc.history)} (valid BCE {fits[name]['best_valid_loss']:.4f}) | {fits[name]['seconds']}s")
    lrs = {"ring": fit_lr(g_train, y_ring), "rto": fit_lr(g_train, y_rto)}

    # ---- 1. ring detection on test-era phones
    test_idx = g_test.phone_nodes(phones.loc[in_test, "phone_h"])
    assert (test_idx >= 0).all()
    yt_ring = phones.loc[in_test, "is_ring"].to_numpy().astype(bool)
    resident = phones.loc[in_test, "resident"].to_numpy() & ~yt_ring
    t = time.time()
    uf = union_find_flags(orders, outcome_cutoff=T80)
    uf_flag = np.array([uf[ph] for ph in phones.index[in_test]])
    uf_m = binary_prf(yt_ring, uf_flag)
    uf_m["residents_condemned"] = int((uf_flag & resident).sum())
    print(f"[union-find] {time.time() - t:.1f}s | precision {uf_m['precision']:.3f} recall {uf_m['recall']:.3f} "
          f"(tp {uf_m['tp']} fp {uf_m['fp']} fn {uf_m['fn']}) | residents condemned {uf_m['residents_condemned']}/{int(resident.sum())}")
    ring_scores = {"gnn": fits["ring"]["scorer"].score(g_test)[test_idx], "logreg": lr_score(lrs["ring"], g_test)[test_idx],
                   "union_find": uf_flag.astype(float)}
    ring = {"phones": int(in_test.sum()), "ring_phones_true": int(yt_ring.sum()), "legit_residents": int(resident.sum()),
            "union_find": {**uf_m, **ranking(yt_ring, ring_scores["union_find"])}}
    for name in ("gnn", "logreg"):
        s = ring_scores[name]
        p_at_r, thr = precision_at_recall(yt_ring, s, uf_m["recall"])
        flag = s >= thr
        ring[name] = {**ranking(yt_ring, s), "precision_at_uf_recall": p_at_r, "threshold_at_uf_recall": thr,
                      "recall_at_uf_precision": recall_at_precision(yt_ring, s, uf_m["precision"]),
                      "residents_condemned_at_uf_recall": int((flag & resident).sum()), "fp_at_uf_recall": int((flag & ~yt_ring).sum())}
        print(f"[ring:{name:7s}] AUC {ring[name]['auc']:.4f} PR-AUC {ring[name]['pr_auc']:.4f} | precision at union-find recall "
              f"{p_at_r:.3f} (fp {ring[name]['fp_at_uf_recall']}, residents {ring[name]['residents_condemned_at_uf_recall']}) "
              f"| recall at union-find precision {ring[name]['recall_at_uf_precision']:.3f}")

    # ---- 2. future RTO of test-era phones
    yt_rto = phones.loc[in_test, "rto_test"].to_numpy().astype(bool)
    rto_scores = {"gnn": fits["rto"]["scorer"].score(g_test)[test_idx], "logreg": lr_score(lrs["rto"], g_test)[test_idx],
                  "gnn_ring_score": ring_scores["gnn"], "union_find_flag": ring_scores["union_find"]}
    rto = {"phones": int(in_test.sum()), "positives": int(yt_rto.sum()), **{k: ranking(yt_rto, v) for k, v in rto_scores.items()}}
    print("[rto] " + " | ".join(f"{k} AUC {v['auc']:.4f} PR-AUC {v['pr_auc']:.4f}" for k, v in rto.items() if isinstance(v, dict)))

    # ---- 3. stacking into the served feature frame
    t = time.time()
    oof = {"ring": out_of_fold_scores(g_train, y_ring, FOLDS, CONFIG), "rto": out_of_fold_scores(g_train, y_rto, FOLDS, CONFIG)}
    print(f"[oof] {FOLDS}-fold cross-fitted training-era scores in {time.time() - t:.1f}s")
    era = np.where(full.ts < T60, "train", np.where(full.ts < T80, "mid", "test"))
    full_h = full.order_id.map(dict(zip(orders.order_id, hashed.phone)))
    for target in ("ring", "rto"):
        col = np.full(len(full), np.nan)
        per_era = {"train": oof[target], "mid": fits[target]["scorer"].score(g_mid), "test": fits[target]["scorer"].score(g_test)}
        for e, g in graphs.items():
            m = era == e
            idx = g.phone_nodes(full_h[m])
            vals = np.where(idx >= 0, per_era[e][np.maximum(idx, 0)], np.nan)
            col[m] = vals
        full[f"gnn_{target}_score"] = col
    missing = {c: int(full[c].isna().sum()) for c in ("gnn_ring_score", "gnn_rto_score")}
    cod = full[full.payment_method == "COD"].sort_values("ts").reset_index(drop=True)
    served = json.loads((MODEL_DIR / "training_summary.json").read_text(encoding="utf-8"))["chakra"]
    stacking = {"served_summary": {k: served[k] for k in ("auc", "pr_auc", "logloss", "best_iter")},
                "baseline": stack(cod, S, []), "with_gnn_ring": stack(cod, S, ["gnn_ring_score"]),
                "with_gnn_ring_rto": stack(cod, S, ["gnn_ring_score", "gnn_rto_score"]), "rows_without_score": missing,
                "note": "training-era rows carry out-of-fold scores; mid/test-era rows are scored by the training-era model on "
                        "that era's end-of-era snapshot (structure not point-in-time, outcomes frozen at the era start)"}
    for k in ("with_gnn_ring", "with_gnn_ring_rto"):
        stacking[k]["delta_auc"] = stacking[k]["auc"] - stacking["baseline"]["auc"]
        stacking[k]["delta_pr_auc"] = stacking[k]["pr_auc"] - stacking["baseline"]["pr_auc"]
    for k in ("baseline", "with_gnn_ring", "with_gnn_ring_rto"):
        m = stacking[k]
        print(f"[stack:{k:18s}] {m['features']} features, {m['best_iter']} trees | test AUC {m['auc']:.4f} PR-AUC {m['pr_auc']:.4f} "
              f"logloss {m['logloss']:.4f}" + (f" | Δ AUC {m['delta_auc']:+.4f} Δ PR-AUC {m['delta_pr_auc']:+.4f} | gain rank {m['extra_gain_rank']}" if "delta_auc" in m else ""))

    out = {
        "protocol": {"train_era_frac": 0.6, "rto_window_frac": [0.4, 0.6], "test_era_frac": WINDOW,
                     "boundaries_day": {"T40": day(T40), "T60": day(T60), "T80": day(T80)},
                     "snapshots": {k: {"nodes": g.n_nodes, "edges": g.n_edges, "order_cutoff_day": None if g.order_cutoff is None else day(g.order_cutoff),
                                       "outcome_cutoff_day": day(g.outcome_cutoff)} for k, g in graphs.items()},
                     "node_features": N_FEATURES, "config": CONFIG.__dict__, "folds": FOLDS, "subsampled": False},
        "training": {k: {kk: vv for kk, vv in v.items() if kk != "scorer"} for k, v in fits.items()},
        "ring_detection_test_era": ring, "future_rto_test_era": rto, "stacking": stacking,
        "seconds_total": round(time.time() - t0, 1),
    }
    (REPORT_DIR / "gnn_embeddings.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[done] {out['seconds_total']}s -> {REPORT_DIR / 'gnn_embeddings.json'}")


if __name__ == "__main__":
    main()
