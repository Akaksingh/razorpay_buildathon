"""Materialise an order stream into the artifacts the gateway and the training scripts read.

Shared by the synthetic generator (scripts/01_generate_data.py) and the merchant
CSV ingester (scripts/ingest_csv.py) so a real export takes exactly the same path:
chronological replay -> point-in-time features -> store + graph snapshots.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR
from .replay import replay
from ..store.feature_store import MemoryStore


def materialise(df: pd.DataFrame, out_dir: Path = DATA_DIR, extra_world: dict | None = None, progress: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    store = MemoryStore()
    feats, graph = replay(df, store=store, progress=progress)
    if progress:
        print(f"[replay] {feats.shape} in {time.time() - t:.1f}s; graph {graph.stats()}")
    # feature columns are canonical where names overlap (items_count, weight_grams, pin_tier, coupon_applied)
    full = df.drop(columns=[c for c in feats.columns if c in df.columns]).join(feats)
    full["is_new_customer"] = full["_phone_is_new"].astype(bool)  # point-in-time truth from the store
    full = full.drop(columns=["_phone_is_new"])

    cod = df[df.payment_method == "COD"]
    clock_ts = float(df.outcome_ts.max()) + 86400
    df.to_pickle(out_dir / "orders.pkl")
    full.to_pickle(out_dir / "features.pkl")
    store.dump(out_dir / "store.pkl", clock_ts=clock_ts)
    graph.dump(out_dir / "graph.pkl")
    world = {
        "orders": int(len(df)), "phones": int(df.customer_phone.nunique()), "devices": int(df.device_fingerprint.nunique()),
        "cod_share": float(len(cod) / max(1, len(df))), "cod_rto_rate": float(cod.rto.mean()) if len(cod) else 0.0,
        "rto_by_cohort": {str(k): float(v) for k, v in cod.groupby("cohort").rto.mean().items()} if "cohort" in cod else {},
        "graph": graph.stats(), "top_rings": graph.rings(top=10), "clock_ts": clock_ts, **(extra_world or {}),
    }
    (out_dir / "world.json").write_text(json.dumps(world, indent=2), encoding="utf-8")
    return world
