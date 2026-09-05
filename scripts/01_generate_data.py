"""Generate the synthetic order world and replay it into point-in-time features.

Outputs (artifacts/data):
    orders.pkl      raw event stream (what a merchant's OMS would hold)
    features.pkl    orders + FEATURE_NAMES columns, point-in-time correct
    store.pkl       feature-store snapshot at end of history (serving warm-start)
    graph.pkl       syndicate graph snapshot
    world.json      summary statistics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chakrashield.config import DATA_DIR
from chakrashield.data.generator import generate
from chakrashield.data.replay import replay
from chakrashield.store.feature_store import MemoryStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, default=60_000)
    ap.add_argument("--days", type=int, default=400)
    ap.add_argument("--customers", type=int, default=18_000)
    ap.add_argument("--rings", type=int, default=45)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    t = time.time()
    df = generate(n_orders=a.orders, days=a.days, seed=a.seed, n_customers=a.customers, n_rings=a.rings)
    print(f"[gen] {len(df):,} orders, {df.customer_phone.nunique():,} phones in {time.time() - t:.1f}s")
    cod = df[df.payment_method == "COD"]
    print(f"[gen] COD share {len(cod) / len(df):.1%}; COD RTO {cod.rto.mean():.1%}; "
          f"prepaid return {df[df.payment_method != 'COD'].rto.mean():.1%}")
    print(cod.groupby("cohort").rto.agg(["mean", "size"]).round(3).to_string())

    t = time.time()
    store = MemoryStore()
    feats, graph = replay(df, store=store, progress=True)
    print(f"[replay] {feats.shape} in {time.time() - t:.1f}s; graph {graph.stats()}")
    # feature columns are canonical where names overlap (items_count, weight_grams, pin_tier, coupon_applied)
    full = df.drop(columns=[c for c in feats.columns if c in df.columns]).join(feats)
    full["is_new_customer"] = full["_phone_is_new"].astype(bool)  # point-in-time truth from the store
    full = full.drop(columns=["_phone_is_new"])

    df.to_pickle(DATA_DIR / "orders.pkl")
    full.to_pickle(DATA_DIR / "features.pkl")
    store.dump(DATA_DIR / "store.pkl", clock_ts=float(df.outcome_ts.max()) + 86400)
    graph.dump(DATA_DIR / "graph.pkl")
    rings = graph.rings(top=10)
    world = {
        "orders": int(len(df)), "phones": int(df.customer_phone.nunique()), "devices": int(df.device_fingerprint.nunique()),
        "cod_share": float(len(cod) / len(df)), "cod_rto_rate": float(cod.rto.mean()),
        "rto_by_cohort": {k: float(v) for k, v in cod.groupby("cohort").rto.mean().items()},
        "shared_addresses": int(df.shared_addr_id.nunique()),
        "shared_residents": int(df.loc[df.shared_addr_id.notna(), "customer_phone"].nunique()),
        "graph": graph.stats(), "top_rings": rings, "clock_ts": float(df.outcome_ts.max()) + 86400,
        "days": a.days, "seed": a.seed,
    }
    (DATA_DIR / "world.json").write_text(json.dumps(world, indent=2), encoding="utf-8")
    print(f"[done] artifacts in {DATA_DIR}")


if __name__ == "__main__":
    main()
