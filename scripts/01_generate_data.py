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
from chakrashield.data.pipeline import materialise


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

    materialise(df, DATA_DIR, extra_world={
        "source": "synthetic", "days": a.days, "seed": a.seed,
        "shared_addresses": int(df.shared_addr_id.nunique()),
        "shared_residents": int(df.loc[df.shared_addr_id.notna(), "customer_phone"].nunique()),
    })
    print(f"[done] artifacts in {DATA_DIR}")


if __name__ == "__main__":
    main()
