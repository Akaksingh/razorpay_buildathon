"""Measure the component-collapse guard on the full world.

Two graphs are built from the same chronological event stream:

    naive   every shared entity is a transitive merge edge (plain union-find)
    guarded typed edges: hard identifiers merge, IP never merges, an address stops
            bridging once it looks public (ADDR_MERGE_CEILING distinct phones)

and compared on what a merchant actually pays for: how many legitimate residents
of hostels / offices / PGs are condemned as ring members, ring precision and
recall at phone level, and the size of the largest component.
"""
from __future__ import annotations

import heapq
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chakrashield.config import DATA_DIR, REPORT_DIR
from chakrashield.data.replay import entity_hashes
from chakrashield.graph.syndicate import ADDR_MERGE_CEILING, SHARED_DEGREE_CEILING, SyndicateGraph


def build(df: pd.DataFrame, guard: bool) -> SyndicateGraph:
    g = SyndicateGraph(guard=guard)
    recs = df.to_dict("records")
    events = [(float(r["ts"]), 0, i) for i, r in enumerate(recs)] + [(float(r["outcome_ts"]), 1, i) for i, r in enumerate(recs)]
    heapq.heapify(events)
    while events:
        _, kind, i = heapq.heappop(events)
        r = recs[i]
        if kind == 0:
            g.ingest(r["order_id"], {k: v for k, v in entity_hashes(r).items() if v}, gmv=float(r["cart_gmv"]))
        else:
            g.outcome(r["order_id"], bool(r["rto"]))
    return g


def phone_level(g: SyndicateGraph, df: pd.DataFrame) -> dict:
    phones = df.drop_duplicates("customer_phone")[["customer_phone", "cohort", "shared_addr_id"]]
    truth = (phones.cohort == "ring").to_numpy()
    flagged = []
    for ph in phones.customer_phone:
        st = g.lookup("phone", entity_hashes({"customer_phone": ph, "device_fingerprint": "", "shipping_address": "", "delivery_pin": ""})["phone"])
        flagged.append(bool(st and st.is_ring))
    flagged = pd.Series(flagged, index=phones.index).to_numpy()
    tp = int((flagged & truth).sum())
    fp = int((flagged & ~truth).sum())
    fn = int((~flagged & truth).sum())
    resident = phones.shared_addr_id.notna().to_numpy() & ~truth
    return {
        "phones": int(len(phones)), "ring_phones_true": int(truth.sum()),
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "tp": tp, "fp": fp, "fn": fn,
        "legit_residents": int(resident.sum()),
        "residents_condemned": int((flagged & resident).sum()),
        "residents_condemned_share": float((flagged & resident).sum() / max(1, resident.sum())),
        "legit_phones_condemned": fp,
    }


def main() -> None:
    import chakrashield.graph.syndicate as sg

    df = pd.read_pickle(DATA_DIR / "orders.pkl")
    ceilings = [int(a) for a in sys.argv[1:]] or [ADDR_MERGE_CEILING]
    out = {"addr_merge_ceiling": ADDR_MERGE_CEILING, "shared_degree_ceiling": SHARED_DEGREE_CEILING, "variants": {}}
    variants = [("naive", False, ADDR_MERGE_CEILING)] + [(f"guarded@{c}" if c != ADDR_MERGE_CEILING else "guarded", True, c) for c in ceilings]
    for name, guard, ceiling in variants:
        sg.ADDR_MERGE_CEILING = ceiling
        t = time.time()
        g = build(df, guard)
        m = {"stats": g.stats(), "addr_merge_ceiling": ceiling, **phone_level(g, df), "seconds": round(time.time() - t, 1)}
        out["variants"][name] = m
        s = m["stats"]
        print(f"[{name:10s}] components {s['components']:,} | largest {s['largest_component']:,} | rings {s['rings']} "
              f"| shared entities {s['shared_entities']} | phone precision {m['precision']:.3f} recall {m['recall']:.3f} "
              f"| legit phones condemned {m['fp']} | hostel residents condemned {m['residents_condemned']}/{m['legit_residents']} "
              f"({m['residents_condemned_share']:.1%})")
    sg.ADDR_MERGE_CEILING = ADDR_MERGE_CEILING
    n, gd = out["variants"]["naive"], out["variants"].get("guarded") or out["variants"][variants[-1][0]]
    out["delta"] = {"legit_phones_spared": n["fp"] - gd["fp"], "residents_spared": n["residents_condemned"] - gd["residents_condemned"],
                    "recall_change": gd["recall"] - n["recall"], "largest_component_ratio": n["stats"]["largest_component"] / max(1, gd["stats"]["largest_component"])}
    (REPORT_DIR / "graph_guard.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[done] {REPORT_DIR / 'graph_guard.json'}")


if __name__ == "__main__":
    main()
