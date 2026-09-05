"""Chronological replay: turn an order stream into point-in-time features.

Two event types are merge-sorted by time: ORDER (features are read *then*
the order is recorded) and OUTCOME (RTO / delivered is written back).
Because outcomes land days after orders, an order placed on day 10 can
only see outcomes from orders that resolved before day 10. This is the
same pipeline that serves production, so there is no train/serve skew.
"""
from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np
import pandas as pd

from ..features.address import score_address
from ..features.vectorizer import FEATURE_NAMES, build_features, graph_features_from_store
from ..features.velocity import clean_str, hash_entity, normalize_address, read_velocity, record_order, record_outcome
from ..graph.syndicate import SyndicateGraph
from ..store.feature_store import MemoryStore


def entity_hashes(row: dict) -> dict[str, str]:
    return {
        "phone": hash_entity("phone", row["customer_phone"]),
        "device": hash_entity("device", row["device_fingerprint"]),
        "addr": hash_entity("addr", normalize_address(row["shipping_address"], row["delivery_pin"])),
        "vpa": hash_entity("vpa", row["vpa"]) if clean_str(row.get("vpa")) else "",
        "ip": hash_entity("ip", row["ip"]) if clean_str(row.get("ip")) else "",
    }


def replay(df: pd.DataFrame, store=None, graph: SyndicateGraph | None = None, progress: bool = False,
           upto_ts: float | None = None) -> tuple[pd.DataFrame, SyndicateGraph]:
    """Return (feature frame aligned with df, populated graph). Mutates the store."""
    store = store if store is not None else MemoryStore()
    graph = graph if graph is not None else SyndicateGraph(store=store)
    recs = df.to_dict("records")
    events = []
    for i, r in enumerate(recs):
        events.append((float(r["ts"]), 0, i))          # ORDER
        events.append((float(r["outcome_ts"]), 1, i))  # OUTCOME (later)
    heapq.heapify(events)
    feats: list[dict | None] = [None] * len(recs)
    n = 0
    while events:
        ts, kind, i = heapq.heappop(events)
        if upto_ts is not None and ts > upto_ts:
            break
        r = recs[i]
        h = entity_hashes(r)
        if kind == 0:
            addr = score_address(r["shipping_address"], r["delivery_pin"])
            vel = read_velocity(store, phone_h=h["phone"], device_h=h["device"], addr_h=h["addr"], pin=r["delivery_pin"], now_ts=ts)
            g = graph_features_from_store(store, {k: v for k, v in h.items() if v})
            f = build_features(
                gmv=r["cart_gmv"], items_count=int(r["items_count"]), weight_grams=float(r["weight_grams"]),
                pin=r["delivery_pin"], address=addr, velocity=vel, graph=g, payment_method=r["payment_method"],
                payment_switch_from=r.get("payment_switch_from"), channel=r["acquisition_channel"],
                coupon_applied=bool(r["coupon_applied"]), checkout_seconds=float(r["checkout_seconds"]),
                hour_of_day=int(r["hour_of_day"]),
            )
            f["_phone_is_new"] = 1.0 if vel.phone_is_new else 0.0
            feats[i] = f
            record_order(store, phone_h=h["phone"], device_h=h["device"], addr_h=h["addr"], pin=r["delivery_pin"], ts=ts, gmv=float(r["cart_gmv"]))
            graph.ingest(r["order_id"], {k: v for k, v in h.items() if v}, gmv=float(r["cart_gmv"]))
            n += 1
            if progress and n % 10000 == 0:
                print(f"  replayed {n} orders")
        else:
            rto = bool(r["rto"])
            record_outcome(store, phone_h=h["phone"], device_h=h["device"], addr_h=h["addr"], pin=r["delivery_pin"], rto=rto)
            graph.outcome(r["order_id"], rto)
    out = pd.DataFrame([f if f is not None else {k: np.nan for k in FEATURE_NAMES} for f in feats])
    return out, graph
