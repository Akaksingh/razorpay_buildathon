"""Feature vector hydration: the single source of truth for feature order.

Both the offline replay (training) and the online gateway (serving) call
``build_features`` with the same inputs, so there is exactly one place
where a feature can be defined. FEATURE_NAMES is persisted next to the
model and checked at load time; a mismatch refuses to serve.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..data import pincodes
from .address import AddressSignals, score_address
from .velocity import VelocityFeatures, hash_entity, normalize_address, read_velocity

FEATURE_NAMES: tuple[str, ...] = (
    "gmv_log", "items_count", "weight_grams",
    "pin_tier", "pin_serviceability", "pin_rto_rate", "pin_orders_log",
    "addr_defect_score", "addr_tokens", "addr_has_house_no", "addr_landmark_only",
    "addr_state_mismatch", "addr_junk",
    "phone_orders_30d", "phone_orders_total_log", "phone_rto_rate", "phone_first_seen_days",
    "phone_distinct_devices",
    "device_distinct_phones", "device_orders_24h", "device_rto_rate",
    "addr_distinct_phones", "addr_rto_rate",
    "pay_switch_from_failed", "is_cod", "channel_risk", "coupon_applied", "checkout_seconds_log",
    "hour_sin", "hour_cos",
    "ring_size_log", "ring_rto_rate", "ring_phones", "ring_is_ring", "entity_max_degree", "entity_shared",
    "gmv_vs_pin_median",
)
N_FEATURES = len(FEATURE_NAMES)

# Ordinal-risk encoding of acquisition channel (prior RTO lift). The booster
# is free to disagree; this just gives it a monotone starting point.
CHANNEL_RISK = {
    "ORGANIC": 0.15, "DIRECT": 0.14, "WHATSAPP": 0.20, "GOOGLE_ADS": 0.22, "MARKETPLACE": 0.25,
    "META_ADS": 0.30, "INFLUENCER": 0.32, "AFFILIATE": 0.38,
}


def build_features(
    *, gmv: float, items_count: int, weight_grams: float, pin: str, address: AddressSignals,
    velocity: VelocityFeatures, graph: dict, payment_method: str, payment_switch_from: str | None,
    channel: str, coupon_applied: bool, checkout_seconds: float, hour_of_day: int,
) -> dict[str, float]:
    pin_info = pincodes.lookup(pin)
    sw = str(payment_switch_from).upper() if isinstance(payment_switch_from, str) else ""
    switched = 1.0 if sw.endswith("FAILED") or sw in ("CARD", "UPI", "NETBANKING") else 0.0
    theta = 2 * math.pi * (hour_of_day % 24) / 24.0
    ring_size = float(graph.get("ring_size", 0) or 0)
    f = {
        "gmv_log": math.log1p(max(0.0, gmv)),
        "items_count": float(items_count),
        "weight_grams": float(weight_grams),
        "pin_tier": float(pin_info.tier),
        "pin_serviceability": float(velocity.pin_serviceability),
        "pin_rto_rate": float(velocity.pin_rto_rate),
        "pin_orders_log": math.log1p(velocity.pin_orders),
        "addr_defect_score": float(address.defect_score),
        "addr_tokens": float(address.tokens),
        "addr_has_house_no": 1.0 if address.has_house_no else 0.0,
        "addr_landmark_only": 1.0 if address.landmark_only else 0.0,
        "addr_state_mismatch": 1.0 if address.state_mismatch else 0.0,
        "addr_junk": 1.0 if address.has_junk else 0.0,
        "phone_orders_30d": float(velocity.phone_orders_30d),
        "phone_orders_total_log": math.log1p(velocity.phone_orders_total),
        "phone_rto_rate": float(velocity.phone_rto_rate),
        "phone_first_seen_days": float(min(velocity.phone_first_seen_days, 730.0)),
        "phone_distinct_devices": float(velocity.phone_distinct_devices),
        "device_distinct_phones": float(velocity.device_distinct_phones),
        "device_orders_24h": float(velocity.device_orders_24h),
        "device_rto_rate": float(velocity.device_rto_rate),
        "addr_distinct_phones": float(velocity.addr_distinct_phones),
        "addr_rto_rate": float(velocity.addr_rto_rate),
        "pay_switch_from_failed": switched,
        "is_cod": 1.0 if payment_method.upper() == "COD" else 0.0,
        "channel_risk": float(CHANNEL_RISK.get(channel.upper(), 0.22)),
        "coupon_applied": 1.0 if coupon_applied else 0.0,
        "checkout_seconds_log": math.log1p(max(0.0, checkout_seconds)),
        "hour_sin": math.sin(theta),
        "hour_cos": math.cos(theta),
        "ring_size_log": math.log1p(ring_size),
        "ring_rto_rate": float(graph.get("ring_rto_rate", 0.0) or 0.0),
        "ring_phones": float(graph.get("ring_phones", 0) or 0),
        "ring_is_ring": 1.0 if graph.get("is_ring") else 0.0,
        "entity_max_degree": float(graph.get("entity_max_degree", 0) or 0),
        "entity_shared": 1.0 if graph.get("entity_shared") else 0.0,
        "gmv_vs_pin_median": float(gmv) / max(50.0, float(velocity.pin_gmv_median_proxy)),
    }
    assert tuple(f.keys()) == FEATURE_NAMES, "feature order drift"
    return f


def to_vector(features: dict[str, float]) -> np.ndarray:
    return np.asarray([[features[k] for k in FEATURE_NAMES]], dtype=np.float32)


def graph_features_from_store(store, entities: dict[str, str]) -> dict:
    """Read ring stats published by the graph worker. Serving never touches the graph."""
    best = {"ring_size": 0, "ring_phones": 0, "ring_rto_rate": 0.0, "is_ring": False, "ring_id": None,
            "ring_orders": 0, "ring_devices": 0}
    max_deg, shared = 0, 0
    for kind, h in entities.items():
        if not h:
            continue
        g = store.hgetall(f"graph:{kind}:{h}")
        if not g:
            continue
        shared = max(shared, int(g.get("shared", 0)))
        size = int(g.get("ring_size", 0))
        rto, dl = float(g.get("ring_rto", 0)), float(g.get("ring_delivered", 0))
        rate = rto / (rto + dl) if (rto + dl) else 0.0
        if (size, rate) > (best["ring_size"], best["ring_rto_rate"]):
            best = {"ring_size": size, "ring_phones": int(g.get("ring_phones", 0)), "ring_rto_rate": rate,
                    "is_ring": bool(int(g.get("is_ring", 0))), "ring_id": g.get("ring_id"),
                    "ring_orders": int(g.get("ring_orders", 0)), "ring_devices": int(g.get("ring_devices", 0))}
        max_deg = max(max_deg, int(g.get("degree", 0)))
    if best["ring_size"] <= 1:
        best.update({"ring_size": 0, "ring_phones": 0, "ring_rto_rate": 0.0, "is_ring": False, "ring_id": None})
    best["entity_max_degree"] = max_deg
    best["entity_shared"] = bool(shared)
    return best


@dataclass
class Hydrated:
    vector: np.ndarray
    features: dict[str, float]
    address: AddressSignals
    velocity: VelocityFeatures
    graph: dict
    hashes: dict[str, str]
    timings_ms: dict[str, float] = field(default_factory=dict)


def hydrate(req, store, now_ts: float | None = None) -> Hydrated:
    """Serving-path hydration for a RiskRequest. Budget: ~2ms on the memory store."""
    t0 = time.perf_counter()
    now = now_ts or time.time()
    pin = req.delivery_pin
    hashes = {
        "phone": hash_entity("phone", req.customer_phone),
        "device": hash_entity("device", req.device_fingerprint_hash),
        "addr": hash_entity("addr", normalize_address(req.shipping_address, pin)),
        "vpa": hash_entity("vpa", req.vpa) if req.vpa else "",
        "ip": hash_entity("ip", req.ip_hash) if req.ip_hash else "",
    }
    t1 = time.perf_counter()
    addr = score_address(req.shipping_address, pin)
    t2 = time.perf_counter()
    vel = read_velocity(store, phone_h=hashes["phone"], device_h=hashes["device"], addr_h=hashes["addr"], pin=pin, now_ts=now)
    t3 = time.perf_counter()
    graph = graph_features_from_store(store, {k: v for k, v in hashes.items() if v})
    t4 = time.perf_counter()
    hour = req.hour_of_day if req.hour_of_day is not None else time.localtime(now).tm_hour
    feats = build_features(
        gmv=req.cart_gmv, items_count=req.items_count, weight_grams=req.weight_grams, pin=pin, address=addr,
        velocity=vel, graph=graph, payment_method=req.payment_method, payment_switch_from=req.payment_switch_from,
        channel=req.acquisition_channel, coupon_applied=req.coupon_applied, checkout_seconds=req.checkout_seconds,
        hour_of_day=hour,
    )
    vec = to_vector(feats)
    t5 = time.perf_counter()
    return Hydrated(
        vector=vec, features=feats, address=addr, velocity=vel, graph=graph, hashes=hashes,
        timings_ms={"hash": (t1 - t0) * 1e3, "address": (t2 - t1) * 1e3, "velocity": (t3 - t2) * 1e3,
                    "graph": (t4 - t3) * 1e3, "vectorize": (t5 - t4) * 1e3},
    )
