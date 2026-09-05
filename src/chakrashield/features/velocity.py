"""Entity velocity features and the write path that maintains them.

Key layout (one hash per entity, plus one set per relationship):

    ent:phone:<h>    orders, rto, delivered, first_ts, last_ts, recent_ts(json list)
    ent:device:<h>   orders, rto, delivered, recent_ts
    ent:addr:<h>     orders, rto, delivered
    ent:pin:<pin>    orders, rto, delivered, gmv_sum
    rel:device:<h>:phones   set of phone hashes seen on this device
    rel:addr:<h>:phones     set of phone hashes shipped to this address
    rel:phone:<h>:devices   set of device hashes used by this phone

Windows are computed from stored timestamps, not key TTLs, so the *same*
code produces point-in-time-correct features when replaying history for
training and when serving live. That property is what makes the offline
metrics believable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict

from ..data import pincodes

_SALT = os.environ.get("CHAKRA_HASH_SALT", "chakrashield-demo-salt")
_RECENT_CAP = 40
_DAY = 86400.0
GLOBAL_RTO_PRIOR = 0.22          # COD base rate used for Bayesian smoothing
_K_ENTITY = 3.0                  # pseudo-count for phone/device/addr
_K_PIN = 25.0                    # pseudo-count for PIN (priors are informative)


def clean_str(raw) -> str:
    """None / NaN / non-str -> '' ; everything else stripped + lower-cased."""
    if raw is None or isinstance(raw, float) or not isinstance(raw, str):
        try:
            if raw is None or (isinstance(raw, float) and raw != raw):
                return ""
        except Exception:
            return ""
        raw = str(raw)
    return raw.strip().lower()


def hash_entity(kind: str, raw) -> str:
    """Salted SHA-256, 16 hex chars. Raw PII never reaches the store."""
    norm = clean_str(raw)
    return hashlib.sha256(f"{_SALT}|{kind}|{norm}".encode("utf-8")).hexdigest()[:16]


def normalize_address(address: str, pin: str) -> str:
    a = re.sub(r"[^a-z0-9]+", " ", (address or "").lower()).strip()
    a = re.sub(r"\s+", " ", a)
    return f"{a}|{pin}"


def k_ent(kind: str, h: str) -> str:
    return f"ent:{kind}:{h}"


def k_rel(kind: str, h: str, other: str) -> str:
    return f"rel:{kind}:{h}:{other}"


@dataclass
class VelocityFeatures:
    phone_orders_30d: int = 0
    phone_orders_total: int = 0
    phone_rto_rate: float = GLOBAL_RTO_PRIOR
    phone_first_seen_days: float = 0.0
    phone_distinct_devices: int = 0
    phone_is_new: bool = True
    device_distinct_phones: int = 0
    device_orders_24h: int = 0
    device_rto_rate: float = GLOBAL_RTO_PRIOR
    addr_distinct_phones: int = 0
    addr_rto_rate: float = GLOBAL_RTO_PRIOR
    pin_rto_rate: float = GLOBAL_RTO_PRIOR
    pin_serviceability: float = 0.9
    pin_orders: int = 0
    pin_gmv_median_proxy: float = 900.0

    def as_dict(self) -> dict:
        return asdict(self)


def _smoothed(rto: float, delivered: float, prior: float, k: float) -> float:
    n = rto + delivered
    return (rto + prior * k) / (n + k) if (n + k) > 0 else prior


def _recent(h: dict) -> list[float]:
    try:
        return json.loads(h.get("recent_ts", "[]"))
    except Exception:
        return []


def read_velocity(store, *, phone_h: str, device_h: str, addr_h: str, pin: str, now_ts: float) -> VelocityFeatures:
    ph = store.hgetall(k_ent("phone", phone_h))
    dv = store.hgetall(k_ent("device", device_h))
    ad = store.hgetall(k_ent("addr", addr_h))
    pn = store.hgetall(k_ent("pin", pin))
    pin_info = pincodes.lookup(pin)

    v = VelocityFeatures()
    if ph:
        rec = _recent(ph)
        v.phone_orders_30d = sum(1 for t in rec if now_ts - t <= 30 * _DAY)
        v.phone_orders_total = int(ph.get("orders", 0))
        v.phone_rto_rate = _smoothed(float(ph.get("rto", 0)), float(ph.get("delivered", 0)), GLOBAL_RTO_PRIOR, _K_ENTITY)
        first = float(ph.get("first_ts", now_ts))
        v.phone_first_seen_days = max(0.0, (now_ts - first) / _DAY)
        v.phone_is_new = v.phone_orders_total == 0
    v.phone_distinct_devices = store.scard(k_rel("phone", phone_h, "devices"))

    if dv:
        rec = _recent(dv)
        v.device_orders_24h = sum(1 for t in rec if now_ts - t <= _DAY)
        v.device_rto_rate = _smoothed(float(dv.get("rto", 0)), float(dv.get("delivered", 0)), GLOBAL_RTO_PRIOR, _K_ENTITY)
    v.device_distinct_phones = store.scard(k_rel("device", device_h, "phones"))

    if ad:
        v.addr_rto_rate = _smoothed(float(ad.get("rto", 0)), float(ad.get("delivered", 0)), GLOBAL_RTO_PRIOR, _K_ENTITY)
    v.addr_distinct_phones = store.scard(k_rel("addr", addr_h, "phones"))

    v.pin_serviceability = pin_info.serviceability_prior
    v.pin_rto_rate = pin_info.rto_prior
    if pn:
        v.pin_orders = int(pn.get("orders", 0))
        v.pin_rto_rate = _smoothed(float(pn.get("rto", 0)), float(pn.get("delivered", 0)), pin_info.rto_prior, _K_PIN)
        attempts = float(pn.get("rto", 0)) + float(pn.get("delivered", 0))
        if attempts > 0:
            v.pin_serviceability = (float(pn.get("delivered", 0)) + pin_info.serviceability_prior * _K_PIN) / (attempts + _K_PIN)
        if v.pin_orders > 0:
            v.pin_gmv_median_proxy = float(pn.get("gmv_sum", 0)) / v.pin_orders
    return v


def record_order(store, *, phone_h: str, device_h: str, addr_h: str, pin: str, ts: float, gmv: float) -> None:
    """Order placed: bump counters, extend recent-ts windows, link entities."""
    for kind, h in (("phone", phone_h), ("device", device_h)):
        key = k_ent(kind, h)
        cur = store.hgetall(key)
        rec = _recent(cur)
        rec.append(ts)
        rec = rec[-_RECENT_CAP:]
        mapping = {"orders": int(cur.get("orders", 0)) + 1, "last_ts": ts, "recent_ts": json.dumps(rec)}
        if "first_ts" not in cur:
            mapping["first_ts"] = ts
        store.hset(key, mapping)
    store.hincrby(k_ent("addr", addr_h), "orders", 1)
    pk = k_ent("pin", pin)
    store.hincrby(pk, "orders", 1)
    cur = store.hgetall(pk)
    store.hset(pk, {"gmv_sum": float(cur.get("gmv_sum", 0)) + gmv})
    store.sadd(k_rel("device", device_h, "phones"), phone_h)
    store.sadd(k_rel("addr", addr_h, "phones"), phone_h)
    store.sadd(k_rel("phone", phone_h, "devices"), device_h)


def record_outcome(store, *, phone_h: str, device_h: str, addr_h: str, pin: str, rto: bool) -> None:
    """Delivery outcome known (days later): update the RTO / delivered tallies."""
    field = "rto" if rto else "delivered"
    for kind, h in (("phone", phone_h), ("device", device_h), ("addr", addr_h), ("pin", pin)):
        store.hincrby(k_ent(kind, h), field, 1)
