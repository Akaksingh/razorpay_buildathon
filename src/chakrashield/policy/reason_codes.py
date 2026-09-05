"""TreeSHAP contributions -> merchant-legible reason codes.

Codes are stable identifiers a merchant can key business rules on
(RSK_ADDR_DEFECT, RSK_RING_MEMBER, ...). The human string is derived from
the actual feature value so the same code reads differently for "3 phones
on this device" and "41 phones on this device".
"""
from __future__ import annotations

import numpy as np

from ..config import MAX_REASON_CODES, REASON_CODE_MIN_ABS_SHAP
from ..features.vectorizer import FEATURE_NAMES

_CODE = {
    "addr_defect_score": ("RSK_ADDR_DEFECT", lambda v, f: f"Address structurally incomplete (defect {v:.2f})"),
    "addr_has_house_no": ("RSK_ADDR_NO_HOUSE_NO", lambda v, f: "No house / flat number in address" if v < 0.5 else "House number present"),
    "addr_landmark_only": ("RSK_ADDR_LANDMARK_ONLY", lambda v, f: "Address is landmark-only (near/opp/behind)"),
    "addr_state_mismatch": ("RSK_ADDR_PIN_MISMATCH", lambda v, f: "City/state text disagrees with PIN"),
    "addr_junk": ("RSK_ADDR_JUNK", lambda v, f: "Address contains junk / placeholder tokens"),
    "pin_tier": ("RSK_PIN_TIER", lambda v, f: f"Tier-{int(v)} delivery PIN"),
    "pin_rto_rate": ("RSK_PIN_RTO_HISTORY", lambda v, f: f"PIN historical RTO {v:.0%}"),
    "pin_serviceability": ("RSK_PIN_SERVICEABILITY", lambda v, f: f"PIN delivery success {v:.0%}"),
    "phone_rto_rate": ("RSK_PHONE_RTO_HISTORY", lambda v, f: f"Phone historical RTO {v:.0%}"),
    "phone_orders_30d": ("RSK_PHONE_VELOCITY", lambda v, f: f"{int(v)} orders from this phone in 30d"),
    "phone_first_seen_days": ("RSK_NEW_CUSTOMER", lambda v, f: "First-time phone" if v < 1 else f"Phone tenure {v:.0f} days"),
    "phone_distinct_devices": ("RSK_PHONE_MULTI_DEVICE", lambda v, f: f"Phone seen on {int(v)} devices"),
    "device_distinct_phones": ("RSK_DEVICE_MULTI_PHONE", lambda v, f: f"{int(v)} phones share this device"),
    "device_orders_24h": ("RSK_DEVICE_VELOCITY", lambda v, f: f"{int(v)} orders from this device in 24h"),
    "device_rto_rate": ("RSK_DEVICE_RTO_HISTORY", lambda v, f: f"Device historical RTO {v:.0%}"),
    "addr_distinct_phones": ("RSK_ADDR_MULTI_PHONE", lambda v, f: f"{int(v)} phones ship to this address"),
    "addr_rto_rate": ("RSK_ADDR_RTO_HISTORY", lambda v, f: f"Address historical RTO {v:.0%}"),
    "pay_switch_from_failed": ("RSK_PAY_FALLBACK_TO_COD", lambda v, f: "Switched to COD after failed prepaid attempt"),
    "channel_risk": ("RSK_HIGH_CAC_CHANNEL", lambda v, f: f"Paid-acquisition channel (prior RTO {v:.0%})"),
    "gmv_log": ("RSK_BASKET_VALUE", lambda v, f: f"Basket ₹{np.expm1(v):,.0f}"),
    "gmv_vs_pin_median": ("RSK_BASKET_VS_LOCALITY", lambda v, f: f"Basket {v:.1f}× locality median"),
    "items_count": ("RSK_MULTI_ITEM", lambda v, f: f"{int(v)} items"),
    "coupon_applied": ("RSK_COUPON", lambda v, f: "Coupon applied"),
    "checkout_seconds_log": ("RSK_CHECKOUT_SPEED", lambda v, f: f"Checkout in {np.expm1(v):.0f}s"),
    "hour_sin": ("RSK_ODD_HOUR", lambda v, f: "Late-night ordering window"),
    "hour_cos": ("RSK_ODD_HOUR", lambda v, f: "Late-night ordering window"),
    "ring_size_log": ("RSK_RING_MEMBER", lambda v, f: f"Entity linked to a {int(np.expm1(v))}-node cluster"),
    "ring_rto_rate": ("RSK_RING_RTO", lambda v, f: f"Linked cluster RTO {v:.0%}"),
    "ring_phones": ("RSK_RING_PHONES", lambda v, f: f"{int(v)} phones in linked cluster"),
    "ring_is_ring": ("RSK_SYNDICATE_SUBGRAPH", lambda v, f: "Matches syndicate subgraph signature"),
    "entity_max_degree": ("RSK_GRAPH_DEGREE", lambda v, f: f"Entity degree {int(v)} in graph"),
    "is_cod": ("RSK_COD", lambda v, f: "Cash on delivery"),
    "weight_grams": ("RSK_PARCEL_WEIGHT", lambda v, f: f"Parcel {v:.0f} g"),
    "phone_orders_total_log": ("RSK_PHONE_TENURE", lambda v, f: f"{int(np.expm1(v))} prior orders on phone"),
    "pin_orders_log": ("RSK_PIN_VOLUME", lambda v, f: f"{int(np.expm1(v))} prior orders in PIN"),
    "addr_tokens": ("RSK_ADDR_LENGTH", lambda v, f: f"Address has {int(v)} tokens"),
}


def reason_codes(contribs: np.ndarray | None, features: dict[str, float], top: int = MAX_REASON_CODES) -> list[dict]:
    if contribs is None:
        return []
    order = np.argsort(-np.abs(contribs))
    out, seen = [], set()
    for i in order:
        name = FEATURE_NAMES[i]
        s = float(contribs[i])
        if abs(s) < REASON_CODE_MIN_ABS_SHAP:
            break
        code, human = _CODE.get(name, (f"RSK_{name.upper()}", lambda v, f: f"{name}={v:.3g}"))
        if code in seen:
            continue
        seen.add(code)
        v = float(features[name])
        out.append({
            "code": code, "feature": name, "value": round(v, 4), "shap": round(s, 4),
            "direction": "RISK_UP" if s > 0 else "RISK_DOWN", "human": human(v, features),
        })
        if len(out) >= top:
            break
    return out
