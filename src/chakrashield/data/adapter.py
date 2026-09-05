"""Merchant CSV adapter: a real order export becomes the internal order stream.

Nothing downstream knows whether the world was generated or exported. A
mapping file names the merchant's columns and how their status values map to
the RTO label; anything the export lacks is filled with an explicit default
that is recorded in the returned report, never silently invented.

Required (after mapping):  order_id, ts, customer_phone, shipping_address,
                           delivery_pin, cart_gmv, payment_method, status
Recommended:               outcome_ts, device_fingerprint, items_count,
                           weight_grams, acquisition_channel, coupon_applied,
                           checkout_seconds, hour_of_day, ip, vpa, email, card_token
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import pincodes

REQUIRED = ("order_id", "ts", "customer_phone", "shipping_address", "delivery_pin", "cart_gmv", "payment_method", "status")
DEFAULTS = {"items_count": 1, "weight_grams": 450.0, "acquisition_channel": "ORGANIC", "coupon_applied": False,
            "checkout_seconds": 90.0, "merchant_margin": 0.18, "cac": 300.0, "outcome_lag_days": 6.0}
PAYMENT_ALIASES = {"COD": "COD", "CASH ON DELIVERY": "COD", "CASH": "COD", "UPI": "UPI", "CARD": "CARD", "CREDIT CARD": "CARD",
                   "DEBIT CARD": "CARD", "NETBANKING": "NETBANKING", "NET BANKING": "NETBANKING", "WALLET": "WALLET", "EMI": "EMI",
                   "PREPAID": "UPI"}

MAPPING_EXAMPLE = {
    "columns": {
        "order_id": "Order ID", "ts": "Order Date", "outcome_ts": "Delivery Date", "customer_phone": "Customer Mobile",
        "customer_email": "Customer Email", "device_fingerprint": "Device ID", "ip": "Client IP", "vpa": "UPI ID",
        "card_token": "Card Token", "shipping_address": "Ship Address", "delivery_pin": "Ship Pincode",
        "cart_gmv": "Order Amount", "items_count": "Qty", "weight_grams": "Weight (g)", "payment_method": "Payment Mode",
        "payment_switch_from": "Failed Payment Mode", "acquisition_channel": "UTM Source", "coupon_applied": "Coupon Code",
        "checkout_seconds": "Checkout Duration", "status": "Status",
    },
    "status": {
        "rto": ["RTO", "Returned to Sender", "Rejected by Buyer", "Undelivered", "Return to Origin"],
        "delivered": ["Delivered", "Shipped - Delivered to Buyer", "Completed"],
        "drop": ["Cancelled", "Pending", "Shipped"],
    },
    "channel_aliases": {"facebook": "META_ADS", "instagram": "META_ADS", "google": "GOOGLE_ADS", "affiliate": "AFFILIATE",
                        "influencer": "INFLUENCER", "whatsapp": "WHATSAPP", "amazon": "MARKETPLACE", "flipkart": "MARKETPLACE",
                        "direct": "DIRECT", "organic": "ORGANIC"},
    "date_format": None,            # e.g. "%d/%m/%Y %H:%M"; None = infer element-wise, day first
    "outcome_date_format": None,    # defaults to date_format
    "defaults": DEFAULTS,
}


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:24]


def load_merchant_orders(csv_path: str | Path, mapping: dict) -> tuple[pd.DataFrame, dict]:
    """Return (orders frame in the internal schema, ingestion report)."""
    raw = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    cols = mapping.get("columns", {})
    missing = [k for k in REQUIRED if cols.get(k) not in raw.columns]
    if missing:
        raise ValueError(f"mapping does not provide required columns {missing}; CSV has {list(raw.columns)[:20]}")
    defaults = {**DEFAULTS, **(mapping.get("defaults") or {})}
    report = {"rows_in": int(len(raw)), "defaults_used": [], "dropped": {}}

    def col(name, default=None):
        c = cols.get(name)
        if c in raw.columns:
            return raw[c]
        if default is not None:
            report["defaults_used"].append(name)
        return pd.Series([default] * len(raw), index=raw.index)

    def parse_dates(s: pd.Series, fmt: str | None) -> pd.Series:
        # an explicit format wins; otherwise parse element-wise, day-first (Indian exports are dd/mm/yyyy)
        if fmt:
            return pd.to_datetime(s, format=fmt, errors="coerce")
        return pd.to_datetime(s, format="mixed", dayfirst=True, errors="coerce")

    def epoch_seconds(s: pd.Series) -> pd.Series:
        return s.astype("datetime64[ns]").astype("int64") / 1e9      # resolution-safe (pandas 3 parses to [us])

    ts = parse_dates(col("ts"), mapping.get("date_format"))
    ok = ts.notna()
    report["dropped"]["unparseable_date"] = int((~ok).sum())
    st = mapping.get("status", {})
    status = col("status").astype(str).str.strip()
    is_rto = status.str.lower().isin([s.lower() for s in st.get("rto", [])])
    is_del = status.str.lower().isin([s.lower() for s in st.get("delivered", [])])
    labelled = is_rto | is_del
    report["dropped"]["unlabelled_status"] = int((~labelled & ok).sum())
    keep = ok & labelled

    out_ts = parse_dates(col("outcome_ts", ""), mapping.get("outcome_date_format") or mapping.get("date_format"))
    lag = pd.to_timedelta(float(defaults["outcome_lag_days"]), unit="D")
    outcome_ts = out_ts.where(out_ts.notna(), ts + lag)
    pay = col("payment_method").astype(str).str.strip().str.upper().map(lambda v: PAYMENT_ALIASES.get(v, v if v in PAYMENT_ALIASES.values() else "COD"))
    aliases = {k.lower(): v for k, v in (mapping.get("channel_aliases") or {}).items()}
    channel = col("acquisition_channel", defaults["acquisition_channel"]).astype(str).str.strip().map(
        lambda v: aliases.get(v.lower(), v.upper() if v.upper() in {"ORGANIC", "DIRECT", "WHATSAPP", "GOOGLE_ADS", "MARKETPLACE", "META_ADS", "INFLUENCER", "AFFILIATE"} else defaults["acquisition_channel"]))
    phone = col("customer_phone").astype(str).str.replace(r"\D", "", regex=True).str[-10:]
    device = col("device_fingerprint", "")
    device = device.where(device.astype(str).str.len() > 0, "fp_" + phone.map(_hash))   # no device id: one device per phone
    pin = col("delivery_pin").astype(str).str.replace(r"\D", "", regex=True).str[:6]
    coupon_raw = col("coupon_applied", "")
    coupon = coupon_raw.astype(str).str.strip().str.lower().map(lambda v: v not in ("", "0", "false", "no", "none", "nan"))
    hour = col("hour_of_day", "")
    hour = pd.to_numeric(hour, errors="coerce").fillna(ts.dt.hour).astype(int)

    df = pd.DataFrame({
        "order_id": col("order_id").astype(str), "ts": epoch_seconds(ts), "outcome_ts": epoch_seconds(outcome_ts),
        "customer_phone": phone, "customer_email": col("customer_email", ""), "account_id": "acc_" + phone.map(_hash).str[:8],
        "device_fingerprint": device, "ip": col("ip", ""), "vpa": col("vpa", ""), "card_token": col("card_token", ""),
        "shipping_address": col("shipping_address").astype(str), "delivery_pin": pin,
        "pin_tier": pin.map(lambda p: pincodes.lookup(p).tier),
        "cart_gmv": pd.to_numeric(col("cart_gmv"), errors="coerce"),
        "items_count": pd.to_numeric(col("items_count", defaults["items_count"]), errors="coerce").fillna(defaults["items_count"]).astype(int),
        "weight_grams": pd.to_numeric(col("weight_grams", defaults["weight_grams"]), errors="coerce").fillna(defaults["weight_grams"]),
        "payment_method": pay, "payment_switch_from": col("payment_switch_from", ""), "acquisition_channel": channel,
        "coupon_applied": coupon,
        "checkout_seconds": pd.to_numeric(col("checkout_seconds", defaults["checkout_seconds"]), errors="coerce").fillna(defaults["checkout_seconds"]),
        "hour_of_day": hour, "rto": is_rto, "disputed": False,
        "merchant_margin": pd.to_numeric(col("merchant_margin", defaults["merchant_margin"]), errors="coerce").fillna(defaults["merchant_margin"]),
        "cac": pd.to_numeric(col("cac", defaults["cac"]), errors="coerce").fillna(defaults["cac"]),
    })
    df = df[keep & df.cart_gmv.notna() & (df.cart_gmv > 0) & (df.delivery_pin.str.len() == 6)].copy()
    report["dropped"]["bad_amount_or_pin"] = int(keep.sum() - len(df))
    for c in ("ip", "vpa", "card_token", "payment_switch_from", "customer_email"):
        df[c] = df[c].replace("", np.nan)
    df = df.sort_values("ts").reset_index(drop=True)
    first = ~df.duplicated("customer_phone")
    df["is_new_customer"] = first
    df["cohort"] = "unknown"
    df["ring_id"] = np.nan
    df["shared_addr_id"] = np.nan
    df["latent_p_rto"] = np.nan
    report.update({"rows_out": int(len(df)), "cod_share": float((df.payment_method == "COD").mean()) if len(df) else 0.0,
                   "cod_rto_rate": float(df.loc[df.payment_method == "COD", "rto"].mean()) if (df.payment_method == "COD").any() else 0.0,
                   "defaults_used": sorted(set(report["defaults_used"]))})
    return df, report


def write_mapping_example(path: str | Path) -> None:
    Path(path).write_text(json.dumps(MAPPING_EXAMPLE, indent=2), encoding="utf-8")
