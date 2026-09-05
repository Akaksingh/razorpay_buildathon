"""Synthetic Indian D2C order stream with organised COD-abuse syndicates.

We simulate a *world*, not a table: persistent customers with devices and
addresses, syndicates that rotate burner phones through shared handsets and
drop addresses, and an impulse-buyer cohort acquired from paid social. RTO
is drawn from a latent logistic model over the causes practitioners
actually observe (address quality, PIN tier, channel, payment fallback,
basket size, ring membership), so the learned model has real structure to
find and the reason codes mean something.

Everything downstream (features, labels, CE3.0 history) is derived from
this event stream by chronological replay -- never from the latent truth.
"""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import pincodes

CHANNELS = ["ORGANIC", "DIRECT", "WHATSAPP", "GOOGLE_ADS", "MARKETPLACE", "META_ADS", "INFLUENCER", "AFFILIATE"]
CHANNEL_W_LEGIT = [0.26, 0.12, 0.08, 0.18, 0.10, 0.16, 0.06, 0.04]
CHANNEL_W_IMPULSE = [0.04, 0.02, 0.06, 0.12, 0.06, 0.38, 0.22, 0.10]
CHANNEL_W_RING = [0.05, 0.03, 0.05, 0.10, 0.12, 0.25, 0.15, 0.25]

_DAY = 86400.0
_T0 = 1_700_000_000.0  # epoch anchor (Nov 2023) so timestamps look real

_FIRST = ["Rahul", "Priya", "Amit", "Sneha", "Vikram", "Anjali", "Rohit", "Neha", "Suresh", "Kavita", "Arjun",
          "Pooja", "Manish", "Divya", "Sanjay", "Meena", "Deepak", "Ritu", "Karan", "Swati", "Nikhil", "Asha"]
_STREETS = ["MG Road", "Nehru Nagar", "Gandhi Colony", "Sector 12", "Indira Nagar", "Station Road", "Rajaji Street",
            "Shivaji Marg", "Anna Nagar", "Vasant Vihar", "Lake View Layout", "Model Town", "Shastri Nagar",
            "Civil Lines", "Koramangala 5th Block", "Andheri West", "Salt Lake Sector V", "Banjara Hills",
            "Jubilee Hills", "Powai", "Hinjewadi Phase 2", "Whitefield", "Dwarka Sector 7", "Malviya Nagar"]
_SOCIETIES = ["Green Meadows", "Prestige Lakeside", "Shanti Apartments", "Sai Residency", "Lodha Splendora",
              "DLF Phase 3", "Brigade Gateway", "Mahindra World City", "Godrej Garden City", "Sobha Dream Acres"]
_LANDMARKS = ["Hanuman Temple", "Bus Stand", "Govt School", "Old Post Office", "Shiv Mandir", "Jama Masjid",
              "Railway Station", "Primary Health Centre", "Panchayat Bhawan", "Water Tank", "Petrol Pump"]
_VILLAGES = ["Rampur", "Sultanpur", "Bhagwanpur", "Kishanganj", "Madhopur", "Chandpur", "Gopalganj", "Narayanpur"]


def _oid(i: int) -> str:
    return f"ord_{hashlib.sha1(str(i).encode()).hexdigest()[:12]}"


def _phone(rng: random.Random) -> str:
    return f"{rng.choice('6789')}{rng.randint(10**8, 10**9 - 1)}"


def _device(rng: random.Random) -> str:
    return "fp_" + "".join(rng.choices("0123456789abcdef", k=24))


def _card(rng: random.Random) -> str:
    return "tok_" + "".join(rng.choices("0123456789abcdef", k=16))


def _good_address(rng: random.Random, pin: str) -> str:
    info = pincodes.lookup(pin)
    forms = [
        f"{rng.randint(1, 450)}, {rng.choice(_STREETS)}, {info.city}, {info.state} {pin}",
        f"Flat {rng.randint(101, 1204)}, {rng.choice(_SOCIETIES)}, {rng.choice(_STREETS)}, {info.city} {pin}",
        f"H.No {rng.randint(1, 99)}-{rng.randint(1, 999)}, {rng.choice(_STREETS)}, near {rng.choice(_LANDMARKS)}, {info.city}",
        f"Plot {rng.randint(1, 300)}, {rng.choice(_STREETS)}, {info.city}, {info.state}",
        f"#{rng.randint(10, 999)}, {rng.randint(1, 12)}th Cross, {rng.choice(_STREETS)}, {info.city} - {pin}",
    ]
    return rng.choice(forms)


def _weak_address(rng: random.Random, pin: str) -> str:
    info = pincodes.lookup(pin)
    forms = [
        f"Near {rng.choice(_LANDMARKS)}, Ward {rng.randint(1, 20)}, {info.city}",
        f"Village {rng.choice(_VILLAGES)}, PO {rng.choice(_VILLAGES)}, {info.state}",
        f"Opp {rng.choice(_LANDMARKS)}, {rng.choice(_STREETS)}",
        f"{rng.choice(_VILLAGES)} gaon, {info.state}",
        f"Behind {rng.choice(_LANDMARKS)} {info.city}",
        f"{rng.choice(_STREETS)}, {info.city}",
    ]
    return rng.choice(forms)


def _junk_address(rng: random.Random, pin: str) -> str:
    return rng.choice(["asdf asdf", "test address", "xxxxx", "na", "home", f"{rng.choice(_VILLAGES)}", "near shop"])


@dataclass
class Customer:
    phone: str
    devices: list[str]
    addresses: list[tuple[str, str]]     # (address text, pin)
    tier: int
    reliability: float                   # latent propensity to accept delivery (0..1)
    address_quality: float               # prob of giving a good address
    channel_w: list[float]
    cohort: str                          # legit | impulse | ring
    card: str | None
    email: str
    account_id: str
    ip: str
    ring_id: str | None = None
    order_rate: float = 1.0
    first_order_ts: float | None = None


@dataclass
class Ring:
    ring_id: str
    devices: list[str]
    addresses: list[tuple[str, str]]
    phones: list[str] = field(default_factory=list)
    ips: list[str] = field(default_factory=list)
    burst_start: float = 0.0
    burst_len: float = 30 * _DAY


def _sample_pin(rng: random.Random, tier_w=(0.42, 0.33, 0.14, 0.11)) -> tuple[str, int]:
    tier = rng.choices([1, 2, 3, 4], weights=tier_w)[0]
    return rng.choice(pincodes.SAMPLE_PINS[tier]), tier


def _make_customer(rng: random.Random, cohort: str, ring: Ring | None = None) -> Customer:
    if cohort == "ring":
        pin, tier = ring.addresses[0][1], pincodes.lookup(ring.addresses[0][1]).tier
        return Customer(
            phone=_phone(rng), devices=[rng.choice(ring.devices)], addresses=list(ring.addresses), tier=tier,
            reliability=rng.betavariate(1.2, 4.0), address_quality=0.25, channel_w=CHANNEL_W_RING, cohort="ring",
            card=None, email=f"user{rng.randint(1000, 999999)}@mail.ru", account_id=f"acc_{rng.randint(10**6, 10**7)}",
            ip=rng.choice(ring.ips), ring_id=ring.ring_id, order_rate=1.2,
        )
    pin, tier = _sample_pin(rng)
    if cohort == "impulse":
        rel = rng.betavariate(2.2, 2.4)
        aq = 0.55
        cw = CHANNEL_W_IMPULSE
        rate = 0.6
    else:
        rel = rng.betavariate(6.0, 1.6) if tier <= 2 else rng.betavariate(4.0, 1.8)
        aq = 0.86 if tier <= 2 else 0.68
        cw = CHANNEL_W_LEGIT
        rate = 1.0
    n_addr = 1 if rng.random() < 0.7 else 2
    addrs = []
    for _ in range(n_addr):
        p = pin if rng.random() < 0.8 else _sample_pin(rng)[0]
        addrs.append((_good_address(rng, p) if rng.random() < aq else _weak_address(rng, p), p))
    name = rng.choice(_FIRST).lower()
    return Customer(
        phone=_phone(rng), devices=[_device(rng) for _ in range(1 if rng.random() < 0.75 else 2)], addresses=addrs,
        tier=tier, reliability=rel, address_quality=aq, channel_w=cw, cohort=cohort,
        card=_card(rng) if rng.random() < 0.55 else None, email=f"{name}{rng.randint(1, 9999)}@gmail.com",
        account_id=f"acc_{rng.randint(10**6, 10**7)}", ip=f"ip_{rng.randint(10**6, 10**7)}", order_rate=rate,
    )


def _rto_prob(*, cohort: str, reliability: float, tier: int, addr_defect: float, channel: str, gmv: float,
              switched: bool, is_new: bool, items: int, hour: int, coupon: bool, pay: str, prior_rtos: int) -> float:
    if pay != "COD":
        z = -3.3 + 0.9 * addr_defect + 0.25 * (tier - 1) + (1.5 if cohort == "ring" else 0.0)
        return 1 / (1 + math.exp(-z))
    z = -2.75
    z += -2.2 * (reliability - 0.5)
    z += 0.32 * (tier - 1)
    z += 1.9 * addr_defect
    z += {"ORGANIC": -0.25, "DIRECT": -0.3, "WHATSAPP": 0.0, "GOOGLE_ADS": 0.05, "MARKETPLACE": 0.1,
          "META_ADS": 0.35, "INFLUENCER": 0.45, "AFFILIATE": 0.6}[channel]
    z += 0.45 * math.log1p(gmv / 800.0)
    z += 1.1 if switched else 0.0
    z += 0.35 if is_new else -0.2
    z += 0.12 * (items - 1)
    z += 0.3 if (hour >= 23 or hour <= 4) else 0.0
    z += 0.2 if coupon else 0.0
    z += 0.55 * min(prior_rtos, 3)
    if cohort == "ring":
        z += 1.6
    return 1 / (1 + math.exp(-z))


def generate(n_orders: int = 60_000, days: int = 400, seed: int = 42, n_customers: int = 18_000,
             n_rings: int = 45) -> pd.DataFrame:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # ---- population --------------------------------------------------------
    rings: list[Ring] = []
    for r in range(n_rings):
        pin, _ = _sample_pin(rng, tier_w=(0.2, 0.3, 0.25, 0.25))
        n_dev = rng.randint(2, 6)
        n_addr = rng.randint(1, 4)
        addrs = [((_weak_address(rng, pin) if rng.random() < 0.7 else _good_address(rng, pin)), pin) for _ in range(n_addr)]
        rings.append(Ring(
            ring_id=f"ring_{r:03d}", devices=[_device(rng) for _ in range(n_dev)], addresses=addrs,
            ips=[f"ip_{rng.randint(10**6, 10**7)}" for _ in range(rng.randint(1, 3))],
            burst_start=rng.uniform(0, days - 60) * _DAY, burst_len=rng.uniform(20, 90) * _DAY,
        ))
    customers: list[Customer] = []
    for _ in range(int(n_customers * 0.80)):
        customers.append(_make_customer(rng, "legit"))
    for _ in range(int(n_customers * 0.12)):
        customers.append(_make_customer(rng, "impulse"))
    for ring in rings:
        for _ in range(rng.randint(6, 40)):
            c = _make_customer(rng, "ring", ring)
            ring.phones.append(c.phone)
            customers.append(c)

    # ---- order stream ------------------------------------------------------
    weights = np.array([c.order_rate for c in customers], dtype=float)
    weights /= weights.sum()
    idx = np_rng.choice(len(customers), size=n_orders, p=weights)
    rows = []
    orders_by_customer: dict[str, list] = {}
    for i, ci in enumerate(idx):
        c = customers[ci]
        if c.cohort == "ring":
            ring = rings[int(c.ring_id.split("_")[1])]
            ts = ring.burst_start + rng.random() * ring.burst_len
            if rng.random() < 0.15:
                ts = rng.uniform(0, days * _DAY)
        else:
            ts = rng.uniform(0, days * _DAY)
        ts = _T0 + ts
        hist = orders_by_customer.setdefault(c.phone, [])
        prior = [h for h in hist if h["ts"] < ts]
        is_new = len(prior) == 0
        prior_rtos = sum(1 for h in prior if h["rto"])

        pay = "COD" if rng.random() < (0.62 if c.cohort != "ring" else 0.93) else rng.choice(["UPI", "UPI", "CARD", "CARD", "NETBANKING", "WALLET"])
        if pay == "CARD" and c.card is None:
            pay = "UPI"
        switched = None
        if pay == "COD" and rng.random() < (0.05 if c.cohort == "legit" else 0.22):
            switched = rng.choice(["CARD_FAILED", "UPI_FAILED"])
        channel = rng.choices(CHANNELS, weights=c.channel_w)[0]
        addr_text, pin = rng.choice(c.addresses)
        if c.cohort != "ring" and rng.random() < 0.03:
            addr_text = _junk_address(rng, pin)
        tier = pincodes.lookup(pin).tier
        base = {1: 1350, 2: 1100, 3: 900, 4: 780}[tier]
        gmv = float(max(149, round(np_rng.lognormal(math.log(base), 0.55) * (1.6 if c.cohort == "ring" else 1.0), -1)))
        items = 1 + int(np_rng.poisson(0.6 if c.cohort != "ring" else 1.4))
        weight = float(max(120, np_rng.normal(380 + 160 * items, 120)))
        hour = int(np_rng.choice(24, p=_hour_profile(c.cohort)))
        coupon = rng.random() < (0.25 if c.cohort != "impulse" else 0.55)
        checkout_s = float(max(8, np_rng.lognormal(math.log(70 if c.cohort != "ring" else 28), 0.5)))
        from ..features.address import score_address  # local import: avoid cycle at module load
        addr_defect = score_address(addr_text, pin).defect_score
        p_rto = _rto_prob(cohort=c.cohort, reliability=c.reliability, tier=tier, addr_defect=addr_defect,
                          channel=channel, gmv=gmv, switched=bool(switched), is_new=is_new, items=items,
                          hour=hour, coupon=coupon, pay=pay, prior_rtos=prior_rtos)
        rto = rng.random() < p_rto
        outcome_ts = ts + rng.uniform(2.5, 9.0) * _DAY
        disputed = False
        if pay == "CARD":
            disputed = rng.random() < (0.006 if c.cohort == "legit" else 0.05)
        row = {
            "order_id": _oid(i), "ts": ts, "outcome_ts": outcome_ts, "customer_phone": c.phone,
            "customer_email": c.email, "account_id": c.account_id, "device_fingerprint": rng.choice(c.devices),
            "ip": c.ip if rng.random() < 0.85 else f"ip_{rng.randint(10**6, 10**7)}",
            "vpa": (f"{c.phone}@upi" if pay == "UPI" or rng.random() < 0.3 else None),
            "card_token": c.card if pay == "CARD" else None, "shipping_address": addr_text, "delivery_pin": pin,
            "pin_tier": tier, "cart_gmv": gmv, "items_count": items, "weight_grams": round(weight, 0),
            "payment_method": pay, "payment_switch_from": switched, "acquisition_channel": channel,
            "coupon_applied": coupon, "checkout_seconds": round(checkout_s, 1), "hour_of_day": hour,
            "is_new_customer": is_new, "cohort": c.cohort, "ring_id": c.ring_id, "latent_p_rto": round(p_rto, 4),
            "rto": rto, "disputed": disputed, "merchant_margin": 0.18, "cac": _cac_for(channel, rng),
        }
        rows.append(row)
        hist.append({"ts": ts, "rto": rto})
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df


def _hour_profile(cohort: str) -> np.ndarray:
    base = np.array([1, 0.6, 0.4, 0.3, 0.3, 0.5, 1.2, 2, 3, 3.5, 4, 4.2, 4.5, 4.3, 4, 4.2, 4.5, 5, 5.5, 6, 6.2, 5.5, 4, 2.2], dtype=float)
    if cohort == "ring":
        base = base * np.array([2.5, 2.5, 2.5, 2, 1.5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.5, 2, 2.5])
    return base / base.sum()


def _cac_for(channel: str, rng: random.Random) -> float:
    base = {"ORGANIC": 90, "DIRECT": 60, "WHATSAPP": 180, "GOOGLE_ADS": 380, "MARKETPLACE": 260,
            "META_ADS": 520, "INFLUENCER": 610, "AFFILIATE": 470}[channel]
    return float(round(base * rng.uniform(0.75, 1.3), 0))
