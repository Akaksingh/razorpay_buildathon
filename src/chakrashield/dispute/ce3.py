"""Visa Compelling Evidence 3.0 -- deterministic evidence compiler.

CE3.0 (effective April 2023) lets a merchant shift liability on a
reason-code 10.4 (card-absent fraud) dispute by proving the cardholder has
an established, undisputed history with the merchant. The rule set is
mechanical, which is exactly why we do not put an LLM anywhere near it:

  1. Disputed transaction is card-not-present, reason code 10.4.
  2. At least TWO prior transactions on the same payment credential.
  3. Each prior transaction is between 120 and 365 days before the
     dispute date.
  4. No prior transaction was itself disputed / reported as fraud.
  5. Each prior transaction shares at least TWO core data elements with
     the disputed transaction, of which at least ONE is the IP address or
     the device ID / fingerprint; the others may be shipping address or
     account / login ID.
  6. Merchant supplies item description and transaction dates.

Every input hash and every match is written into the packet and the packet
is content-addressed (SHA-256) so a network reviewer can re-derive it.
``render.py`` turns the same dict into a printable document; the JSON here
stays the contract and the only thing the hash covers.

The Mastercard difference
-------------------------
This compiler is Visa-specific by design. Mastercard handles the same
situation -- a card-not-present "I did not authorise this" claim from a
cardholder with a real history at the merchant -- differently in ways that
change what a merchant can automate:

* Vocabulary and flow. Mastercard says chargeback, second presentment,
  pre-arbitration and arbitration where Visa says dispute, dispute
  response, pre-arbitration and arbitration. The card-not-present fraud
  chargeback is Mastercard reason code 4837 (No Cardholder Authorization);
  Visa's is 10.4.
* No mechanical liability shift. Visa CE3.0 is a rule: when the six
  criteria above are met the merchant is not liable, and the issuer's
  acceptance is not a matter of judgement. Mastercard's chargeback rules
  for 4837 let the merchant present compelling evidence in the second
  presentment -- and evidence of earlier undisputed transactions on the
  same card is a recognised form of it -- but the issuer evaluates that
  evidence, and a refusal goes to pre-arbitration rather than being
  blocked by rule. Mastercard's general rules also do not, to our
  knowledge, fix a 120-365 day window or a two-transaction minimum; the
  numbers in this module must not be reused for a 4837 response.
* Data at authorisation, not at dispute time. Mastercard's First-Party
  Trust programme (announced 2024, first rolled out in the United States)
  protects merchants from first-party-misuse chargebacks when identity and
  delivery data elements -- of the same kind CE3.0 matches: device, IP,
  account and shipping identifiers -- were supplied with the transaction
  and match the issuer's own view. The consequence for a system like this
  one is that the ledger must feed the authorisation message, because a
  packet assembled after the chargeback arrives, which is what CE3.0
  allows, is too late. The exact data elements, thresholds, effective
  dates and regions of that programme change by bulletin and are not
  reproduced here.
* Pre-dispute channel. Visa's pre-dispute leg of CE3.0 runs through
  Verifi Order Insight; Mastercard's equivalent is Ethoca Consumer
  Clarity. Both let the issuer show the cardholder merchant-supplied order
  detail before a dispute is raised, but the Mastercard channel is not tied
  to a liability rule the way Order Insight is tied to CE3.0.

Practical upshot: the ledger search in ``compile_ce3`` is still worth running
for a Mastercard 4837 chargeback, because the prior-transaction evidence is
persuasive, but the packet must be labelled as compelling evidence for a
second presentment, not as a CE3.0 liability shift, and the criteria table
must not be presented as pass/fail against Mastercard rules.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

_DAY = 86400.0
MIN_PRIOR_DAYS = 120
MAX_PRIOR_DAYS = 365
MIN_PRIOR_TXNS = 2
MIN_MATCH_ELEMENTS = 2
PRIMARY_ELEMENTS = ("ip_hash", "device_hash")
SECONDARY_ELEMENTS = ("addr_hash", "account_id")


@dataclass
class TxnRecord:
    transaction_id: str
    ts: float
    card_token: str
    ip_hash: str
    device_hash: str
    addr_hash: str
    account_id: str
    amount: float
    items: str
    disputed: bool
    delivered: bool

    def elements(self) -> dict[str, str]:
        return {"ip_hash": self.ip_hash, "device_hash": self.device_hash, "addr_hash": self.addr_hash,
                "account_id": self.account_id}


class TransactionLedger:
    """In-memory index of historical card transactions keyed by credential."""

    def __init__(self) -> None:
        self._by_id: dict[str, TxnRecord] = {}
        self._by_card: dict[str, list[TxnRecord]] = {}

    @classmethod
    def from_frame(cls, df: pd.DataFrame) -> "TransactionLedger":
        from ..features.velocity import hash_entity, normalize_address

        led = cls()
        card = df[df["payment_method"] == "CARD"]
        for r in card.itertuples(index=False):
            rec = TxnRecord(
                transaction_id=r.order_id, ts=float(r.ts), card_token=str(r.card_token),
                ip_hash=hash_entity("ip", r.ip), device_hash=hash_entity("device", r.device_fingerprint),
                addr_hash=hash_entity("addr", normalize_address(r.shipping_address, r.delivery_pin)),
                account_id=str(r.account_id), amount=float(r.cart_gmv),
                items=f"{int(r.items_count)} item(s), {r.weight_grams:.0f} g", disputed=bool(r.disputed),
                delivered=not bool(r.rto),
            )
            led.add(rec)
        return led

    def add(self, rec: TxnRecord) -> None:
        self._by_id[rec.transaction_id] = rec
        self._by_card.setdefault(rec.card_token, []).append(rec)

    def get(self, txn_id: str) -> TxnRecord | None:
        return self._by_id.get(txn_id)

    def by_card(self, token: str) -> list[TxnRecord]:
        return list(self._by_card.get(token, []))

    def __len__(self) -> int:
        return len(self._by_id)

    def sample_disputable(self, n: int = 10) -> list[str]:
        """Transactions that have >=2 prior card txns (for demo pickers)."""
        out = []
        for token, txns in self._by_card.items():
            txns = sorted(txns, key=lambda t: t.ts)
            if len(txns) >= 3:
                out.append(txns[-1].transaction_id)
            if len(out) >= n:
                break
        return out


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def compile_ce3(ledger: TransactionLedger, transaction_id: str, dispute_reason_code: str = "10.4",
                dispute_date: str | None = None) -> dict:
    disputed = ledger.get(transaction_id)
    if disputed is None:
        return {"transaction_id": transaction_id, "eligible": False, "standard": "Visa CE3.0",
                "criteria": {}, "evidence": {}, "reason": "transaction not found in ledger", "packet_hash": ""}

    d_ts = datetime.strptime(dispute_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() \
        if dispute_date else disputed.ts + 45 * _DAY
    criteria: dict[str, dict] = {}
    criteria["reason_code_10_4"] = {"pass": dispute_reason_code == "10.4", "value": dispute_reason_code}

    window_lo, window_hi = d_ts - MAX_PRIOR_DAYS * _DAY, d_ts - MIN_PRIOR_DAYS * _DAY
    candidates = [t for t in ledger.by_card(disputed.card_token) if t.transaction_id != disputed.transaction_id]
    in_window = [t for t in candidates if window_lo <= t.ts <= window_hi]
    undisputed = [t for t in in_window if not t.disputed]

    matched = []
    for t in sorted(undisputed, key=lambda x: x.ts, reverse=True):
        prim = [e for e in PRIMARY_ELEMENTS if t.elements()[e] and t.elements()[e] == disputed.elements()[e]]
        sec = [e for e in SECONDARY_ELEMENTS if t.elements()[e] and t.elements()[e] == disputed.elements()[e]]
        elems = prim + sec
        if len(elems) >= MIN_MATCH_ELEMENTS and len(prim) >= 1:
            matched.append({
                "transaction_id": t.transaction_id, "date": _iso(t.ts), "days_before_dispute": round((d_ts - t.ts) / _DAY),
                "amount_inr": t.amount, "items": t.items, "delivered": t.delivered,
                "matched_elements": elems, "element_hashes": {e: t.elements()[e] for e in elems},
            })
    criteria["prior_txns_in_120_365d_window"] = {"pass": len(in_window) >= MIN_PRIOR_TXNS, "count": len(in_window),
                                                 "window": [_iso(window_lo), _iso(window_hi)]}
    criteria["prior_txns_undisputed"] = {"pass": len(undisputed) >= MIN_PRIOR_TXNS, "count": len(undisputed)}
    criteria["prior_txns_with_2_matching_elements_incl_ip_or_device"] = {"pass": len(matched) >= MIN_PRIOR_TXNS, "count": len(matched)}
    eligible = all(c["pass"] for c in criteria.values())

    evidence = {
        "disputed_transaction": {
            "transaction_id": disputed.transaction_id, "date": _iso(disputed.ts), "amount_inr": disputed.amount,
            "items": disputed.items, "credential": disputed.card_token[:8] + "…",
            "elements": disputed.elements(),
        },
        "dispute_date": _iso(d_ts),
        "prior_transactions": matched[:MIN_PRIOR_TXNS] if eligible else matched,
        "additional_prior_transactions": max(0, len(matched) - MIN_PRIOR_TXNS) if eligible else 0,
        "merchant_narrative": (
            f"Cardholder credential {disputed.card_token[:8]}… has {len(matched)} undisputed prior "
            f"transactions 120–365 days before the dispute, each sharing "
            f"{'device fingerprint / IP' if matched else 'no'} and shipping/account identifiers with the "
            f"disputed transaction. Liability shift requested under Visa CE3.0."
        ) if eligible else "CE3.0 criteria not satisfied; respond with standard compelling evidence.",
    }
    packet = {"transaction_id": transaction_id, "standard": "Visa CE3.0", "criteria": criteria, "evidence": evidence}
    canon = json.dumps(packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    packet_hash = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    reason = "eligible: two or more qualifying prior transactions" if eligible else \
        next((f"failed: {k}" for k, v in criteria.items() if not v["pass"]), "failed")
    return {**packet, "eligible": eligible, "reason": reason, "packet_hash": packet_hash}
