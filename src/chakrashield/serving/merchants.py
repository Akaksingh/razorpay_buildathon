"""Per-merchant configuration: economics, friction budget, control band, shadow mode, API key.

The resolver prices every decision in the merchant's own rupees, so two
merchants sending the same order must get different thresholds: a 32 %-margin
beauty brand with a Rs.650 CAC has a higher tau*(x) than an 18 %-margin
generalist, and a brand whose buyers balk at a deposit needs a larger delta_s
prior until the learner has data. Those constants live here, keyed by
``merchant_id``, and are resolved once per request with a dict lookup so the
hot path pays nothing for multi-tenancy.

Shadow mode is how a merchant trials the engine with zero customer impact:
the scorer, conformal gate, resolver, learner lookup, drift monitor and
propensity ledger all run exactly as in production, the response carries the
action the policy *would* have taken, but the served decision is always
ALLOW_COD. Because every shadow order ships, every one of them earns an
untreated delivery label with propensity 1 -- the trial doubles as an
unbiased calibration set for that merchant's first retraining, and the
ledger can replay the P&L the engine would have protected.

API keys are optional and enforced only when ``CHAKRA_REQUIRE_API_KEY=1``, so
the local demo and the test suite run without headers. Keys sit in the same
JSON file for the hackathon; a deployment would keep them in a secret store
and load them here.
"""
from __future__ import annotations

import hmac
import json
import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from ..config import ECONOMICS, ROOT, Economics

DEFAULT_MERCHANT_ID = "demo_merchant"
MERCHANTS_PATH = Path(os.environ.get("CHAKRA_MERCHANTS", ROOT / "config" / "merchants.json"))
_ECON_FIELDS = frozenset(f.name for f in fields(Economics))


def api_key_required() -> bool:
    """Read at request time so a deployment (or a test) can flip enforcement without a restart."""
    return os.environ.get("CHAKRA_REQUIRE_API_KEY", "0").strip() == "1"


@dataclass(frozen=True)
class MerchantConfig:
    merchant_id: str
    label: str = ""
    economics: Economics = field(default_factory=lambda: ECONOMICS)
    #: the Economics fields this merchant changes, for the response and /v1/merchants
    overrides: dict = field(default_factory=dict)
    friction_budget: float | None = None     # applied when the request does not set one
    epsilon: float | None = None             # None -> the global CHAKRA_EPSILON control band
    shadow: bool = False
    api_key: str | None = None

    @classmethod
    def from_entry(cls, merchant_id: str, entry: dict, base: Economics) -> "MerchantConfig":
        overrides = dict(entry.get("economics") or {})
        unknown = sorted(set(overrides) - _ECON_FIELDS)
        if unknown:
            raise ValueError(f"merchant {merchant_id!r}: unknown Economics fields {unknown}")
        budget = entry.get("friction_budget")
        if budget is not None and not 0.0 <= float(budget) <= 1.0:
            raise ValueError(f"merchant {merchant_id!r}: friction_budget must be in [0, 1]")
        eps = entry.get("epsilon")
        if eps is not None and not 0.0 <= float(eps) <= 1.0:
            raise ValueError(f"merchant {merchant_id!r}: epsilon must be in [0, 1]")
        return cls(merchant_id=merchant_id, label=str(entry.get("label", "")),
                   economics=replace(base, **overrides) if overrides else base, overrides=overrides,
                   friction_budget=None if budget is None else float(budget),
                   epsilon=None if eps is None else float(eps), shadow=bool(entry.get("shadow", False)),
                   api_key=entry.get("api_key") or None)

    def public(self) -> dict:
        """Everything about the merchant except the key."""
        return {"id": self.merchant_id, "label": self.label, "shadow": self.shadow, "epsilon": self.epsilon,
                "friction_budget": self.friction_budget, "economics": dict(self.overrides),
                "has_api_key": self.api_key is not None}

    def response_block(self, requested_id: str, known: bool) -> dict:
        """The ``merchant`` block of a risk response."""
        out = {"id": self.merchant_id, "known": known, "shadow": self.shadow, "economics": dict(self.overrides),
               "economics_source": "merchant" if self.overrides else "default"}
        if not known:
            out["requested_id"] = requested_id
            out["note"] = f"unknown merchant_id {requested_id!r}; {self.merchant_id} defaults applied"
        return out


class MerchantRegistry:
    """Merchant id -> MerchantConfig, with a guaranteed default to fall back on."""

    def __init__(self, merchants: dict[str, MerchantConfig], default_id: str = DEFAULT_MERCHANT_ID, path: Path | None = None) -> None:
        if default_id not in merchants:
            merchants = {default_id: MerchantConfig(merchant_id=default_id, label="repo defaults"), **merchants}
        self._merchants = merchants
        self.default_id = default_id
        self.path = path

    @classmethod
    def builtin(cls) -> "MerchantRegistry":
        """Only the default merchant, on the global economics. What a process gets before the file is read."""
        return cls({}, path=None)

    @classmethod
    def load(cls, path: Path | None = None, base: Economics | None = None) -> "MerchantRegistry":
        path = Path(path or MERCHANTS_PATH)
        base = base or ECONOMICS
        if not path.exists():
            return cls.builtin()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls({mid: MerchantConfig.from_entry(mid, entry, base) for mid, entry in raw.items()}, path=path)

    def resolve(self, merchant_id: str) -> tuple[MerchantConfig, bool]:
        """(config, known). An unknown id gets the default merchant and known=False; one dict lookup."""
        m = self._merchants.get(merchant_id)
        if m is not None:
            return m, True
        return self._merchants[self.default_id], False

    def authorize(self, merchant: MerchantConfig, known: bool, presented: str | None) -> bool:
        """True iff the presented key matches the merchant's. Unknown merchants and merchants
        without a key cannot authenticate: when keys are required, both are refused."""
        if not known or merchant.api_key is None or not presented:
            return False
        return hmac.compare_digest(merchant.api_key.encode("utf-8"), presented.encode("utf-8"))

    def public(self) -> list[dict]:
        return [m.public() for m in self._merchants.values()]

    def __len__(self) -> int:
        return len(self._merchants)

    def __contains__(self, merchant_id: str) -> bool:
        return merchant_id in self._merchants
