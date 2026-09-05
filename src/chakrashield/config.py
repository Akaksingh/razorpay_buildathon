"""Central configuration for ChakraShield.

Everything that is a *business* constant (unit economics, conformal alpha,
latency budgets, file locations) lives here so the model layer, the policy
layer and the serving layer cannot drift apart.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("CHAKRA_ARTIFACTS", ROOT / "artifacts"))
MODEL_DIR = ARTIFACTS / "models"
DATA_DIR = ARTIFACTS / "data"
REPORT_DIR = ARTIFACTS / "reports"

for _d in (MODEL_DIR, DATA_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Conformal prediction
# ---------------------------------------------------------------------------
#: Target miscoverage rate. alpha=0.10 -> 90% class-conditional coverage.
CONFORMAL_ALPHA = float(os.environ.get("CHAKRA_ALPHA", 0.10))

# ---------------------------------------------------------------------------
# Latency budget (milliseconds). The gateway self-reports a breach rather than
# silently blowing the SLA -- a risk engine that is slow is a risk engine that
# gets bypassed by the checkout team.
# ---------------------------------------------------------------------------
LATENCY_BUDGET_MS = float(os.environ.get("CHAKRA_LATENCY_BUDGET_MS", 25.0))

# ---------------------------------------------------------------------------
# Unit economics defaults (INR). Overridable per-merchant at request time.
# Calibrated against public Indian D2C benchmarks: RTO on COD runs 18-30%,
# forward+reverse logistics on a sub-500g parcel is ~Rs.110-140 each way.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Economics:
    # --- cost of a *false negative*: we allowed COD on an order that RTO'd ---
    forward_shipping: float = 95.0     # first-mile + line-haul + last-mile attempt
    reverse_shipping: float = 110.0    # RTO leg is priced above forward by most 3PLs
    packaging_cost: float = 18.0       # non-recoverable once shipped
    restocking_cost: float = 22.0      # QC, re-bagging, shelf return
    holding_cost_rate: float = 0.015   # cost of capital while inventory is in transit
    inventory_lockup_lambda: float = 0.02  # lambda: opportunity cost as frac of GMV

    # --- cost of a *false positive*: we frictioned an order that was good ---
    default_margin: float = 0.18       # M: contribution margin on GMV
    default_cac: float = 420.0         # CAC: blended paid-acquisition cost
    cac_insult_kappa: float = 0.55     # kappa: share of CAC destroyed by a hard block
                                       # (a blocked customer is damaged, not always lost)

    # --- behavioural response to each intervention -------------------------
    stepup_deposit: float = 49.0       # the refundable UPI shipping token
    stepup_abandon_rate: float = 0.11  # delta_s: good customers lost to a Rs.49 prompt
    prepaid_abandon_rate: float = 0.38 # delta_p: good customers lost to a hard COD block
    stepup_rto_residual: float = 0.10  # rho: RTO risk surviving a paid deposit
    prepaid_rto_residual: float = 0.04 # prepaid orders still RTO, but cash is held
    prepaid_rto_unit_cost: float = 60.0  # refund + logistics on a prepaid return
    stepup_friction_cost: float = 6.0  # UPI collect MDR + support contacts

    # --- friction budget --------------------------------------------------
    #: lambda_f: rupees charged to every non-ALLOW action. The Lagrange multiplier
    #: of a merchant's "at most X% of orders may be frictioned" constraint. 0 = pure
    #: expected-cost minimisation. Per-request override via RiskRequest.
    friction_shadow_price: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: Path | None = None) -> "Economics":
        path = path or (ROOT / "config" / "economics.json")
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as fh:
                return cls(**json.load(fh))
        return cls()


ECONOMICS = Economics.load()

# ---------------------------------------------------------------------------
# Reason-code emission
# ---------------------------------------------------------------------------
MAX_REASON_CODES = 4
#: A SHAP contribution must move log-odds by at least this much to be surfaced.
#: Stops the API from emitting noise codes that a merchant ops team will ignore.
REASON_CODE_MIN_ABS_SHAP = 0.05
