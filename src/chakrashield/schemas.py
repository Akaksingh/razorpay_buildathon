"""Pydantic contracts for the risk gateway. This *is* the API documentation."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

PaymentMethod = Literal["COD", "UPI", "CARD", "NETBANKING", "WALLET", "EMI"]
Channel = Literal["ORGANIC", "META_ADS", "GOOGLE_ADS", "AFFILIATE", "INFLUENCER", "WHATSAPP", "MARKETPLACE", "DIRECT"]


class RiskRequest(BaseModel):
    """Everything the checkout SDK knows at the moment the buyer taps 'Place order'."""

    order_id: Optional[str] = Field(None, description="Merchant order id (idempotency key)")
    merchant_id: str = Field("demo_merchant")
    customer_phone: str = Field(..., min_length=6, description="Raw phone; hashed at ingress, never stored raw")
    customer_email: Optional[str] = None
    delivery_pin: str = Field(..., min_length=6, max_length=6)
    shipping_address: str = Field(..., min_length=1)
    cart_gmv: float = Field(..., gt=0, description="INR")
    items_count: int = Field(1, ge=1)
    weight_grams: float = Field(450.0, gt=0)
    device_fingerprint_hash: str = Field(..., min_length=4)
    ip_hash: Optional[str] = None
    vpa: Optional[str] = Field(None, description="UPI VPA if the buyer has used UPI before / offers it")
    payment_method: PaymentMethod = "COD"
    payment_switch_from: Optional[str] = Field(None, description="e.g. CARD_FAILED when buyer fell back to COD")
    acquisition_channel: Channel = "ORGANIC"
    coupon_applied: bool = False
    checkout_seconds: float = Field(90.0, ge=0)
    hour_of_day: Optional[int] = Field(None, ge=0, le=23)
    is_new_customer: Optional[bool] = Field(None, description="Override; else inferred from store")

    # merchant economics (defaults come from config)
    merchant_margin: Optional[float] = Field(None, ge=0, le=1)
    cac: Optional[float] = Field(None, ge=0)
    logistics_loss: Optional[float] = Field(None, ge=0)
    holding_cost: Optional[float] = Field(None, ge=0)

    @field_validator("delivery_pin")
    @classmethod
    def _pin_digits(cls, v: str) -> str:
        d = "".join(c for c in v if c.isdigit())
        if len(d) != 6:
            raise ValueError("delivery_pin must be 6 digits")
        return d


class ReasonCode(BaseModel):
    code: str
    feature: str
    value: float | str
    shap: float = Field(..., description="Log-odds contribution (TreeSHAP)")
    direction: Literal["RISK_UP", "RISK_DOWN"]
    human: str


class ConformalBand(BaseModel):
    alpha: float
    prediction_set: list[int]
    certainty: str
    nonconformity: dict[str, float]
    quantiles: dict[str, float]


class RiskResponse(BaseModel):
    order_id: Optional[str]
    decision: str
    action_label: str
    friction_level: int
    p_loss: float = Field(..., description="Calibrated P(RTO | x)")
    p_raw: float = Field(..., description="Uncalibrated booster output (diagnostic)")
    tau_star: float = Field(..., description="ALLOW vs hard-block indifference point C_FP/(C_FN+C_FP)")
    tau_soft: float = Field(..., description="ALLOW vs step-up indifference point (closed form)")
    conformal: ConformalBand
    expected_costs: dict[str, float]
    expected_saving_vs_allow: float
    admissible_actions: list[str]
    reason_codes: list[ReasonCode]
    economics: dict
    graph: dict
    address: dict
    velocity: dict
    features: dict
    hashes: dict
    rationale: str
    latency_ms: dict
    model_version: str
    scorer_backend: str


class DisputeRequest(BaseModel):
    transaction_id: str
    dispute_reason_code: str = Field("10.4", description="Visa reason code; CE3.0 applies to 10.4 (card-absent fraud)")
    dispute_date: Optional[str] = Field(None, description="ISO date; defaults to today")


class DisputeResponse(BaseModel):
    transaction_id: str
    eligible: bool
    standard: str
    criteria: dict
    evidence: dict
    reason: str
    packet_hash: str
