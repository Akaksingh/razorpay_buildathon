"""Instance-dependent unit economics for a single COD checkout.

The whole engine is built on one idea: a misclassification is not a unit
error, it is a rupee amount that depends on *this* order. Two functions
carry that idea:

    C_FN(x) = cost of allowing COD on an order that will RTO
            = L_logistics + lambda * V
    C_FP(x) = cost of frictioning an order that would have delivered
            = M * V + kappa * CAC

and the Bayes-optimal binary rule that falls out of them:

    block  iff  (1 - p) * C_FP  <  p * C_FN
           iff  p  >  C_FP / (C_FN + C_FP)  =:  tau*(x)

tau*(x) is *not* a tuned hyper-parameter. It is the indifference point of the
merchant's own P&L for this specific order, which is why the API returns it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config import ECONOMICS, Economics


@dataclass
class TransactionContext:
    """Everything the resolver needs to price a decision. Model-free."""

    gmv: float                       # V: cart value in INR
    merchant_margin: float           # M: contribution margin (0.18 = 18%)
    cac: float                       # customer acquisition cost in INR
    p_loss: float                    # P(RTO | x) from the calibrated scorer
    is_new_customer: bool = True     # first order => full CAC at stake
    weight_grams: float = 450.0      # drives logistics cost tiering
    addr_defect: float = 0.0         # structural address defect score in [0, 1]
    econ: Economics = field(default_factory=lambda: ECONOMICS)

    # Optional merchant overrides (per-request), default to Economics values
    logistics_loss: Optional[float] = None
    holding_cost: Optional[float] = None
    friction_shadow_price: Optional[float] = None   # lambda_f override; None -> Economics.friction_shadow_price

    # ------------------------------------------------------------------ costs
    @property
    def logistics(self) -> float:
        """Forward + reverse shipping + packaging + restocking, weight-tiered."""
        if self.logistics_loss is not None:
            return float(self.logistics_loss)
        e = self.econ
        tier = 1.0 if self.weight_grams <= 500 else (1.35 if self.weight_grams <= 2000 else 1.9)
        return (e.forward_shipping + e.reverse_shipping) * tier + e.packaging_cost + e.restocking_cost

    @property
    def holding(self) -> float:
        """Cost of capital on inventory sitting in a 3PL hub for ~10 days."""
        if self.holding_cost is not None:
            return float(self.holding_cost)
        return self.econ.holding_cost_rate * self.gmv

    @property
    def cost_fn(self) -> float:
        """C_FN(x): what we lose if we allow COD and the parcel comes back."""
        return self.logistics + self.holding + self.econ.inventory_lockup_lambda * self.gmv

    @property
    def cost_fp(self) -> float:
        """C_FP(x): what we lose if we friction a customer who would have paid.

        Only a *new* customer carries the CAC insult -- an existing customer
        who bounces off one friction step is annoyed, not un-acquired.
        """
        cac_at_risk = self.cac * self.econ.cac_insult_kappa if self.is_new_customer else 0.15 * self.cac
        return self.merchant_margin * self.gmv + cac_at_risk

    # ------------------------------------------------- intent vs deliverability
    @property
    def address_attribution(self) -> float:
        """a(x): share of this order's RTO risk that no deposit can fix.

        A refundable deposit works through commitment and self-selection. It
        does nothing for a parcel the rider cannot find. We attribute a
        convex share of the risk to the address (defect^2: a landmark-only
        address is usually still deliverable with a phone call; junk is not).
        """
        d = min(1.0, max(0.0, float(self.addr_defect)))
        return d * d

    @property
    def stepup_rto_residual(self) -> float:
        """rho_eff = rho + (1 - rho) * a(x)."""
        r = self.econ.stepup_rto_residual
        return r + (1.0 - r) * self.address_attribution

    @property
    def prepaid_rto_residual(self) -> float:
        r = self.econ.prepaid_rto_residual
        return r + (1.0 - r) * self.address_attribution

    # -------------------------------------------------------------- threshold
    @property
    def shadow_price(self) -> float:
        """lambda_f: rupees charged to every non-ALLOW action.

        It is the Lagrange multiplier of a friction budget ("at most X% of
        orders may be frictioned"): minimising E[cost] + lambda_f * 1[friction]
        for the smallest lambda_f that meets the budget is the constrained
        optimum. 0 means pure expected-cost minimisation.
        """
        v = self.friction_shadow_price if self.friction_shadow_price is not None else self.econ.friction_shadow_price
        return max(0.0, float(v))

    @property
    def tau_star(self) -> float:
        """Instance-dependent optimal threshold tau*(x) = (C_FP + lambda_f) / (C_FN + C_FP).

        With no friction budget (lambda_f = 0) this is the textbook cost ratio;
        a shadow price raises the point at which a block becomes worth it.
        """
        denom = self.cost_fn + self.cost_fp
        if denom <= 0:
            return 0.5
        return float(min(1.0, (self.cost_fp + self.shadow_price) / denom))

    @property
    def tau_soft(self) -> float:
        """Indifference point between ALLOW and STEP_UP (closed form).

        Solve  p*C_FN = (1-p)*delta_s*C_FP + p*rho*C_FN + (1-(1-p)*delta_s)*f  for p:

            p = (delta_s*C_FP + f*(1-delta_s)) / (C_FN*(1-rho) + delta_s*C_FP - f*delta_s)

        This is always below tau_star: a refundable deposit is cheap to ask
        for, so it becomes worth asking for long before a hard block would.
        A friction shadow price lambda_f enters the numerator (the step-up
        must now also pay for its share of the budget).
        """
        e = self.econ
        num = e.stepup_abandon_rate * self.cost_fp + e.stepup_friction_cost * (1 - e.stepup_abandon_rate) + self.shadow_price
        den = self.cost_fn * (1 - self.stepup_rto_residual) + e.stepup_abandon_rate * self.cost_fp \
            - e.stepup_friction_cost * e.stepup_abandon_rate
        if den <= 0:
            return 1.0
        return float(min(1.0, max(0.0, num / den)))

    # ------------------------------------------------------- expected costs
    def expected_cost_allow(self) -> float:
        """E[cost | ALLOW_COD] = p * C_FN."""
        return self.p_loss * self.cost_fn

    def expected_cost_stepup(self) -> float:
        """E[cost | STEP_UP_DEPOSIT].

        A refundable deposit splits the population: a fraction delta_s of *good*
        customers abandon (we eat C_FP on them), the rest pay and proceed with
        RTO risk collapsed to rho * p (someone who parted with Rs.49 over UPI
        answers the doorbell). Fraudulent / non-serious buyers self-select out
        at zero logistics cost -- that is the whole point of the intervention.
        """
        e = self.econ
        p, q = self.p_loss, 1.0 - self.p_loss
        good_abandon = q * e.stepup_abandon_rate * self.cost_fp
        proceed_mass = 1.0 - q * e.stepup_abandon_rate  # everyone else proceeds
        residual_rto = p * self.stepup_rto_residual * self.cost_fn
        friction = proceed_mass * e.stepup_friction_cost
        return good_abandon + residual_rto + friction

    def expected_cost_prepaid(self) -> float:
        """E[cost | FORCE_PREPAID]: heavy good-customer abandonment, near-zero RTO."""
        e = self.econ
        q = 1.0 - self.p_loss
        good_abandon = q * e.prepaid_abandon_rate * self.cost_fp
        residual = (1.0 - q * e.prepaid_abandon_rate) * self.prepaid_rto_residual * e.prepaid_rto_unit_cost
        return good_abandon + residual

    def expected_costs(self) -> dict[str, float]:
        return {
            "ALLOW_COD": self.expected_cost_allow(),
            "STEP_UP_DEPOSIT": self.expected_cost_stepup(),
            "FORCE_PREPAID": self.expected_cost_prepaid(),
        }

    def as_dict(self) -> dict:
        return {
            "gmv": self.gmv,
            "merchant_margin": self.merchant_margin,
            "cac": self.cac,
            "is_new_customer": self.is_new_customer,
            "logistics": round(self.logistics, 2),
            "holding": round(self.holding, 2),
            "cost_fn": round(self.cost_fn, 2),
            "cost_fp": round(self.cost_fp, 2),
            "tau_star": round(self.tau_star, 4),
            "tau_soft": round(self.tau_soft, 4),
            "p_loss": round(self.p_loss, 4),
            "addr_defect": round(self.addr_defect, 3),
            "address_attribution": round(self.address_attribution, 3),
            "stepup_rto_residual_eff": round(self.stepup_rto_residual, 3),
            "friction_shadow_price": round(self.shadow_price, 2),
        }
