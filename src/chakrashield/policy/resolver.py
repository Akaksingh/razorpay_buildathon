"""Dynamic Action Resolver: conformal set  x  unit economics  ->  action.

The conformal prediction set C(x) tells us what the model is *certified* to
know at miscoverage alpha. The unit economics tell us what each action
*costs*. The resolver combines them:

    C(x) = {0}     certified deliverable  -> only ALLOW is admissible
    C(x) = {1}     certified RTO          -> ALLOW is inadmissible
    C(x) = {0,1}   genuinely ambiguous    -> all actions admissible, pick the
                                             expected-cost argmin; this is where
                                             tau*(x) decides between soft and hard
    C(x) = {}      novel / out-of-support -> neither label conforms. That is the
                                             signature of a fresh syndicate pattern,
                                             so we step up rather than guess.

Within the admissible set we choose the action with the lowest expected
rupee cost. tau*(x) is reported alongside because it is the ALLOW/BLOCK
indifference point and is what an ops analyst wants to see next to p.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

from .economics import TransactionContext

ALLOW = "ALLOW_COD"
STEP_UP = "STEP_UP_DEPOSIT"
PREPAID = "FORCE_PREPAID"
ALL_ACTIONS: tuple[str, ...] = (ALLOW, STEP_UP, PREPAID)

# Customer-facing rendering of each action (what the checkout SDK shows)
ACTION_UX = {
    ALLOW: {"label": "Cash on Delivery available", "friction": 0},
    STEP_UP: {"label": "Confirm with a refundable ₹49 UPI shipping deposit", "friction": 1},
    PREPAID: {"label": "Prepaid only for this order (UPI / Card / EMI)", "friction": 2},
}


@dataclass
class Decision:
    action: str
    conformal_set: list[int]
    certainty: str                 # CERTIFIED_LOW | CERTIFIED_HIGH | AMBIGUOUS | NOVEL
    p_loss: float
    tau_star: float
    tau_soft: float
    expected_costs: dict[str, float]
    expected_saving_vs_allow: float
    admissible: list[str]
    rationale: str
    ux: dict = field(default_factory=dict)
    shadow_price: float = 0.0          # lambda_f applied (friction budget)
    budget_changed_action: bool = False

    def as_dict(self) -> dict:
        return {
            "action": self.action,
            "friction_shadow_price": round(self.shadow_price, 2),
            "budget_changed_action": self.budget_changed_action,
            "conformal_set": self.conformal_set,
            "certainty": self.certainty,
            "p_loss": round(self.p_loss, 4),
            "tau_star": round(self.tau_star, 4),
            "tau_soft": round(self.tau_soft, 4),
            "expected_costs": {k: round(v, 2) for k, v in self.expected_costs.items()},
            "expected_saving_vs_allow": round(self.expected_saving_vs_allow, 2),
            "admissible_actions": self.admissible,
            "rationale": self.rationale,
            "ux": self.ux,
        }


class DynamicRiskResolver:
    """Stateless. Every input is on the request; every output is explainable."""

    @staticmethod
    def compute_optimal_threshold(ctx: TransactionContext) -> float:
        return ctx.tau_star

    @staticmethod
    def _admissible(conformal_set: Sequence[int]) -> tuple[list[str], str]:
        s = set(int(v) for v in conformal_set)
        if s == {0}:
            return [ALLOW], "CERTIFIED_LOW"
        if s == {1}:
            return [STEP_UP, PREPAID], "CERTIFIED_HIGH"
        if s == {0, 1}:
            return list(ALL_ACTIONS), "AMBIGUOUS"
        # empty set: both labels are non-conforming at level alpha
        return [STEP_UP, PREPAID], "NOVEL"

    @classmethod
    def resolve_action(cls, ctx: TransactionContext, conformal_set: Iterable[int],
                       friction_shadow_price: float | None = None) -> Decision:
        if friction_shadow_price is not None:
            ctx = replace(ctx, friction_shadow_price=friction_shadow_price)
        cset = sorted(set(int(v) for v in conformal_set))
        admissible, certainty = cls._admissible(cset)
        costs = ctx.expected_costs()
        lam = ctx.shadow_price
        tau = ctx.tau_star
        tau_soft = ctx.tau_soft

        # Lagrangian of a friction budget: every non-ALLOW action also pays the shadow price lambda_f.
        # argmin over admissible actions; ties resolve to the *least* friction. Because STEP_UP and
        # PREPAID pay the same lambda_f, raising it can only move a decision toward ALLOW.
        penalised = {a: costs[a] + (lam if a != ALLOW else 0.0) for a in costs}
        best = min(admissible, key=lambda a: (round(penalised[a], 6), ALL_ACTIONS.index(a)))
        unpriced = min(admissible, key=lambda a: (round(costs[a], 6), ALL_ACTIONS.index(a)))
        saving = costs[ALLOW] - costs[best]

        if certainty == "CERTIFIED_LOW":
            why = (f"Conformal set {{0}}: deliverability certified at 1-α. "
                   f"P(RTO)={ctx.p_loss:.2f} ≤ τ*={tau:.2f}. Frictionless COD.")
        elif certainty == "CERTIFIED_HIGH":
            why = (f"Conformal set {{1}}: RTO certified at 1-α. P(RTO)={ctx.p_loss:.2f} > τ*={tau:.2f}. "
                   f"{'Soft step-up' if best == STEP_UP else 'Prepaid mandate'} minimises expected loss "
                   f"(₹{costs[best]:.0f} vs ₹{costs[ALLOW]:.0f} on ALLOW).")
        elif certainty == "NOVEL":
            why = (f"Conformal set ∅: neither label conforms at α — pattern is outside the "
                   f"calibrated support (novel syndicate signature). Escalating to "
                   f"{'step-up' if best == STEP_UP else 'prepaid'} rather than guessing.")
        else:  # AMBIGUOUS
            if best == ALLOW:
                why = (f"Conformal set {{0,1}}: ambiguous. P(RTO)={ctx.p_loss:.2f} ≤ τ_soft={tau_soft:.2f} — "
                       f"CAC insult (₹{ctx.cost_fp:.0f}) outweighs shipping exposure (₹{ctx.cost_fn:.0f}). "
                       f"Allowing COD, flagged for post-delivery review.")
            elif best == STEP_UP:
                rel = "below" if ctx.p_loss <= tau else "above"
                why = (f"Conformal set {{0,1}}: ambiguous. P(RTO)={ctx.p_loss:.2f} is {rel} the hard-block "
                       f"point τ*={tau:.2f} but above τ_soft={tau_soft:.2f}. A refundable "
                       f"₹{ctx.econ.stepup_deposit:.0f} UPI deposit costs ₹{costs[STEP_UP]:.0f} expected vs "
                       f"₹{costs[ALLOW]:.0f} on ALLOW, without the ₹{ctx.cost_fp:.0f} insult of a hard block.")
            else:
                why = (f"Conformal set {{0,1}} but P(RTO)={ctx.p_loss:.2f} ≫ τ*={tau:.2f}; "
                       f"expected loss under step-up (₹{costs[STEP_UP]:.0f}) still exceeds prepaid "
                       f"(₹{costs[PREPAID]:.0f}). Prepaid mandate.")
        if best == PREPAID and ctx.address_attribution >= 0.4:
            why += (f" A deposit buys commitment, not deliverability: {ctx.address_attribution:.0%} of this "
                    f"order's risk is attributed to the address, so step-up residual RTO is "
                    f"{ctx.stepup_rto_residual:.0%}.")
        if lam > 0 and unpriced != best:
            why += (f" Friction budget: a shadow price of ₹{lam:.0f} per frictioned order tipped "
                    f"{unpriced} → {best}.")

        return Decision(
            action=best,
            conformal_set=cset,
            certainty=certainty,
            p_loss=ctx.p_loss,
            tau_star=tau,
            tau_soft=tau_soft,
            expected_costs=costs,
            expected_saving_vs_allow=saving,
            admissible=admissible,
            rationale=why,
            ux=ACTION_UX[best],
            shadow_price=lam,
            budget_changed_action=bool(lam > 0 and unpriced != best),
        )
