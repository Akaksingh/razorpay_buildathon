"""Counterfactual P&L simulator for evaluating checkout policies.

Given the *true* outcome of each order (would it have delivered or RTO'd)
and the action a policy would have taken, compute the merchant's realised
contribution in rupees. Everything is an exact conditional expectation
over the buyer's behavioural response, so two runs give the same number
and there is no Monte-Carlo noise to hide behind.

The behavioural parameters here are deliberately *separate* from the ones
the resolver assumes (config.Economics). The evaluation sweeps them so the
headline claim survives the resolver being wrong about buyer behaviour.
The one structural rule shared with the resolver is that a deposit cannot
fix an undeliverable address: residual RTO after a paid step-up is
rho + (1 - rho) * a(x), with a(x) the order's address attribution.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .economics import TransactionContext
from .resolver import ALLOW, PREPAID, STEP_UP


@dataclass
class BehaviourSim:
    stepup_good_abandon: float = 0.11    # good buyers who quit at a Rs.49 UPI prompt
    stepup_bad_abandon: float = 0.65     # would-be-RTO buyers who quit at the prompt (costs nothing)
    stepup_rto_residual: float = 0.10    # of bad buyers who paid the deposit, share that still refuses (intent-driven)
    prepaid_good_abandon: float = 0.38
    prepaid_bad_abandon: float = 0.85
    prepaid_rto_residual: float = 0.04
    deposit: float = 49.0                # forfeited on refusal, refunded on delivery
    friction_cost: float = 6.0
    prepaid_rto_unit_cost: float = 60.0

    def as_dict(self) -> dict:
        return asdict(self)

    def rho_stepup(self, ctx: TransactionContext) -> float:
        return self.stepup_rto_residual + (1 - self.stepup_rto_residual) * ctx.address_attribution

    def rho_prepaid(self, ctx: TransactionContext) -> float:
        return self.prepaid_rto_residual + (1 - self.prepaid_rto_residual) * ctx.address_attribution


def order_pnl(action: str, y: int, ctx: TransactionContext, sim: BehaviourSim) -> float:
    margin = ctx.merchant_margin * ctx.gmv
    cac_insult = ctx.cost_fp - margin
    if y == 0:  # would have delivered
        if action == ALLOW:
            return margin
        if action == STEP_UP:
            return (1 - sim.stepup_good_abandon) * (margin - sim.friction_cost) + sim.stepup_good_abandon * (-cac_insult)
        return (1 - sim.prepaid_good_abandon) * margin + sim.prepaid_good_abandon * (-cac_insult)
    # y == 1: would have RTO'd
    if action == ALLOW:
        return -ctx.cost_fn
    if action == STEP_UP:
        pay, rho = 1 - sim.stepup_bad_abandon, sim.rho_stepup(ctx)
        return pay * (rho * (-ctx.cost_fn + sim.deposit) + (1 - rho) * (margin - sim.friction_cost))
    pay, rho = 1 - sim.prepaid_bad_abandon, sim.rho_prepaid(ctx)
    return pay * (rho * (-sim.prepaid_rto_unit_cost) + (1 - rho) * margin)


def simulate(actions: np.ndarray, y: np.ndarray, ctxs: list[TransactionContext], sim: BehaviourSim) -> dict:
    actions = np.asarray(actions)
    y = np.asarray(y, dtype=int)
    pnl = np.array([order_pnl(a, int(yy), c, sim) for a, yy, c in zip(actions, y, ctxs)])
    gmv = np.array([c.gmv for c in ctxs])
    rho_s = np.array([sim.rho_stepup(c) for c in ctxs])
    rho_p = np.array([sim.rho_prepaid(c) for c in ctxs])
    n = len(y)
    good, bad = y == 0, y == 1
    # expected RTOs actually shipped under each action
    shipped_rto = np.where(actions == ALLOW, bad * 1.0,
                  np.where(actions == STEP_UP, bad * (1 - sim.stepup_bad_abandon) * rho_s,
                           bad * (1 - sim.prepaid_bad_abandon) * rho_p))
    good_lost = np.where(actions == ALLOW, 0.0,
                np.where(actions == STEP_UP, good * sim.stepup_good_abandon, good * sim.prepaid_good_abandon))
    return {
        "n": int(n),
        "pnl_total": float(pnl.sum()),
        "pnl_per_order": float(pnl.mean()),
        "pnl_per_1000_gmv": float(1000 * pnl.sum() / gmv.sum()),
        "actions": {a: int((actions == a).sum()) for a in (ALLOW, STEP_UP, PREPAID)},
        "action_share": {a: float((actions == a).mean()) for a in (ALLOW, STEP_UP, PREPAID)},
        "rto_shipped_expected": float(shipped_rto.sum()),
        "rto_prevented_expected": float(bad.sum() - shipped_rto.sum()),
        "good_customers_lost_expected": float(good_lost.sum()),
        "good_frictioned": int(((actions != ALLOW) & good).sum()),
        "bad_frictioned": int(((actions != ALLOW) & bad).sum()),
        "pnl_good": float(pnl[good].sum()),
        "pnl_bad": float(pnl[bad].sum()),
    }
