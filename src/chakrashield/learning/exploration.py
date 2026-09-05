"""Epsilon-greedy shadow control band: the fix for censored labels.

Every order the engine frictions never ships, so it never earns a delivery
label. Retrain on what shipped and the training distribution is survival-
biased: the high-risk boundary, the one region where calibration matters
most, disappears from the data. Over cycles the model forgets what a bad
order looks like and lets more of them through.

The remedy is a small, deterministic, *logged* randomisation: a share
``epsilon`` of flagged orders is routed through frictionless COD anyway,
tagged ``is_control_cohort``, and written to the ledger with the propensity
of the action that was served. Retraining then re-weights each shipped order
by 1 / propensity (Horvitz-Thompson), which is an unbiased estimate of risk
over the *whole* population, not just the survivors.
"""
from __future__ import annotations

import hashlib

ALLOW = "ALLOW_COD"


def control_draw(key: str, epsilon: float) -> tuple[bool, float]:
    """Deterministic uniform draw from the order key; explored iff u < epsilon.

    Deterministic so that a retry of the same order gets the same answer and
    the ledger can be replayed exactly.
    """
    u = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64
    return (u < epsilon), u


def propensity(policy_action: str, served_action: str, epsilon: float) -> float:
    """P(served action | x) under the epsilon-greedy policy."""
    if policy_action == ALLOW:
        return 1.0                       # nothing to explore: allow was the policy
    if served_action == ALLOW:
        return epsilon                   # control cohort
    return 1.0 - epsilon                 # the policy's own friction action


def ipw_weight(served_action: str, prop: float, cap: float = 100.0) -> float:
    """Training weight of a *shipped* order for the RTO model.

    Only orders served frictionless COD carry an untreated delivery label. A
    policy-allow order has propensity 1 (weight 1); a control-cohort order has
    propensity epsilon (weight 1/epsilon), capped for variance control.
    """
    if served_action != ALLOW or prop <= 0:
        return 0.0
    return float(min(cap, 1.0 / prop))
