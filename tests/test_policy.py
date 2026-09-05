import numpy as np
import pytest

from chakrashield.config import Economics
from chakrashield.policy.economics import TransactionContext
from chakrashield.policy.resolver import ALLOW, PREPAID, STEP_UP, DynamicRiskResolver


def ctx(p=0.3, gmv=1500.0, cac=400.0, margin=0.15, new=True, weight=450.0, econ=None):
    return TransactionContext(gmv=gmv, merchant_margin=margin, cac=cac, p_loss=p, is_new_customer=new,
                              weight_grams=weight, econ=econ or Economics())


def test_tau_star_is_cost_ratio():
    c = ctx()
    assert c.tau_star == pytest.approx(c.cost_fp / (c.cost_fn + c.cost_fp))
    assert 0.0 < c.tau_star < 1.0


def test_tau_star_is_bayes_indifference_point():
    """At p = tau*, expected cost of ALLOW equals expected cost of a hard block on a good order."""
    c = ctx()
    p = c.tau_star
    assert p * c.cost_fn == pytest.approx((1 - p) * c.cost_fp)


def test_tau_soft_below_tau_star_and_is_allow_stepup_indifference():
    c = ctx()
    assert c.tau_soft < c.tau_star
    c2 = ctx(p=c.tau_soft)
    assert c2.expected_cost_allow() == pytest.approx(c2.expected_cost_stepup(), rel=1e-6)


def test_threshold_moves_with_economics():
    assert ctx(cac=900).tau_star > ctx(cac=100).tau_star          # more CAC at stake -> more tolerant
    assert ctx(weight=3000).tau_star < ctx(weight=300).tau_star   # heavier parcel -> less tolerant
    assert ctx(new=False).tau_star < ctx(new=True).tau_star       # returning customer carries no CAC insult


def test_conformal_gating():
    assert DynamicRiskResolver.resolve_action(ctx(p=0.95), [0]).action == ALLOW
    d = DynamicRiskResolver.resolve_action(ctx(p=0.05), [1])
    assert d.action != ALLOW and d.certainty == "CERTIFIED_HIGH"
    d = DynamicRiskResolver.resolve_action(ctx(p=0.3), [])
    assert d.action != ALLOW and d.certainty == "NOVEL"
    assert DynamicRiskResolver.resolve_action(ctx(p=0.01), [0, 1]).action == ALLOW
    assert DynamicRiskResolver.resolve_action(ctx(p=0.99), [0, 1]).action == PREPAID


def test_ambiguous_allow_iff_below_tau_soft():
    for p in np.linspace(0.01, 0.99, 50):
        c = ctx(p=float(p))
        d = DynamicRiskResolver.resolve_action(c, [0, 1])
        if abs(p - c.tau_soft) > 1e-3:
            assert (d.action == ALLOW) == (p < c.tau_soft), (p, c.tau_soft, d.action)


def test_deposit_cannot_fix_an_undeliverable_address():
    """Same p, same economics: a junk address flips the soft step-up into a prepaid mandate."""
    good_addr = TransactionContext(gmv=1500, merchant_margin=0.15, cac=400, p_loss=0.6, addr_defect=0.1, econ=Economics())
    junk_addr = TransactionContext(gmv=1500, merchant_margin=0.15, cac=400, p_loss=0.6, addr_defect=1.0, econ=Economics())
    assert good_addr.stepup_rto_residual < 0.15 and junk_addr.stepup_rto_residual == pytest.approx(1.0)
    assert DynamicRiskResolver.resolve_action(good_addr, [0, 1]).action == STEP_UP
    d = DynamicRiskResolver.resolve_action(junk_addr, [0, 1])
    assert d.action == PREPAID and "deliverability" in d.rationale
    assert junk_addr.tau_soft > good_addr.tau_soft   # step-up is worth less, so it is asked for later


def test_argmin_is_cheapest_admissible():
    c = ctx(p=0.5)
    d = DynamicRiskResolver.resolve_action(c, [0, 1])
    assert d.action == min(d.expected_costs, key=d.expected_costs.get)
    assert d.expected_saving_vs_allow == pytest.approx(d.expected_costs[ALLOW] - d.expected_costs[d.action])


def test_decision_serialises():
    d = DynamicRiskResolver.resolve_action(ctx(), [0, 1]).as_dict()
    for k in ("action", "conformal_set", "certainty", "p_loss", "tau_star", "tau_soft", "expected_costs", "rationale", "ux"):
        assert k in d


def test_friction_shadow_price_never_adds_friction():
    """Raising lambda_f can only move a decision toward ALLOW: it is the Lagrangian of a friction budget."""
    from chakrashield.policy.resolver import ACTION_UX
    for p in (0.15, 0.3, 0.45, 0.6, 0.8):
        prev = None
        for lam in (0.0, 10.0, 40.0, 150.0, 1000.0):
            d = DynamicRiskResolver.resolve_action(ctx(p=p), [0, 1], friction_shadow_price=lam)
            f = ACTION_UX[d.action]["friction"]
            assert prev is None or f <= prev
            assert d.shadow_price == lam
            prev = f
        assert d.action == ALLOW          # at a prohibitive price the ambiguous order is allowed


def test_shadow_price_raises_both_indifference_points():
    a = ctx(p=0.3)
    b = TransactionContext(gmv=1500.0, merchant_margin=0.15, cac=400.0, p_loss=0.3, is_new_customer=True,
                           weight_grams=450.0, econ=Economics(), friction_shadow_price=60.0)
    assert b.tau_star > a.tau_star and b.tau_soft > a.tau_soft
    assert b.tau_star == pytest.approx((a.cost_fp + 60.0) / (a.cost_fn + a.cost_fp))


def test_certified_high_is_frictioned_regardless_of_price():
    d = DynamicRiskResolver.resolve_action(ctx(p=0.8), [1], friction_shadow_price=1e6)
    assert d.action in (STEP_UP, PREPAID)


def test_budget_changed_action_flag():
    p = 0.35
    free = DynamicRiskResolver.resolve_action(ctx(p=p), [0, 1])
    assert free.action == STEP_UP and free.budget_changed_action is False
    priced = DynamicRiskResolver.resolve_action(ctx(p=p), [0, 1], friction_shadow_price=1000.0)
    assert priced.action == ALLOW and priced.budget_changed_action is True and "Friction budget" in priced.rationale
