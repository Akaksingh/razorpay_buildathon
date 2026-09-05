from chakrashield.learning.exploration import ALLOW, control_draw, ipw_weight, propensity
from chakrashield.learning.ledger import DecisionLedger


def test_control_draw_is_deterministic_and_hits_the_rate():
    a1, u1 = control_draw("ord_123", 0.02)
    a2, u2 = control_draw("ord_123", 0.02)
    assert (a1, u1) == (a2, u2) and 0.0 <= u1 < 1.0
    n = 40_000
    hits = sum(control_draw(f"ord_{i}", 0.02)[0] for i in range(n))
    assert abs(hits / n - 0.02) < 0.004


def test_propensity_and_ipw_weights():
    eps = 0.02
    assert propensity(ALLOW, ALLOW, eps) == 1.0
    assert propensity("STEP_UP_DEPOSIT", "STEP_UP_DEPOSIT", eps) == 1 - eps
    assert propensity("FORCE_PREPAID", ALLOW, eps) == eps
    assert ipw_weight(ALLOW, 1.0) == 1.0
    assert ipw_weight(ALLOW, eps) == 50.0                 # a control-cohort order stands in for 50 blocked ones
    assert ipw_weight(ALLOW, 0.001, cap=100.0) == 100.0   # capped for variance control
    assert ipw_weight("STEP_UP_DEPOSIT", 1 - eps) == 0.0  # frictioned orders carry no untreated label


def test_ledger_round_trip(tmp_path):
    led = DecisionLedger(tmp_path / "decisions.jsonl")
    led.log_decision({"order_id": "o1", "policy_action": "STEP_UP_DEPOSIT", "served_action": ALLOW, "is_control_cohort": True, "propensity": 0.02})
    led.log_decision({"order_id": "o2", "policy_action": ALLOW, "served_action": ALLOW, "is_control_cohort": False, "propensity": 1.0})
    led.log_outcome("o1", rto=True)
    s = led.stats()
    assert s["decisions"] == 2 and s["control_cohort"] == 1 and s["outcomes"] == 1 and s["served"][ALLOW] == 2
    dec, out = DecisionLedger.load(tmp_path / "decisions.jsonl")
    assert len(dec) == 2 and len(out) == 1 and bool(out.iloc[0]["rto"]) is True
    again = DecisionLedger(tmp_path / "decisions.jsonl")             # counters survive a restart
    assert again.stats()["decisions"] == 2
