import random

import pytest

from chakrashield.config import Economics
from chakrashield.learning.response import BehaviourLearner, segment_key


def simulate_stepups(learner, segment, n, delta_s, delta_bad, rho, rng):
    for _ in range(n):
        p = rng.uniform(0.05, 0.9)
        bad = rng.random() < p
        abandoned = rng.random() < (delta_bad if bad else delta_s)
        rto = None if abandoned else (bad and rng.random() < rho)
        learner.observe_stepup(segment, p, abandoned, rto)


def test_segment_key_buckets():
    assert segment_key("META_ADS", 4, 2899) == "SOCIAL|T4|high"
    assert segment_key("organic", 1, 499) == "ORGANIC|T1|low"
    assert segment_key("GOOGLE_ADS", 2, 1500) == "SEARCH|T2|mid"


def test_starts_at_prior_then_learns_the_truth():
    econ = Economics()
    L = BehaviourLearner(prior=econ)
    seg = "SOCIAL|T3|mid"
    e0 = L.estimate(seg)
    assert e0.source == "prior" and e0.delta_s == econ.stepup_abandon_rate
    base, est = L.economics_for(seg, econ)
    assert base is econ                                  # no data: the resolver runs on the configured prior
    rng = random.Random(7)
    simulate_stepups(L, seg, 3000, delta_s=0.30, delta_bad=0.55, rho=0.20, rng=rng)
    e = L.estimate(seg)
    assert e.source == "segment"
    assert abs(e.delta_s - 0.30) < 0.05 and abs(e.delta_bad - 0.55) < 0.06 and abs(e.rho - 0.20) < 0.07
    econ2, _ = L.economics_for(seg, econ)
    assert econ2.stepup_abandon_rate == pytest.approx(e.delta_s) and econ2.stepup_rto_residual == pytest.approx(e.rho)


def test_thin_segment_borrows_from_global():
    L = BehaviourLearner(prior=Economics())
    rng = random.Random(3)
    simulate_stepups(L, "SEARCH|T2|mid", 2000, 0.20, 0.60, 0.10, rng)
    thin = L.estimate("ORGANIC|T1|low")
    assert thin.source == "global" and abs(thin.delta_s - 0.20) < 0.06
    simulate_stepups(L, "ORGANIC|T1|low", 10, 0.05, 0.60, 0.10, rng)
    assert L.estimate("ORGANIC|T1|low").source == "global"          # below min_n_segment: still borrowing


def test_snapshot_round_trip(tmp_path):
    L = BehaviourLearner(prior=Economics())
    simulate_stepups(L, "SOCIAL|T4|high", 500, 0.25, 0.7, 0.15, random.Random(1))
    L.snapshot(tmp_path / "b.json")
    L2 = BehaviourLearner(prior=Economics()).load(tmp_path / "b.json")
    assert L2.estimate("SOCIAL|T4|high").as_dict() == L.estimate("SOCIAL|T4|high").as_dict()
    assert L2.summary()["observations"] == 500
