import numpy as np

from chakrashield.models.conformal import ConformalCalibrator


def _world(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.beta(1.2, 3.5, size=n)              # calibrated by construction
    y = (rng.random(n) < p).astype(int)
    return p, y


def test_class_conditional_coverage_holds():
    p_cal, y_cal = _world(6000, 1)
    p_te, y_te = _world(6000, 2)
    alpha = 0.10
    c = ConformalCalibrator.fit(p_cal, y_cal, alpha)
    ev = c.evaluate(p_te, y_te)
    assert ev["coverage_class0"] >= 1 - alpha - 0.025
    assert ev["coverage_class1"] >= 1 - alpha - 0.025
    assert 0 < ev["frac_singleton"] < 1


def test_sets_are_well_formed_and_monotone():
    p_cal, y_cal = _world(3000, 3)
    c = ConformalCalibrator.fit(p_cal, y_cal, 0.1)
    for p in np.linspace(0, 1, 101):
        s = c.predict_one(float(p))
        assert s in ([], [0], [1], [0, 1])
    # 0 is in the set for small p and 1 is in the set for large p
    assert 0 in c.predict_one(0.0) and 1 in c.predict_one(1.0)


def test_round_trip(tmp_path):
    p, y = _world(1000, 4)
    c = ConformalCalibrator.fit(p, y, 0.2)
    c.save(tmp_path / "c.json")
    c2 = ConformalCalibrator.load(tmp_path / "c.json")
    assert (c2.q0, c2.q1, c2.alpha) == (c.q0, c.q1, c.alpha)
