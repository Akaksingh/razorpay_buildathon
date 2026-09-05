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


# --- alternative conditionings (scripts/12_conformal_variants.py) ---------------------------

from chakrashield.models.conformal import GroupConformalCalibrator, MarginalConformalCalibrator, coverage_by_group


def _grouped_world(n, seed, n_groups=4):
    """Calibrated by construction, but each group has its own risk profile (like PIN tiers)."""
    rng = np.random.default_rng(seed)
    g = rng.integers(1, n_groups + 1, size=n)
    p = np.array([rng.beta(1.0 + 0.6 * gg, 4.5 - 0.6 * gg) for gg in g])
    y = (rng.random(n) < p).astype(int)
    return p, y, g


def test_marginal_coverage_holds_but_minority_class_is_not_protected():
    p_cal, y_cal = _world(6000, 5)
    p_te, y_te = _world(6000, 6)
    alpha = 0.10
    m = MarginalConformalCalibrator.fit(p_cal, y_cal, alpha)
    ev = m.evaluate(p_te, y_te)
    assert ev["coverage_marginal"] >= 1 - alpha - 0.025
    # the marginal promise is met by covering the majority class; the minority class is left exposed,
    # which is the reason the served layer conditions on class
    assert ev["coverage_class0"] > 1 - alpha
    assert ev["coverage_class1"] < 1 - alpha - 0.10


def test_group_conditional_coverage_holds_in_every_group():
    p_cal, y_cal, g_cal = _grouped_world(8000, 7)
    p_te, y_te, g_te = _grouped_world(8000, 8)
    alpha = 0.10
    gc = GroupConformalCalibrator.fit(p_cal, y_cal, g_cal, alpha)
    by_group = coverage_by_group(gc.predict_set(p_te, g_te), y_te, g_te)
    assert set(by_group) == {1, 2, 3, 4}
    for m in by_group.values():
        assert m["coverage_class0"] >= 1 - alpha - 0.03
        assert m["coverage_class1"] >= 1 - alpha - 0.03
    ev = gc.evaluate(p_te, y_te, g_te)
    assert ev["coverage_class1"] >= 1 - alpha - 0.025 and ev["frac_empty"] == 0.0


def test_group_variant_reduces_to_class_conditional_with_one_group_and_for_unseen_groups():
    p, y = _world(3000, 9)
    cc = ConformalCalibrator.fit(p, y, 0.1)
    gc = GroupConformalCalibrator.fit(p, y, np.ones(len(p), dtype=int), 0.1)
    assert gc.quantiles(1) == (cc.q0, cc.q1)
    assert gc.quantiles(42) == (cc.q0, cc.q1)          # unseen group -> class-conditional fallback
    grid = np.linspace(0, 1, 51)
    assert gc.predict_set(grid, 42) == cc.predict_set(grid)


def test_sets_are_monotone_in_alpha_for_every_variant():
    p_cal, y_cal, g_cal = _grouped_world(4000, 10)
    alphas = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
    grid = np.linspace(0, 1, 201)
    fits = {
        "marginal": [MarginalConformalCalibrator.fit(p_cal, y_cal, a).predict_set(grid) for a in alphas],
        "class": [ConformalCalibrator.fit(p_cal, y_cal, a).predict_set(grid) for a in alphas],
        "class_x_group": [GroupConformalCalibrator.fit(p_cal, y_cal, g_cal, a).predict_set(np.tile(grid, 4), np.repeat([1, 2, 3, 4], len(grid)))
                          for a in alphas],
    }
    for name, per_alpha in fits.items():
        for wide, narrow in zip(per_alpha, per_alpha[1:]):      # smaller alpha -> superset
            assert all(set(n) <= set(w) for w, n in zip(wide, narrow)), name
        assert any(len(s) == 2 for s in per_alpha[0]) and all(len(s) <= 1 for s in per_alpha[-1]), name
