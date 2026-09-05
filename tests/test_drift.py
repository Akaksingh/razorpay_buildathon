import random

from chakrashield.monitoring.drift import CERTAINTIES, ConformalDriftMonitor, DriftBaseline, psi
from chakrashield.store.feature_store import MemoryStore

BASE = DriftBaseline(share={"CERTIFIED_LOW": 0.32, "AMBIGUOUS": 0.50, "CERTIFIED_HIGH": 0.18, "NOVEL": 0.0},
                     p_hist=[0.10] * 5 + [0.06] * 5 + [0.03] * 5 + [0.01] * 5, n=3868, q0=0.30, q1=0.88)


def make(clock_start=1_000_000.0):
    t = {"now": clock_start}
    m = ConformalDriftMonitor(MemoryStore(), BASE, window_s=300, keep=12, clock=lambda: t["now"])
    return m, t


def stream(m, t, n, mix, rng, p_fn):
    keys = list(mix)
    for _ in range(n):
        c = rng.choices(keys, weights=[mix[k] for k in keys])[0]
        m.record(c, p_fn(c, rng))
        t["now"] += 1.0


def base_p(c, rng):
    return {"CERTIFIED_LOW": rng.uniform(0.0, 0.25), "AMBIGUOUS": rng.uniform(0.15, 0.5), "CERTIFIED_HIGH": rng.uniform(0.5, 0.95), "NOVEL": 0.5}[c]


def test_stationary_traffic_is_ok():
    m, t = make()
    stream(m, t, 1500, BASE.share, random.Random(1), base_p)
    s = m.snapshot()
    assert s["status"] in ("OK", "WARN") and s["rolling_n"] == 1500
    assert not any(a["severity"] == "critical" for a in s["alerts"])
    assert not BASE.empty_sets_possible          # q0 + q1 > 1: the empty-set alarm cannot fire yet


def test_festival_burst_raises_alerts():
    m, t = make()
    stream(m, t, 400, BASE.share, random.Random(2), base_p)
    burst = {"CERTIFIED_LOW": 0.10, "AMBIGUOUS": 0.30, "CERTIFIED_HIGH": 0.60, "NOVEL": 0.0}
    stream(m, t, 1200, burst, random.Random(3), base_p)
    s = m.snapshot()
    codes = {a["code"] for a in s["alerts"]}
    assert "MODEL_RISK_MIX_SHIFT" in codes and "MODEL_EPISTEMIC_DRIFT" in codes and "MODEL_SCORE_PSI" in codes
    assert s["status"] == "ALERT"


def test_empty_sets_trigger_epistemic_drift():
    m, t = make()
    novel = {"CERTIFIED_LOW": 0.30, "AMBIGUOUS": 0.48, "CERTIFIED_HIGH": 0.17, "NOVEL": 0.05}
    stream(m, t, 800, novel, random.Random(4), base_p)
    s = m.snapshot()
    a = [x for x in s["alerts"] if x["code"] == "MODEL_EPISTEMIC_DRIFT" and "empty" in x["message"]]
    assert a and a[0]["severity"] == "critical"


def test_windows_slide_and_warming():
    m, t = make()
    assert m.snapshot()["status"] == "WARMING"
    stream(m, t, 20, BASE.share, random.Random(5), base_p)
    w = m.windows()
    assert len(w) == 12 and sum(x["n"] for x in w) == 20
    t["now"] += 300 * 20                                   # an hour and more passes: the old window falls out
    assert sum(x["n"] for x in m.windows()) == 0


def test_psi_is_zero_for_identical_histograms():
    assert psi(BASE.p_hist, BASE.p_hist) == 0.0
    assert psi([1.0] + [0.0] * 19, BASE.p_hist) > 1.0
    assert set(CERTAINTIES) == {"CERTIFIED_LOW", "AMBIGUOUS", "CERTIFIED_HIGH", "NOVEL"}
