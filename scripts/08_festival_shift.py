"""Domain shift: a festival sale hits a model calibrated on ordinary traffic.

A second world is generated with a different seed and a shifted regime (far more
impulse buyers acquired from paid social, more syndicates, higher latent RTO). It is
replayed into its own store for point-in-time features and scored with the *served*
model. Two questions are answered with numbers:

  1. What does the label-free drift monitor see, and how soon?  The conformal set mix
     and calibrated-p PSI are streamed into ConformalDriftMonitor at one order per
     second (300 orders per 5-minute window); alerts are recorded per window.
  2. What did the labels later confirm?  AUC, ECE and class-conditional conformal
     coverage under the shift, and after recalibrating isotonic + conformal on the
     first 30% of the new regime (labels that would exist a week into the sale).

A same-regime control world (new seed, same knobs) separates "new data" from "new regime".
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.metrics import roc_auc_score

from chakrashield.config import CONFORMAL_ALPHA, MODEL_DIR, REPORT_DIR
from chakrashield.data.generator import generate
from chakrashield.data.replay import replay
from chakrashield.features.vectorizer import FEATURE_NAMES
from chakrashield.models.calibration import IsotonicKnots, expected_calibration_error
from chakrashield.models.conformal import ConformalCalibrator
from chakrashield.monitoring.drift import CERTAINTIES, ConformalDriftMonitor, DriftBaseline
from chakrashield.runtime.scorer import Scorer
from chakrashield.store.feature_store import MemoryStore

# ring count scales with the population (45 rings per 18,000 customers in the training world)
SCENARIOS = {
    "control (new seed, same regime)": dict(seed=43, n_rings=18),
    "festival sale (paid-social surge, 2x rings, +0.6 logit RTO)": dict(seed=43, impulse_share=0.30, n_rings=36, rto_logit_shift=0.6),
}
CERT_OF = {(0,): "CERTIFIED_LOW", (1,): "CERTIFIED_HIGH", (0, 1): "AMBIGUOUS", (): "NOVEL"}


def certainty(sets) -> list[str]:
    return [CERT_OF[tuple(sorted(s))] for s in sets]


def run(name: str, knobs: dict, scorer: Scorer, baseline: DriftBaseline) -> dict:
    t = time.time()
    df = generate(n_orders=20_000, days=120, n_customers=7_000, **knobs)
    feats, _ = replay(df, store=MemoryStore())
    full = df.drop(columns=[c for c in feats.columns if c in df.columns]).join(feats)
    cod = full[full.payment_method == "COD"].sort_values("ts").reset_index(drop=True)
    X = cod[list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    y = cod["rto"].to_numpy(dtype=int)
    raw = scorer._booster.predict(X)
    p = np.clip(scorer._iso.apply(raw), 1e-3, 1 - 1e-3)
    conf = scorer.conformal
    sets = conf.predict_set(p)
    cert = certainty(sets)

    # --- the label-free monitor, streamed at one order per second -------------------------
    clock = {"t": 1_000_000.0}
    mon = ConformalDriftMonitor(MemoryStore(), baseline, clock=lambda: clock["t"])
    timeline, first_alert = [], None
    for i, (c, pp) in enumerate(zip(cert, p)):
        mon.record(c, float(pp))
        clock["t"] += 1.0
        if (i + 1) % 300 == 0:
            s = mon.snapshot()
            timeline.append({"orders": i + 1, "status": s["status"], "alerts": [a["code"] for a in s["alerts"]],
                             "share": s["rolling_share"], "psi": s["score_psi"]})
            if first_alert is None and s["status"] == "ALERT":
                first_alert = i + 1
    final = mon.snapshot()

    # --- what the labels confirm ------------------------------------------------------------
    before = conf.evaluate(p, y)
    n_recal = int(0.3 * len(y))
    iso2 = IsotonicKnots.fit(raw[:n_recal], y[:n_recal])
    p2 = np.clip(iso2.apply(raw), 1e-3, 1 - 1e-3)
    conf2 = ConformalCalibrator.fit(p2[:n_recal], y[:n_recal], CONFORMAL_ALPHA)
    after = conf2.evaluate(p2[n_recal:], y[n_recal:])
    shares = {c: float(np.mean([x == c for x in cert])) for c in CERTAINTIES}
    out = {
        "knobs": knobs, "orders_cod": int(len(y)), "rto_rate": float(y.mean()), "seconds": round(time.time() - t, 1),
        "auc": float(roc_auc_score(y, p)), "ece": expected_calibration_error(p, y),
        "coverage_before": {k: before[k] for k in ("coverage_class0", "coverage_class1", "frac_ambiguous", "frac_empty")},
        "coverage_after_recalibration": {k: after[k] for k in ("coverage_class0", "coverage_class1", "frac_ambiguous", "frac_empty")},
        "ece_after_recalibration": expected_calibration_error(p2[n_recal:], y[n_recal:]),
        "certainty_share": shares, "monitor_final": {k: final[k] for k in ("status", "rolling_share", "score_psi", "alerts")},
        "first_alert_after_orders": first_alert, "timeline": timeline,
    }
    print(f"[{name[:14]}] RTO {out['rto_rate']:.1%} | AUC {out['auc']:.3f} ECE {out['ece']:.3f} | coverage {before['coverage_class0']:.3f}/{before['coverage_class1']:.3f} "
          f"-> recalibrated {after['coverage_class0']:.3f}/{after['coverage_class1']:.3f} | certified-high {shares['CERTIFIED_HIGH']:.1%} ambiguous {shares['AMBIGUOUS']:.1%} "
          f"| monitor {final['status']} psi {final['score_psi']:.2f} first alert @ {first_alert} orders | {[a['code'] for a in final['alerts']]}")
    return out


def main() -> None:
    scorer = Scorer(MODEL_DIR).load()
    ev = json.loads((REPORT_DIR / "evaluation.json").read_text(encoding="utf-8"))
    baseline = DriftBaseline.from_reports(ev, q0=scorer.conformal.q0, q1=scorer.conformal.q1)
    print(f"[baseline] certainty share {{{', '.join(f'{k} {v:.1%}' for k, v in baseline.share.items())}}} | q0 {baseline.q0:.3f} q1 {baseline.q1:.3f} "
          f"| empty sets possible: {baseline.empty_sets_possible}")
    report = {"baseline": {"share": baseline.share, "q0": baseline.q0, "q1": baseline.q1, "empty_sets_possible": baseline.empty_sets_possible,
                           "test_auc": ev["model_metrics"]["chakra"]["auc"], "test_ece": ev["model_metrics"]["chakra"]["ece_calibrated"],
                           "test_coverage": {k: ev["conformal_test"][k] for k in ("coverage_class0", "coverage_class1")}},
              "scenarios": {name: run(name, knobs, scorer, baseline) for name, knobs in SCENARIOS.items()}}
    (REPORT_DIR / "domain_shift.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[done] {REPORT_DIR / 'domain_shift.json'}")


if __name__ == "__main__":
    main()
