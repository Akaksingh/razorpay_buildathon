"""Zero-latency distribution-shift monitor built on conformal set dynamics.

Classic drift checks (PSI / KS on raw features) need batched feature windows
and say nothing about whether *inference* degraded. The conformal calibrator
gives a label-free signal on every request: which prediction set the order
landed in. Under exchangeability the mix of {certified-low, ambiguous,
certified-high, empty} sets is stationary and known from calibration; when
live traffic decouples from the calibration distribution the mix moves
immediately, days before any delivery label exists.

Three alarms, all computed from sliding windows kept in the feature store
(Redis or in-process, TTL-expired, one HINCRBY per request):

    MODEL_EPISTEMIC_DRIFT   empty-set share above ``empty_threshold``, or the
                            ambiguous share departs from its calibration
                            baseline by more than ``z_threshold`` sigma
    MODEL_RISK_MIX_SHIFT    certified-RTO share more than ``mix_ratio`` x its
                            baseline (a festival burst, or a new ring)
    MODEL_SCORE_PSI         population stability index of calibrated p against
                            the calibration histogram above ``psi_threshold``

On empty sets: C(x) = {} needs p > q0 and p < 1 - q1, i.e. q0 + q1 < 1. With a
weak scorer the Mondrian quantiles overlap (q0 + q1 > 1) and empty sets are
impossible; the alarm is then carried by the ambiguity and PSI signals, and
the empty-set alarm becomes live automatically once the scorer sharpens.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

CERTAINTIES = ("CERTIFIED_LOW", "AMBIGUOUS", "CERTIFIED_HIGH", "NOVEL")
N_BINS = 20
_EPS = 1e-4


@dataclass
class DriftBaseline:
    share: dict[str, float]                 # calibration-time share of each certainty
    p_hist: list[float]                     # N_BINS-bin histogram of calibrated p, as probabilities
    n: int = 0
    q0: float = 0.0
    q1: float = 0.0

    @property
    def empty_sets_possible(self) -> bool:
        return self.q0 + self.q1 < 1.0

    @classmethod
    def from_reports(cls, evaluation: dict | None, q0: float = 0.0, q1: float = 0.0) -> "DriftBaseline":
        ev = evaluation or {}
        cert = ev.get("certainty") or {}
        n = sum(int(v.get("n", 0)) for v in cert.values())
        if n > 0:
            share = {k: int(cert.get(k, {}).get("n", 0)) / n for k in CERTAINTIES}
        else:
            share = {"CERTIFIED_LOW": 0.3, "AMBIGUOUS": 0.5, "CERTIFIED_HIGH": 0.2, "NOVEL": 0.0}
        counts = (ev.get("p_loss_hist") or {}).get("counts") or [1] * N_BINS
        counts = [float(c) for c in counts][:N_BINS] + [0.0] * max(0, N_BINS - len(counts))
        tot = sum(counts) or 1.0
        return cls(share=share, p_hist=[c / tot for c in counts], n=n, q0=q0, q1=q1)


def psi(live: list[float], base: list[float]) -> float:
    """Population stability index between two probability histograms."""
    out = 0.0
    for a, b in zip(live, base):
        a, b = max(a, _EPS), max(b, _EPS)
        out += (a - b) * math.log(a / b)
    return float(out)


@dataclass
class ConformalDriftMonitor:
    store: object
    baseline: DriftBaseline
    window_s: int = 300                    # 5-minute buckets
    keep: int = 12                         # one rolling hour
    empty_threshold: float = 0.03
    z_threshold: float = 4.0
    mix_ratio: float = 2.0
    psi_threshold: float = 0.25
    min_n: int = 50
    clock: object = field(default=time.time)

    # ------------------------------------------------------------- writes
    def _key(self, bucket: int) -> str:
        return f"drift:w:{bucket}"

    def record(self, certainty: str, p_loss: float, now: float | None = None) -> None:
        b = int((now if now is not None else self.clock()) // self.window_s)
        ttl = self.window_s * (self.keep + 2)
        key = self._key(b)
        self.store.hincrby(key, certainty if certainty in CERTAINTIES else "NOVEL", 1, ttl=ttl)
        self.store.hincrby(key, f"h{min(N_BINS - 1, max(0, int(p_loss * N_BINS)))}", 1, ttl=ttl)
        self.store.hincrby(key, "n", 1, ttl=ttl)

    # -------------------------------------------------------------- reads
    def windows(self, now: float | None = None) -> list[dict]:
        b = int((now if now is not None else self.clock()) // self.window_s)
        out = []
        for k in range(b - self.keep + 1, b + 1):
            h = self.store.hgetall(self._key(k)) or {}
            n = int(h.get("n", 0))
            counts = {c: int(h.get(c, 0)) for c in CERTAINTIES}
            hist = [int(h.get(f"h{i}", 0)) for i in range(N_BINS)]
            out.append({"bucket": k, "start_ts": k * self.window_s, "n": n, "counts": counts,
                        "share": {c: (counts[c] / n if n else 0.0) for c in CERTAINTIES}, "hist": hist})
        return out

    def snapshot(self, now: float | None = None) -> dict:
        wins = self.windows(now)
        n = sum(w["n"] for w in wins)
        counts = {c: sum(w["counts"][c] for w in wins) for c in CERTAINTIES}
        share = {c: (counts[c] / n if n else 0.0) for c in CERTAINTIES}
        hist = [sum(w["hist"][i] for w in wins) for i in range(N_BINS)]
        live_hist = [h / n for h in hist] if n else [0.0] * N_BINS
        score_psi = psi(live_hist, self.baseline.p_hist) if n else 0.0
        alerts = self._alerts(share, n, score_psi)
        status = "WARMING" if n < self.min_n else ("ALERT" if any(a["severity"] == "critical" for a in alerts)
                                                 else ("WARN" if alerts else "OK"))
        return {
            "status": status, "window_s": self.window_s, "keep": self.keep, "rolling_n": n,
            "rolling_share": share, "score_psi": round(score_psi, 4), "alerts": alerts,
            "baseline": {"share": self.baseline.share, "n": self.baseline.n, "q0": self.baseline.q0, "q1": self.baseline.q1,
                         "empty_sets_possible": self.baseline.empty_sets_possible},
            "thresholds": {"empty": self.empty_threshold, "z": self.z_threshold, "mix_ratio": self.mix_ratio,
                           "psi": self.psi_threshold, "min_n": self.min_n},
            "windows": [{k: v for k, v in w.items() if k != "hist"} for w in wins],
            "live_hist": live_hist,
        }

    def _alerts(self, share: dict[str, float], n: int, score_psi: float) -> list[dict]:
        if n < self.min_n:
            return []
        alerts: list[dict] = []
        empty = share["NOVEL"]
        if empty > self.empty_threshold:
            alerts.append({"code": "MODEL_EPISTEMIC_DRIFT", "severity": "critical", "value": round(empty, 4),
                           "threshold": self.empty_threshold,
                           "message": f"{empty:.1%} of live orders fall in an empty conformal set: neither label conforms at alpha. "
                                      f"Traffic has decoupled from the calibration set."})
        amb0 = self.baseline.share.get("AMBIGUOUS", 0.5)
        sd = math.sqrt(max(amb0 * (1 - amb0), 1e-6) / n)
        z = (share["AMBIGUOUS"] - amb0) / sd
        if abs(z) > self.z_threshold:
            alerts.append({"code": "MODEL_EPISTEMIC_DRIFT", "severity": "critical" if abs(z) > 2 * self.z_threshold else "warning",
                           "value": round(share["AMBIGUOUS"], 4), "threshold": round(amb0, 4), "z": round(z, 2),
                           "message": f"Ambiguous-set share {share['AMBIGUOUS']:.1%} vs calibration {amb0:.1%} (z = {z:+.1f}). "
                                      f"The model is {'less' if z > 0 else 'more'} certain about live traffic than it was calibrated to be."})
        hi0 = self.baseline.share.get("CERTIFIED_HIGH", 0.2)
        if hi0 > 0 and share["CERTIFIED_HIGH"] > self.mix_ratio * hi0:
            alerts.append({"code": "MODEL_RISK_MIX_SHIFT", "severity": "warning", "value": round(share["CERTIFIED_HIGH"], 4),
                           "threshold": round(self.mix_ratio * hi0, 4),
                           "message": f"Certified-RTO share {share['CERTIFIED_HIGH']:.1%} is {share['CERTIFIED_HIGH'] / hi0:.1f}x its baseline "
                                      f"({hi0:.1%}): a burst of high-risk traffic (festival sale, new ring)."})
        if score_psi > self.psi_threshold:
            alerts.append({"code": "MODEL_SCORE_PSI", "severity": "critical" if score_psi > 2 * self.psi_threshold else "warning",
                           "value": round(score_psi, 4), "threshold": self.psi_threshold,
                           "message": f"PSI of calibrated P(RTO) vs the calibration histogram is {score_psi:.2f} (> {self.psi_threshold})."})
        return alerts
