"""Learn buyer response to interventions per segment, instead of assuming it.

The resolver prices every action with four behavioural numbers the merchant
knows least: delta_s (good buyers who abandon at the Rs.49 prompt), delta_bad
(would-be-RTO buyers who abandon), rho (residual RTO among buyers who paid the
deposit) and delta_p (good buyers lost to a prepaid mandate). None of them is
directly observable -- at a step-up we see abandon / pay and, for payers,
deliver / RTO, but never whether the buyer was "good".

Identification uses the calibrated risk score as the instrument. For a
stepped-up order with abandonment a_i in {0, 1}:

    E[a_i | p_i] = (1 - p_i) * delta_s + p_i * delta_bad

so within a segment (delta_s, delta_bad) is a two-parameter regression through
the origin on (1 - p, p), solved in closed form from sufficient statistics
with a ridge toward the prior (kappa pseudo-observations). Residual RTO among
payers is E[rto_i | paid, p_i] = rho * w_i with

    w_i = p_i (1 - delta_bad) / [(1 - p_i)(1 - delta_s) + p_i (1 - delta_bad)]

(the posterior share of bad buyers among payers), again ridge least squares.
Estimates shrink two levels: segment -> global -> prior, so a thin segment
borrows strength and a new merchant starts exactly at the configured prior.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from ..config import Economics

CHANNEL_GROUP = {"ORGANIC": "ORGANIC", "DIRECT": "ORGANIC", "WHATSAPP": "ORGANIC", "GOOGLE_ADS": "SEARCH",
                 "MARKETPLACE": "SEARCH", "META_ADS": "SOCIAL", "INFLUENCER": "SOCIAL", "AFFILIATE": "SOCIAL"}


def segment_key(channel: str, pin_tier: int, gmv: float) -> str:
    band = "low" if gmv < 1000 else ("mid" if gmv < 2500 else "high")
    return f"{CHANNEL_GROUP.get(str(channel).upper(), 'SEARCH')}|T{int(pin_tier)}|{band}"


@dataclass
class _Accum:
    # step-up abandonment regression on (1-p, p)
    n: int = 0
    s11: float = 0.0
    s12: float = 0.0
    s22: float = 0.0
    t1: float = 0.0
    t2: float = 0.0
    # residual RTO among payers
    n_paid: int = 0
    ww: float = 0.0
    wr: float = 0.0
    # prepaid abandonment regression
    n_prep: int = 0
    q11: float = 0.0
    q12: float = 0.0
    q22: float = 0.0
    u1: float = 0.0
    u2: float = 0.0

    def add_stepup(self, p: float, a: float) -> None:
        q = 1.0 - p
        self.n += 1
        self.s11 += q * q
        self.s12 += q * p
        self.s22 += p * p
        self.t1 += a * q
        self.t2 += a * p

    def add_paid(self, w: float, rto: float) -> None:
        self.n_paid += 1
        self.ww += w * w
        self.wr += w * rto

    def add_prepaid(self, p: float, a: float) -> None:
        q = 1.0 - p
        self.n_prep += 1
        self.q11 += q * q
        self.q12 += q * p
        self.q22 += p * p
        self.u1 += a * q
        self.u2 += a * p


def _ridge2(s11, s12, s22, t1, t2, kappa, prior: tuple[float, float]) -> tuple[float, float]:
    """Solve (S + kappa I) d = t + kappa d0 for a 2x2 system."""
    a, b, c = s11 + kappa, s12, s22 + kappa
    r1, r2 = t1 + kappa * prior[0], t2 + kappa * prior[1]
    det = a * c - b * b
    if det <= 1e-12:
        return prior
    d1 = (c * r1 - b * r2) / det
    d2 = (a * r2 - b * r1) / det
    clip = lambda v: float(min(0.95, max(0.005, v)))
    return clip(d1), clip(d2)


@dataclass
class BehaviourEstimate:
    segment: str
    delta_s: float
    delta_bad: float
    rho: float
    delta_p: float
    n_stepup: int
    n_paid: int
    n_prepaid: int
    source: str                     # prior | global | segment

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("delta_s", "delta_bad", "rho", "delta_p"):
            d[k] = round(d[k], 4)
        return d


@dataclass
class BehaviourLearner:
    prior: Economics
    kappa_global: float = 150.0
    kappa_segment: float = 40.0
    kappa_rho: float = 30.0
    min_n_segment: int = 25
    min_n_global: int = 30
    segments: dict[str, _Accum] = field(default_factory=dict)
    glob: _Accum = field(default_factory=_Accum)
    observations: int = 0

    # --------------------------------------------------------------- priors
    @property
    def prior_stepup(self) -> tuple[float, float]:
        return self.prior.stepup_abandon_rate, 0.65        # (delta_s, delta_bad) before any data

    @property
    def prior_prepaid(self) -> tuple[float, float]:
        return self.prior.prepaid_abandon_rate, 0.85

    def _seg(self, segment: str) -> _Accum:
        return self.segments.setdefault(segment, _Accum())

    # ---------------------------------------------------------- observations
    def observe_stepup(self, segment: str, p: float, abandoned: bool, rto: bool | None = None) -> None:
        """A step-up was served: the buyer abandoned, or paid and then delivered / refused."""
        p = float(min(0.999, max(0.001, p)))
        a = 1.0 if abandoned else 0.0
        for acc in (self._seg(segment), self.glob):
            acc.add_stepup(p, a)
        if not abandoned and rto is not None:
            ds, db = self.prior_stepup                      # posterior bad-share among payers, at the prior
            w = p * (1 - db) / ((1 - p) * (1 - ds) + p * (1 - db))
            for acc in (self._seg(segment), self.glob):
                acc.add_paid(w, 1.0 if rto else 0.0)
        self.observations += 1

    def observe_prepaid(self, segment: str, p: float, abandoned: bool) -> None:
        p = float(min(0.999, max(0.001, p)))
        a = 1.0 if abandoned else 0.0
        for acc in (self._seg(segment), self.glob):
            acc.add_prepaid(p, a)
        self.observations += 1

    # -------------------------------------------------------------- estimates
    def _global_estimate(self) -> tuple[tuple[float, float], float, tuple[float, float]]:
        g = self.glob
        su = _ridge2(g.s11, g.s12, g.s22, g.t1, g.t2, self.kappa_global, self.prior_stepup)
        rho = (g.wr + self.kappa_rho * self.prior.stepup_rto_residual) / (g.ww + self.kappa_rho)
        pp = _ridge2(g.q11, g.q12, g.q22, g.u1, g.u2, self.kappa_global, self.prior_prepaid)
        return su, float(min(0.99, max(0.01, rho))), pp

    def estimate(self, segment: str) -> BehaviourEstimate:
        gsu, grho, gpp = self._global_estimate()
        acc = self.segments.get(segment)
        if self.glob.n < self.min_n_global:
            su, rho, pp, src = self.prior_stepup, self.prior.stepup_rto_residual, self.prior_prepaid, "prior"
        elif acc is None or acc.n < self.min_n_segment:
            su, rho, pp, src = gsu, grho, gpp, "global"
        else:
            su = _ridge2(acc.s11, acc.s12, acc.s22, acc.t1, acc.t2, self.kappa_segment, gsu)
            rho = (acc.wr + self.kappa_rho * grho) / (acc.ww + self.kappa_rho)
            rho = float(min(0.99, max(0.01, rho)))
            pp = _ridge2(acc.q11, acc.q12, acc.q22, acc.u1, acc.u2, self.kappa_segment, gpp)
            src = "segment"
        return BehaviourEstimate(segment=segment, delta_s=su[0], delta_bad=su[1], rho=rho, delta_p=pp[0],
                                 n_stepup=acc.n if acc else 0, n_paid=acc.n_paid if acc else 0,
                                 n_prepaid=acc.n_prep if acc else 0, source=src)

    def economics_for(self, segment: str, base: Economics) -> tuple[Economics, BehaviourEstimate]:
        """The resolver's economics with the learned behaviour swapped in for this segment."""
        est = self.estimate(segment)
        if est.source == "prior":
            return base, est
        return replace(base, stepup_abandon_rate=est.delta_s, stepup_rto_residual=est.rho,
                       prepaid_abandon_rate=est.delta_p), est

    # --------------------------------------------------------------- reports
    def summary(self) -> dict:
        gsu, grho, gpp = self._global_estimate()
        rows = sorted((self.estimate(s).as_dict() for s in self.segments), key=lambda r: -r["n_stepup"])
        return {"observations": self.observations, "segments": len(self.segments),
                "prior": {"delta_s": self.prior.stepup_abandon_rate, "delta_bad": 0.65, "rho": self.prior.stepup_rto_residual,
                          "delta_p": self.prior.prepaid_abandon_rate},
                "global": {"delta_s": round(gsu[0], 4), "delta_bad": round(gsu[1], 4), "rho": round(grho, 4),
                           "delta_p": round(gpp[0], 4), "n_stepup": self.glob.n, "n_paid": self.glob.n_paid, "n_prepaid": self.glob.n_prep},
                "rows": rows}

    def snapshot(self, path: Path) -> None:
        data = {"observations": self.observations, "glob": asdict(self.glob), "segments": {k: asdict(v) for k, v in self.segments.items()}}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    def load(self, path: Path) -> "BehaviourLearner":
        p = Path(path)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            self.observations = int(data.get("observations", 0))
            self.glob = _Accum(**data.get("glob", {}))
            self.segments = {k: _Accum(**v) for k, v in data.get("segments", {}).items()}
        return self
