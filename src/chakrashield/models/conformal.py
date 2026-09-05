"""Inductive (split) conformal prediction, Mondrian by class.

For a calibrated scorer p(x) = P(y=1|x) define the nonconformity of
labelling x as y:   s(x, y) = 1 - p(y | x).

On a held-out conformal set {(x_i, y_i)} that the scorer never saw, take
class-conditional quantiles

    q_c = the ceil((n_c + 1)(1 - alpha)) / n_c  empirical quantile of
          { s(x_i, y_i) : y_i = c }

Then the prediction set  C(x) = { y : s(x, y) <= q_y }  satisfies

    P( y_test in C(x_test) | y_test = c ) >= 1 - alpha       for each c,

under exchangeability, with *no* assumptions on the model. Class
conditioning matters here: RTO is the minority class and a marginal
guarantee could be met by covering deliverables well and RTOs poorly.

Outcomes:  {0} certified deliverable, {1} certified RTO, {0,1} ambiguous,
{} neither label conforms (novel input). All four are actionable.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class ConformalCalibrator:
    alpha: float
    q0: float          # quantile for class 0 nonconformity
    q1: float          # quantile for class 1 nonconformity
    n0: int
    n1: int
    empirical_cov0: float = 0.0
    empirical_cov1: float = 0.0

    @staticmethod
    def _quantile(scores: np.ndarray, alpha: float) -> float:
        n = len(scores)
        if n == 0:
            return 1.0
        k = math.ceil((n + 1) * (1 - alpha))
        k = min(max(k, 1), n)
        return float(np.sort(scores)[k - 1])

    @classmethod
    def fit(cls, p_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> "ConformalCalibrator":
        p_cal = np.asarray(p_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int)
        s0 = 1.0 - (1.0 - p_cal[y_cal == 0])     # s(x, 0) = 1 - P(y=0|x) = p
        s1 = 1.0 - p_cal[y_cal == 1]             # s(x, 1) = 1 - p
        c = cls(alpha=alpha, q0=cls._quantile(s0, alpha), q1=cls._quantile(s1, alpha),
                n0=int((y_cal == 0).sum()), n1=int((y_cal == 1).sum()))
        sets = c.predict_set(p_cal)
        c.empirical_cov0 = float(np.mean([0 in s for s, y in zip(sets, y_cal) if y == 0])) if c.n0 else 0.0
        c.empirical_cov1 = float(np.mean([1 in s for s, y in zip(sets, y_cal) if y == 1])) if c.n1 else 0.0
        return c

    def nonconformity(self, p: float) -> dict[str, float]:
        return {"s0": float(p), "s1": float(1.0 - p)}

    def predict_set(self, p: np.ndarray | float) -> list[list[int]]:
        p = np.atleast_1d(np.asarray(p, dtype=float))
        out = []
        for v in p:
            s = []
            if v <= self.q0:            # s(x,0) = p <= q0
                s.append(0)
            if (1.0 - v) <= self.q1:    # s(x,1) = 1-p <= q1
                s.append(1)
            out.append(s)
        return out

    def predict_one(self, p: float) -> list[int]:
        return self.predict_set(p)[0]

    def evaluate(self, p: np.ndarray, y: np.ndarray) -> dict:
        sets = self.predict_set(p)
        y = np.asarray(y, dtype=int)
        cov0 = float(np.mean([0 in s for s, yy in zip(sets, y) if yy == 0]))
        cov1 = float(np.mean([1 in s for s, yy in zip(sets, y) if yy == 1]))
        sizes = np.array([len(s) for s in sets])
        return {
            "alpha": self.alpha, "coverage_class0": cov0, "coverage_class1": cov1,
            "coverage_marginal": float(np.mean([yy in s for s, yy in zip(sets, y)])),
            "frac_singleton": float(np.mean(sizes == 1)), "frac_ambiguous": float(np.mean(sizes == 2)),
            "frac_empty": float(np.mean(sizes == 0)), "q0": self.q0, "q1": self.q1,
        }

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ConformalCalibrator":
        with open(path, "r", encoding="utf-8") as fh:
            return cls(**json.load(fh))


# ---------------------------------------------------------------------------
# Alternative conditionings. Neither is served; scripts/12_conformal_variants.py
# measures what each buys in coverage, ambiguity and resolver P&L against the
# class-conditional calibrator above.
# ---------------------------------------------------------------------------


def _pooled_scores(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """s(x_i, y_i) for the observed label, written exactly as ConformalCalibrator.fit writes it."""
    return np.where(y == 1, 1.0 - p, 1.0 - (1.0 - p))


def set_metrics(sets: list[list[int]], y: np.ndarray) -> dict:
    """Coverage per class and set-size mix; the part of evaluate() every variant shares."""
    y = np.asarray(y, dtype=int)
    sizes = np.array([len(s) for s in sets])
    hit = np.array([yy in s for s, yy in zip(sets, y)])
    return {
        "coverage_class0": float(hit[y == 0].mean()) if (y == 0).any() else 0.0,
        "coverage_class1": float(hit[y == 1].mean()) if (y == 1).any() else 0.0,
        "coverage_marginal": float(hit.mean()),
        "frac_singleton": float(np.mean(sizes == 1)), "frac_ambiguous": float(np.mean(sizes == 2)),
        "frac_empty": float(np.mean(sizes == 0)),
    }


@dataclass
class MarginalConformalCalibrator:
    """One quantile over the pooled nonconformity scores: P(y in C(x)) >= 1 - alpha, no per-class promise.

    The cheapest conformal layer there is, and the one most textbooks describe. Its guarantee can be
    met by covering the 78 % of deliverables well and the RTOs poorly, which is exactly the failure a
    COD engine cannot afford; it is kept as the control the class-conditional layer is compared against.
    """
    alpha: float
    q: float
    n: int

    @classmethod
    def fit(cls, p_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> "MarginalConformalCalibrator":
        p_cal = np.asarray(p_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int)
        return cls(alpha=alpha, q=ConformalCalibrator._quantile(_pooled_scores(p_cal, y_cal), alpha), n=int(len(y_cal)))

    def predict_set(self, p: np.ndarray | float) -> list[list[int]]:
        p = np.atleast_1d(np.asarray(p, dtype=float))
        return [[c for c, s in ((0, v), (1, 1.0 - v)) if s <= self.q] for v in p]

    def evaluate(self, p: np.ndarray, y: np.ndarray) -> dict:
        return {"alpha": self.alpha, **set_metrics(self.predict_set(p), y), "q": self.q}


@dataclass
class GroupConformalCalibrator:
    """Mondrian by class x group (here: PIN tier): a quantile per (group, class) cell.

    Class conditioning promises 1 - alpha coverage of RTOs overall; it does not promise it in tier-4
    PINs, where the scorer's errors concentrate. Conditioning on the group as well makes the promise
    per cell, at the price of smaller calibration cells (noisier quantiles, wider sets). A group unseen
    at calibration falls back to the class-conditional quantiles.
    """
    alpha: float
    q0: dict[int, float]
    q1: dict[int, float]
    n0: dict[int, int]
    n1: dict[int, int]
    fallback: ConformalCalibrator

    @classmethod
    def fit(cls, p_cal: np.ndarray, y_cal: np.ndarray, g_cal: np.ndarray, alpha: float) -> "GroupConformalCalibrator":
        p_cal = np.asarray(p_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int)
        g_cal = np.asarray(g_cal, dtype=int)
        q0, q1, n0, n1 = {}, {}, {}, {}
        for g in sorted(set(g_cal.tolist())):
            m = g_cal == g
            q0[g] = ConformalCalibrator._quantile(1.0 - (1.0 - p_cal[m & (y_cal == 0)]), alpha)
            q1[g] = ConformalCalibrator._quantile(1.0 - p_cal[m & (y_cal == 1)], alpha)
            n0[g], n1[g] = int((m & (y_cal == 0)).sum()), int((m & (y_cal == 1)).sum())
        return cls(alpha=alpha, q0=q0, q1=q1, n0=n0, n1=n1, fallback=ConformalCalibrator.fit(p_cal, y_cal, alpha))

    def quantiles(self, g: int) -> tuple[float, float]:
        g = int(g)
        if g in self.q0:
            return self.q0[g], self.q1[g]
        return self.fallback.q0, self.fallback.q1

    def predict_set(self, p: np.ndarray | float, g: np.ndarray | int) -> list[list[int]]:
        p = np.atleast_1d(np.asarray(p, dtype=float))
        g = np.broadcast_to(np.atleast_1d(np.asarray(g, dtype=int)), p.shape)
        out = []
        for v, gg in zip(p, g):
            q0, q1 = self.quantiles(gg)
            out.append([c for c, s, q in ((0, v, q0), (1, 1.0 - v, q1)) if s <= q])
        return out

    def evaluate(self, p: np.ndarray, y: np.ndarray, g: np.ndarray) -> dict:
        return {"alpha": self.alpha, **set_metrics(self.predict_set(p, g), y),
                "q0": {int(k): v for k, v in self.q0.items()}, "q1": {int(k): v for k, v in self.q1.items()}}


def coverage_by_group(sets: list[list[int]], y: np.ndarray, g: np.ndarray) -> dict[int, dict]:
    """Class-conditional coverage inside each group, for any variant's sets."""
    y = np.asarray(y, dtype=int)
    g = np.asarray(g, dtype=int)
    out = {}
    for gg in sorted(set(g.tolist())):
        m = g == gg
        sub = [s for s, keep in zip(sets, m) if keep]
        out[int(gg)] = {"n0": int((m & (y == 0)).sum()), "n1": int((m & (y == 1)).sum()), **{k: v for k, v in set_metrics(sub, y[m]).items()}}
    return out
