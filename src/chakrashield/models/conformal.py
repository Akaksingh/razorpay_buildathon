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
