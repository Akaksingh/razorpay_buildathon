"""Isotonic recalibration stored as interpolation knots.

Cost weighting shifts the booster's probability scale (it over-states the
class it was told is expensive). tau*(x) is only meaningful against a
calibrated P(RTO|x), so we fit isotonic regression on a disjoint split and
persist the monotone step function as (x, y) knots. The runtime applies it
with np.interp -- no sklearn on the hot path, and it is trivially portable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass
class IsotonicKnots:
    x: list[float]
    y: list[float]

    @classmethod
    def fit(cls, p_raw: np.ndarray, y: np.ndarray) -> "IsotonicKnots":
        iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip", increasing=True)
        iso.fit(np.asarray(p_raw, dtype=float), np.asarray(y, dtype=float))
        xs = np.asarray(iso.X_thresholds_, dtype=float)
        ys = np.asarray(iso.y_thresholds_, dtype=float)
        return cls(x=xs.tolist(), y=ys.tolist())

    def apply(self, p_raw: np.ndarray | float) -> np.ndarray:
        return np.interp(np.asarray(p_raw, dtype=float), self.x, self.y)

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"x": self.x, "y": self.y}, fh)

    @classmethod
    def load(cls, path: Path) -> "IsotonicKnots":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(x=d["x"], y=d["y"])


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)
