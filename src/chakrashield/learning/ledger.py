"""Append-only decision ledger with propensities (the analytical-store stand-in).

In production this is a ClickHouse / warehouse table fed off the request
path. Here it is newline-delimited JSON, written by the async worker, one
record per served decision and one per delivery outcome. Everything a
retraining job needs is on the decision record: the feature vector at
scoring time, the resolver's action, the action actually served, whether
the order was a control-cohort pass-through, and the propensity of the
served action. Outcomes are joined by order id.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pandas as pd


class DecisionLedger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counts = {"decisions": 0, "control_cohort": 0, "outcomes": 0, "served": {}}
        if self.path.exists():
            self._recount()

    def _recount(self) -> None:
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    self._tally(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def _tally(self, rec: dict) -> None:
        c = self._counts
        if rec.get("kind") == "decision":
            c["decisions"] += 1
            c["control_cohort"] += int(bool(rec.get("is_control_cohort")))
            c["served"][rec.get("served_action")] = c["served"].get(rec.get("served_action"), 0) + 1
        elif rec.get("kind") == "outcome":
            c["outcomes"] += 1

    def _append(self, rec: dict) -> None:
        line = json.dumps(rec, separators=(",", ":"), ensure_ascii=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._tally(rec)

    def log_decision(self, rec: dict) -> None:
        self._append({"kind": "decision", "logged_at": time.time(), **rec})

    def log_outcome(self, order_id: str, rto: bool, extra: dict | None = None) -> None:
        self._append({"kind": "outcome", "logged_at": time.time(), "order_id": order_id, "rto": bool(rto), **(extra or {})})

    def stats(self) -> dict:
        with self._lock:
            return {"path": str(self.path), **json.loads(json.dumps(self._counts))}

    @staticmethod
    def load(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(decisions, outcomes) frames; decisions carry the feature vector as a list."""
        dec, out = [], []
        p = Path(path)
        if not p.exists():
            return pd.DataFrame(), pd.DataFrame()
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                (dec if r.get("kind") == "decision" else out).append(r)
        return pd.DataFrame(dec), pd.DataFrame(out)
