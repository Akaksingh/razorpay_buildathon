"""Hot-path scorer: ONNX Runtime probability + exact TreeSHAP reason codes.

Two model handles are kept warm on purpose:

* the .onnx graph, evaluated by ONNX Runtime with one intra-op thread, is
  what produces P(RTO|x). It is the thing we benchmark and the thing that
  ships to the edge.
* the native LightGBM booster is used *only* for ``pred_contrib`` -- exact
  TreeSHAP log-odds attributions -- because explanation is a first-class
  output of the API, not a debugging afterthought. If the booster is
  missing we still score; we just cannot explain.

Every load checks that the persisted feature list matches the code's
FEATURE_NAMES, so a stale artifact refuses to serve rather than silently
scoring garbage.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import CONFORMAL_ALPHA, MODEL_DIR
from ..features.vectorizer import FEATURE_NAMES
from ..models.calibration import IsotonicKnots
from ..models.conformal import ConformalCalibrator


@dataclass
class ScoreResult:
    p_raw: float
    p_loss: float
    conformal_set: list[int]
    nonconformity: dict[str, float]
    contribs: np.ndarray | None          # shape (n_features,), log-odds
    bias: float
    timings_ms: dict[str, float] = field(default_factory=dict)


class Scorer:
    def __init__(self, model_dir: Path = MODEL_DIR, name: str = "chakra_rto") -> None:
        self.model_dir = Path(model_dir)
        self.name = name
        self.backend = "none"
        self.version = "unloaded"
        self._sess = None
        self._input_name = None
        self._booster = None
        self._iso: IsotonicKnots | None = None
        self._conf: ConformalCalibrator | None = None
        self._loaded = False
        self._shap_lock = threading.Lock()   # LightGBM pred_contrib is not safe to call concurrently

    # ------------------------------------------------------------- loading
    def load(self) -> "Scorer":
        meta_path = self.model_dir / f"{self.name}.txt.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"model artifacts not found in {self.model_dir}; run scripts/train.py")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        persisted = tuple(meta["feature_names"])
        if persisted != FEATURE_NAMES:
            raise RuntimeError("feature list drift between artifact and code; retrain before serving")
        self.version = meta.get("version", "dev")

        onnx_path = self.model_dir / f"{self.name}.onnx"
        if onnx_path.exists():
            try:
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
                self._input_name = self._sess.get_inputs()[0].name
                self.backend = "onnxruntime"
            except Exception:
                self._sess = None

        txt_path = self.model_dir / f"{self.name}.txt"
        if txt_path.exists():
            import lightgbm as lgb

            self._booster = lgb.Booster(model_file=str(txt_path))
            if self._sess is None:
                self.backend = "lightgbm"
        if self._sess is None and self._booster is None:
            raise FileNotFoundError("neither ONNX nor LightGBM artifact present")

        self._iso = IsotonicKnots.load(self.model_dir / f"{self.name}.isotonic.json")
        self._conf = ConformalCalibrator.load(self.model_dir / f"{self.name}.conformal.json")
        self._loaded = True
        self.warmup()
        return self

    def warmup(self, n: int = 20) -> None:
        x = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
        for _ in range(n):
            self._predict_raw(x)
            if self._booster is not None:
                self._booster.predict(x, pred_contrib=True)

    @property
    def conformal(self) -> ConformalCalibrator:
        assert self._conf is not None
        return self._conf

    # ------------------------------------------------------------- scoring
    def _predict_raw(self, x: np.ndarray) -> float:
        if self._sess is not None:
            outs = self._sess.run(None, {self._input_name: x.astype(np.float32)})
            for o in outs:
                if isinstance(o, list) and o and isinstance(o[0], dict):
                    return float(o[0].get(1, o[0].get("1", 0.0)))
                arr = np.asarray(o)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype.kind == "f":
                    return float(arr[0, 1])
            raise RuntimeError("unexpected ONNX output layout")
        return float(self._booster.predict(x)[0])

    def score(self, x: np.ndarray, explain: bool = True) -> ScoreResult:
        assert self._loaded, "call load() first"
        t0 = time.perf_counter()
        p_raw = self._predict_raw(x)
        t1 = time.perf_counter()
        p = float(np.clip(self._iso.apply(p_raw), 0.001, 0.999))
        cset = self._conf.predict_one(p)
        t2 = time.perf_counter()
        contribs, bias = None, 0.0
        if explain and self._booster is not None:
            with self._shap_lock:
                c = self._booster.predict(x, pred_contrib=True)[0]
            contribs, bias = c[:-1], float(c[-1])
        t3 = time.perf_counter()
        return ScoreResult(
            p_raw=p_raw, p_loss=p, conformal_set=cset, nonconformity=self._conf.nonconformity(p),
            contribs=contribs, bias=bias,
            timings_ms={"onnx_infer": (t1 - t0) * 1e3, "calibrate_conformal": (t2 - t1) * 1e3,
                        "treeshap": (t3 - t2) * 1e3 if contribs is not None else 0.0},
        )

    def explain(self, x: np.ndarray) -> tuple[np.ndarray | None, float, float]:
        """Exact TreeSHAP log-odds contributions, deferred.

        Returns (contribs, bias, elapsed_ms). Kept separate from score() so the
        gateway can skip it for ALLOW decisions: an allow needs no defence, a
        step-up or a block must carry reason codes.
        """
        if self._booster is None:
            return None, 0.0, 0.0
        t0 = time.perf_counter()
        # Serialised on purpose. The gateway runs requests in a threadpool, and concurrent
        # pred_contrib calls on one Booster corrupted the native heap under load
        # (STATUS_HEAP_CORRUPTION at 4 clients; see scripts/13_load_test.py). ONNX scoring
        # stays parallel; only the ~2 ms TreeSHAP pass takes the lock.
        with self._shap_lock:
            c = self._booster.predict(x, pred_contrib=True)[0]
        return c[:-1], float(c[-1]), (time.perf_counter() - t0) * 1e3

    def score_batch(self, X: np.ndarray) -> np.ndarray:
        """Calibrated P(RTO) for evaluation; uses ONNX if present."""
        if self._sess is not None:
            outs = self._sess.run(None, {self._input_name: X.astype(np.float32)})
            raw = None
            for o in outs:
                arr = np.asarray(o)
                if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype.kind == "f":
                    raw = arr[:, 1]
            if raw is None:
                raw = np.array([d.get(1, 0.0) for d in outs[1]])
        else:
            raw = self._booster.predict(X)
        return np.clip(self._iso.apply(raw), 0.001, 0.999)
