"""LightGBM -> ONNX export with parity verification.

The serving path runs ONNX Runtime with a single intra-op thread: a
35-feature, ~500-tree ensemble evaluates in well under a millisecond and
threading only adds jitter at batch size 1. We assert parity against the
native booster before writing the file, so the artifact on disk is the
model that was evaluated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def export_onnx(booster, n_features: int, out_path: Path, sample: np.ndarray | None = None, atol: float = 1e-4) -> dict:
    import onnx
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    initial_types = [("input", FloatTensorType([None, n_features]))]
    try:
        model = convert_lightgbm(booster, initial_types=initial_types, zipmap=False, target_opset=15)
    except TypeError:  # older onnxmltools without zipmap kwarg
        model = convert_lightgbm(booster, initial_types=initial_types, target_opset=15)
    onnx.checker.check_model(model)
    onnx.save_model(model, str(out_path))

    report = {"path": str(out_path), "opset": 15, "bytes": Path(out_path).stat().st_size}
    if sample is not None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(out_path), so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        outs = sess.run(None, {name: sample.astype(np.float32)})
        probs = _extract_prob(outs)
        ref = booster.predict(sample, num_iteration=booster.best_iteration or None)
        diff = np.abs(probs - ref)
        report.update({"parity_max_abs_diff": float(diff.max()), "parity_mean_abs_diff": float(diff.mean()),
                       "parity_ok": bool(diff.max() < atol), "n_checked": int(len(ref))})
        if diff.max() >= atol:
            raise RuntimeError(f"ONNX parity failed: max |diff| = {diff.max():.2e} >= {atol}")
    with open(str(out_path) + ".json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    return report


def _extract_prob(outs) -> np.ndarray:
    """ONNX tree classifiers emit [label, probabilities]; probabilities may be a
    (n,2) tensor or a list of dicts (ZipMap). Normalise to P(y=1)."""
    for o in outs:
        if isinstance(o, list) and o and isinstance(o[0], dict):
            return np.array([d.get(1, d.get("1", 0.0)) for d in o], dtype=float)
        arr = np.asarray(o)
        if arr.ndim == 2 and arr.shape[1] == 2 and arr.dtype.kind == "f":
            return arr[:, 1].astype(float)
    raise RuntimeError("could not locate probability output in ONNX outputs")
