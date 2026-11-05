"""LSTM on exactly five LASSO-selected temporal features and reviewed labels."""

import json
import io
import os
import tempfile
from pathlib import Path

import numpy as np

from .io import load_jsonl
from .feature_inventory import CONF_THR, _torso_scale, pose_feature_values
from .tracking import stable_track_ids, to_dense
from .training_features import PREDICTOR_IDS

SECONDS = 5
MAX_GAP_S = .25
WINDOW_SECONDS = 3
HOP_SECONDS = .25
STEPS = 9
INPUT_SIZE = 5
INPUT_VERSION = "lasso-feature-windows-v1"


def middle_sequences(run_dir, merges=None, feature_ids=None):
    """Measure feature statistics in nine 3-second windows inside the middle 5s.

    Cache all eligible candidates before selection. ``select_sequences`` then
    supplies only LASSO's five ordered columns to the model. Missing measurements
    remain NaN until preprocessing is fitted on reviewed training samples.
    """
    feature_ids = list(PREDICTOR_IDS if feature_ids is None else feature_ids)
    if not feature_ids or len(set(feature_ids)) != len(feature_ids) or not set(feature_ids) <= set(PREDICTOR_IDS):
        raise ValueError("Feature sequences require distinct eligible pose features")
    run_dir = Path(run_dir)
    det = load_jsonl(run_dir / "tracking.jsonl")
    if len(det.t) < 2 or not np.all(np.diff(det.t) > 0):
        raise ValueError("Tracking needs strictly increasing video timestamps")
    dt = float(np.median(np.diff(det.t)))
    duration = float(det.t[-1] - det.t[0] + dt)
    if duration < SECONDS - 1e-6:
        raise ValueError(f"Clip is {duration:.2f}s; a middle 5-second window requires at least 5s")
    start = float(det.t[0] + (duration - SECONDS) / 2)
    end = start + SECONDS
    ids, count = stable_track_ids(det, merges)
    dense = to_dense(det.kps, ids, count)
    # Crop before computing motion derivatives or statistics.
    inside = (det.t >= start - 1e-6) & (det.t < end - 1e-6)
    dense = dense[inside]
    times = det.t[inside]
    tracks = {}
    for tid in range(count):
        poses = dense[:, tid]
        coverage = float((poses[:, :, 2] >= CONF_THR).mean()) if len(poses) else 0.
        if coverage < .5:
            continue
        values = np.full((STEPS, len(feature_ids)), np.nan, np.float32)
        supported_windows = 0
        for step in range(STEPS):
            lo = start + step * HOP_SECONDS
            window = (times >= lo - 1e-6) & (times < lo + WINDOW_SECONDS - 1e-6)
            part, t = poses[window], times[window]
            if len(t) < 8 or (part[:, :, 2] >= CONF_THR).mean() < .5:
                continue
            scales = np.array([_torso_scale(pose) for pose in part])
            observed = t[np.isfinite(scales) & (scales > 0)]
            if len(observed) < 4 or np.any(np.diff(observed) > MAX_GAP_S + 1e-6):
                continue  # Never bridge long missing spans to manufacture feature measurements.
            measured = pose_feature_values(part, t, scales)
            values[step] = [measured.get(fid) if measured.get(fid) is not None else np.nan
                            for fid in feature_ids]
            supported_windows += 1
        if supported_windows >= (STEPS+1)//2:
            tracks[tid] = {"sequence": values, "coverage": coverage, "feature_ids": feature_ids}
    return tracks, {"start_s": start, "end_s": end, "frames": STEPS,
                    "window_seconds": WINDOW_SECONDS, "hop_seconds": HOP_SECONDS,
                    "feature_ids": feature_ids, "input_version": INPUT_VERSION,
                    "channels": f"{len(feature_ids)} pose-feature statistics; model selects five columns",
                    "max_interpolation_gap_s": MAX_GAP_S}


def select_sequences(samples, selected_features):
    """Select five named feature columns; no scores, coefficients or labels enter x."""
    if len(selected_features) != INPUT_SIZE or len(set(selected_features)) != INPUT_SIZE \
            or not set(selected_features) <= set(PREDICTOR_IDS):
        raise ValueError("LSTM requires exactly five distinct LASSO-selected eligible features")
    result = []
    for sample in samples:
        ids = sample["feature_ids"]
        sequence = np.asarray(sample["sequence"])
        if sequence.shape != (STEPS, len(ids)) or not set(selected_features) <= set(ids):
            raise ValueError("Feature sequence does not contain the selected LASSO columns")
        x = sequence[:, [ids.index(fid) for fid in selected_features]]
        if np.isinf(x).any():
            raise ValueError("Feature sequences cannot contain infinite measurements")
        if np.isfinite(x).mean() >= .5:
            result.append({**sample, "sequence": x, "feature_ids": list(selected_features)})
    return result


def _raw_inputs(samples, selected_features):
    if len(selected_features) != INPUT_SIZE or len(set(selected_features)) != INPUT_SIZE \
            or not set(selected_features) <= set(PREDICTOR_IDS):
        raise ValueError("LSTM requires exactly five distinct LASSO-selected eligible features")
    if not samples or any(s.get("feature_ids") != list(selected_features) for s in samples):
        raise ValueError("LSTM input feature order must match the LASSO selection")
    x = np.stack([s["sequence"] for s in samples]).astype(np.float32)
    if x.shape[1:] != (STEPS, INPUT_SIZE) or np.isinf(x).any():
        raise ValueError("LSTM requires nine timesteps of exactly five feature measurements")
    return x


def _transform(x, preprocessing):
    medians, mean, std = (np.asarray(preprocessing[key], dtype=np.float32)
                          for key in ("medians", "mean", "std"))
    if any(v.shape != (INPUT_SIZE,) or not np.isfinite(v).all() for v in (medians, mean, std)) \
            or np.any(std <= 0):
        raise ValueError("Invalid LSTM feature preprocessing")
    return (np.where(np.isfinite(x), x, medians) - mean) / std


def make_model():
    import torch

    class FeatureLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(INPUT_SIZE, 32, batch_first=True)
            self.output = torch.nn.Linear(32, 1)

        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            return self.output(hidden[-1]).squeeze(-1)

    return FeatureLSTM()


def train_lstm(samples, output_dir, selected_features, fingerprint, epochs=60):
    import torch

    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    raw = _raw_inputs(samples, selected_features)
    flat = raw.reshape(-1, INPUT_SIZE)
    if not np.isfinite(flat).any(axis=0).all():
        raise ValueError("Every selected feature needs observed reviewed training measurements")
    medians = np.nanmedian(flat, axis=0)
    filled = np.where(np.isfinite(flat), flat, medians)
    std = filled.std(axis=0)
    preprocessing = {"medians": medians.tolist(), "mean": filled.mean(axis=0).tolist(),
                     "std": np.where(std > 1e-8, std, 1.).tolist()}
    x = torch.tensor(_transform(raw, preprocessing), dtype=torch.float32)
    y = torch.tensor([s["label"] for s in samples], dtype=torch.float32)
    if set(y.tolist()) != {0., 1.}:
        raise ValueError("LSTM requires reviewed dancer and non-dancer examples")
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.001)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(len(y) - y.sum()) / y.sum())
    losses = []
    model.train()
    # One sequence per track; batches never split the time dimension.
    for _ in range(epochs):
        order = torch.randperm(len(y))
        total = 0.
        for batch in order.split(16):
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        losses.append(total / len(y))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "supervised_feature_sequences.npz", x=x.numpy(), raw=raw, y=y.numpy(),
                        feature_ids=np.asarray(selected_features), input_version=INPUT_VERSION,
                        runs=np.asarray([s["run"] for s in samples]),
                        track_ids=np.asarray([s["track_id"] for s in samples]))
    fd, temporary = tempfile.mkstemp(dir=output_dir, suffix=".pt")
    os.close(fd)
    try:
        torch.save({"state_dict": model.state_dict(), "input_size": INPUT_SIZE, "hidden_size": 32,
                    "steps": STEPS, "seconds": SECONDS, "input_version": INPUT_VERSION,
                    "window_seconds": WINDOW_SECONDS, "hop_seconds": HOP_SECONDS,
                    "preprocessing": preprocessing,
                    "selected_lasso_features": selected_features, "fingerprint": fingerprint}, temporary)
        os.replace(temporary, output_dir / "feature_lstm.pt")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    report = {"status": "trained", "epochs": epochs, "samples": len(samples),
              "runs": sorted({s["run"] for s in samples}), "loss": losses,
              "final_training_loss": losses[-1], "fingerprint": fingerprint,
              "evaluation": "Rolling fit uses all currently reviewed usable sequences. No independent test accuracy estimated.",
              "labels": {"dancer": 1, "non_dancer": 0},
              "input": "Five LASSO-selected feature statistics in nine overlapping 3-second windows within the middle 5 seconds",
              "input_size": INPUT_SIZE, "steps": STEPS, "input_version": INPUT_VERSION,
              "window_seconds": WINDOW_SECONDS, "hop_seconds": HOP_SECONDS,
              "preprocessing": preprocessing,
              "selected_lasso_features": selected_features}
    (output_dir / "feature_lstm_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def predict_lstm(samples, checkpoint_bytes, fingerprint):
    """Score usable tracks using the exact versioned checkpoint, in bounded batches."""
    import torch
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if checkpoint.get("fingerprint") != fingerprint:
        raise ValueError("LSTM checkpoint does not match the current training inputs")
    if checkpoint.get("input_version") != INPUT_VERSION or checkpoint.get("input_size") != INPUT_SIZE:
        raise ValueError("LSTM checkpoint does not use the five selected feature inputs; retrain")
    model = make_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    result = {}
    with torch.inference_mode():
        for offset in range(0, len(samples), 32):
            batch = samples[offset:offset+32]
            x = _transform(_raw_inputs(batch, checkpoint["selected_lasso_features"]), checkpoint["preprocessing"])
            probabilities = torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).tolist()
            for sample, probability in zip(batch, probabilities):
                if not np.isfinite(probability):
                    raise ValueError("LSTM produced a non-finite probability")
                result[(sample["run"], sample["track_id"])] = {
                    "probability": probability,
                    "predicted_label": "dancer" if probability >= .5 else "non_dancer"}
    return result
