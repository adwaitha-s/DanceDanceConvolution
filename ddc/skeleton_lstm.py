"""Supervised middle-five-second sequences. No video pixels enter this model."""

import json
from pathlib import Path

import numpy as np

from .io import load_jsonl
from .normalize import normalize
from .skeleton import BODY
from .tracking import stable_track_ids, to_dense

FPS = 20
SECONDS = 5
MAX_GAP_S = .25


def middle_sequences(run_dir, merges=None):
    """Return (track sequences, crop metadata); long gaps stay masked, never bridged."""
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
    # Crop before smoothing so frames outside the requested window cannot influence inputs.
    inside = (det.t >= start - 1e-6) & (det.t < end - 1e-6)
    norm, _, _, _ = normalize(dense[inside], smooth_win=3)
    times = det.t[inside]
    target = start + np.arange(FPS * SECONDS) / FPS
    tracks = {}
    for tid in range(count):
        xy = np.zeros((len(target), len(BODY), 2), np.float32)
        mask = np.zeros((len(target), len(BODY)), np.float32)
        for joint in range(len(BODY)):
            points = norm[:, tid, joint, :2]
            valid = np.all(np.isfinite(points), axis=1)
            observed_t = times[valid]
            if len(observed_t) < 2:
                continue
            pos = np.searchsorted(observed_t, target, side="left")
            lo, hi = np.clip(pos - 1, 0, len(observed_t)-1), np.clip(pos, 0, len(observed_t)-1)
            short = (observed_t[hi] - observed_t[lo]) <= MAX_GAP_S
            supported = (target >= observed_t[0]) & (target <= observed_t[-1]) & short
            for axis in range(2):
                xy[supported, joint, axis] = np.interp(target[supported], observed_t, points[valid, axis])
            mask[supported, joint] = 1
        coverage = float(mask.mean())
        if coverage >= .5:
            tracks[tid] = {"sequence": np.concatenate([xy.reshape(len(target), -1), mask], axis=1),
                           "coverage": coverage}
    return tracks, {"start_s": start, "end_s": end, "fps": FPS, "frames": FPS * SECONDS,
                    "channels": "34 hip-centered torso-scaled xy + 17 visibility masks",
                    "max_interpolation_gap_s": MAX_GAP_S}


def make_model():
    import torch

    class SkeletonLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(51, 32, batch_first=True)
            self.output = torch.nn.Linear(32, 1)

        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            return self.output(hidden[-1]).squeeze(-1)

    return SkeletonLSTM()


def train_lstm(samples, output_dir, selected_features, fingerprint, epochs=60):
    import torch

    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    x = torch.tensor(np.stack([s["sequence"] for s in samples]), dtype=torch.float32)
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
    np.savez_compressed(output_dir / "supervised_skeleton_sequences.npz", x=x.numpy(), y=y.numpy(),
                        runs=np.asarray([s["run"] for s in samples]),
                        track_ids=np.asarray([s["track_id"] for s in samples]))
    torch.save({"state_dict": model.state_dict(), "input_size": 51, "hidden_size": 32,
                "fps": FPS, "seconds": SECONDS, "joint_order": BODY,
                "selected_lasso_features": selected_features, "fingerprint": fingerprint},
               output_dir / "skeleton_lstm.pt")
    report = {"status": "trained", "epochs": epochs, "samples": len(samples),
              "runs": sorted({s["run"] for s in samples}), "loss": losses,
              "final_training_loss": losses[-1], "fingerprint": fingerprint,
              "evaluation": "Rolling fit uses all currently reviewed usable sequences. No independent test accuracy estimated.",
              "labels": {"dancer": 1, "non_dancer": 0},
              "input": "Middle 5 seconds; skeleton xy and visibility masks only",
              "selected_lasso_features": selected_features}
    (output_dir / "lstm_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
