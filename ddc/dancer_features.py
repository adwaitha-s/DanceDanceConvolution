"""Feature collection and evaluation for dancer-vs-non-dancer tracks.

The detector only finds people; it cannot know who is participating in the
dance.  This module turns each stabilized track in a ``tracking.jsonl`` run
into five motion/pose features, then compares three feature-selection
strategies against human supplied dancer/non-dancer labels:

* filter   -- univariate Fisher score;
* wrapper  -- greedy forward selection using leave-one-out accuracy;
* embedded -- sparse (L1) logistic-regression coefficients.

The public CLI deliberately requires labels for real runs.  ``--demo`` is a
synthetic verification dataset only; it is useful for checking the pipeline,
not for estimating real-world accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np

from . import io as ddc_io
from .skeleton import CONF_THR, J
from .tracking import stabilize, to_dense


FEATURE_NAMES = (
    "wrist_height",
    "wrist_speed",
    "limb_extension",
    "limb_motion",
    "pose_motion",
)


@dataclass(frozen=True)
class FeatureRow:
    """One track's aggregate pose features, ready to join with a label."""

    track_id: int
    frames_present: int
    values: tuple[float, float, float, float, float]

    def as_dict(self) -> dict[str, float | int]:
        return {
            "track_id": self.track_id,
            "frames_present": self.frames_present,
            **dict(zip(FEATURE_NAMES, self.values)),
        }


def _mean_or_nan(values: Iterable[float]) -> float:
    values = np.asarray(list(values), float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _torso_scale(pose: np.ndarray) -> float:
    """Shoulder-to-hip distance; returns NaN when the torso is not reliable."""
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    ids = [J[name] for name in required]
    if not np.all(pose[ids, 2] >= CONF_THR):
        return float("nan")
    shoulders = pose[[J["left_shoulder"], J["right_shoulder"]], :2].mean(axis=0)
    hips = pose[[J["left_hip"], J["right_hip"]], :2].mean(axis=0)
    return float(np.linalg.norm(shoulders - hips))


def _visible_speed(points: np.ndarray, scales: np.ndarray, t: np.ndarray) -> list[float]:
    """Frame-to-frame speed in torso lengths/sec for one or more joints."""
    values: list[float] = []
    for f in range(1, len(points)):
        dt = float(t[f] - t[f - 1])
        if not (dt > 0 and np.isfinite(scales[f]) and np.isfinite(scales[f - 1])):
            continue
        before, after = points[f - 1], points[f]
        valid = (before[:, 2] >= CONF_THR) & (after[:, 2] >= CONF_THR)
        if valid.any():
            distance = np.linalg.norm(after[valid, :2] - before[valid, :2], axis=1).mean()
            values.append(float(distance / (dt * (scales[f] + scales[f - 1]) / 2)))
    return values


def collect_track_features(tracking_jsonl: str | Path) -> list[FeatureRow]:
    """Collect five scale-normalized features for every persistent person track.

    Features deliberately describe movement and articulated pose rather than
    image location, so a bystander at a different distance or place in frame
    does not automatically look like a dancer.
    """
    det = ddc_io.load_jsonl(tracking_jsonl)
    if det.n_frames == 0:
        raise ValueError("tracking.jsonl is empty")
    track_ids, n_tracks = stabilize(det.kps)
    if n_tracks == 0:
        raise ValueError("no persistent people found")
    dense = to_dense(det.kps, track_ids, n_tracks)
    rows: list[FeatureRow] = []
    wrist_ids = [J["left_wrist"], J["right_wrist"]]
    limb_pairs = [
        (J["left_shoulder"], J["left_wrist"]),
        (J["right_shoulder"], J["right_wrist"]),
        (J["left_hip"], J["left_ankle"]),
        (J["right_hip"], J["right_ankle"]),
    ]
    motion_ids = [
        J["left_elbow"], J["right_elbow"], J["left_knee"], J["right_knee"],
    ]

    for tid in range(n_tracks):
        poses = dense[:, tid]
        scales = np.array([_torso_scale(p) for p in poses])
        wrist_heights: list[float] = []
        limb_extensions: list[float] = []
        for f, p in enumerate(poses):
            scale = scales[f]
            if not np.isfinite(scale) or scale <= 0:
                continue
            hips = p[[J["left_hip"], J["right_hip"]], :2].mean(axis=0)
            for wi in wrist_ids:
                if p[wi, 2] >= CONF_THR:
                    # Image y increases downward: a wrist above the hips is positive.
                    wrist_heights.append(float((hips[1] - p[wi, 1]) / scale))
            for parent, child in limb_pairs:
                if p[parent, 2] >= CONF_THR and p[child, 2] >= CONF_THR:
                    limb_extensions.append(float(np.linalg.norm(p[parent, :2] - p[child, :2]) / scale))

        wrist_speed = _visible_speed(poses[:, wrist_ids], scales, det.t)
        limb_motion = _visible_speed(poses[:, motion_ids], scales, det.t)
        all_motion = _visible_speed(poses, scales, det.t)
        rows.append(FeatureRow(
            track_id=tid,
            frames_present=int(np.isfinite(poses[:, :, 0]).any(axis=1).sum()),
            values=(
                _mean_or_nan(wrist_heights),
                _mean_or_nan(wrist_speed),
                _mean_or_nan(limb_extensions),
                _mean_or_nan(limb_motion),
                _mean_or_nan(all_motion),
            ),
        ))
    return rows


def write_feature_csv(rows: Iterable[FeatureRow], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "frames_present", *FEATURE_NAMES])
        writer.writeheader()
        writer.writerows(row.as_dict() for row in rows)
    return path


def load_labels(path: str | Path) -> dict[int, int]:
    """Read ``track_id,label`` CSV labels; accepted labels are dancer/non_dancer."""
    labels: dict[int, int] = {}
    positive = {"dancer", "dance", "1", "true", "yes"}
    negative = {"non_dancer", "non-dancer", "nondancer", "bystander", "0", "false", "no"}
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"track_id", "label"}.issubset(reader.fieldnames):
            raise ValueError("labels CSV needs track_id,label columns")
        for row in reader:
            value = (row["label"] or "").strip().lower()
            if value in positive:
                labels[int(row["track_id"])] = 1
            elif value in negative:
                labels[int(row["track_id"])] = 0
            else:
                raise ValueError(f"unknown label {row['label']!r}; use dancer or non_dancer")
    return labels


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-9] = 1.0
    return (train - mean) / std, (test - mean) / std


def _centroid_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> int:
    train_x, test_x = _standardize(train_x, test_x[None, :])
    centroids = np.array([train_x[train_y == label].mean(axis=0) for label in (0, 1)])
    return int(np.argmin(((centroids - test_x[0]) ** 2).sum(axis=1)))


def loo_accuracy(x: np.ndarray, y: np.ndarray, columns: Iterable[int]) -> float:
    """Leave-one-track-out nearest-centroid accuracy for selected columns."""
    columns = list(columns)
    if not columns or len(x) < 4 or min((y == 0).sum(), (y == 1).sum()) < 2:
        return float("nan")
    hits = 0
    for held_out in range(len(x)):
        keep = np.arange(len(x)) != held_out
        if min((y[keep] == 0).sum(), (y[keep] == 1).sum()) == 0:
            continue
        pred = _centroid_predict(x[keep][:, columns], y[keep], x[held_out, columns])
        hits += pred == y[held_out]
    return hits / len(x)


def filter_select(x: np.ndarray, y: np.ndarray, n_features: int = 3) -> list[int]:
    """Rank features by their between-class / within-class variance (Fisher) score."""
    scores = []
    for col in range(x.shape[1]):
        a, b = x[y == 0, col], x[y == 1, col]
        score = (a.mean() - b.mean()) ** 2 / (a.var() + b.var() + 1e-12)
        scores.append(float(score))
    return list(np.argsort(scores)[::-1][:n_features])


def wrapper_select(x: np.ndarray, y: np.ndarray, n_features: int = 3) -> list[int]:
    """Greedy forward feature selection scored with leave-one-out accuracy."""
    selected: list[int] = []
    remaining = set(range(x.shape[1]))
    while remaining and len(selected) < n_features:
        candidate = max(
            remaining,
            key=lambda col: (loo_accuracy(x, y, [*selected, col]), -col),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return selected


def _fit_l1_logistic(x: np.ndarray, y: np.ndarray, l1: float = 0.08,
                     iterations: int = 800) -> np.ndarray:
    """Small proximal-gradient L1 logistic regression, without a sklearn dependency."""
    z, _ = _standardize(x, x)
    coef = np.zeros(z.shape[1])
    intercept = 0.0
    # A conservative bound for logistic-loss curvature keeps iterations stable.
    spectral = float(np.linalg.norm(z, ord=2) ** 2 / max(1, len(z)))
    step = 1.0 / max(0.25 * spectral, 1e-3)
    for _ in range(iterations):
        logits = np.clip(z @ coef + intercept, -30, 30)
        prob = 1.0 / (1.0 + np.exp(-logits))
        residual = prob - y
        next_intercept = intercept - step * residual.mean()
        raw = coef - step * (z.T @ residual / len(z))
        next_coef = np.sign(raw) * np.maximum(np.abs(raw) - step * l1, 0.0)
        if np.max(np.abs(next_coef - coef)) < 1e-7 and abs(next_intercept - intercept) < 1e-7:
            coef, intercept = next_coef, next_intercept
            break
        coef, intercept = next_coef, next_intercept
    return coef


def embedded_select(x: np.ndarray, y: np.ndarray, n_features: int = 3) -> list[int]:
    """Select features with the largest absolute L1-logistic coefficients."""
    importance = np.abs(_fit_l1_logistic(x, y))
    if not np.any(importance):
        return filter_select(x, y, n_features)
    return list(np.argsort(importance)[::-1][:n_features])


def _nested_accuracy(x: np.ndarray, y: np.ndarray, selector, n_features: int) -> float:
    """Unbiased selection-method accuracy: select features inside each LOO fold."""
    hits = 0
    for held_out in range(len(x)):
        keep = np.arange(len(x)) != held_out
        train_x, train_y = x[keep], y[keep]
        cols = selector(train_x, train_y, n_features)
        pred = _centroid_predict(train_x[:, cols], train_y, x[held_out, cols])
        hits += pred == y[held_out]
    return hits / len(x)


def evaluate_features(rows: Iterable[FeatureRow], labels: dict[int, int], n_features: int = 3) -> dict:
    """Evaluate individual features and the three selection methods.

    Tracks without labels and tracks with missing feature measurements are
    skipped.  At least two examples of each class are needed for a meaningful
    leave-one-track-out evaluation.
    """
    matched = [row for row in rows if row.track_id in labels and np.isfinite(row.values).all()]
    if len(matched) < 4:
        raise ValueError("need at least four labeled tracks with complete features")
    x = np.asarray([row.values for row in matched], float)
    y = np.asarray([labels[row.track_id] for row in matched], int)
    if set(y) != {0, 1} or min((y == 0).sum(), (y == 1).sum()) < 2:
        raise ValueError("need at least two dancers and two non_dancers")
    n_features = max(1, min(int(n_features), x.shape[1]))
    selectors = {"Filter": filter_select, "Wrapper": wrapper_select, "Embedded": embedded_select}
    selections = {name: fn(x, y, n_features) for name, fn in selectors.items()}
    individual = {FEATURE_NAMES[i]: loo_accuracy(x, y, [i]) for i in range(x.shape[1])}
    methods = {name: _nested_accuracy(x, y, fn, n_features) for name, fn in selectors.items()}
    best_name, best_score = max({**individual, **methods}.items(), key=lambda item: item[1])
    return {
        "samples": len(matched),
        "dancers": int(y.sum()),
        "non_dancers": int((y == 0).sum()),
        "features_per_method": n_features,
        "individual_accuracy": individual,
        "method_accuracy": methods,
        "selected_features": {name: [FEATURE_NAMES[i] for i in cols] for name, cols in selections.items()},
        "best": {"name": best_name, "accuracy": best_score},
    }


def write_accuracy_graph(report: dict, path: str | Path) -> Path:
    """Write a dependency-free SVG bar chart of all feature/method accuracies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = [*report["individual_accuracy"].items(), *report["method_accuracy"].items()]
    width, height, left, bottom = 940, 490, 70, 92
    plot_w, plot_h = width - left - 35, height - bottom - 55
    bar_w = plot_w / max(len(items), 1) * 0.62
    bars = []
    labels = []
    grid = []
    for tick in np.arange(0, 1.01, 0.25):
        y = bottom + plot_h * (1 - tick)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - 35}" y2="{y:.1f}" class="grid"/>')
        grid.append(f'<text x="{left - 10}" y="{y + 4:.1f}" class="tick">{tick:.2f}</text>')
    for i, (name, value) in enumerate(items):
        value = 0.0 if not np.isfinite(value) else float(value)
        x = left + (i + 0.5) * plot_w / len(items)
        y = bottom + plot_h * (1 - value)
        color = "#3867d6" if i < len(report["individual_accuracy"]) else "#20a39e"
        bars.append(
            f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{bottom + plot_h - y:.1f}" rx="3" fill="{color}"/>'
            f'<text x="{x:.1f}" y="{y - 7:.1f}" class="value">{value:.1%}</text>'
        )
        labels.append(f'<text x="{x:.1f}" y="{bottom + plot_h + 23}" class="label">{escape(name)}</text>')
    title = "Dancer vs non-dancer accuracy (leave-one-track-out)"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Arial, sans-serif; fill: #1f2937; }}
  .title {{ font-size: 19px; font-weight: 700; }} .subtitle {{ font-size: 12px; fill: #64748b; }}
  .grid {{ stroke: #dbe3ee; stroke-width: 1; }} .tick {{ font-size: 11px; text-anchor: end; }}
  .label {{ font-size: 11px; text-anchor: middle; }} .value {{ font-size: 11px; text-anchor: middle; font-weight: 700; }}
</style>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="30" class="title">{title}</text>
<text x="{left}" y="50" class="subtitle">Blue: individual pose features · Teal: selection methods using {report['features_per_method']} features</text>
{''.join(grid)}
<line x1="{left}" y1="{bottom + plot_h}" x2="{width - 35}" y2="{bottom + plot_h}" stroke="#64748b"/>
{''.join(bars)}
{''.join(labels)}
<text x="{left}" y="{height - 18}" class="subtitle">{report['samples']} labeled tracks ({report['dancers']} dancers, {report['non_dancers']} non-dancers)</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")
    return path


def evaluate_run(tracking_jsonl: str | Path, labels_csv: str | Path, out_dir: str | Path,
                 n_features: int = 3) -> dict:
    """Collect features, evaluate labels, and write CSV, JSON, and SVG artifacts."""
    out_dir = Path(out_dir)
    rows = collect_track_features(tracking_jsonl)
    features_path = write_feature_csv(rows, out_dir / "track_features.csv")
    report = evaluate_features(rows, load_labels(labels_csv), n_features=n_features)
    graph_path = write_accuracy_graph(report, out_dir / "feature_selection_accuracy.svg")
    report.update({"features_csv": str(features_path), "graph": str(graph_path)})
    report_path = out_dir / "feature_selection_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_json"] = str(report_path)
    return report


def _demo_pose(dancer: bool, frame: int, cx: float, cy: float, scale: float) -> np.ndarray:
    """A deliberately obvious synthetic pose generator for CLI smoke testing."""
    p = np.zeros((17, 3), float)
    p[:, 2] = 0.95
    coords = {
        "nose": (0, -2.3), "left_eye": (-.1, -2.4), "right_eye": (.1, -2.4),
        "left_ear": (-.2, -2.3), "right_ear": (.2, -2.3),
        "left_shoulder": (-.5, -1), "right_shoulder": (.5, -1),
        "left_elbow": (-.65, -.1), "right_elbow": (.65, -.1),
        "left_wrist": (-.65, .75), "right_wrist": (.65, .75),
        "left_hip": (-.3, 0), "right_hip": (.3, 0),
        "left_knee": (-.3, 1.5), "right_knee": (.3, 1.5),
        "left_ankle": (-.3, 3), "right_ankle": (.3, 3),
    }
    if dancer:
        phase = frame * 0.42 + cx * 0.01
        coords["left_elbow"] = (-.8, -1.0 + .35 * math.sin(phase))
        coords["right_elbow"] = (.8, -1.0 + .35 * math.cos(phase))
        coords["left_wrist"] = (-1.05, -1.9 + .7 * math.sin(phase))
        coords["right_wrist"] = (1.05, -1.9 + .7 * math.cos(phase))
        coords["left_knee"] = (-.45, 1.45 + .2 * math.cos(phase))
        coords["right_knee"] = (.45, 1.45 + .2 * math.sin(phase))
    for name, xy in coords.items():
        p[J[name], :2] = np.asarray(xy) * scale + [cx, cy]
    return p


def write_synthetic_demo(path: str | Path, labels_path: str | Path, frames: int = 90) -> None:
    """Write synthetic tracking output with 4 dancers and 4 stationary bystanders."""
    from .skeleton import BODY

    path, labels_path = Path(path), Path(labels_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    people_specs = [
        (True, 120, 270), (True, 320, 270), (True, 520, 270), (True, 720, 270),
        (False, 120, 600), (False, 320, 600), (False, 520, 600), (False, 720, 600),
    ]
    with path.open("w") as f:
        for frame in range(frames):
            people = []
            for pid, (is_dancer, cx, cy) in enumerate(people_specs):
                pose = _demo_pose(is_dancer, frame, cx, cy, 48)
                xy = pose[:, :2]
                people.append({
                    "id": pid, "bbox": [float(xy[:, 0].min() - 8), float(xy[:, 1].min() - 8),
                                        float(xy[:, 0].max() + 8), float(xy[:, 1].max() + 8)],
                    "conf": 0.95,
                    "keypoints": {name: pose[i].round(4).tolist() for i, name in enumerate(BODY)},
                })
            f.write(json.dumps({"frame": frame, "t": frame / 30, "people": people}) + "\n")
    with labels_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "label"])
        writer.writerows((pid, "dancer" if is_dancer else "non_dancer")
                         for pid, (is_dancer, _, _) in enumerate(people_specs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate dancer/non-dancer pose features")
    ap.add_argument("tracking_jsonl", nargs="?", help="pose detector output for one run")
    ap.add_argument("labels_csv", nargs="?", help="CSV with track_id,label")
    ap.add_argument("--out", default="reports/dancer_features", help="output directory")
    ap.add_argument("--features", type=int, default=3, help="features selected by each method")
    ap.add_argument("--demo", action="store_true", help="run only the clearly synthetic verification dataset")
    args = ap.parse_args()
    out_dir = Path(args.out)
    if args.demo:
        tracking = out_dir / "synthetic_tracking.jsonl"
        labels = out_dir / "synthetic_labels.csv"
        write_synthetic_demo(tracking, labels)
    elif args.tracking_jsonl and args.labels_csv:
        tracking, labels = Path(args.tracking_jsonl), Path(args.labels_csv)
    else:
        ap.error("provide tracking_jsonl and labels_csv, or use --demo")
    report = evaluate_run(tracking, labels, out_dir, args.features)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
