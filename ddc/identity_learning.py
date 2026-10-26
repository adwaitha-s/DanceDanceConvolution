"""Learn fragment identity from explicit merge reviews, independently of dancer labels.

The logistic pair classifier stores plain JSON weights, never pickled code. New
analyses freeze its decisions in their run directory so retraining cannot renumber
people in previously reviewed runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
from filelock import FileLock

from .appearance import appearance_distance, load_appearances

VERSION = 1
REVIEW_FILE = "merge_review.json"
PREDICTION_FILE = "identity_predictions.json"
MODEL_FILE = "identity_model.json"
FEATURES = ["clothing_distance", "clothing_available", "torso_size_change",
            "body_extent_change", "extent_available", "motion_excess"]
THRESHOLD = .85
MAX_GAP_S = 15.


def _write(path, data):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prediction_stamp(path):
    snapshot = Path(path).parent / PREDICTION_FILE
    return hashlib.sha256(snapshot.read_bytes()).hexdigest() if snapshot.exists() else None


def prediction_groups(path):
    from .tracking import baseline_stamp
    snapshot = Path(path).parent / PREDICTION_FILE
    if not snapshot.exists():
        return []
    saved = json.loads(snapshot.read_text(encoding="utf-8"))
    if saved.get("version") != VERSION or saved.get("baseline_stamp") != baseline_stamp(path):
        return []
    return saved["groups"]


def _track_groups(det, ids, count):
    from .tracking import _make_group
    appearances = load_appearances(det)
    boxes = det.bbox
    occurrences = [[] for _ in range(count)]
    for f, frame in enumerate(ids):
        for i, tid in enumerate(frame):
            if tid >= 0:
                occurrences[tid].append((f, i))
    return [_make_group(occ, det.kps, [tid], appearances, boxes)
            for tid, occ in enumerate(occurrences)]


def _extent(group):
    values = group["extents"]
    valid = values[np.isfinite(values) & (values > 0)] if values is not None else []
    return float(np.median(valid)) if len(valid) else None


def pair_features(a, b, fps):
    """Identity evidence shared by positive and negative pairs; no track numbers/labels.

    Use near-in-time observations for motion and whole-track clothing/size summaries.
    Overlap is a hard inference constraint, not an input the classifier can memorize.
    """
    fa, fb = a["frames"], b["frames"]
    nearest = np.searchsorted(fa, fb)
    left, right = np.maximum(nearest - 1, 0), np.minimum(nearest, len(fa) - 1)
    pick = np.where(np.abs(fb - fa[left]) <= np.abs(fa[right] - fb), left, right)
    gaps = np.abs(fb - fa[pick])
    sel = np.argsort(gaps, kind="stable")[:8]
    sizes = np.maximum(1., np.minimum(a["sizes"][pick[sel]], b["sizes"][sel]))
    distances = np.linalg.norm(a["centers"][pick[sel]] - b["centers"][sel], axis=1) / sizes
    excess = float(np.median(np.maximum(0., distances - gaps[sel] / fps)))
    color = appearance_distance(a["appearance"], b["appearance"])
    ea, eb = _extent(a), _extent(b)
    size_change = abs(float(np.log(np.median(a["sizes"]) / np.median(b["sizes"]))))
    features = [color if color is not None else .5, float(color is not None),
                size_change, abs(float(np.log(ea / eb))) if ea and eb else 0.,
                float(ea is not None and eb is not None), excess]
    return features if np.isfinite(features).all() else None


def _overlap(a, b):
    return a["frames"][-1] >= b["frames"][0] and b["frames"][-1] >= a["frames"][0]


def _simultaneous(a, b):
    return bool(np.intersect1d(a["frames"], b["frames"]).size)


def _fps(det):
    return float((len(det.t) - 1) / (det.t[-1] - det.t[0])) if len(det.t) > 1 and det.t[-1] > det.t[0] else 30.


def _examples(groups, merges, fps, identity=None):
    """`identity` (track -> person, None for a track the user's edits split across people)
    overrides `merges`; tracks without one identity are left out of every pair."""
    if identity is None:
        identity = {i: i for i in range(len(groups))}
        for merged in merges:
            for tid in merged:
                identity[tid] = min(merged)
    samples = []
    for i, j in combinations(range(len(groups)), 2):
        if identity.get(i) is None or identity.get(j) is None:
            continue
        same = identity[i] == identity[j]
        # Untouched, disjoint fragments are unknown, never invented negatives.
        if not same and not _simultaneous(groups[i], groups[j]):
            continue
        features = pair_features(groups[i], groups[j], fps)
        if features is not None:
            samples.append({"pair": [i, j], "features": features, "label": int(same)})
    return samples


def train_model(runs_dir):
    """Replace this run's examples on every save; use only current explicit reviews."""
    from .tracking import tracking_stamp
    samples, reviews = [], []
    for path in sorted(Path(runs_dir).glob(f"*/{REVIEW_FILE}")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        tracking = path.parent / "tracking.jsonl"
        if not tracking.exists() or saved.get("version") != VERSION:
            continue
        if saved.get("tracking_stamp") != tracking_stamp(tracking):
            continue
        samples.extend(saved["examples"])
        reviews.append(path.parent.name)
    positive = sum(s["label"] == 1 for s in samples)
    negative = len(samples) - positive
    report = {"version": VERSION, "features": FEATURES, "positive_pairs": positive,
              "negative_pairs": negative, "reviewed_runs": reviews,
              "threshold": THRESHOLD, "status": "collecting"}
    if positive and negative:
        from scipy.optimize import minimize
        from scipy.special import expit
        x = np.asarray([s["features"] for s in samples], float)
        y = np.asarray([s["label"] for s in samples])
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.)
        standardized = (x - mean) / scale
        weights = np.where(y == 1, len(y) / (2 * positive), len(y) / (2 * negative))

        def objective(parameters):
            # Balanced binary cross entropy with L2 regularization (C=1).
            # The intercept is unpenalized; logaddexp avoids overflow.
            coefficients, intercept = parameters[:-1], parameters[-1]
            logits = standardized @ coefficients + intercept
            residual = weights * (expit(logits) - y)
            loss = np.sum(weights * (np.logaddexp(0., logits) - y * logits))
            loss += .5 * (coefficients @ coefficients)
            gradient = np.r_[standardized.T @ residual + coefficients, residual.sum()]
            return float(loss), gradient

        fitted = minimize(objective, np.zeros(x.shape[1] + 1), jac=True,
                          method="L-BFGS-B", options={"maxiter": 1000, "gtol": 1e-8})
        if not fitted.success or not np.isfinite(fitted.x).all():
            raise RuntimeError(f"Identity model optimization failed: {fitted.message}")
        report.update(status="ready", mean=mean.tolist(), scale=scale.tolist(),
                      coefficients=fitted.x[:-1].tolist(), intercept=float(fitted.x[-1]))
    report["fingerprint"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    _write(Path(runs_dir) / MODEL_FILE, report)
    return report


def save_merge_review(run_dir, merges):
    """Save the identities shown in Dancers and refit without changing that run's ids."""
    from .io import load_jsonl
    from .feature_inventory import normalize_merges
    from .identity import apply_identity, track_identity
    from .tracking import stable_track_ids, tracking_stamp
    run_dir = Path(run_dir)
    det = load_jsonl(run_dir / "tracking.jsonl")
    ids, count = stable_track_ids(det)
    # Validate before writing anything, including repeated membership and unknown ids.
    final, _ = apply_identity(det.kps, ids, merges)
    merges = normalize_merges(merges)
    examples = _examples(_track_groups(det, ids, count), [], _fps(det), track_identity(ids, final))
    saved = {"version": VERSION, "tracking_stamp": tracking_stamp(det.path),
             "groups": merges, "examples": examples}
    with FileLock(str(run_dir.parent / ".identity-learning.lock"), timeout=30):
        _write(run_dir / REVIEW_FILE, saved)
        _write(run_dir / "merges.json", merges)
        try:
            report = train_model(run_dir.parent)
        except Exception as exc:
            # The human correction is already durable even if fitting fails.
            return {"status": "training_failed", "error": str(exc)}
    return report


def same_person_probability(features, model):
    x = (np.asarray(features) - model["mean"]) / model["scale"]
    value = float(x @ model["coefficients"] + model["intercept"])
    return float(1. / (1. + np.exp(-np.clip(value, -30., 30.))))


def predict_groups(groups, model, fps):
    """Only add high-confidence, physically plausible, non-overlapping links.

    Require every pair across two proposed groups to agree. This prevents a
    chain of individually plausible links from absorbing a different person.
    """
    candidates = {}
    for i, j in combinations(range(len(groups)), 2):
        a, b = groups[i], groups[j]
        if _overlap(a, b):
            continue
        gap = max(a["frames"][0] - b["frames"][-1], b["frames"][0] - a["frames"][-1]) / fps
        if gap > MAX_GAP_S:
            continue
        features = pair_features(a, b, fps)
        if features is None or features[0] > .6 or features[2] > np.log(2.5) or features[-1] > 2.:
            continue
        probability = same_person_probability(features, model)
        if probability >= model["threshold"]:
            candidates[i, j] = probability
    members = {i: [i] for i in range(len(groups))}
    owner = {i: i for i in members}
    for (i, j), _ in sorted(candidates.items(), key=lambda item: (-item[1], item[0])):
        x, y = owner[i], owner[j]
        if x == y:
            continue
        if not all(tuple(sorted((a, b))) in candidates for a in members[x] for b in members[y]):
            continue
        joined = members.pop(y)
        members[x].extend(joined)
        for tid in joined:
            owner[tid] = x
    return sorted(sorted(group) for group in members.values() if len(group) > 1)


def initialize_predictions(run_dir):
    """Freeze learned identities for a newly analyzed run, before any labels exist."""
    from .io import load_jsonl
    from .tracking import baseline_stamp, stable_track_ids
    run_dir = Path(run_dir)
    snapshot = run_dir / PREDICTION_FILE
    if snapshot.exists() or any((run_dir / name).exists() for name in
                                (REVIEW_FILE, "merges.json", "labels.csv")):
        return []  # Existing human track numbers take precedence.
    model_path = run_dir.parent / MODEL_FILE
    if not model_path.exists():
        return []
    model = json.loads(model_path.read_text(encoding="utf-8"))
    if model.get("version") != VERSION or model.get("status") != "ready" or model.get("features") != FEATURES:
        return []
    det = load_jsonl(run_dir / "tracking.jsonl")
    ids, count = stable_track_ids(det)
    groups = predict_groups(_track_groups(det, ids, count), model, _fps(det))
    _write(snapshot, {"version": VERSION, "baseline_stamp": baseline_stamp(det.path),
                      "model_fingerprint": model["fingerprint"], "groups": groups})
    return groups
