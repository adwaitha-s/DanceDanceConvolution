"""Persistent dancer-feature registry built from already detected pose tracks.

The registry is deliberately label-driven: predictions populate the UI checklist,
while the user's untouched or corrected checklist becomes the next labelled sample.
No function in this module invokes pose detection.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .dancer_features import load_labels
from .feature_inventory import (ELIGIBLE_FEATURE_IDS, FEATURE_SPECS, SPEC_BY_ID, _fit_l1,
                                collect_feature_inventory, normalize_merges,
                                write_inventory_artifacts)


def _sigmoid(value: float) -> float:
    return float(1 / (1 + math.exp(-max(-30, min(30, value)))))


def _run_directories(runs_dir: Path) -> list[Path]:
    return sorted((p for p in runs_dir.iterdir() if p.is_dir() and (p / "tracking.jsonl").exists()),
                  key=lambda p: p.name)


def load_merges(run_dir: Path) -> list[list[int]]:
    """Manual track merges saved with a run's labels (empty if none)."""
    try:
        return normalize_merges(json.loads((run_dir / "merges.json").read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return []


def _load_run_rows(run_dir: Path, merge_groups=None) -> list[dict]:
    """Feature rows for a run, numbered exactly as the Find people list is.

    Saved labels are keyed by track id, and ids depend on the run's manual merges, so rows
    are always built with the merges the labels were saved under (``merges.json``) unless
    the caller passes the live ``merge_groups`` of a run it is about to review.
    """
    merges = load_merges(run_dir) if merge_groups is None else normalize_merges(merge_groups)
    inventory = run_dir / "feature_inventory.json"
    if inventory.exists() and merge_groups is None:
        try:
            payload = json.loads(inventory.read_text(encoding="utf-8"))
            if normalize_merges(payload.get("merge_groups")) == merges:
                return payload["tracks"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return collect_feature_inventory(run_dir / "tracking.jsonl", merges)


def _training_records(runs_dir: Path, exclude_run: str | None = None) -> list[dict]:
    records = []
    for run_dir in _run_directories(runs_dir):
        if run_dir.name == exclude_run:
            continue
        label_path = run_dir / "labels.csv"
        if not label_path.exists():
            continue
        try:
            labels = load_labels(label_path)
        except ValueError:
            continue
        for row in _load_run_rows(run_dir):
            if row["track_id"] in labels:
                records.append({"run": run_dir.name, "track_id": int(row["track_id"]),
                                "label": int(labels[row["track_id"]]), "values": row["values"],
                                "frames_present": int(row["frames_present"])})
    return records


def _fit_registry(records: list[dict], n_features: int = 5) -> dict:
    """Fit a registry model constrained to its five strongest features.

    The first L1 fit ranks every eligible pose signal. A second fit then uses
    only that ranking's strongest features, so the five predictors displayed in
    ``dancer_feature_registry.xlsx`` are precisely the inputs used to score a
    new Compare-tab checklist.
    """
    if len(records) < 4:
        raise ValueError("need at least four validated tracks to train the registry")
    y = np.asarray([r["label"] for r in records], int)
    if set(y) != {0, 1} or min(int((y == 0).sum()), int((y == 1).sum())) < 2:
        raise ValueError("need at least two validated dancers and two validated non-dancers")
    candidate_ids, columns, medians = [], [], []
    for feature_id in ELIGIBLE_FEATURE_IDS:
        raw = np.asarray([r["values"].get(feature_id, np.nan) for r in records], float)
        finite = raw[np.isfinite(raw)]
        if finite.size < 2:
            continue
        median = float(np.median(finite))
        candidate_ids.append(feature_id)
        columns.append(np.where(np.isfinite(raw), raw, median))
        medians.append(median)
    if not columns:
        raise ValueError("validated tracks do not have usable predictor values")
    x = np.asarray(columns, float).T
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-9] = 1.0
    ranking_coef, _ = _fit_l1((x - mean) / std, y, l1=.025, iterations=2400)
    if not np.any(np.abs(ranking_coef)):
        ranking_coef = (x[y == 1].mean(0) - x[y == 0].mean(0)) / std
    order = np.argsort(np.abs(ranking_coef))[::-1]
    selected_idx = order[:min(n_features, len(order))]

    # Refit after selection rather than merely hiding the other coefficients.
    # This makes the workbook's top-five table an auditable description of the
    # classifier that drives the checklist.
    selected_x = x[:, selected_idx]
    selected_mean, selected_std = selected_x.mean(0), selected_x.std(0)
    selected_std[selected_std < 1e-9] = 1.0
    coef, intercept = _fit_l1((selected_x - selected_mean) / selected_std, y,
                              l1=.025, iterations=2400)
    if not np.any(np.abs(coef)):
        coef = ((selected_x[y == 1].mean(0) - selected_x[y == 0].mean(0)) /
                selected_std)
    # Keep the returned lists in descending influence order. The same ordered
    # five-element lists are used by _score_row and the workbook builder.
    final_order = np.argsort(np.abs(coef))[::-1]
    selected_idx = selected_idx[final_order]
    selected_mean = selected_mean[final_order]
    selected_std = selected_std[final_order]
    coef = coef[final_order]
    selected = []
    for rank, (i, weight) in enumerate(zip(selected_idx, coef), 1):
        selected.append({
            "rank": rank, "feature_id": candidate_ids[i],
            "feature_name": SPEC_BY_ID[candidate_ids[i]].name,
            "category": SPEC_BY_ID[candidate_ids[i]].category,
            "standardized_coefficient": float(weight),
            "absolute_weight": float(abs(weight)),
            "direction": "more dancer-like" if weight >= 0 else "more non-dancer-like",
        })
    return {
        "feature_ids": [candidate_ids[i] for i in selected_idx],
        "medians": [medians[i] for i in selected_idx],
        "mean": selected_mean.tolist(), "std": selected_std.tolist(),
        "coefficient": coef.tolist(), "intercept": float(intercept), "selected_features": selected,
        "labelled_tracks": len(records), "dancers": int(y.sum()), "non_dancers": int((y == 0).sum()),
        "runs": sorted({r["run"] for r in records}),
    }


def _score_row(row: dict, model: dict) -> dict:
    ids = model["feature_ids"]
    values = np.asarray([row["values"].get(fid, np.nan) for fid in ids], float)
    fallback = np.asarray(model["medians"], float)
    values = np.where(np.isfinite(values), values, fallback)
    z = (values - np.asarray(model["mean"])) / np.asarray(model["std"])
    coef = np.asarray(model["coefficient"])
    probability = _sigmoid(float(z @ coef + model["intercept"]))
    lookup = {fid: i for i, fid in enumerate(ids)}
    predictors = []
    for spec in model["selected_features"]:
        index = lookup[spec["feature_id"]]
        predictors.append({**spec, "raw_value": float(values[index]),
                           "standardized_value": float(z[index]),
                           "logit_contribution": float(z[index] * coef[index])})
    return {"track_id": int(row["track_id"]), "frames_present": int(row["frames_present"]),
            "dancer_probability": probability,
            "predicted_label": "dancer" if probability >= .5 else "non_dancer",
            "top_predictors": predictors}


def build_registry(runs_dir: str | Path, n_features: int = 5) -> dict:
    """Recompute the cross-run model and workbook payload from all validated runs."""
    runs_dir = Path(runs_dir)
    records = _training_records(runs_dir)
    model = _fit_registry(records, n_features=n_features)
    runs = []
    for run_dir in _run_directories(runs_dir):
        label_path = run_dir / "labels.csv"
        labels = load_labels(label_path) if label_path.exists() else {}
        validation_path = run_dir / "prediction_validation.json"
        validation = {}
        if validation_path.exists():
            try:
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
            except ValueError:
                pass
        tracks = []
        local_records = []
        for row in _load_run_rows(run_dir):
            scored = _score_row(row, model)
            # Preserve the prediction shown to the reviewer. Once a reviewed
            # run joins training data, a refit can change its in-sample score;
            # that must not rewrite the label that was originally validated.
            prediction_labels = validation.get("predictor_labels", {})
            prediction_probabilities = validation.get("predictor_probabilities", {})
            track_key = str(row["track_id"])
            if track_key in prediction_labels:
                scored["predicted_label"] = prediction_labels[track_key]
            if track_key in prediction_probabilities:
                scored["dancer_probability"] = prediction_probabilities[track_key]
            if row["track_id"] in labels:
                scored["reviewed_label"] = "dancer" if labels[row["track_id"]] else "non_dancer"
                local_records.append({"run": run_dir.name, "track_id": int(row["track_id"]),
                                      "label": int(labels[row["track_id"]]), "values": row["values"],
                                      "frames_present": int(row["frames_present"])})
            else:
                scored["reviewed_label"] = ""
            tracks.append(scored)
        try:
            local_model = _fit_registry(local_records, n_features=n_features)
            local_top5 = local_model["selected_features"]
        except ValueError:
            local_top5 = []
        runs.append({"run": run_dir.name, "validation": validation, "tracks": tracks,
                     "run_top5": local_top5, "scoring_top5": model["selected_features"],
                     "labelled_track_count": len(local_records)})
    model["generated_at"] = datetime.now(timezone.utc).isoformat()
    model["runs_detail"] = runs
    model["feature_catalog"] = [spec.__dict__ for spec in FEATURE_SPECS]
    # A run's five strongest local weights are kept alongside every Labels
    # sheet. The global model is refit on all labelled tracks, rather than
    # averaging coefficients from clips with unequal numbers of labels.
    model["run_top5_method"] = (
        "Each local top-five list is retained for inspection. The overall model is refit "
        "from every validated track, so a run contributes its labels and feature values "
        "without giving a small run the same vote as a larger one."
    )
    (runs_dir / "feature_model_registry.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    (runs_dir / "feature_registry_workbook_payload.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    return model


def predict_run(runs_dir: str | Path, run_name: str,
                merge_groups=None) -> tuple[dict, list[dict]]:
    """Score all saved tracks in a run from *other* runs' validated statistics.

    The run being scored is held out of training.  Otherwise a run that was
    already reviewed would be scored by a model that has memorised its own
    labels, and "validated correct" would be meaningless.  This is a read-only
    fit; ``build_registry`` (which writes the workbook payload) is not called.
    """
    runs_dir = Path(runs_dir)
    model = _fit_registry(_training_records(runs_dir, exclude_run=run_name))
    model["held_out_run"] = run_name
    run_dir = runs_dir / run_name
    return model, [_score_row(row, model) for row in _load_run_rows(run_dir, merge_groups)]


def validate_checklist(runs_dir: str | Path, run_name: str, selected_track_ids: list[int],
                       predicted: list[dict], merge_groups=None) -> dict:
    """Persist a completed checklist as training labels and rebuild the registry payload.

    ``merge_groups`` are the manual merges the checklist's track ids were numbered under.
    They are saved beside ``labels.csv`` so the labels keep pointing at the same people.
    """
    runs_dir, run_dir = Path(runs_dir), Path(runs_dir) / run_name
    merges = normalize_merges(merge_groups)
    (run_dir / "merges.json").write_text(json.dumps(merges), encoding="utf-8")
    all_ids = {int(row["track_id"]) for row in _load_run_rows(run_dir)}
    selected = {int(item) for item in selected_track_ids}
    if not selected.issubset(all_ids):
        raise ValueError("checklist contains an unknown track")
    prediction_map = {int(item["track_id"]): item["predicted_label"] for item in predicted}
    corrections = (sorted(tid for tid in all_ids
                          if ("dancer" if tid in selected else "non_dancer") != prediction_map.get(tid))
                   if prediction_map else [])
    with (run_dir / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "label"])
        writer.writeheader()
        writer.writerows({"track_id": tid, "label": "dancer" if tid in selected else "non_dancer"}
                         for tid in sorted(all_ids))
    validation = {"run": run_name, "validated_at": datetime.now(timezone.utc).isoformat(),
                  "status": ("validated_correct" if not corrections else "corrected")
                  if prediction_map else "reviewed_without_prediction",
                  "checked_dancer_tracks": sorted(selected), "corrected_track_ids": corrections,
                  "predictor_label_count": len(prediction_map),
                  "predictor_labels": {str(item["track_id"]): item["predicted_label"]
                                       for item in predicted},
                  "predictor_probabilities": {str(item["track_id"]): item["dancer_probability"]
                                              for item in predicted}}
    (run_dir / "prediction_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    # Keep the run-local inventory compatible with the original embedded report.
    write_inventory_artifacts(run_dir / "tracking.jsonl", run_dir / "labels.csv", run_dir, merges)
    model = build_registry(runs_dir)
    return {"validation": validation, "model": model}


def rebuild_registry_workbook(runs_dir: str | Path) -> Path:
    """Rebuild ``dancer_feature_registry.xlsx`` from every validated run; never reruns detection."""
    runs_dir = Path(runs_dir)
    try:
        from .registry_workbook import write_registry_workbook
        model = build_registry(runs_dir)
        return write_registry_workbook(model, runs_dir / "dancer_feature_registry.xlsx")
    except ImportError as e:
        raise RuntimeError(f"{e}; run `pip install -r requirements.txt`") from e
    except (ValueError, OSError) as e:
        raise RuntimeError(f"workbook builder failed: {e}") from e
