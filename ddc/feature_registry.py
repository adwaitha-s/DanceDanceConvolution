"""Merge and human-review records for saved pose tracks.

LASSO feature selection and LSTM prediction live in ``ddc.training``. This
module keeps track identity checks and reviewed labels shared by the GUI and
training pipeline.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .feature_inventory import normalize_merges
from .io import load_jsonl
from .tracking import TRACKING_VERSION, stable_track_ids, tracking_stamp


def load_merges(run_dir: Path) -> list[list[int]]:
    """Return saved merges only while they refer to the current track identities."""
    run_dir = Path(run_dir)
    try:
        merges = normalize_merges(json.loads((run_dir / "merges.json").read_text(encoding="utf-8")))
        review = run_dir / "merge_review.json"
        stale_merge_review = False
        if review.exists():
            saved = json.loads(review.read_text(encoding="utf-8"))
            if (saved.get("tracking_stamp") == tracking_stamp(run_dir / "tracking.jsonl")
                    and saved.get("groups") == merges):
                return merges
            stale_merge_review = saved.get("tracking_stamp") != tracking_stamp(run_dir / "tracking.jsonl")
        validation = run_dir / "prediction_validation.json"
        if validation.exists():
            saved = json.loads(validation.read_text(encoding="utf-8"))
            if (saved.get("tracking_version") != TRACKING_VERSION
                    or saved.get("tracking_stamp") != tracking_stamp(run_dir / "tracking.jsonl")):
                return []
        elif stale_merge_review:
            return []
        return merges
    except (OSError, ValueError, TypeError):
        return []


def review_is_current(run_dir: Path) -> bool:
    """Saved checklist IDs are meaningful only under the tracker that created them."""
    run_dir = Path(run_dir)
    review = run_dir / "training_review.json"
    validation = run_dir / "prediction_validation.json"
    try:
        if review.exists():
            from .training import signature
            return json.loads(review.read_text(encoding="utf-8")).get("signature") == signature(run_dir)
        if validation.exists():
            saved = json.loads(validation.read_text(encoding="utf-8"))
            return (saved.get("tracking_version") == TRACKING_VERSION
                    and saved.get("tracking_stamp") == tracking_stamp(run_dir / "tracking.jsonl")
                    and (saved.get("merge_groups") == load_merges(run_dir)
                         if "merge_groups" in saved else not (run_dir / "merge_review.json").exists()))
    except (OSError, ValueError, TypeError):
        return False
    # Hand-authored label CSVs have no saved UI identity state.
    return True


def validate_checklist(runs_dir: str | Path, run_name: str, selected_track_ids: list[int],
                       predicted: list[dict], merge_groups=None, review_action="saved_review") -> dict:
    """Save reviewed dancer labels against the displayed, merged track IDs."""
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_name
    merges = normalize_merges(merge_groups)
    det = load_jsonl(run_dir / "tracking.jsonl")
    _, n_tracks = stable_track_ids(det, merges)
    all_ids = set(range(n_tracks))
    selected = {int(item) for item in selected_track_ids}
    if not selected.issubset(all_ids):
        raise ValueError("checklist contains an unknown track")

    prediction_map = {int(item["track_id"]): item["predicted_label"] for item in predicted}
    corrections = (sorted(tid for tid in all_ids
                          if ("dancer" if tid in selected else "non_dancer") != prediction_map.get(tid))
                   if prediction_map else [])
    (run_dir / "merges.json").write_text(json.dumps(merges), encoding="utf-8")
    with (run_dir / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "label"])
        writer.writeheader()
        writer.writerows({"track_id": tid, "label": "dancer" if tid in selected else "non_dancer"}
                         for tid in sorted(all_ids))
    validation = {"run": run_name, "validated_at": datetime.now(timezone.utc).isoformat(),
                  "tracking_version": TRACKING_VERSION,
                  "tracking_stamp": tracking_stamp(run_dir / "tracking.jsonl"),
                  "merge_groups": merges,
                  "review_action": review_action,
                  "status": ("validated_correct" if not corrections else "corrected")
                  if prediction_map else "reviewed_without_prediction",
                  "checked_dancer_tracks": sorted(selected), "corrected_track_ids": corrections,
                  "predictor_label_count": len(prediction_map),
                  "predictor_labels": {str(item["track_id"]): item["predicted_label"]
                                       for item in predicted},
                  "predictor_probabilities": {str(item["track_id"]): item.get("dancer_probability")
                                              for item in predicted}}
    (run_dir / "prediction_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    from .training import refresh_after_review, signature, _write
    _write(run_dir / "training_review.json", {"signature": signature(run_dir),
           "reviewed_at": datetime.now(timezone.utc).isoformat(), "source": "UI"})
    training_note = refresh_after_review(runs_dir)
    return {"validation": validation, "training_note": training_note}
