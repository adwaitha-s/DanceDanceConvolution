"""Read human reviewed dancer and non-dancer track labels."""

from __future__ import annotations

import csv
from pathlib import Path


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
