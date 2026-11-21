"""Synthetic pose tracks shared by the model and GUI tests."""

import csv
import json
import math
from pathlib import Path

import numpy as np

from ddc.skeleton import BODY, J


def _demo_pose(dancer: bool, frame: int, cx: float, cy: float, scale: float) -> np.ndarray:
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
    """Write a deterministic run with four dancers and four stationary people."""
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
