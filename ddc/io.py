"""Load tracking.jsonl (RTMW / YOLO schema) into dense arrays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .skeleton import BODY


@dataclass
class Detections:
    t: np.ndarray            # (F,) seconds
    kps: list                # per frame: (P_f, 17, 3) float array [x, y, conf]
    ids: list                # per frame: (P_f,) raw detector ids
    raw: list                # per frame: the original people dicts (full keypoints)

    @property
    def n_frames(self) -> int:
        return len(self.t)


def load_jsonl(path: str | Path) -> Detections:
    t, kps, ids, raw = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            people = rec.get("people")
            if people is None:
                raise ValueError(
                    f"{path}: no 'people' key -- only the RTMW/YOLO JSONL schema is supported")
            arr = np.full((len(people), len(BODY), 3), np.nan)
            for pi, p in enumerate(people):
                for ji, name in enumerate(BODY):
                    v = p["keypoints"].get(name)
                    if v is not None:
                        arr[pi, ji] = v
            t.append(rec["t"])
            kps.append(arr)
            ids.append(np.array([p.get("id", i) for i, p in enumerate(people)]))
            raw.append(people)
    return Detections(np.asarray(t, float), kps, ids, raw)
