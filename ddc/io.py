"""Load tracking.jsonl (RTMW / YOLO schema) into dense arrays.

Parsing a long run's JSONL is slow (a 6-minute, 30-dancer run is ~1 GB), so the
first load writes a columnar cache next to it (`<stem>.poses.npz`) holding only
what the analysis needs: timestamps, body keypoints, ids, bboxes and detection
confidences. Later loads read the cache. The full per-person dicts (hands, feet)
are only parsed if something asks for `Detections.raw`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .skeleton import BODY

CACHE_VERSION = 1


@dataclass
class Detections:
    t: np.ndarray            # (F,) seconds
    kps: list                # per frame: (P_f, 17, 3) float array [x, y, conf]
    ids: list                # per frame: (P_f,) raw detector ids
    bbox: list               # per frame: (P_f, 4) [x0, y0, x1, y1], NaN if missing
    conf: list               # per frame: (P_f,) detection confidence (0 if missing)
    source: Path | None = None   # the JSONL this came from
    key: str = ""            # identifies the JSONL contents; used to key derived caches
    _raw: list | None = field(default=None, repr=False)

    @property
    def n_frames(self) -> int:
        return len(self.t)

    @property
    def raw(self) -> list:
        """Per frame: the original people dicts (full keypoints). Parsed lazily."""
        if self._raw is None:
            if self.source is None:
                raise ValueError("raw detections unavailable: no source file")
            self._raw = [rec["people"] for rec in _records(self.source)]
        return self._raw


def _records(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _source_key(path: Path) -> str:
    st = path.stat()
    return f"v{CACHE_VERSION}|{st.st_size}|{st.st_mtime_ns}"


def cache_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".poses.npz")


def _parse(path: Path):
    t, kps, ids, bbox, conf, raw = [], [], [], [], [], []
    for rec in _records(path):
        people = rec.get("people")
        if people is None:
            raise ValueError(
                f"{path}: no 'people' key -- only the RTMW/YOLO JSONL schema is supported")
        arr = np.full((len(people), len(BODY), 3), np.nan)
        box = np.full((len(people), 4), np.nan)
        for pi, p in enumerate(people):
            for ji, name in enumerate(BODY):
                v = p["keypoints"].get(name)
                if v is not None:
                    arr[pi, ji] = v
            if p.get("bbox"):
                box[pi] = p["bbox"][:4]
        t.append(rec["t"])
        kps.append(arr)
        ids.append(np.array([p.get("id", i) for i, p in enumerate(people)], int))
        bbox.append(box)
        conf.append(np.array([p.get("conf", 0.0) for p in people], float))
        raw.append(people)
    return np.asarray(t, float), kps, ids, bbox, conf, raw


def _split(flat: np.ndarray, counts: np.ndarray) -> list:
    return np.split(flat, np.cumsum(counts)[:-1]) if len(counts) else []


def _write_cache(cp: Path, key: str, t, kps, ids, bbox, conf) -> None:
    counts = np.array([len(a) for a in kps], int)
    tmp = cp.with_name(cp.name + ".tmp.npz")
    try:
        np.savez(tmp, key=np.array(key), t=t, counts=counts,
                 kps=np.concatenate(kps) if kps else np.zeros((0, len(BODY), 3)),
                 ids=np.concatenate(ids) if ids else np.zeros(0, int),
                 bbox=np.concatenate(bbox) if bbox else np.zeros((0, 4)),
                 conf=np.concatenate(conf) if conf else np.zeros(0))
        os.replace(tmp, cp)
    except OSError:          # read-only run dir: just don't cache
        tmp.unlink(missing_ok=True)


def _read_cache(cp: Path, key: str):
    try:
        with np.load(cp) as z:
            if str(z["key"]) != key:
                return None
            counts = z["counts"]
            return (z["t"], _split(z["kps"], counts), _split(z["ids"], counts),
                    _split(z["bbox"], counts), _split(z["conf"], counts))
    except (OSError, KeyError, ValueError):
        return None


def load_jsonl(path: str | Path, use_cache: bool = True) -> Detections:
    path = Path(path)
    key = _source_key(path)
    cp = cache_path(path)
    if use_cache and cp.exists():
        hit = _read_cache(cp, key)
        if hit is not None:
            return Detections(*hit, source=path, key=key)
    t, kps, ids, bbox, conf, raw = _parse(path)
    if use_cache:
        _write_cache(cp, key, t, kps, ids, bbox, conf)
    return Detections(t, kps, ids, bbox, conf, source=path, key=key, _raw=raw)
