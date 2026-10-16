"""Clothing descriptors for identity association; never classifier predictors."""

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .skeleton import CONF_THR, J


_CACHE = OrderedDict()


def source_video(tracking_path):
    sources = sorted(p for p in Path(tracking_path).parent.glob("source.*")
                     if p.is_file() and p.suffix.lower() != ".stamp")
    return sources[0] if sources else None


def source_stamp(tracking_path):
    source = source_video(tracking_path)
    if source is None:
        return None
    stat = source.stat()
    return [source.name, stat.st_size, stat.st_mtime_ns]


def clothing_descriptor(image, pose):
    """Two HSV histograms sampled inside the torso and thighs, excluding background."""
    result = np.full((2, 192), np.nan, np.float32)
    ok = (pose[:, 2] >= CONF_THR) & np.isfinite(pose[:, :2]).all(axis=1)
    torso = [J[n] for n in ("left_shoulder", "right_shoulder", "right_hip", "left_hip")]
    legs = [(J[f"{side}_hip"], J[f"{side}_knee"]) for side in ("left", "right")]
    selected = [i for i in torso + [j for leg in legs for j in leg] if ok[i]]
    if len(selected) < 3:
        return result
    points = pose[selected, :2]
    h, w = image.shape[:2]
    lo = np.maximum(np.floor(points.min(0) - 5).astype(int), 0)
    hi = np.minimum(np.ceil(points.max(0) + 6).astype(int), [w, h])
    if np.any(hi <= lo):
        return result
    hsv = cv2.cvtColor(image[lo[1]:hi[1], lo[0]:hi[0]], cv2.COLOR_BGR2HSV)
    masks = np.zeros((2, *hsv.shape[:2]), np.uint8)
    if ok[torso].all():
        pts = pose[torso, :2]
        pts = (pts - pts.mean(0)) * .7 + pts.mean(0) - lo
        cv2.fillConvexPoly(masks[0], cv2.convexHull(pts.astype(np.int32)), 255)
        width = max(2, int(np.linalg.norm(pose[torso[0], :2] - pose[torso[1], :2]) * .25))
    else:
        width = max(2, int(np.linalg.norm(points.max(0) - points.min(0)) * .06))
    for hip, knee in legs:
        if ok[hip] and ok[knee]:
            a, b = pose[[hip, knee], :2] - lo
            cv2.line(masks[1], tuple((.8*a + .2*b).astype(int)),
                     tuple((.2*a + .8*b).astype(int)), 255, width)
    for i, mask in enumerate(masks):
        if np.count_nonzero(mask) < 10:
            continue
        hist = cv2.calcHist([hsv], [0, 1, 2], mask, [12, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
        result[i] = hist / hist.sum()
    return result


def appearance_distance(a, b):
    if a is None or b is None:
        return None
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if not valid.any():
        return None
    overlap = np.sqrt(a[valid] * b[valid]).sum(axis=1)
    return float(np.sqrt(np.maximum(0, 1 - np.minimum(1, overlap))).mean())


def combine_appearances(samples):
    valid = [a for a in samples if a is not None]
    if not valid:
        return None
    values = np.stack(valid)
    counts = np.isfinite(values).sum(axis=0)
    return np.divide(np.nansum(values, axis=0), counts,
                     out=np.full(values.shape[1:], np.nan), where=counts > 0)


def load_appearances(det):
    """Decode a saved source once, using the pipeline's frame alignment and a small cache."""
    path = getattr(det, "path", None)
    source = source_video(path) if path is not None else None
    if source is None:
        return None
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns, tuple(source_stamp(path)))
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    from video_pipeline import _cfr_source, open_video
    frames = []
    with _cfr_source(str(source)) as video:
        cap = open_video(video)
        try:
            for poses in det.kps:
                ok, image = cap.read()
                if not ok:
                    return None  # Partial/missing video: use the geometric fallback consistently.
                frames.append([clothing_descriptor(image, p) for p in poses])
        finally:
            cap.release()
    _CACHE[key] = frames
    while len(_CACHE) > 2:
        _CACHE.popitem(last=False)
    return frames
