"""End-to-end: tracking.jsonl -> composite, deviation, overlay video, timeline.

CLI:  python -m ddc.analyze runs/<ts> [--video path] [--leave-one-out/--no-leave-one-out]
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import io as ddc_io
from .deviation import Deviation, compute_deviation, worst_moments
from .normalize import normalize
from .reference import reference_from_solo, reference_from_track
from .skeleton import BODY
from .tracking import stabilize, to_dense


@dataclass
class TrackInfo:
    """One stabilized track found in the clip, for the user to vet before scoring.

    Nothing in the detector or tracker knows "dancer" from "bystander" -- anyone
    tracked for more than a few frames is included by default. `detect_tracks`
    surfaces every track with a thumbnail so a person can rule out audience
    members, people walking through the background, etc. before `analyze_run`
    builds the composite/deviation from only the tracks they keep.
    """
    track_id: int
    label: str
    frames_present: int
    t_start: float
    t_end: float
    thumbnail: np.ndarray   # RGB uint8 crop


def _find_saved_source(run_dir: Path) -> Path | None:
    """A previously `_save_source_video`-d upload for this run, if any."""
    matches = sorted(run_dir.glob("source.*"))
    return matches[0] if matches else None


def _save_source_video(run_dir: Path, video_path) -> Path:
    """Persist an uploaded source video into the run dir (as `source.<ext>`) so
    later Compare runs on this run reuse it automatically -- an undimmed
    background -- without asking the user to re-upload it every time.
    """
    video_path = Path(video_path)
    dest = run_dir / f"source{video_path.suffix or '.mp4'}"
    for old in run_dir.glob("source.*"):
        if old != dest:
            old.unlink()
    if dest.resolve() != video_path.resolve():
        shutil.copy2(video_path, dest)
    return dest


def _pick_background(run_dir: Path, video_path) -> tuple[Path | None, float]:
    """(background_path, dim). A given/saved source video is used undimmed;
    the run's own overlay.mp4 (which already has a skeleton drawn on it) is
    used as a last-resort background, dimmed so the new deviation skeleton
    drawn on top of it stays legible."""
    if video_path:
        return _save_source_video(run_dir, video_path), 1.0
    saved = _find_saved_source(run_dir)
    if saved is not None:
        return saved, 1.0
    overlay = run_dir / "overlay.mp4"
    return (overlay, 0.3) if overlay.exists() else (None, 1.0)


def _crop_thumbnail(cap: cv2.VideoCapture | None, frame_idx: int, bbox, pad: float = 0.3,
                    size: int = 200) -> np.ndarray:
    blank = np.full((size, size, 3), 40, np.uint8)
    if cap is None or not cap.isOpened():
        return blank
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return blank
    h, w = frame.shape[:2]
    crop = frame
    if bbox:
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0, y1 - y0
        x0, y0 = max(0, int(x0 - bw * pad)), max(0, int(y0 - bh * pad))
        x1, y1 = min(w, int(x1 + bw * pad)), min(h, int(y1 + bh * pad))
        if x1 > x0 and y1 > y0:
            crop = frame[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = size / max(ch, cw, 1)
    crop = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


def detect_tracks(run_dir, video_path=None) -> list[TrackInfo]:
    """Stabilize tracks and return one thumbnail + summary per track.

    Meant to run before `analyze_run`: show the result to the user so they can
    exclude anyone who isn't actually a dancer (`analyze_run`'s `selected_tracks`).
    Track ids are deterministic (same jsonl -> same `stabilize()` output), so ids
    collected here line up with the ones `analyze_run` will produce.
    """
    run_dir = Path(run_dir)
    det = ddc_io.load_jsonl(run_dir / "tracking.jsonl")
    if det.n_frames == 0:
        raise ValueError("tracking.jsonl is empty")
    ids, n_tracks = stabilize(det.kps)
    if n_tracks == 0:
        raise ValueError("no persistent people found (all detections were short-lived)")

    occurrences: list[list[tuple[int, int]]] = [[] for _ in range(n_tracks)]
    for f, a in enumerate(ids):
        for i, tid in enumerate(a):
            if tid >= 0:
                occurrences[tid].append((f, i))

    bg_path, _ = _pick_background(run_dir, video_path)
    cap = cv2.VideoCapture(str(bg_path)) if bg_path else None
    tracks = []
    for tid, occ in enumerate(occurrences):
        fs = [f for f, _ in occ]
        t_start, t_end = float(det.t[fs[0]]), float(det.t[fs[-1]])
        best_f, best_i = max(occ, key=lambda fi: det.raw[fi[0]][fi[1]].get("conf", 0.0))
        thumb = _crop_thumbnail(cap, best_f, det.raw[best_f][best_i].get("bbox"))
        tracks.append(TrackInfo(
            track_id=tid, frames_present=len(occ), t_start=t_start, t_end=t_end, thumbnail=thumb,
            label=f"Person {tid + 1}  ({len(occ)} frames, {t_start:.1f}s–{t_end:.1f}s)",
        ))
    if cap is not None:
        cap.release()
    return tracks


@dataclass
class Result:
    run_dir: Path
    t: np.ndarray
    dense: np.ndarray
    norm: np.ndarray
    origin: np.ndarray
    scale: np.ndarray
    rot: np.ndarray
    dev: Deviation
    n_tracks: int
    summary: list          # per-dancer dicts
    overlay_path: str | None = None
    csv_path: str | None = None


REFERENCE_MODES = ("composite", "dancer", "solo")


def _ensure_solo_run(run_dir: Path, solo_video) -> Path:
    """Pose-track the solo reference video into `<run_dir>/reference_solo/`, reusing
    a previous result if the same file (name, size, mtime) was already analyzed."""
    from video_pipeline import analyze_video_rtmw

    solo_video = Path(solo_video)
    if not solo_video.exists():
        raise ValueError(f"solo reference video not found: {solo_video}")
    out = run_dir / "reference_solo"
    jsonl, stamp = out / "tracking.jsonl", out / "source.stamp"
    st = solo_video.stat()
    key = f"{solo_video.name}|{st.st_size}|{int(st.st_mtime)}"
    if jsonl.exists() and stamp.exists() and stamp.read_text() == key:
        return out
    out.mkdir(parents=True, exist_ok=True)
    stamp.unlink(missing_ok=True)
    analyze_video_rtmw(str(solo_video), str(jsonl), str(out / "overlay.mp4"))
    stamp.write_text(key)
    return out


def analyze_run(run_dir, video_path=None, leave_one_out=None, rotate=False,
                smooth=5, render=True, selected_tracks=None, on_progress=None,
                reference_mode="composite", reference_dancer=None, solo_video=None,
                solo_offset=0.0) -> Result:
    """`selected_tracks`: track ids (from `detect_tracks`) to keep as dancers; a
    bystander/audience member's track dropped here never reaches the composite
    or any deviation score. None keeps every stabilized track (old behavior).

    `reference_mode` picks what deviation is measured against: "composite" (default,
    consensus of all dancers), "dancer" (`reference_dancer`, an index into the kept
    dancers, is the target and isn't scored itself), or "solo" (`solo_video` is
    pose-tracked and used as the target; `solo_offset` seconds into it lines up
    with t=0 of this run)."""
    if reference_mode not in REFERENCE_MODES:
        raise ValueError(f"unknown reference_mode {reference_mode!r}")
    run_dir = Path(run_dir)
    det = ddc_io.load_jsonl(run_dir / "tracking.jsonl")
    if det.n_frames == 0:
        raise ValueError("tracking.jsonl is empty")
    ids, n_tracks = stabilize(det.kps)
    if n_tracks == 0:
        raise ValueError("no persistent dancers found (all detections were short-lived)")
    dense = to_dense(det.kps, ids, n_tracks)
    if selected_tracks is not None:
        if not selected_tracks:
            raise ValueError("no dancers selected -- pick at least one person")
        if any(t < 0 or t >= n_tracks for t in selected_tracks):
            raise ValueError(
                f"selected track ids {sorted(selected_tracks)} don't match this run's "
                f"{n_tracks} detected people -- click \"Find people\" again")
        dense = dense[:, sorted(selected_tracks)]
        n_tracks = len(selected_tracks)
    norm, origin, scale, rot = normalize(dense, rotate=rotate)
    reference = ref_track = None
    if reference_mode == "dancer":
        if n_tracks < 2:
            raise ValueError("need at least 2 dancers to use one as the reference")
        if reference_dancer is None or not 0 <= reference_dancer < n_tracks:
            raise ValueError("pick which dancer is the reference")
        ref_track = int(reference_dancer)
        reference = reference_from_track(norm, ref_track, smooth)
    elif reference_mode == "solo":
        if not solo_video:
            raise ValueError("upload a solo reference video")
        reference = reference_from_solo(_ensure_solo_run(run_dir, solo_video), det.t,
                                        solo_offset, rotate, smooth)
        if not np.isfinite(reference[..., 0]).any():
            raise ValueError("the solo reference video doesn't overlap this video's timeline "
                             "-- check the offset")
    dev = compute_deviation(norm, leave_one_out=leave_one_out, smooth=smooth,
                            reference=reference, reference_track=ref_track)

    moments = worst_moments(dev, det.t)
    summary = []
    for d in range(n_tracks):
        if d == ref_track:
            summary.append({"dancer": f"Dancer {d + 1}", "frames_present": 0,
                            "mean_dev": float("nan"), "worst_joint": "Reference",
                            "worst_moments": []})
            continue
        s = dev.score[:, d]
        mj = np.nanmean(dev.joint[:, d], axis=0)
        summary.append({
            "dancer": f"Dancer {d + 1}",
            "frames_present": int(np.isfinite(s).sum()),
            "mean_dev": float(np.nanmean(s)) if np.isfinite(s).any() else float("nan"),
            "worst_joint": BODY[int(np.nanargmax(mj))] if np.isfinite(mj).any() else "-",
            "worst_moments": [(round(tm, 1), round(sc, 2), BODY[j]) for tm, sc, j in moments[d]],
        })

    csv_path = run_dir / "deviation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t", "dancer", "score"] + BODY)
        for fr in range(det.n_frames):
            for d in range(n_tracks):
                if np.isfinite(dev.score[fr, d]):
                    w.writerow([fr, f"{det.t[fr]:.3f}", d + 1, f"{dev.score[fr, d]:.4f}"]
                               + [f"{v:.4f}" if np.isfinite(v) else "" for v in dev.joint[fr, d]])

    res = Result(run_dir, det.t, dense, norm, origin, scale, rot, dev, n_tracks, summary,
                 csv_path=str(csv_path))
    if render:
        from .render import render_overlay
        bg, dim = _pick_background(run_dir, video_path)
        res.overlay_path = render_overlay(dense, dev, origin, scale, rot, det.t,
                                          run_dir / "composite_overlay.mp4", bg, dim)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--video")
    ap.add_argument("--rotate", action="store_true")
    ap.add_argument("--leave-one-out", dest="loo", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--list-tracks", action="store_true",
                    help="list detected people (id, frames, time range) and exit")
    ap.add_argument("--tracks", help="comma-separated track ids to keep as dancers, e.g. 0,2")
    ap.add_argument("--reference", choices=REFERENCE_MODES, default="composite",
                    help="what to measure deviation against (default: composite of all dancers)")
    ap.add_argument("--reference-dancer", type=int,
                    help="with --reference dancer: 0-based index among the kept dancers")
    ap.add_argument("--solo-video", help="with --reference solo: the solo reference video")
    ap.add_argument("--solo-offset", type=float, default=0.0,
                    help="seconds into the solo video that line up with t=0 of this run")
    a = ap.parse_args()
    if a.list_tracks:
        for tr in detect_tracks(a.run_dir, a.video):
            print(f"{tr.track_id}: {tr.label}")
        return
    selected = [int(x) for x in a.tracks.split(",")] if a.tracks else None
    r = analyze_run(a.run_dir, a.video, a.loo, a.rotate, render=not a.no_render,
                    selected_tracks=selected, reference_mode=a.reference,
                    reference_dancer=a.reference_dancer, solo_video=a.solo_video,
                    solo_offset=a.solo_offset)
    print(f"{r.n_tracks} dancers, {len(r.t)} frames, reference={r.dev.mode}, "
          f"leave_one_out={r.dev.leave_one_out}")
    for s in r.summary:
        print(s)
    print("csv:", r.csv_path, "\noverlay:", r.overlay_path)


if __name__ == "__main__":
    main()
