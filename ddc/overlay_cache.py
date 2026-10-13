"""On-demand, cached dancer overlays.

Rendering the dancers-only overlay re-reads and re-encodes the whole source video, which
takes minutes on a real clip.  It is therefore never done as a side effect of scoring: the
UI asks for it, and a finished render is reused whenever nothing it depends on has changed.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .feature_inventory import _source_video, normalize_merges, render_classifier_overlay

# Bump when the drawing code changes so old cached renders are not reused.
RENDER_VERSION = 1
DEFAULT_MAX_HEIGHT = 720
KEEP_PER_RUN = 3


def overlay_key(run_dir: Path, source: Path, merge_groups, dancer_ids, max_height) -> str:
    """Hash of everything the rendered pixels depend on."""
    tracking = run_dir / "tracking.jsonl"
    blob = json.dumps({
        "version": RENDER_VERSION, "max_height": max_height,
        "source": [source.name, source.stat().st_size, int(source.stat().st_mtime)],
        "tracking": [tracking.stat().st_size, int(tracking.stat().st_mtime)],
        "merges": normalize_merges(merge_groups),
        "dancers": sorted(int(i) for i in dancer_ids),
    }, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _prune(overlay_dir: Path, keep: int) -> None:
    renders = sorted((p for p in overlay_dir.glob("*.mp4") if ".partial" not in p.name), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in renders[keep:]:
        old.unlink(missing_ok=True)
        old.with_suffix(".jsonl").unlink(missing_ok=True)


def render_dancer_overlay(run_dir, dancer_ids, all_track_ids, video_path=None, merge_groups=None,
                          max_height: int | None = DEFAULT_MAX_HEIGHT, keep: int = KEEP_PER_RUN,
                          progress=None) -> dict:
    """Overlay showing only ``dancer_ids``; reuses an identical earlier render if there is one.

    Returns {"overlay", "json", "cached", "seconds"}.  ``all_track_ids`` are the track ids
    currently shown, so everyone not in ``dancer_ids`` is treated as a non-dancer.
    """
    run_dir = Path(run_dir)
    source = _source_video(run_dir, video_path)
    key = overlay_key(run_dir, source, merge_groups, dancer_ids, max_height)
    overlay_dir = run_dir / "overlays"
    final = overlay_dir / f"{key}.mp4"
    if final.exists():
        os.utime(final)  # most recently used survives pruning
        return {"overlay": str(final), "json": str(final.with_suffix(".jsonl")),
                "cached": True, "seconds": 0.0}

    chosen = {int(i) for i in dancer_ids}
    predictions = [{"track_id": int(t), "predicted_label": "dancer" if int(t) in chosen else "non_dancer"}
                   for t in all_track_ids]
    # Render beside the final name and rename on success, so an interrupted render can
    # never be mistaken for a finished one.
    partial = overlay_dir / f"{key}.{os.getpid()}.partial.mp4"
    started = time.time()
    try:
        render_classifier_overlay(run_dir, show_non_dancers=False, video_path=source,
                                  predictions=predictions, merge_groups=merge_groups,
                                  output=partial, max_height=max_height, progress=progress)
        os.replace(partial.with_suffix(".jsonl"), final.with_suffix(".jsonl"))
        os.replace(partial, final)
    finally:
        for leftover in (partial, partial.with_suffix(".jsonl"), partial.with_suffix(".raw.mp4")):
            leftover.unlink(missing_ok=True)
    _prune(overlay_dir, keep)
    return {"overlay": str(final), "json": str(final.with_suffix(".jsonl")),
            "cached": False, "seconds": time.time() - started}
