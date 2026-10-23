"""Local web UI: upload a dance video, pick the dancers, see who deviates and when.

One page, three steps:
  1. Video    -- upload a clip (pose tracking runs once) or reopen a previous run
  2. Dancers  -- vet the automatically predicted dancers, choose the reference
  3. Results  -- deviation overlay video next to the timeline, kept in sync

Usage:
    python app.py
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import time
from html import escape
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from ddc.analyze import _save_source_video, analyze_run, detect_tracks
from ddc.coaching import SCORE_EXPLANATION, feedback_moments, joint_feedback, key_moments
from ddc.overlay_cache import render_dancer_overlay
from ddc.feature_registry import (predict_run, rebuild_registry_workbook,
                                  validate_checklist)
from ddc.plots import timeline_figure, tracks_figure
from ddc.skeleton import BODY
from video_pipeline import analyze_video_mediapipe, analyze_video_rtmw, analyze_video_yolo

MODEL_YOLO = "YOLO (body only, fast)"
MODEL_MEDIAPIPE = "MediaPipe (body + hands, tracking only -- can't be compared)"
MODEL_RTMW = "RTMW (body + hands, multi-person, recommended)"

REF_COMPOSITE = "Composite of all dancers"
REF_DANCER = "One of the dancers in this video"
REF_SOLO = "Separate solo video"
REF_MODES = {REF_COMPOSITE: "composite", REF_DANCER: "dancer", REF_SOLO: "solo"}

RUNS_DIR = Path(os.environ.get("DDC_RUNS_DIR") or Path(__file__).parent / "runs")

TAB_VIDEO, TAB_DANCERS, TAB_RESULTS = "video", "dancers", "results"
NO_RUN_HINT = "*Analyze a video or open a previous run in step 1 first.*"
NO_RESULT_HINT = "*Pick the dancers in step 2 and click **Compare dancers** to see results here.*"


def _friendly(what: str):
    """Turn unexpected failures in a long callback into a readable toast instead of a
    raw traceback (gr.Error raised on purpose passes straight through)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except gr.Error:
                raise
            except subprocess.CalledProcessError as e:
                raise gr.Error(f"{what} failed: {Path(str(e.cmd[0])).name} could not process the "
                               "video (is the file a playable video?).")
            except Exception as e:
                raise gr.Error(f"{what} failed: {e}")
        return wrapper
    return deco


def _frame_progress(progress, lo: float, hi: float, desc: str):
    """`on_progress(done, total)` callback that maps a frame loop onto [lo, hi] of the bar."""
    def cb(done, total):
        if done % 5 and done != total:
            return
        frac = min(done / total, 1.0) if total > 0 else 0.0
        progress(lo + (hi - lo) * frac,
                 desc=f"{desc} ({done}/{total} frames)" if total > 0 else f"{desc} ({done} frames)")
    return cb


def _run_label(run_dir: Path) -> str:
    """Human-readable name for a run: when, which video, how long."""
    try:
        when = time.strftime("%b %d %H:%M", time.strptime(run_dir.name, "%Y%m%d-%H%M%S"))
    except ValueError:
        when = run_dir.name
    video, duration = None, None
    try:
        meta = json.loads((run_dir / "meta.json").read_text())
        video, duration = meta.get("video"), meta.get("duration_s")
    except (OSError, ValueError):
        # Runs made before meta.json existed: the overlay still gives the length.
        overlay = run_dir / "overlay.mp4"
        if overlay.exists():
            cap = cv2.VideoCapture(str(overlay))
            fps, frames = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            duration = frames / fps if fps > 0 and frames > 0 else None
    parts = [when] + ([video] if video else []) + ([f"{duration:.0f} s"] if duration else [])
    return " · ".join(parts)


def _comparable(run_dir: Path) -> bool:
    tracking = run_dir / "tracking.jsonl"
    if not tracking.exists() or tracking.stat().st_size == 0:
        return False
    try:
        return json.loads((run_dir / "meta.json").read_text()).get("model") != MODEL_MEDIAPIPE
    except (OSError, ValueError):
        return True


def list_runs() -> list[tuple[str, str]]:
    """(label, run name) for every run that can be compared, newest first."""
    if not RUNS_DIR.exists():
        return []
    names = sorted((p.name for p in RUNS_DIR.iterdir() if p.is_dir() and _comparable(p)),
                   reverse=True)
    return [(_run_label(RUNS_DIR / n), n) for n in names]


def _runs_update(value=None):
    runs = list_runs()
    if value is None and runs:
        value = runs[0][1]
    return gr.update(choices=runs, value=value)


@_friendly("Pose tracking")
def run_analysis(video_path: str, model_choice: str, track: bool, num_people: int,
                 progress=gr.Progress()):
    """Step 1: pose-track the upload into a new run and keep the source video with it,
    so the later steps never ask for the same file again.

    Returns (overlay, jsonl, note, run dropdown update, run to open or None)."""
    if not video_path:
        raise gr.Error("Upload a video first.")

    progress(0, desc="Preparing video...")
    run_dir = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = run_dir / "tracking.jsonl"
    out_video = run_dir / "overlay.mp4"

    on_progress = _frame_progress(progress, 0.02, 0.95, "Tracking poses")
    if model_choice == MODEL_YOLO:
        summary = analyze_video_yolo(video_path, str(out_jsonl), str(out_video), track=track,
                                     on_progress=on_progress)
    elif model_choice == MODEL_RTMW:
        summary = analyze_video_rtmw(video_path, str(out_jsonl), str(out_video),
                                     on_progress=on_progress)
    else:
        summary = analyze_video_mediapipe(
            video_path, str(out_jsonl), str(out_video),
            num_poses=num_people, num_hands=num_people * 2, on_progress=on_progress,
        )
    if not summary["frames"]:
        raise gr.Error("No frames could be read from that video.")

    progress(0.97, desc="Saving...")
    _save_source_video(run_dir, video_path)
    duration = summary["frames"] / summary["fps"] if summary["fps"] else 0.0
    (run_dir / "meta.json").write_text(json.dumps({
        "video": Path(video_path).name, "model": model_choice,
        "frames": summary["frames"], "fps": summary["fps"], "duration_s": round(duration, 2),
    }, indent=2))

    note = (f"Tracked **{summary['frames']} frames** ({duration:.1f} s), up to "
            f"**{summary['max_people_detected']} people** in a frame.")
    if model_choice == MODEL_MEDIAPIPE:
        note += (" MediaPipe output is tracking-only: use the files below, or analyze with "
                 "RTMW or YOLO to compare dancers.")
        return str(out_video), str(out_jsonl), note, gr.skip(), None
    return str(out_video), str(out_jsonl), note, _runs_update(run_dir.name), run_dir.name


_LAST: dict = {}


def _members(n_auto: int, groups: list[list[int]]) -> list[list[int]]:
    """Automatic track ids behind each displayed person, in display order (matches the
    renumbering in ddc.tracking.merge_track_ids: merged people sort by lowest member)."""
    target = {t: t for t in range(n_auto)}
    for g in groups:
        for t in g:
            target[t] = min(g)
    return [[t for t in range(n_auto) if target[t] == k] for k in sorted(set(target.values()))]


def _predict_people(run_name, groups, tracks):
    """Score the people currently shown and pick the predicted dancers.

    Returns (selected labels, note).  Scoring uses the
    same merged track ids as the list on screen and holds this run out of training.
    """
    labels = [tr.label for tr in tracks]
    try:
        model, predictions = predict_run(RUNS_DIR, run_name, merge_groups=groups)
    except ValueError as e:
        # A first-ever run cannot be predicted until both classes have two human
        # labels.  Keep the manual fallback explicit rather than silently training.
        _LAST["predictions"] = []
        return (labels,
                f"Automatic screening needs validated examples first: {e}. All tracks are selected for review.")

    predicted_ids = frozenset(item["track_id"] for item in predictions
                              if item["predicted_label"] == "dancer")
    selected = [tr.label for tr in tracks if tr.track_id in predicted_ids]
    _LAST["predictions"] = predictions
    _LAST["model"] = model
    _LAST["predicted_selected_ids"] = predicted_ids

    # The workbook is a convenience: a failure must not discard the predictions above.
    # The preview video is NOT rendered here -- it re-encodes the whole clip and takes
    # a while, so it is only built when asked (see run_render_overlay).
    problems = []
    try:
        # Add a Labels-style worksheet for this run immediately. Its local
        # top-five weights populate once the checklist is validated.
        rebuild_registry_workbook(RUNS_DIR)
    except RuntimeError as e:
        problems.append(f"Registry workbook refresh needs attention: {e}")

    weights = "; ".join(f"{item['feature_name']}: {item['standardized_coefficient']:+.3f}"
                        for item in model["selected_features"])
    note = (f"Automatic checklist: {len(selected)} predicted dancer(s). Top-five model has "
            f"{model['labelled_tracks']} validated track labels across {len(model['runs'])} run(s) "
            f"(this run is held out of training). "
            f"The checklist uses exactly these five standardized weights: {weights}. "
            "Leave it unchanged and Compare to validate every prediction, or save edits to record corrections. "
            + " ".join(problems))
    return selected, note.strip()


def _review_status(track_choice) -> str:
    """One line saying whether the ticked dancers are recorded in the label registry."""
    ids = _LAST.get("track_ids")
    if not ids:
        return ""
    current = frozenset(ids[c] for c in (track_choice or []) if c in ids)
    if _LAST.get("saved_selection") == current:
        return "✅ Labels saved."
    if _LAST.get("predictions") and current == _LAST.get("predicted_selected_ids"):
        return "Matches the automatic selection -- it is confirmed when you compare."
    if not _LAST.get("predictions") and "saved_selection" not in _LAST:
        return "Not recorded yet -- click *Save reviewed labels* to teach the automatic selection."
    return "🟠 **Unsaved changes** -- click *Save reviewed labels* to record them."


def _show_people(run_name, groups, progress=None):
    """(Re)render the people list for the current merges and remember it.

    Merging renumbers the people, so predictions are recomputed for the new numbering and
    any earlier save/validation state is dropped.
    """
    try:
        tracks = detect_tracks(RUNS_DIR / run_name, merge_groups=groups or None)
    except ValueError as e:
        raise gr.Error(str(e))
    labels = [tr.label for tr in tracks]
    _LAST["track_run"] = run_name
    _LAST["track_labels"] = labels
    _LAST["track_ids"] = {tr.label: tr.track_id for tr in tracks}
    _LAST["merges"] = groups
    for stale in ("saved_selection", "saved_validation", "predicted_selected_ids"):
        _LAST.pop(stale, None)
    if progress is not None:
        progress(0.5, desc="Predicting dancers...")
    selected, note = _predict_people(run_name, groups, tracks)
    if _LAST.get("predictions"):
        headline = (f"**{len(selected)} of {len(labels)} people** were automatically selected as "
                    "dancers. Untick anyone who isn't dancing, tick anyone who was missed.")
    else:
        headline = ("Automatic selection isn't available yet, so **everyone is ticked**. "
                    "Untick anyone who isn't dancing.")
    gallery = [(tr.thumbnail, tr.label) for tr in tracks]
    return (gallery, gr.update(choices=labels, value=selected), gr.update(choices=labels, value=None),
            gr.update(choices=labels, value=[]), tracks_figure(tracks),
            gr.update(value=None, visible=False), note, headline,  # preview is on demand
            _review_status(selected), gr.update(interactive=True))


N_PEOPLE_OUTPUTS = 10   # length of the tuple _show_people returns


def begin_review(run_name):
    """Jump to step 2 for `run_name` and blank everything left over from another run."""
    if not run_name:
        return [gr.skip()] * 27
    return (gr.Tabs(selected=TAB_DANCERS),
            f"### {_run_label(RUNS_DIR / run_name)}",
            # step 2
            None, gr.update(choices=[], value=[]), gr.update(choices=[], value=None),
            gr.update(choices=[], value=[]), None, gr.update(value=None, visible=False), "",
            "Finding people...", "", gr.update(interactive=False),
            # step 3
            NO_RESULT_HINT, None, None, [], [], "", None, "", [], [], "", "", gr.update(open=False),
            gr.update(choices=["All dancers"], value="All dancers"), "")


@_friendly("Finding people")
def run_detect(run_name, progress=gr.Progress()):
    """Find every stabilized person track in the run and show a thumbnail
    for each, so the user can rule out bystanders/audience before anything is
    scored -- the tracker has no notion of "dancer" vs. "person on camera",
    it keeps anyone visible for more than ~15 frames. Fragments of one person
    that the tracker lost and re-found are joined automatically; "Merge" below
    fixes any it missed.
    """
    if not run_name:
        return [gr.skip()] * N_PEOPLE_OUTPUTS
    progress(0.1, desc="Finding people in the clip...")
    _LAST.clear()
    out = _show_people(run_name, [], progress)
    _LAST["n_auto"] = len(out[0])
    return out


def _require_people(run_name):
    labels = _LAST.get("track_labels")
    if not run_name or not labels or _LAST.get("track_run") != run_name:
        raise gr.Error("Open a run in step 1 first.")
    return labels


@_friendly("Merging")
def run_merge(run_name, merge_choice, progress=gr.Progress()):
    """Treat the ticked people as one person (adds to any earlier merges)."""
    labels = _require_people(run_name)
    if len(merge_choice or []) < 2:
        raise gr.Error("Tick at least two people to merge.")
    progress(0.1, desc="Merging...")
    members = _members(_LAST["n_auto"], _LAST.get("merges", []))
    picked = sorted(labels.index(c) for c in merge_choice)
    joined = sorted(t for i in picked for t in members[i])
    groups = [m for i, m in enumerate(members) if len(m) > 1 and i not in picked] + [joined]
    return _show_people(run_name, groups, progress)


@_friendly("Resetting merges")
def run_reset_merges(run_name, progress=gr.Progress()):
    _require_people(run_name)
    progress(0.1, desc="Resetting merges...")
    return _show_people(run_name, [], progress)


def update_ref_choices(track_choice, current):
    """Reference-dancer dropdown offers only the people currently ticked as dancers."""
    choices = list(track_choice or [])
    return gr.update(choices=choices, value=current if current in choices else None)


@_friendly("Rendering the preview")
def run_render_overlay(run_name, track_choice, progress=gr.Progress()):
    """Render (or reuse) the preview showing only the people currently ticked as dancers."""
    _require_people(run_name)
    if not track_choice:
        raise gr.Error("Tick at least one person as a dancer to preview.")
    dancers = [_LAST["track_ids"][c] for c in track_choice]
    result = render_dancer_overlay(
        RUNS_DIR / run_name, dancers, list(_LAST["track_ids"].values()),
        merge_groups=_LAST.get("merges"), progress=lambda f, d: progress(f, desc=d))
    how = ("reused an identical earlier render" if result["cached"]
           else f"rendered in {result['seconds']:.0f} s")
    return gr.update(value=result["overlay"], visible=True,
                     label=f"Preview: ticked dancers only ({how})")


def save_review(run_name, track_choice, progress=gr.Progress()):
    """Persist the current checklist without starting pose comparison."""
    _require_people(run_name)
    selected = [_LAST["track_ids"][choice] for choice in (track_choice or [])]
    progress(.25, desc="Saving reviewed dancer labels...")
    try:
        outcome = validate_checklist(RUNS_DIR, run_name, selected, _LAST.get("predictions", []),
                                     merge_groups=_LAST.get("merges"))
        _LAST["saved_selection"] = frozenset(selected)
        _LAST["saved_validation"] = outcome["validation"]
        workbook = rebuild_registry_workbook(RUNS_DIR)
    except (RuntimeError, ValueError) as e:
        raise gr.Error(f"Could not save the reviewed checklist: {e}")
    progress(1, desc="Reviewed labels saved")
    validation = outcome["validation"]
    state = validation["status"].replace("_", " ")
    return (f"✅ **Labels saved.** Predictor labels {state}; "
            f"{len(validation['corrected_track_ids'])} track correction(s) recorded. "
            f"Updated {workbook.name}.")


def results_overview(res, names, practice_rows):
    """A small, escaped score summary; the full analysis stays in the detail panel."""
    score = f"{res.coaching.overall:.1f}" if res.coaching.overall is not None else "N/A"
    ref = res.dev.reference_track
    reference = (f"Reference: {names[ref]} · not scored" if ref is not None else
                 "Compared with the solo reference" if res.dev.mode == "solo" else
                 "Compared with the group · differences do not establish who is correct")
    cards = []
    for name, dancer in zip(names, res.coaching.dancers):
        if dancer["reference"]:
            continue
        value = f'{dancer["score"]:.1f}' if dancer["score"] is not None else "N/A"
        coverage = f'{dancer["valid_frames"]}/{dancer["total_frames"]} comparable frames'
        cards.append(f'<div class="ddc-dancer"><h3>{escape(name)}</h3>'
                     f'<p class="ddc-dancer-score">{value}<span> / 100</span></p>'
                     f'<p class="ddc-muted">{coverage}</p></div>')
    note = ""
    if any(d["coverage"] < 0.5 for d in res.coaching.dancers if not d["reference"]):
        note += '<p class="ddc-coverage-note">Limited pose coverage. Open detailed feedback before interpreting these scores.</p>'
    if not practice_rows:
        note += '<p class="ddc-muted">No practice moments to show. Detailed feedback includes pose coverage and individual joint notes.</p>'
    return (f'<section class="ddc-overview" aria-label="Movement similarity overview">'
            f'<div class="ddc-overall"><div><p class="ddc-eyebrow">Movement similarity</p>'
            f'<p class="ddc-overall-score">{score}<span> / 100</span></p></div>'
            f'<div class="ddc-score-context"><p>{escape(reference)}</p>'
            f'<p class="ddc-muted">A practice guide, not a validated accuracy percentage.</p></div></div>'
            f'<div class="ddc-dancer-grid">{"".join(cards)}</div>{note}</section>')


def practice_preview(key_rows):
    """Render every detected priority moment as an accessible replay card."""
    if not key_rows:
        return ('<p class="ddc-empty">No qualifying practice moments found. '
                'Open detailed feedback to check pose coverage and individual notes.</p>')
    cards = []
    for rank, row in enumerate(key_rows, 1):
        dancers, peak, start, end, region, joint = row[:6]
        cards.append(
            f'<li><button type="button" class="ddc-moment" data-ddc-time="{float(peak):.6f}" '
            f'aria-label="Replay moment {rank}: {escape(str(joint), quote=True)} at {peak:.2f} seconds">'
            f'<span class="ddc-moment-number">{rank:02d}</span>'
            f'<span class="ddc-moment-copy"><strong>Review {escape(str(joint))}</strong>'
            f'<span>{escape(str(dancers))}</span>'
            f'<span class="ddc-moment-range">{start:.2f}–{end:.2f} s · {escape(str(region))}</span></span>'
            f'<span class="ddc-moment-play"><span>{peak:.2f} s</span><span aria-hidden="true">↗</span></span>'
            '</button></li>')
    return ('<ol class="ddc-moments" aria-label="All priority practice moments">'
            + ''.join(cards) + '</ol>')


@_friendly("Comparison")
def run_compare(run_name, track_choice, loo_choice, rotate, smooth,
                ref_choice, ref_dancer, solo_video, solo_offset, progress=gr.Progress()):
    _require_people(run_name)
    if not track_choice:
        raise gr.Error("Select at least one person as a dancer.")
    selected = [_LAST["track_ids"][c] for c in track_choice]
    loo = {"Auto (on for 3+ dancers)": None, "On": True, "Off": False}[loo_choice]
    mode = REF_MODES[ref_choice]
    ref_idx = None
    if mode == "dancer":
        if ref_dancer not in track_choice:
            raise gr.Error("Pick which dancer is the reference.")
        ref_idx = sorted(selected).index(_LAST["track_ids"][ref_dancer])
    elif mode == "solo" and not solo_video:
        raise gr.Error("Upload a solo reference video.")
    progress(0.05, desc="Analyzing solo reference video..." if mode == "solo"
             else "Building reference and deviation...")
    try:
        res = analyze_run(RUNS_DIR / run_name, leave_one_out=loo,
                          rotate=rotate, smooth=int(smooth), selected_tracks=selected,
                          reference_mode=mode, reference_dancer=ref_idx,
                          solo_video=solo_video, solo_offset=float(solo_offset or 0),
                          merge_groups=_LAST.get("merges") or None,
                          on_progress=_frame_progress(progress, 0.15, 0.95,
                                                      "Rendering deviation overlay"))
    except ValueError as e:
        raise gr.Error(str(e))
    _LAST["res"] = res
    validation_note = ""
    if _LAST.get("predictions"):
        baseline = _LAST.get("predicted_selected_ids", frozenset())
        current = frozenset(selected)
        if current == baseline:
            # An untouched checklist is the explicit all-correct confirmation.
            # Save it here so a reviewer can validate straight from Compare.
            try:
                outcome = validate_checklist(RUNS_DIR, run_name, selected, _LAST["predictions"],
                                             merge_groups=_LAST.get("merges"))
                _LAST["saved_selection"] = current
                _LAST["saved_validation"] = outcome["validation"]
                workbook = rebuild_registry_workbook(RUNS_DIR)
                validation_note = (" Predictor labels validated correct; 0 track corrections stored. "
                                   f"Accumulating workbook updated: {workbook.name}.")
            except (RuntimeError, ValueError) as e:
                validation_note = f" Checklist validation could not be saved: {e}"
        elif _LAST.get("saved_selection") != current:
            validation_note = (" Checklist changes have not been saved. Click **Save reviewed labels** "
                               "in step 2 to update the registry's Reviewed label, Validation status, and "
                               "Correction recorded columns.")
        else:
            validation = _LAST.get("saved_validation", {})
            validation_note = (f" Reviewed corrections are saved: "
                               f"{len(validation.get('corrected_track_ids', []))} track correction(s).")
    progress(0.97, desc="Plotting...")

    # Dancers are renumbered 1..N among the ticked people, so say who is who whenever
    # "Dancer 2" isn't "Person 2".
    person = {tid: label.split("  (")[0] for label, tid in _LAST["track_ids"].items()}
    people = [person[tid] for tid in sorted(selected)]
    renumbered = any(p != f"Person {d + 1}" for d, p in enumerate(people))
    names = [f"Dancer {d + 1} ({p})" if renumbered else f"Dancer {d + 1}"
             for d, p in enumerate(people)]

    _LAST["names"] = names
    rows, moments = [], []
    for d, s in enumerate(res.summary):
        if d == res.dev.reference_track:
            continue
        c = res.coaching.dancers[d]
        rows.append([names[d], round(c["score"], 1) if c["score"] is not None else "N/A",
                     f'{c["valid_frames"]}/{c["total_frames"]} ({c["coverage"]:.0%})',
                     round(s["mean_dev"], 3) if np.isfinite(s["mean_dev"]) else "N/A",
                     c["strength"], c["improvement"]])
        moments += [[names[d], t, joint, score] for t, score, joint in s["worst_moments"]]
    moments.sort(key=lambda m: -m[3])

    note = ""
    if res.dev.reference_track is not None:
        note = f"Reference: **{names[res.dev.reference_track]}** (not scored)."
    if res.n_tracks == 2 and mode == "composite":
        note = ("Only 2 dancers: the composite is their midpoint, so deviation is symmetric "
                "(it measures how far apart they are, not who is off).")
    overall = res.coaching.overall
    overall_label = f"{overall:.1f}/100" if overall is not None else "N/A — insufficient comparable pose data"
    note = f"**Overall movement similarity: {overall_label}** (heuristic).\n\n" + note
    feedback = [[names[d], tm, message] for d, tm, message in
                feedback_moments(res.dev, res.norm, res.coaching, res.t, res.rotate)]
    priority = key_moments(res.norm, res.t)
    key_rows = [[f"{names[m.dancers[0]]} & {names[m.dancers[1]]}", m.peak,
                 m.start, m.end, m.region, m.joint.replace("_", " "),
                 round(m.mean_deviation, 3), m.comparable_joints]
                for m in priority]
    _LAST["report_rows"] = (rows, moments, feedback, key_rows)
    _LAST["report_pairs"] = [tuple(names[d] for d in m.dancers) for m in priority]
    max_t = float(res.t[-1])
    return (gr.Tabs(selected=TAB_RESULTS), f"### {_run_label(RUNS_DIR / run_name)}",
            res.overlay_path, timeline_figure(res.dev, res.t), rows, moments,
            note + validation_note, res.csv_path, gr.update(maximum=max_t, value=0),
            frame_detail(0), _review_status(track_choice), feedback, key_rows,
            results_overview(res, names, key_rows), practice_preview(key_rows), gr.update(open=False),
            gr.update(choices=["All dancers"] + names, value="All dancers"),
            dancer_report("All dancers"))


def dancer_report(dancer):
    """Readable individual report without implying that reference dancers are scored."""
    names = _LAST.get("names", [])
    if dancer == "All dancers":
        return '<p class="ddc-empty">Choose a dancer above to open their individual report.</p>'
    if dancer not in names or _LAST.get("res") is None:
        return '<p class="ddc-empty">Compare dancers to load an individual report.</p>'
    c = _LAST["res"].coaching.dancers[names.index(dancer)]
    if c["reference"]:
        body = '<p>Reference dancer · not scored. Pairwise practice moments involving this dancer are listed below.</p>'
    else:
        score = f'{c["score"]:.1f} / 100' if c["score"] is not None else 'N/A · insufficient comparable pose data'
        body = (f'<p class="ddc-dancer-score">{score}</p>'
                f'<p>{c["valid_frames"]}/{c["total_frames"]} comparable frames ({c["coverage"]:.0%})</p>'
                f'<p><strong>Strength / closest region:</strong> {escape(c["strength"])}</p>'
                f'<p><strong>Area to review:</strong> {escape(c["improvement"])}</p>'
                '<p class="ddc-muted">Movement similarity is a heuristic, not a validated accuracy percentage.</p>')
        if c["coverage"] < .5:
            body += '<p>Limited pose coverage; interpret this score cautiously.</p>'
    return f'<section class="ddc-dancer" aria-label="Individual dancer report"><h3>{escape(dancer)}</h3>{body}</section>'


def filter_dancer_report(dancer, t_sec):
    """Filter by exact dancer identity, keeping the cached full analysis intact."""
    rows, moments, feedback, priority = _LAST.get("report_rows", ([], [], [], []))
    if dancer != "All dancers":
        rows, moments, feedback = ([r for r in table if r[0] == dancer]
                                  for table in (rows, moments, feedback))
        priority = [r for r, pair in zip(priority, _LAST.get("report_pairs", []))
                    if dancer in pair]
    return rows, moments, feedback, priority, frame_detail(t_sec, dancer), dancer_report(dancer)


def jump_to_moment(rows, evt: gr.SelectData):
    """Clicking a worst-moment row moves the time slider (and so the video) there."""
    return float(rows.iloc[evt.index[0], 1])


def frame_detail(t_sec, dancer="All dancers"):
    res = _LAST.get("res")
    if res is None:
        return "Run an analysis first."
    f = int(np.argmin(np.abs(res.t - t_sec)))
    lines = [f"t={res.t[f]:.2f}s (frame {f}), {int(res.dev.n_present[f])} dancer(s) present"]
    names = _LAST.get("names", [f"Dancer {i + 1}" for i in range(res.n_tracks)])
    for d in range(res.n_tracks):
        if dancer != "All dancers" and names[d] != dancer:
            continue
        if d == res.dev.reference_track:
            lines.append(f"{names[d]}: reference (not scored)")
            continue
        name = _LAST.get("names", [f"Dancer {i + 1}" for i in range(res.n_tracks)])[d]
        j = res.coaching.joint[f, d]
        if not np.isfinite(j).any():
            lines.append(f"{name}: insufficient comparable pose data (dancer or reference missing, "
                         "too few reliable joints, or no peer for the composite).")
            continue
        score = res.coaching.frame_score[f, d]
        lines.append(f"{name}: movement similarity {score:.1f}/100; "
                     f"deviation {res.coaching.deviation[f, d]:.2f} torso lengths.")
        top = np.flatnonzero(np.isfinite(j))
        top = top[np.argsort(-j[top])][:3]
        lines.extend(joint_feedback(res.dev, res.norm, res.coaching, f, d, i, res.t[f], res.rotate)
                     for i in top)
    return "\n".join(lines)


# Runs in the browser before `frame_detail`: any change of the time slider (dragging it,
# clicking a worst-moment row) also moves the video. `ddcSeek` is defined in _SYNC_JS.
_SEEK_ON_SLIDER_JS = "(t, dancer) => { if (window.ddcSeek) window.ddcSeek(t); return [t, dancer]; }"
# Each step starts at the top of the page, not wherever the previous step was scrolled to.
_SCROLL_TOP_JS = "() => { window.scrollTo({top: 0, behavior: 'smooth'}); }"
_SCROLL_TOP_JS_KEEP = "(run) => { window.scrollTo({top: 0, behavior: 'smooth'}); return run; }"

with gr.Blocks(title="DanceDanceConvolution") as demo:
    active_run = gr.State(None)   # the run steps 2 and 3 are working on

    gr.Markdown("# DanceDanceConvolution\n"
                "A clearer view of your movement. A little more focus for your next practice.",
                elem_id="ddc_header")
    with gr.Tabs() as tabs:
        with gr.Tab("1 · Video", id=TAB_VIDEO):
            with gr.Row():
                with gr.Column(scale=3):
                    video_in = gr.Video(label="Dance video", elem_id="ddc_in_video")
                    with gr.Accordion("Pose model (advanced)", open=False):
                        model_choice = gr.Radio(
                            [MODEL_RTMW, MODEL_YOLO, MODEL_MEDIAPIPE],
                            value=MODEL_RTMW, label="Model",
                        )
                        track = gr.Checkbox(
                            value=True, label="Persistent per-person IDs (YOLO ByteTrack)",
                            visible=False,
                        )
                        num_people = gr.Slider(
                            minimum=1, maximum=12, step=1, value=4, visible=False,
                            label="Number of people in video (MediaPipe only -- YOLO/RTMW have no cap)",
                        )
                    run_btn = gr.Button("Analyze video", variant="primary")
                    analysis_note = gr.Markdown()
                with gr.Column(scale=2):
                    gr.Markdown("**Or open a video you already analyzed**")
                    run_dd = gr.Dropdown(list_runs(), value=(list_runs() or [(None, None)])[0][1],
                                         label="Previous runs")
                    open_btn = gr.Button("Open run")
                    with gr.Accordion("Raw tracking output", open=False):
                        video_out = gr.Video(label="Pose overlay video")
                        json_out = gr.File(label="Tracking JSONL (per-frame keypoints)")

        with gr.Tab("2 · Dancers", id=TAB_DANCERS):
            run_title = gr.Markdown(NO_RUN_HINT)
            with gr.Row():
                with gr.Column(scale=3):
                    people_headline = gr.Markdown()
                    people_gallery = gr.Gallery(label="Detected people", columns=4, height=220,
                                                object_fit="contain")
                    with gr.Accordion("Same person listed twice? Merge", open=False):
                        gr.Markdown("Fragments are joined automatically when the tracker lost "
                                    "someone and found them again. If the same person still "
                                    "appears more than once, tick them here and merge. Merging "
                                    "renumbers people, so the dancer predictions are recomputed.")
                        merge_select = gr.CheckboxGroup(label="People to merge into one", choices=[])
                        with gr.Row():
                            merge_btn = gr.Button("Merge selected", size="sm")
                            reset_merge_btn = gr.Button("Reset merges", size="sm")
                    track_select = gr.CheckboxGroup(label="Include as dancer", choices=[])
                    with gr.Row():
                        review_status = gr.Markdown()
                        save_review_btn = gr.Button("Save reviewed labels", size="sm", scale=0,
                                                    min_width=180)
                    with gr.Accordion("How the automatic selection was made", open=False):
                        predictor_note = gr.Markdown()
                with gr.Column(scale=2):
                    render_overlay_btn = gr.Button(
                        "Render preview of the ticked dancers (takes a while)", size="sm")
                    predicted_overlay = gr.Video(label="Preview: predicted dancers only",
                                                 elem_id="ddc_pred_video", visible=False)
                    tracks_plot = gr.Plot(label="When each person is visible")

            with gr.Group():
                ref_choice = gr.Radio([REF_COMPOSITE, REF_DANCER, REF_SOLO], value=REF_COMPOSITE,
                                      label="Compare each dancer against")
                ref_dancer = gr.Dropdown(
                    [], value=None, visible=False,
                    label="Reference dancer (everyone else is scored against this person)")
                solo_video = gr.Video(label="Solo reference video", visible=False)
                solo_offset = gr.Number(
                    value=0, visible=False,
                    label="Solo offset (s): seconds into the solo video that line up with the "
                          "start of the group video (negative = solo starts later)")
            with gr.Accordion("Advanced settings", open=False):
                loo_choice = gr.Radio(
                    ["Auto (on for 3+ dancers)", "On", "Off"], value="Auto (on for 3+ dancers)",
                    label="Leave-one-out composite",
                    info="Score each dancer against the consensus of the other dancers, so "
                         "their own mistakes don't pull the reference toward them.")
                rotate = gr.Checkbox(
                    value=False, label="Normalize torso rotation",
                    info="Turn every dancer to face the same way before comparing. Use when "
                         "dancers face different directions on purpose.")
                smooth = gr.Slider(
                    1, 15, step=2, value=5, label="Composite smoothing (frames)",
                    info="Averages the reference pose over this many frames. Higher is "
                         "steadier but blurs fast moves.")
            cmp_btn = gr.Button("Compare dancers", variant="primary", interactive=False)
            cmp_progress = gr.Markdown(min_height=40)   # where the Compare progress bar shows

        with gr.Tab("3 · Results", id=TAB_RESULTS):
            result_title = gr.Markdown(NO_RESULT_HINT, elem_id="ddc_result_title")
            overview = gr.HTML(elem_id="ddc_score_overview")
            with gr.Row(elem_id="ddc_practice_row"):
                with gr.Column(scale=3, elem_id="ddc_video_col"):
                    cmp_video = gr.Video(label="Your movement · green = close, red = different",
                                         elem_id="ddc_cmp_video")
                    t_slider = gr.Slider(0, 1, step=0.05, value=0, label="Inspect time (s)",
                                         elem_id="ddc_t_slider")
                with gr.Column(scale=2, min_width=280, elem_id="ddc_practice_col"):
                    gr.Markdown("### Practice moments\n"
                                "Every detected priority moment, ranked by pose difference. Choose a card to replay.")
                    practice_tbl = gr.HTML(elem_id="ddc_practice_preview")
                    gr.Markdown("Pose differences can be intentional. Review the video before "
                                "deciding what to change.", elem_classes="ddc_small_note")

            with gr.Accordion("View detailed feedback", open=False, elem_id="ddc_details") as details_panel:
                gr.Markdown("### A closer look\n"
                            "Explore each dancer, compare moments, or download the full analysis.")
                report_dancer = gr.Dropdown(["All dancers"], value="All dancers",
                                            label="Feedback report for", interactive=True)
                individual_report = gr.HTML()
                cmp_note = gr.Markdown()
                summary_tbl = gr.Dataframe(
                    headers=["Dancer", "Similarity / 100", "Comparable frames", "Mean raw deviation",
                             "Strength / closest region", "Area to review"],
                    interactive=False, wrap=True, label="Per-dancer coaching summary (heuristic scores)")
                timeline = gr.Plot(label="Group timeline (all dancers) — click to inspect a moment", elem_id="ddc_timeline")
                detail = gr.Textbox(label="At this moment", lines=6)
                gr.Markdown("### Priority practice moments for the selected dancer\n"
                            "Up to five sustained pairwise pose differences, ranked by peak mean "
                            "deviation. The review range gives you the surrounding section to practice. "
                            "These do not establish who is early or late.")
                key_moments_tbl = gr.Dataframe(
                    headers=["Dancers", "Peak time (s)", "Review from (s)", "Review to (s)",
                             "Largest region difference", "Joint to review", "Mean pose difference",
                             "Comparable joints"], interactive=False,
                    label="Priority practice moments — click a row to inspect")
                moments_tbl = gr.Dataframe(
                    headers=["Dancer", "Time (s)", "Joint", "Deviation"], interactive=False,
                    label="Worst moments — click a row to inspect")
                feedback_tbl = gr.Dataframe(
                    headers=["Dancer", "Time (s)", "Suggestion"], interactive=False, wrap=True,
                    label="Personalized feedback — click a row to inspect")
                with gr.Accordion("How scores and feedback work", open=False):
                    gr.Markdown(SCORE_EXPLANATION)
                    gr.Markdown("Priority moments need two dancers sharing at least six reliable "
                                "joints, with at least one differing by 0.15 torso lengths for 0.25 "
                                "seconds. Reference dancers can be compared without receiving a score. "
                                "An empty list means no qualifying event was found with enough data; "
                                "it does not certify synchronization.")
                csv_out = gr.File(label="Download full group per-frame analysis · CSV")

    people_outputs = [people_gallery, track_select, ref_dancer, merge_select, tracks_plot,
                      predicted_overlay, predictor_note, people_headline, review_status, cmp_btn]
    result_outputs = [result_title, cmp_video, timeline, summary_tbl, moments_tbl, cmp_note,
                      csv_out, detail, feedback_tbl, key_moments_tbl, overview, practice_tbl, details_panel, report_dancer, individual_report]
    review_reset_outputs = [tabs, run_title] + people_outputs + result_outputs

    demo.load(_runs_update, outputs=run_dd)
    model_choice.change(
        lambda choice: (gr.update(visible=choice == MODEL_YOLO),
                        gr.update(visible=choice == MODEL_MEDIAPIPE)),
        inputs=model_choice, outputs=[track, num_people],
    )

    # Both ways into step 2 share one chain: set the active run, clear the old run's
    # leftovers and switch tab, then find people.
    run_btn.click(
        run_analysis,
        inputs=[video_in, model_choice, track, num_people],
        outputs=[video_out, json_out, analysis_note, run_dd, active_run],
        show_progress_on=analysis_note,
    ).success(begin_review, inputs=active_run, outputs=review_reset_outputs,
              show_progress="hidden",
              js=_SCROLL_TOP_JS_KEEP
    ).success(run_detect, inputs=active_run, outputs=people_outputs,
              show_progress_on=people_gallery)
    open_btn.click(
        lambda run: run, inputs=run_dd, outputs=active_run,
    ).success(begin_review, inputs=active_run, outputs=review_reset_outputs,
              show_progress="hidden",
              js=_SCROLL_TOP_JS_KEEP
    ).success(run_detect, inputs=active_run, outputs=people_outputs,
              show_progress_on=people_gallery)

    merge_btn.click(run_merge, inputs=[active_run, merge_select], outputs=people_outputs,
                    show_progress_on=people_gallery)
    reset_merge_btn.click(run_reset_merges, inputs=active_run, outputs=people_outputs,
                          show_progress_on=people_gallery)
    track_select.change(update_ref_choices, inputs=[track_select, ref_dancer],
                        outputs=ref_dancer)
    track_select.change(_review_status, inputs=track_select, outputs=review_status)
    ref_choice.change(
        lambda c: (gr.update(visible=c == REF_DANCER), gr.update(visible=c == REF_SOLO),
                   gr.update(visible=c == REF_SOLO), gr.update(visible=c == REF_COMPOSITE)),
        inputs=ref_choice, outputs=[ref_dancer, solo_video, solo_offset, loo_choice])
    save_review_btn.click(save_review, inputs=[active_run, track_select], outputs=review_status)
    render_overlay_btn.click(run_render_overlay, inputs=[active_run, track_select],
                             outputs=predicted_overlay, show_progress_on=predicted_overlay)
    cmp_btn.click(run_compare,
                  inputs=[active_run, track_select, loo_choice, rotate, smooth,
                          ref_choice, ref_dancer, solo_video, solo_offset],
                  outputs=[tabs, result_title, cmp_video, timeline, summary_tbl, moments_tbl,
                           cmp_note, csv_out, t_slider, detail, review_status, feedback_tbl,
                           key_moments_tbl, overview, practice_tbl, details_panel, report_dancer, individual_report],
                  show_progress_on=cmp_progress
                  ).success(lambda: None, js=_SCROLL_TOP_JS)
    report_dancer.change(filter_dancer_report, inputs=[report_dancer, t_slider],
                         outputs=[summary_tbl, moments_tbl, feedback_tbl, key_moments_tbl,
                                  detail, individual_report])
    t_slider.change(frame_detail, inputs=[t_slider, report_dancer], outputs=detail, js=_SEEK_ON_SLIDER_JS)
    moments_tbl.select(jump_to_moment, inputs=moments_tbl, outputs=t_slider
    ).then(frame_detail, inputs=[t_slider, report_dancer], outputs=detail, js=_SEEK_ON_SLIDER_JS)

    feedback_tbl.select(jump_to_moment, inputs=feedback_tbl, outputs=t_slider
    ).then(frame_detail, inputs=[t_slider, report_dancer], outputs=detail, js=_SEEK_ON_SLIDER_JS)
    key_moments_tbl.select(jump_to_moment, inputs=key_moments_tbl, outputs=t_slider
    ).then(frame_detail, inputs=[t_slider, report_dancer], outputs=detail, js=_SEEK_ON_SLIDER_JS)



# Local fonts, warm paper/ink colors, and restrained accents in both color modes.
_THEME = gr.themes.Base(
    primary_hue="stone", secondary_hue="stone", neutral_hue="stone",
    font=["ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "Consolas", "monospace"],
    radius_size="sm",
).set(
    body_background_fill="#f5f3ee", body_background_fill_dark="#171a22",
    body_text_color="#252939", body_text_color_dark="#eeedf5",
    body_text_color_subdued="#697082", body_text_color_subdued_dark="#b2b6c8",
    background_fill_primary="#fffefa", background_fill_primary_dark="#222632",
    background_fill_secondary="#ebeaf1", background_fill_secondary_dark="#2a3040",
    block_background_fill="#fffefa", block_background_fill_dark="#222632",
    block_border_color="#dedde5", block_border_color_dark="#3c4254",
    border_color_primary="#dedde5", border_color_primary_dark="#3c4254",
    input_background_fill="#fffefa", input_background_fill_dark="#222632",
    block_shadow="none", block_shadow_dark="none",
    button_primary_background_fill="#515d87", button_primary_background_fill_dark="#c3cbed",
    button_primary_background_fill_hover="#3f4b73", button_primary_background_fill_hover_dark="#dde2fa",
    button_primary_text_color="#fffefa", button_primary_text_color_dark="#171a22",
    button_primary_border_color="#515d87", button_primary_border_color_dark="#c3cbed",
    button_primary_shadow="none", button_primary_shadow_dark="none",
)

_CSS = """
.gradio-container { max-width: 1240px !important; padding: 32px 28px 56px !important; }
#ddc_header { padding: 24px 28px; margin-bottom: 24px; border: 1px solid var(--border-color-primary); border-radius: 18px; background: var(--background-fill-primary); }
#ddc_header h1 { font-size: clamp(24px, 3vw, 34px); font-weight: 600; letter-spacing: -0.045em; }
#ddc_header p, .ddc_small_note p { color: var(--body-text-color-subdued); }
.tabs > .tab-nav { gap: 8px; padding-bottom: 12px; }
#ddc_result_title h3 { font-size: 13px; font-weight: 400; color: var(--body-text-color-subdued); margin: 8px 0; }
#ddc_score_overview, #ddc_practice_preview { padding: 0; border: 0; background: transparent; }
.ddc-overview { color: var(--body-text-color); display: grid; grid-template-columns: 1.2fr 1fr; gap: 16px; padding: 12px 0 24px; }
.ddc-overall { padding: 24px; border-radius: 18px; background: var(--button-primary-background-fill); color: var(--button-primary-text-color); display: flex; flex-direction: column; justify-content: space-between; gap: 16px; }
.ddc-overview p, .ddc-overview h3 { margin: 0; }
.ddc-overview .ddc-overall p, .ddc-overview .ddc-overall span { color: var(--button-primary-text-color) !important; }
.ddc-eyebrow { font-size: 12px; letter-spacing: .09em; text-transform: uppercase; }
.ddc-overall-score { font-size: clamp(48px, 7vw, 72px); font-weight: 500; line-height: 1.2; letter-spacing: -.06em; font-variant-numeric: tabular-nums; }
.ddc-overall-score span, .ddc-dancer-score span { font-size: 14px; font-weight: 400; letter-spacing: 0; opacity: .75; }
.ddc-score-context { font-size: 13px; line-height: 1.65; }
.ddc-overview .ddc-muted { font-size: 12px; line-height: 1.7; opacity: .8; }
.ddc-dancer-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 12px; }
.ddc-dancer { padding: 20px; border: 1px solid var(--border-color-primary); border-radius: 18px; background: var(--background-fill-primary); }
.ddc-dancer h3 { font-size: 13px; font-weight: 500; }
.ddc-dancer-score { font-size: 32px; line-height: 1.8; letter-spacing: -.035em; }
.ddc-overview > .ddc-coverage-note, .ddc-overview > .ddc-muted { grid-column: 1 / -1; }
#ddc_practice_row { gap: 24px; align-items: flex-start; }
#ddc_practice_col { gap: 14px; }
#ddc_practice_col h3, #ddc_details h3 { font-size: 21px; font-weight: 600; letter-spacing: -.035em; }
#ddc_practice_col p { font-size: 13px; line-height: 1.65; }
.ddc_small_note p { font-size: 12px !important; line-height: 1.65; }
.ddc-moments { list-style: none !important; margin: 0 !important; padding: 0 !important; display: grid; gap: 10px; }
.ddc-moment { width: 100%; display: flex; align-items: flex-start; gap: 14px; padding: 18px; text-align: left; color: var(--body-text-color); background: var(--background-fill-primary); border: 1px solid var(--border-color-primary); border-radius: 16px; cursor: pointer; transition: border-color .15s, background .15s; font: inherit; }
.ddc-moment:hover { border-color: var(--button-primary-background-fill); background: var(--background-fill-secondary); }
.ddc-moment:focus-visible { outline: 2px solid var(--body-text-color); outline-offset: 3px; }
.ddc-moment-number { font-size: 12px; color: var(--body-text-color-subdued); padding-top: 3px; font-variant-numeric: tabular-nums; }
.ddc-moment-copy { display: grid; gap: 6px; flex: 1; min-width: 0; overflow-wrap: anywhere; }
.ddc-moment-copy strong { font-size: 15px; font-weight: 600; }
.ddc-moment-copy > span { font-size: 12px; color: var(--body-text-color-subdued); line-height: 1.5; }
.ddc-moment-range { font-variant-numeric: tabular-nums; }
.ddc-moment-play { display: grid; gap: 12px; justify-items: end; white-space: nowrap; font-size: 12px; font-variant-numeric: tabular-nums; }
.ddc-moment-play > span:last-child { font-size: 20px; }
.ddc-empty { padding: 20px; border: 1px dashed var(--border-color-primary); border-radius: 16px; color: var(--body-text-color-subdued); }
#ddc_cmp_video { border-radius: 18px; overflow: hidden; }
#ddc_cmp_video video { max-height: 58vh; width: 100%; object-fit: contain; }
#ddc_pred_video video, #ddc_in_video video { max-height: 60vh; object-fit: contain; }
#ddc_video_col { position: sticky; top: 12px; align-self: flex-start; }
#ddc_details { margin-top: 24px; border: 1px solid var(--border-color-primary); border-radius: 16px; }
#ddc_details > button { padding: 20px; font-size: 15px; }
#ddc_details > button:focus-visible { outline: 2px solid var(--body-text-color); outline-offset: 3px; }
#ddc_timeline .ddc-playhead { position: absolute; width: 2px; margin-left: -1px; background: #8895c5; opacity: .85; pointer-events: none; z-index: 5; display: none; }
@media (max-width: 700px) {
  .gradio-container { padding: 20px 16px 40px !important; }
  #ddc_header { padding: 20px; }
  .ddc-overview { grid-template-columns: 1fr; }
  .ddc-overall { padding: 22px; }
  #ddc_video_col { position: static; }
  #ddc_practice_row { gap: 16px; }
}
"""

# Video <-> timeline sync for the Results step. gr.Plot has no click/select event of its
# own (only 'change') and nothing in Gradio reports the playing video's time without a
# server round trip, so this is done client-side against the DOM:
#   * click on the Plotly timeline -> seek the video, move the "Inspect time" slider
#     (firing its native release event so `frame_detail` runs the normal Gradio way)
#   * video playing/seeking -> a playhead line drawn over the plot follows it. It is a
#     positioned <div>, not a Plotly shape, so the heatmap isn't redrawn every frame.
#   * video paused or scrubbed -> the slider (and so the frame detail) follows.
# Slider -> video goes through `_SEEK_ON_SLIDER_JS` on the slider's own listener.
# The figure's div and the <video> are recreated on every "Compare" run, so this polls
# for them (cheap) rather than hooking once. Must be injected via `launch(head=...)` --
# `demo.load(js=...)` with no `fn` is a documented no-op in Gradio 6, it never actually runs.
_SYNC_JS = """
<script>
(() => {
  const plot = () => document.querySelector('#ddc_timeline .js-plotly-plot');
  const video = () => document.querySelector('#ddc_cmp_video video');

  window.ddcSeek = (t) => {
    const v = video();
    t = Number(t);
    if (!v || !isFinite(t)) return;
    // Tolerance stops slider -> video -> slider echoing (the slider snaps to 0.05 s).
    if (Math.abs(v.currentTime - t) > 0.06) v.currentTime = t;
    drawPlayhead();
  };

  function setSlider(t) {
    const input = document.querySelector('#ddc_t_slider input[type=range]');
    if (!input || Math.abs(Number(input.value) - t) < 0.03) return;
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(input, String(t));
    // Gradio's Slider only round-trips to Python on release (pointerup/mouseup),
    // not on a plain 'input'/'change' dispatch -- fire both so the value sticks
    // and the frame-detail callback actually runs.
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
    input.dispatchEvent(new PointerEvent('pointerup', {bubbles: true}));
    input.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
  }

  document.addEventListener('click', (event) => {
    const card = event.target.closest?.('button[data-ddc-time]');
    if (!card || !card.closest('#ddc_practice_preview')) return;
    const t = Number(card.dataset.ddcTime);
    if (!Number.isFinite(t) || t < 0) return;
    window.ddcSeek(t);
    setSlider(t);
  });

  function drawPlayhead() {
    const gd = plot(), v = video();
    if (!gd || !gd._fullLayout || !gd._fullLayout.xaxis) return;
    const host = gd.querySelector('.svg-container') || gd;
    let line = host.querySelector('.ddc-playhead');
    if (!line) {
      line = document.createElement('div');
      line.className = 'ddc-playhead';
      host.appendChild(line);
    }
    const fl = gd._fullLayout, xa = fl.xaxis;
    const ys = Object.keys(fl).filter((k) => /^yaxis\\d*$/.test(k)).map((k) => fl[k]);
    const x = v ? xa.l2p(v.currentTime) : NaN;
    if (!isFinite(x) || x < 0 || x > xa._length || !ys.length) {
      line.style.display = 'none';
      return;
    }
    const top = Math.min(...ys.map((a) => a._offset));
    const bottom = Math.max(...ys.map((a) => a._offset + a._length));
    line.style.display = 'block';
    line.style.left = (xa._offset + x) + 'px';
    line.style.top = top + 'px';
    line.style.height = (bottom - top) + 'px';
  }

  function follow() {
    const v = video();
    drawPlayhead();
    if (v && !v.paused && !v.ended) requestAnimationFrame(follow);
  }

  function attach() {
    const gd = plot();
    if (gd && gd.on && !gd._ddcHooked) {
      gd._ddcHooked = true;
      gd.on('plotly_click', (ev) => {
        if (!ev.points || !ev.points.length) return;
        const t = Number(ev.points[0].x);
        window.ddcSeek(t);
        setSlider(t);
      });
      gd.on('plotly_relayout', drawPlayhead);
      gd.on('plotly_afterplot', drawPlayhead);
      drawPlayhead();
    }
    const v = video();
    if (v && !v._ddcHooked) {
      v._ddcHooked = true;
      v.addEventListener('play', follow);
      v.addEventListener('timeupdate', drawPlayhead);
      v.addEventListener('seeked', () => { drawPlayhead(); if (v.paused) setSlider(v.currentTime); });
      v.addEventListener('pause', () => setSlider(v.currentTime));
      drawPlayhead();
    }
  }
  setInterval(attach, 500);
  window.addEventListener('resize', drawPlayhead);
})();
</script>
"""

if __name__ == "__main__":
    RUNS_DIR.mkdir(exist_ok=True)
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)), share=False, show_error=True,
                head=_SYNC_JS, css=_CSS, theme=_THEME, allowed_paths=[str(RUNS_DIR)])
