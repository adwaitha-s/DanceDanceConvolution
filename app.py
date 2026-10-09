"""Local web UI: upload a video, get back tracking JSON + an overlay video.

Wraps video_pipeline.py (YOLO, MediaPipe, or RTMW) in a small Gradio app for
the video-analysis workflow -- upload a clip once, get the annotated video
and the per-frame keypoint JSONL to feed into downstream processing.

Usage:
    python app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr
import numpy as np

from ddc.analyze import analyze_run, detect_tracks
from ddc.feature_inventory import render_classifier_overlay
from ddc.feature_registry import (predict_run, rebuild_registry_workbook,
                                  validate_checklist)
from ddc.plots import timeline_figure
from ddc.skeleton import BODY
from video_pipeline import analyze_video_mediapipe, analyze_video_rtmw, analyze_video_yolo

MODEL_YOLO = "YOLO (body only, fast)"
MODEL_MEDIAPIPE = "MediaPipe (body + hands, hands unreliable)"
MODEL_RTMW = "RTMW (body + hands, multi-person, recommended)"

RUNS_DIR = Path(__file__).parent / "runs"


def run_analysis(video_path: str, model_choice: str, track: bool, num_people: int,
                  progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video first.")

    progress(0, desc="Starting...")
    run_dir = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = run_dir / "tracking.jsonl"
    out_video = run_dir / "overlay.mp4"

    progress(0.1, desc=f"Running {model_choice} pose detection...")
    if model_choice == MODEL_YOLO:
        summary = analyze_video_yolo(video_path, str(out_jsonl), str(out_video), track=track)
    elif model_choice == MODEL_RTMW:
        summary = analyze_video_rtmw(video_path, str(out_jsonl), str(out_video))
    else:
        summary = analyze_video_mediapipe(
            video_path, str(out_jsonl), str(out_video),
            num_poses=num_people, num_hands=num_people * 2,
        )

    progress(1.0, desc="Done")
    summary_text = "\n".join(f"{k}: {v}" for k, v in summary.items())
    return str(out_video), str(out_jsonl), summary_text


def list_runs() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted((p.name for p in RUNS_DIR.iterdir() if (p / "tracking.jsonl").exists()),
                  reverse=True)


_LAST: dict = {}


def run_detect(run_name, source_video, progress=gr.Progress()):
    """Step 1: find every stabilized person track in the run and show a thumbnail
    for each, so the user can rule out bystanders/audience before anything is
    scored -- the tracker has no notion of "dancer" vs. "person on camera",
    it keeps anyone visible for more than ~15 frames.
    """
    if not run_name:
        raise gr.Error("Pick a run first (run an analysis on the Pose tracking tab).")
    progress(0.3, desc="Finding people in the clip...")
    try:
        tracks = detect_tracks(RUNS_DIR / run_name, source_video)
    except ValueError as e:
        raise gr.Error(str(e))
    _LAST.clear()
    _LAST["track_run"] = run_name
    _LAST["track_labels"] = [tr.label for tr in tracks]
    _LAST["track_ids"] = {tr.label: tr.track_id for tr in tracks}
    gallery = [(tr.thumbnail, tr.label) for tr in tracks]
    choices = [tr.label for tr in tracks]
    try:
        progress(.55, desc="Scoring saved pose tracks from validated runs...")
        model, predictions = predict_run(RUNS_DIR, run_name)
        prediction_map = {item["track_id"]: item for item in predictions}
        selected = [tr.label for tr in tracks
                    if prediction_map.get(tr.track_id, {}).get("predicted_label") == "dancer"]
        # The predicted-dancer overlay stays in this tab.  It is generated from
        # existing JSON poses; no detector call is made here.
        rendered = render_classifier_overlay(RUNS_DIR / run_name, show_non_dancers=False,
                                             video_path=source_video, predictions=predictions)
        try:
            # Add a Labels-style worksheet for this run immediately. Its local
            # top-five weights populate once the checklist is validated.
            rebuild_registry_workbook(RUNS_DIR)
            workbook_status = " Registry workbook refreshed with this run's Labels sheet."
        except RuntimeError as workbook_error:
            workbook_status = f" Registry workbook refresh needs attention: {workbook_error}"
        _LAST["predictions"] = predictions
        _LAST["model"] = model
        _LAST["predicted_selected_ids"] = frozenset(
            tr.track_id for tr in tracks
            if prediction_map.get(tr.track_id, {}).get("predicted_label") == "dancer")
        weights = "; ".join(f"{item['feature_name']}: {item['standardized_coefficient']:+.3f}"
                            for item in model["selected_features"])
        note = (f"Automatic checklist: {len(selected)} predicted dancer(s). Top-five model has "
                f"{model['labelled_tracks']} validated track labels across {len(model['runs'])} run(s). "
                f"The checklist uses exactly these five standardized weights: {weights}. "
                "Leave it unchanged and Compare to validate every prediction, or save edits to record corrections." +
                workbook_status)
        progress(1, desc="Predicted dancer overlay ready")
        return gallery, gr.update(choices=choices, value=selected), rendered["overlay"], note
    except ValueError as e:
        # A first-ever run cannot be predicted until both classes have two human
        # labels.  Keep the manual fallback explicit rather than silently training.
        _LAST["predictions"] = []
        return (gallery, gr.update(choices=choices, value=choices), None,
                f"Automatic screening needs validated examples first: {e}. All tracks are selected for review.")


def save_review(run_name, track_choice, progress=gr.Progress()):
    """Persist the current checklist without starting pose comparison."""
    labels = _LAST.get("track_labels")
    if not labels or _LAST.get("track_run") != run_name:
        raise gr.Error('Click "1. Find people" before saving the reviewed checklist.')
    selected = [_LAST["track_ids"][choice] for choice in (track_choice or [])]
    progress(.25, desc="Saving reviewed dancer labels...")
    try:
        outcome = validate_checklist(RUNS_DIR, run_name, selected, _LAST.get("predictions", []))
        _LAST["saved_selection"] = frozenset(selected)
        _LAST["saved_validation"] = outcome["validation"]
        workbook = rebuild_registry_workbook(RUNS_DIR)
    except (RuntimeError, ValueError) as e:
        raise gr.Error(f"Could not save the reviewed checklist: {e}")
    progress(1, desc="Reviewed labels saved")
    validation = outcome["validation"]
    state = validation["status"].replace("_", " ")
    return (f"**Reviewed labels saved.** Predictor labels {state}; "
            f"{len(validation['corrected_track_ids'])} track correction(s) recorded. "
            f"Updated {workbook.name}.")


def run_compare(run_name, source_video, track_choice, loo_choice, rotate, smooth,
                progress=gr.Progress()):
    if not run_name:
        raise gr.Error("Pick a run first (run an analysis on the Pose tracking tab).")
    labels = _LAST.get("track_labels")
    if not labels or _LAST.get("track_run") != run_name:
        raise gr.Error('Click "1. Find people" first, then uncheck anyone who isn\'t a dancer.')
    if not track_choice:
        raise gr.Error("Select at least one person as a dancer.")
    selected = [_LAST["track_ids"][c] for c in track_choice]
    loo = {"Auto (on for 3+ dancers)": None, "On": True, "Off": False}[loo_choice]
    progress(0.1, desc="Building composite and deviation...")
    try:
        res = analyze_run(RUNS_DIR / run_name, source_video, leave_one_out=loo,
                          rotate=rotate, smooth=int(smooth), selected_tracks=selected)
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
                outcome = validate_checklist(RUNS_DIR, run_name, selected, _LAST["predictions"])
                _LAST["saved_selection"] = current
                _LAST["saved_validation"] = outcome["validation"]
                workbook = rebuild_registry_workbook(RUNS_DIR)
                validation_note = (" Predictor labels validated correct; 0 track corrections stored. "
                                   f"Accumulating workbook updated: {workbook.name}.")
            except (RuntimeError, ValueError) as e:
                validation_note = f" Checklist validation could not be saved: {e}"
        elif _LAST.get("saved_selection") != current:
            validation_note = (" Checklist changes have not been saved. Click **Save reviewed labels** "
                               "to update the registry's Reviewed label, Validation status, and "
                               "Correction recorded columns.")
        else:
            validation = _LAST.get("saved_validation", {})
            validation_note = (f" Reviewed corrections are saved: "
                               f"{len(validation.get('corrected_track_ids', []))} track correction(s).")
    progress(0.9, desc="Plotting...")
    rows = [[s["dancer"], s["frames_present"], round(s["mean_dev"], 3), s["worst_joint"],
             "; ".join(f"{t}s ({j})" for t, _, j in s["worst_moments"])] for s in res.summary]
    note = ""
    if res.n_tracks == 2:
        note = ("Only 2 dancers: the composite is their midpoint, so deviation is symmetric "
                "(it measures how far apart they are, not who is off).")
    max_t = float(res.t[-1])
    return (res.overlay_path, timeline_figure(res.dev, res.t), rows, note + validation_note, res.csv_path,
            gr.update(maximum=max_t, value=0))


def frame_detail(t_sec):
    res = _LAST.get("res")
    if res is None:
        return "Run an analysis first."
    f = int(np.argmin(np.abs(res.t - t_sec)))
    lines = [f"t={res.t[f]:.2f}s (frame {f}), {int(res.dev.n_present[f])} dancer(s) present"]
    for d in range(res.n_tracks):
        j = res.dev.joint[f, d]
        if not np.isfinite(j).any():
            lines.append(f"Dancer {d + 1}: not visible")
            continue
        top = np.argsort(-np.nan_to_num(j, nan=-1))[:3]
        lines.append(f"Dancer {d + 1}: score {res.dev.score[f, d]:.2f}; worst: " +
                     ", ".join(f"{BODY[i]} {j[i]:.2f}" for i in top))
    return "\n".join(lines)


with gr.Blocks(title="DanceDanceConvolution - Pose Tracking") as demo:
    with gr.Tabs():
        with gr.Tab("Pose tracking"):
            gr.Markdown(
                "# Pose tracking\n"
                "Upload a video to get an annotated overlay and a JSON Lines file of "
                "per-frame keypoints for downstream processing."
            )
            with gr.Row():
                with gr.Column():
                    video_in = gr.Video(label="Input video")
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
                    run_btn = gr.Button("Analyze", variant="primary")
                with gr.Column():
                    video_out = gr.Video(label="Overlay video")
                    json_out = gr.File(label="Tracking JSONL")
                    summary_out = gr.Textbox(label="Summary", lines=4)

            model_choice.change(
                lambda choice: (gr.update(visible=choice == MODEL_YOLO),
                                gr.update(visible=choice == MODEL_MEDIAPIPE)),
                inputs=model_choice, outputs=[track, num_people],
            )
            run_btn.click(
                run_analysis,
                inputs=[video_in, model_choice, track, num_people],
                outputs=[video_out, json_out, summary_out],
            )

        with gr.Tab("Compare"):
            gr.Markdown(
                "# Composite pose and deviation\n"
                "Builds a per-frame consensus pose from all dancers, then shows how far "
                "each dancer is from it. Skeleton colour: green = close, red = far "
                "(distance in torso lengths). White ghost = composite."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    run_dd = gr.Dropdown(list_runs(), value=(list_runs() or [None])[0],
                                         label="Run")
                    refresh_btn = gr.Button("Refresh runs", size="sm")
                    src_video = gr.Video(
                        label="Original video (optional; otherwise the run's overlay is dimmed as background)")

                    gr.Markdown(
                        "**1. Find and predict dancers.** The checklist is automatically selected "
                        "from the accumulating feature model. Non-dancers are deselected in the "
                        "same tab and hidden from the predicted-dancer pose overlay. Leave it alone "
                        "to validate the labels; make changes to correct and retrain the model."
                    )
                    detect_btn = gr.Button("1. Find people")
                    people_gallery = gr.Gallery(label="Detected people", columns=4, height=180,
                                                object_fit="contain")
                    with gr.Row():
                        track_select = gr.CheckboxGroup(label="Include as dancer", choices=[], scale=4)
                        save_review_btn = gr.Button("Save reviewed labels", size="sm", scale=1)
                    predictor_note = gr.Markdown()
                    predicted_overlay = gr.Video(label="Predicted dancers pose overlay")

                    loo_choice = gr.Radio(["Auto (on for 3+ dancers)", "On", "Off"],
                                          value="Auto (on for 3+ dancers)",
                                          label="Leave-one-out composite")
                    rotate = gr.Checkbox(value=False, label="Normalize torso rotation")
                    smooth = gr.Slider(1, 15, step=2, value=5, label="Composite smoothing (frames)")
                    cmp_btn = gr.Button("2. Compare", variant="primary")
                with gr.Column(scale=2):
                    cmp_video = gr.Video(label="Deviation overlay", elem_id="ddc_cmp_video")
                    cmp_note = gr.Markdown()
            timeline = gr.Plot(label="Timeline (click to jump the video to that moment)",
                               elem_id="ddc_timeline")
            summary_tbl = gr.Dataframe(
                headers=["Dancer", "Frames present", "Mean deviation", "Worst joint", "Worst moments"],
                interactive=False)
            with gr.Row():
                t_slider = gr.Slider(0, 1, step=0.05, value=0, label="Inspect time (s)",
                                     elem_id="ddc_t_slider")
                detail = gr.Textbox(label="Frame detail", lines=4)
            csv_out = gr.File(label="Per-frame deviation CSV")

            refresh_btn.click(lambda: gr.update(choices=list_runs()), outputs=run_dd)
            detect_btn.click(run_detect, inputs=[run_dd, src_video],
                             outputs=[people_gallery, track_select, predicted_overlay, predictor_note])
            save_review_btn.click(save_review, inputs=[run_dd, track_select], outputs=predictor_note)
            cmp_btn.click(run_compare,
                          inputs=[run_dd, src_video, track_select, loo_choice, rotate, smooth],
                          outputs=[cmp_video, timeline, summary_tbl, cmp_note, csv_out, t_slider])
            t_slider.change(frame_detail, inputs=t_slider, outputs=detail)


# Click-to-seek for the Compare tab: gr.Plot has no click/select event of its own
# (only 'change'), so a plain click on the Plotly timeline can't drive a Python
# callback directly. Instead this hooks Plotly's own 'plotly_click' event on the
# figure's div and, client-side only (no server round trip, so it's instant):
#   1. seeks the deviation-overlay video to the clicked time via `<video>.currentTime`
#   2. moves the "Inspect time" slider to match and fires its native release event,
#      which *does* reach the slider's own `.change()` handler (`frame_detail`) the
#      normal Gradio way, so the worst-joint readout below the timeline updates too.
# The figure's div is recreated on every "Compare" run, so this polls for it (cheap)
# rather than hooking once. Must be injected via `launch(head=...)` -- `demo.load(js=...)`
# with no `fn` is a documented no-op in Gradio 6, it never actually runs.
_CLICK_TO_SEEK_JS = """
<script>
function ddcAttachTimelineClick() {
  const gd = document.querySelector('#ddc_timeline .js-plotly-plot');
  if (!gd || gd._ddcHooked || !gd.on) return;
  gd._ddcHooked = true;
  gd.on('plotly_click', (ev) => {
    if (!ev.points || !ev.points.length) return;
    const tClicked = ev.points[0].x;
    const video = document.querySelector('#ddc_cmp_video video');
    if (video) video.currentTime = tClicked;
    const input = document.querySelector('#ddc_t_slider input[type=range]');
    if (!input) return;
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setter.call(input, String(tClicked));
    // Gradio's Slider only round-trips to Python on release (pointerup/mouseup),
    // not on a plain 'input'/'change' dispatch -- fire both so the value sticks
    // and the frame-detail callback actually runs.
    input.dispatchEvent(new Event('input', {bubbles: true}));
    input.dispatchEvent(new Event('change', {bubbles: true}));
    input.dispatchEvent(new PointerEvent('pointerup', {bubbles: true}));
    input.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
  });
}
setInterval(ddcAttachTimelineClick, 500);
</script>
"""

if __name__ == "__main__":
    RUNS_DIR.mkdir(exist_ok=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True,
                head=_CLICK_TO_SEEK_JS)
