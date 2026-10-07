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
from ddc.plots import timeline_figure
from ddc.skeleton import BODY
from video_pipeline import analyze_video_mediapipe, analyze_video_rtmw, analyze_video_yolo

MODEL_YOLO = "YOLO (body only, fast)"
MODEL_MEDIAPIPE = "MediaPipe (body + hands, hands unreliable)"
MODEL_RTMW = "RTMW (body + hands, multi-person, recommended)"

REF_COMPOSITE = "Composite of all dancers"
REF_DANCER = "One of the dancers in this video"
REF_SOLO = "Separate solo video"
REF_MODES = {REF_COMPOSITE: "composite", REF_DANCER: "dancer", REF_SOLO: "solo"}

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
    gallery = [(tr.thumbnail, tr.label) for tr in tracks]
    choices = [tr.label for tr in tracks]
    return gallery, gr.update(choices=choices, value=choices), gr.update(choices=choices, value=None)


def update_ref_choices(track_choice, current):
    """Reference-dancer dropdown offers only the people currently ticked as dancers."""
    choices = list(track_choice or [])
    return gr.update(choices=choices, value=current if current in choices else None)


def run_compare(run_name, source_video, track_choice, loo_choice, rotate, smooth,
                ref_choice, ref_dancer, solo_video, solo_offset, progress=gr.Progress()):
    if not run_name:
        raise gr.Error("Pick a run first (run an analysis on the Pose tracking tab).")
    labels = _LAST.get("track_labels")
    if not labels or _LAST.get("track_run") != run_name:
        raise gr.Error('Click "1. Find people" first, then uncheck anyone who isn\'t a dancer.')
    if not track_choice:
        raise gr.Error("Select at least one person as a dancer.")
    selected = [labels.index(c) for c in track_choice]
    loo = {"Auto (on for 3+ dancers)": None, "On": True, "Off": False}[loo_choice]
    mode = REF_MODES[ref_choice]
    ref_idx = None
    if mode == "dancer":
        if ref_dancer not in track_choice:
            raise gr.Error("Pick which dancer is the reference.")
        ref_idx = sorted(selected).index(labels.index(ref_dancer))
    elif mode == "solo" and not solo_video:
        raise gr.Error("Upload a solo reference video.")
    progress(0.1, desc="Analyzing solo reference video..." if mode == "solo"
             else "Building reference and deviation...")
    try:
        res = analyze_run(RUNS_DIR / run_name, source_video, leave_one_out=loo,
                          rotate=rotate, smooth=int(smooth), selected_tracks=selected,
                          reference_mode=mode, reference_dancer=ref_idx,
                          solo_video=solo_video, solo_offset=float(solo_offset or 0))
    except ValueError as e:
        raise gr.Error(str(e))
    _LAST["res"] = res
    progress(0.9, desc="Plotting...")
    rows = [[s["dancer"], s["frames_present"], round(s["mean_dev"], 3), s["worst_joint"],
             "; ".join(f"{t}s ({j})" for t, _, j in s["worst_moments"])] for s in res.summary]
    note = ""
    if res.n_tracks == 2 and mode == "composite":
        note = ("Only 2 dancers: the composite is their midpoint, so deviation is symmetric "
                "(it measures how far apart they are, not who is off).")
    max_t = float(res.t[-1])
    return (res.overlay_path, timeline_figure(res.dev, res.t), rows, note, res.csv_path,
            gr.update(maximum=max_t, value=0))


def frame_detail(t_sec):
    res = _LAST.get("res")
    if res is None:
        return "Run an analysis first."
    f = int(np.argmin(np.abs(res.t - t_sec)))
    lines = [f"t={res.t[f]:.2f}s (frame {f}), {int(res.dev.n_present[f])} dancer(s) present"]
    for d in range(res.n_tracks):
        if d == res.dev.reference_track:
            lines.append(f"Dancer {d + 1}: reference")
            continue
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
                "Measures how far each dancer is from a reference pose, frame by frame: a "
                "consensus of all dancers, one chosen dancer, or a separate solo video. "
                "Skeleton colour: green = close, red = far (distance in torso lengths). "
                "White ghost = reference."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    run_dd = gr.Dropdown(list_runs(), value=(list_runs() or [None])[0],
                                         label="Run")
                    refresh_btn = gr.Button("Refresh runs", size="sm")
                    src_video = gr.Video(
                        label="Original video (optional; otherwise the run's overlay is dimmed as background)")

                    gr.Markdown(
                        "**1. Find people.** The tracker keeps anyone visible for more than "
                        "~15 frames -- it doesn't know who's actually dancing. Uncheck anyone "
                        "here who's a bystander, audience, or otherwise not a dancer before "
                        "comparing; unchecked people are excluded from the composite entirely."
                    )
                    detect_btn = gr.Button("1. Find people")
                    people_gallery = gr.Gallery(label="Detected people", columns=4, height=180,
                                                object_fit="contain")
                    track_select = gr.CheckboxGroup(label="Include as dancer", choices=[])

                    ref_choice = gr.Radio([REF_COMPOSITE, REF_DANCER, REF_SOLO], value=REF_COMPOSITE,
                                          label="Deviation reference")
                    ref_dancer = gr.Dropdown(
                        [], value=None, visible=False,
                        label="Reference dancer (everyone else is scored against this person)")
                    solo_video = gr.Video(label="Solo reference video", visible=False)
                    solo_offset = gr.Number(
                        value=0, visible=False,
                        label="Solo offset (s): seconds into the solo video that line up with the "
                              "start of the group video (negative = solo starts later)")
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
                             outputs=[people_gallery, track_select, ref_dancer])
            track_select.change(update_ref_choices, inputs=[track_select, ref_dancer],
                                outputs=ref_dancer)
            ref_choice.change(
                lambda c: (gr.update(visible=c == REF_DANCER), gr.update(visible=c == REF_SOLO),
                           gr.update(visible=c == REF_SOLO), gr.update(visible=c == REF_COMPOSITE)),
                inputs=ref_choice, outputs=[ref_dancer, solo_video, solo_offset, loo_choice])
            cmp_btn.click(run_compare,
                          inputs=[run_dd, src_video, track_select, loo_choice, rotate, smooth,
                                  ref_choice, ref_dancer, solo_video, solo_offset],
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
