"""Local web UI: upload a video, get back tracking JSON + an overlay video.

Wraps video_pipeline.py (YOLO or MediaPipe) in a small Gradio app for the
video-analysis workflow -- upload a clip once, get the annotated video and
the per-frame keypoint JSONL to feed into downstream processing.

Usage:
    python app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from video_pipeline import analyze_video_mediapipe, analyze_video_yolo

RUNS_DIR = Path(__file__).parent / "runs"


def run_analysis(video_path: str, model_choice: str, track: bool, progress=gr.Progress()):
    if not video_path:
        raise gr.Error("Upload a video first.")

    progress(0, desc="Starting...")
    run_dir = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = run_dir / "tracking.jsonl"
    out_video = run_dir / "overlay.mp4"

    progress(0.1, desc=f"Running {model_choice} pose detection...")
    if model_choice == "YOLO (body only, fast)":
        summary = analyze_video_yolo(video_path, str(out_jsonl), str(out_video), track=track)
    else:
        summary = analyze_video_mediapipe(video_path, str(out_jsonl), str(out_video))

    progress(1.0, desc="Done")
    summary_text = "\n".join(f"{k}: {v}" for k, v in summary.items())
    return str(out_video), str(out_jsonl), summary_text


with gr.Blocks(title="DanceDanceConvolution - Pose Tracking") as demo:
    gr.Markdown(
        "# Pose tracking\n"
        "Upload a video to get an annotated overlay and a JSON Lines file of "
        "per-frame keypoints for downstream processing."
    )
    with gr.Row():
        with gr.Column():
            video_in = gr.Video(label="Input video")
            model_choice = gr.Radio(
                ["YOLO (body only, fast)", "MediaPipe (body + hands)"],
                value="YOLO (body only, fast)", label="Model",
            )
            track = gr.Checkbox(
                value=True, label="Persistent per-person IDs (YOLO ByteTrack)",
                visible=True,
            )
            run_btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            video_out = gr.Video(label="Overlay video")
            json_out = gr.File(label="Tracking JSONL")
            summary_out = gr.Textbox(label="Summary", lines=4)

    model_choice.change(
        lambda choice: gr.update(visible=choice.startswith("YOLO")),
        inputs=model_choice, outputs=track,
    )
    run_btn.click(
        run_analysis,
        inputs=[video_in, model_choice, track],
        outputs=[video_out, json_out, summary_out],
    )


if __name__ == "__main__":
    RUNS_DIR.mkdir(exist_ok=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)
