"""Video-in -> tracking JSON + overlay video pipeline.

Wraps the detection logic already in pose_demo.py (YOLO) and
mediapipe_pose_demo.py (MediaPipe) to process a full video file end-to-end,
with no live preview window: read every frame, run pose detection, write an
annotated overlay video, and write one JSON Lines record per frame. Used by
app.py (the web UI) and reusable standalone.

Unlike the live demo scripts (which timestamp records by wall-clock time),
records here are timestamped by position in the video (frame_idx / fps),
since that is what matters for post-hoc analysis of a recorded clip.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2

import pose_demo
import mediapipe_pose_demo as mpd


def _reencode_h264(raw_path: Path, final_path: Path) -> Path:
    """Re-encode to H.264/yuv420p with ffmpeg so browsers can play it inline.

    Falls back to the raw (mp4v) file if ffmpeg isn't available.
    """
    if shutil.which("ffmpeg") is None:
        raw_path.replace(final_path)
        return final_path
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw_path),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(final_path)],
        check=True,
    )
    raw_path.unlink()
    return final_path


def analyze_video_yolo(
    video_path: str,
    out_jsonl: str,
    out_video: str,
    model: str = "yolo11n-pose.pt",
    device: str = "cpu",
    conf: float = 0.5,
    imgsz: int = 640,
    track: bool = True,
) -> dict:
    """Run YOLO pose over every frame of video_path. Returns a summary dict."""
    yolo = pose_demo.YOLO(model)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = Path(out_video)
    raw_video = out_video.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    predict_kwargs = dict(conf=conf, imgsz=imgsz, device=device, verbose=False)
    frame_idx = 0
    max_people = 0
    try:
        with open(out_jsonl, "w") as f:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if track:
                    results = yolo.track(frame, persist=True, **predict_kwargs)
                else:
                    results = yolo.predict(frame, **predict_kwargs)
                result = results[0]

                record = pose_demo.frame_to_record(frame_idx, frame_idx / fps, result)
                f.write(json.dumps(record) + "\n")
                max_people = max(max_people, len(record["people"]))

                writer.write(result.plot())
                frame_idx += 1
    finally:
        cap.release()
        writer.release()

    _reencode_h264(raw_video, out_video)
    return {"frames": frame_idx, "fps": round(fps, 2), "max_people_detected": max_people}


def analyze_video_mediapipe(
    video_path: str,
    out_jsonl: str,
    out_video: str,
    num_poses: int = 1,
    num_hands: int = 2,
    min_conf: float = 0.5,
) -> dict:
    """Run MediaPipe pose+hands over every frame of video_path. Returns a summary dict."""
    pose_model = mpd.ensure_model(mpd.MODELS_DIR / "pose_landmarker_lite.task", mpd.POSE_MODEL_URL)
    hand_model = mpd.ensure_model(mpd.MODELS_DIR / "hand_landmarker.task", mpd.HAND_MODEL_URL)

    pose_landmarker = mpd.PoseLandmarker.create_from_options(mpd.PoseLandmarkerOptions(
        base_options=mpd.BaseOptions(model_asset_path=str(pose_model)),
        running_mode=mpd.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_conf,
    ))
    hand_landmarker = mpd.HandLandmarker.create_from_options(mpd.HandLandmarkerOptions(
        base_options=mpd.BaseOptions(model_asset_path=str(hand_model)),
        running_mode=mpd.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_conf,
    ))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path!r}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = Path(out_video)
    raw_video = out_video.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    max_people = 0
    max_hands = 0
    try:
        with open(out_jsonl, "w") as f:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mpd.mp.Image(image_format=mpd.mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(frame_idx / fps * 1000)

                pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

                record = {
                    "frame": frame_idx,
                    "t": round(frame_idx / fps, 4),
                    "poses": mpd.pose_to_record(pose_result, width, height),
                    "hands": mpd.hands_to_record(hand_result, width, height),
                }
                f.write(json.dumps(record) + "\n")
                max_people = max(max_people, len(record["poses"]))
                max_hands = max(max_hands, len(record["hands"]))

                for pose in record["poses"]:
                    pts = [(int(x), int(y)) for x, y, *_ in pose["landmarks"].values()]
                    mpd.draw_landmarks(frame, pts, mpd.POSE_CONNECTIONS, (0, 255, 0))
                for hand in record["hands"]:
                    pts = [(int(x), int(y)) for x, y, *_ in hand["landmarks"].values()]
                    color = (255, 0, 0) if hand["handedness"] == "Left" else (0, 0, 255)
                    mpd.draw_landmarks(frame, pts, mpd.HAND_CONNECTIONS, color)
                writer.write(frame)
                frame_idx += 1
    finally:
        cap.release()
        writer.release()
        pose_landmarker.close()
        hand_landmarker.close()

    _reencode_h264(raw_video, out_video)
    return {
        "frames": frame_idx, "fps": round(fps, 2),
        "max_people_detected": max_people, "max_hands_detected": max_hands,
    }
