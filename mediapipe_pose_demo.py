"""MediaPipe pose + hand landmark demo -- like pose_demo.py, but with hands.

Ultralytics YOLO pose only outputs the 17 COCO body keypoints (no fingers).
This script uses MediaPipe's PoseLandmarker (33 body points) and
HandLandmarker (21 points per hand) together to also track hand/finger
landmarks, and streams both as JSON Lines for downstream processing, same
pattern as pose_demo.py.

Model files (~13 MB total) are auto-downloaded into ./models on first run.

Usage:
    python mediapipe_pose_demo.py                          # webcam, writes pose_stream.jsonl
    python mediapipe_pose_demo.py --source video.mp4 --out out.jsonl
    python mediapipe_pose_demo.py --source photo.jpg --no-display
    python mediapipe_pose_demo.py --stdout --no-display | python examples/example_consumer.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions,
    RunningMode,
)

MODELS_DIR = Path(__file__).parent / "models"
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                   "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                   "hand_landmarker/float16/latest/hand_landmarker.task")

POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

HAND_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_finger_mcp", "index_finger_pip", "index_finger_dip", "index_finger_tip",
    "middle_finger_mcp", "middle_finger_pip", "middle_finger_dip", "middle_finger_tip",
    "ring_finger_mcp", "ring_finger_pip", "ring_finger_dip", "ring_finger_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def ensure_model(path: Path, url: str) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {path.name} ...", file=sys.stderr)
        urllib.request.urlretrieve(url, path)
    return path


def draw_landmarks(frame, points_px, connections, color):
    for a, b in connections:
        if a < len(points_px) and b < len(points_px):
            cv2.line(frame, points_px[a], points_px[b], color, 2)
    for x, y in points_px:
        cv2.circle(frame, (x, y), 3, color, -1)


def pose_to_record(result, width: int, height: int) -> list[dict]:
    poses = []
    for i, landmarks in enumerate(result.pose_landmarks):
        entry = {"id": i, "landmarks": {}}
        for name, lm in zip(POSE_LANDMARK_NAMES, landmarks):
            entry["landmarks"][name] = [
                round(lm.x * width, 1), round(lm.y * height, 1),
                round(lm.z, 4), round(lm.visibility, 3),
            ]
        poses.append(entry)
    return poses


def hands_to_record(result, width: int, height: int) -> list[dict]:
    hands = []
    for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        top = handedness[0]
        entry = {
            "handedness": top.category_name,
            "score": round(top.score, 3),
            "landmarks": {},
        }
        for name, lm in zip(HAND_LANDMARK_NAMES, landmarks):
            entry["landmarks"][name] = [
                round(lm.x * width, 1), round(lm.y * height, 1), round(lm.z, 4),
            ]
        hands.append(entry)
    return hands


def open_source(source: str):
    try:
        return int(source)
    except ValueError:
        return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0",
                        help="camera index, video file, image file, or stream URL (default: 0)")
    parser.add_argument("--num-poses", type=int, default=1, help="max bodies to track")
    parser.add_argument("--num-hands", type=int, default=2, help="max hands to track")
    parser.add_argument("--min-conf", type=float, default=0.5, help="min detection confidence")
    parser.add_argument("--out", default="pose_stream.jsonl",
                        help="JSON Lines file to write pose signals to (default: pose_stream.jsonl)")
    parser.add_argument("--stdout", action="store_true",
                        help="also print each frame's JSON record to stdout, e.g. for piping "
                             "into another process")
    parser.add_argument("--no-display", action="store_true",
                        help="skip the OpenCV preview window (useful for headless runs)")
    args = parser.parse_args()

    pose_model = ensure_model(MODELS_DIR / "pose_landmarker_lite.task", POSE_MODEL_URL)
    hand_model = ensure_model(MODELS_DIR / "hand_landmarker.task", HAND_MODEL_URL)

    pose_landmarker = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(pose_model)),
        running_mode=RunningMode.VIDEO,
        num_poses=args.num_poses,
        min_pose_detection_confidence=args.min_conf,
    ))
    hand_landmarker = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(hand_model)),
        running_mode=RunningMode.VIDEO,
        num_hands=args.num_hands,
        min_hand_detection_confidence=args.min_conf,
    ))

    source = open_source(args.source)
    is_image = isinstance(source, str) and Path(source).suffix.lower() in (
        ".jpg", ".jpeg", ".png", ".bmp", ".webp")

    out_path = Path(args.out)
    out_file = out_path.open("w")

    cap = None
    if not is_image:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(
                f"Could not open source {source!r}. If this is a camera index, grant "
                "camera access to your terminal in System Settings > Privacy & Security "
                "> Camera."
            )

    print(f"Writing pose signals to {out_path.resolve()}", file=sys.stderr)
    if not args.no_display:
        print("Press q or Esc in the preview window to quit.", file=sys.stderr)

    frame_idx = 0
    prev = time.time()
    fps = 0.0
    try:
        while True:
            if is_image:
                frame = cv2.imread(source)
                if frame is None:
                    raise SystemExit(f"Could not read image {source!r}")
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            record = {
                "frame": frame_idx,
                "t": round(time.time(), 4),
                "poses": pose_to_record(pose_result, width, height),
                "hands": hands_to_record(hand_result, width, height),
            }
            line = json.dumps(record)
            out_file.write(line + "\n")
            out_file.flush()
            if args.stdout:
                print(line, flush=True)

            if not args.no_display:
                for pose in record["poses"]:
                    pts = [(int(x), int(y)) for x, y, *_ in pose["landmarks"].values()]
                    draw_landmarks(frame, pts, POSE_CONNECTIONS, (0, 255, 0))
                for hand in record["hands"]:
                    pts = [(int(x), int(y)) for x, y, *_ in hand["landmarks"].values()]
                    color = (255, 0, 0) if hand["handedness"] == "Left" else (0, 0, 255)
                    draw_landmarks(frame, pts, HAND_CONNECTIONS, color)

                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
                prev = now
                cv2.putText(frame, f"{fps:5.1f} FPS", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("MediaPipe pose+hands - press q to quit", frame)
                key = cv2.waitKey(1 if not is_image else 0) & 0xFF
                if key in (ord("q"), 27):
                    break

            frame_idx += 1
            if is_image:
                break
    finally:
        out_file.close()
        if cap is not None:
            cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        pose_landmarker.close()
        hand_landmarker.close()

    print(f"Wrote {frame_idx} frame record(s) to {out_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
