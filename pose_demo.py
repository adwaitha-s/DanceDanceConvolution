"""YOLO pose-detection demo that also emits keypoint signals for downstream processing.

For every processed frame this writes one JSON object (JSON Lines format) describing
each detected person's bounding box and 17 COCO keypoints, so another process (a
consumer script, a socket client, a numpy/pandas pipeline, a convolutional model
over pose sequences, ...) can read the stream instead of re-running detection.

Usage:
    python pose_demo.py                          # webcam 0, cpu, writes pose_stream.jsonl
    python pose_demo.py --source video.mp4 --out out.jsonl
    python pose_demo.py --source photo.jpg --no-display
    python pose_demo.py --track --stdout | python examples/example_consumer.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# Order Ultralytics' pose models emit keypoints in (COCO-17).
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def frame_to_record(frame_idx: int, timestamp: float, result) -> dict:
    """Convert one Ultralytics pose Result into a JSON-serializable record."""
    people = []
    boxes = result.boxes
    kpts = result.keypoints

    has_ids = boxes is not None and boxes.id is not None
    n = len(boxes) if boxes is not None else 0

    for i in range(n):
        xyxy = boxes.xyxy[i].tolist()
        conf = float(boxes.conf[i])
        person = {
            "id": int(boxes.id[i]) if has_ids else i,
            "bbox": [round(v, 1) for v in xyxy],
            "conf": round(conf, 3),
            "keypoints": {},
        }
        if kpts is not None:
            xy = kpts.xy[i].tolist()
            kconf = kpts.conf[i].tolist() if kpts.conf is not None else [1.0] * len(xy)
            for name, (x, y), c in zip(KEYPOINT_NAMES, xy, kconf):
                person["keypoints"][name] = [round(x, 1), round(y, 1), round(float(c), 3)]
        people.append(person)

    return {"frame": frame_idx, "t": round(timestamp, 4), "people": people}


def open_source(source: str):
    """Resolve --source into an int camera index if it looks like one, else a path/URL."""
    try:
        return int(source)
    except ValueError:
        return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0",
                        help="camera index, video file, image file, or stream URL (default: 0)")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="pose model weights")
    parser.add_argument("--device", default="cpu",
                        help="cpu, mps, or 0 for CUDA. Ultralytics has a known MPS bug for "
                             "pose models, so cpu is the default.")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size")
    parser.add_argument("--track", action="store_true",
                        help="assign persistent person IDs across frames (ByteTrack)")
    parser.add_argument("--out", default="pose_stream.jsonl",
                        help="JSON Lines file to write pose signals to (default: pose_stream.jsonl)")
    parser.add_argument("--stdout", action="store_true",
                        help="also print each frame's JSON record to stdout, e.g. for piping "
                             "into another process")
    parser.add_argument("--no-display", action="store_true",
                        help="skip the OpenCV preview window (useful for headless runs)")
    args = parser.parse_args()

    print(f"Loading {args.model} on {args.device} ...", file=sys.stderr)
    model = YOLO(args.model)

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

    predict_kwargs = dict(conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)

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

            if args.track:
                results = model.track(frame, persist=True, **predict_kwargs)
            else:
                results = model.predict(frame, **predict_kwargs)
            result = results[0]

            record = frame_to_record(frame_idx, time.time(), result)
            line = json.dumps(record)
            out_file.write(line + "\n")
            out_file.flush()
            if args.stdout:
                print(line, flush=True)

            if not args.no_display:
                annotated = result.plot()
                now = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
                prev = now
                cv2.putText(annotated, f"{fps:5.1f} FPS", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("YOLO pose - press q to quit", annotated)
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

    print(f"Wrote {frame_idx} frame record(s) to {out_path.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
