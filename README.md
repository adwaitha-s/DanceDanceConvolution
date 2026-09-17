# DanceDanceConvolution

Pose-detection demos: run pose estimation on a webcam, video, or image, show
an annotated preview, and stream per-frame keypoint signals as JSON Lines so
a separate process (e.g. a model consuming pose-sequence windows) can pick
them up. Two detectors are included:

- **`pose_demo.py`** -- YOLO11 pose. Fast, 17-keypoint COCO body pose only
  (no hands).
- **`mediapipe_pose_demo.py`** -- MediaPipe PoseLandmarker + HandLandmarker.
  33-keypoint body pose *plus* 21 landmarks per hand (fingers included).

## Setup

```bash
pip install -r requirements.txt
```

`pose_demo.py`'s weights (`yolo11n-pose.pt`) and `mediapipe_pose_demo.py`'s
models (`models/*.task`) download automatically on first run.

## Run the YOLO demo (body only, fast)

```bash
python pose_demo.py
```

Opens your default camera, shows a live annotated preview, and writes one
JSON record per frame to `pose_stream.jsonl`. Press `q` or `Esc` in the
preview window to quit.

Useful options:

```bash
python pose_demo.py --source video.mp4 --out out.jsonl   # process a video file
python pose_demo.py --source photo.jpg --no-display       # process a single image
python pose_demo.py --track                                # persistent per-person IDs (ByteTrack)
python pose_demo.py --stdout --no-display | python examples/example_consumer.py
```

`--device` defaults to `cpu`. Ultralytics has a known MPS bug for pose
models on Apple Silicon GPUs, so `mps` is opt-in via `--device mps`.

## Run the MediaPipe demo (body + hands)

```bash
python mediapipe_pose_demo.py
```

Same behavior as `pose_demo.py` -- live preview, `q`/`Esc` to quit, writes
`pose_stream.jsonl` -- but also draws and streams hand/finger landmarks.

```bash
python mediapipe_pose_demo.py --source video.mp4 --out out.jsonl
python mediapipe_pose_demo.py --num-poses 2 --num-hands 4   # track 2 people
python mediapipe_pose_demo.py --stdout --no-display | python examples/example_consumer.py
```

## Pose signal format (`pose_demo.py`)

Each line of the output file is one JSON object:

```json
{"frame": 12, "t": 1699999999.1234, "people": [
  {"id": 0, "bbox": [x1, y1, x2, y2], "conf": 0.93,
   "keypoints": {"nose": [x, y, conf], "left_shoulder": [x, y, conf], ...}}
]}
```

`keypoints` has all 17 COCO keypoints (`nose`, `left_eye`, `right_eye`,
`left_ear`, `right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`,
`right_elbow`, `left_wrist`, `right_wrist`, `left_hip`, `right_hip`,
`left_knee`, `right_knee`, `left_ankle`, `right_ankle`), each as
`[x_pixels, y_pixels, confidence]`. `id` is a stable per-person track ID when
run with `--track`, otherwise a per-frame detection index.

See [`examples/example_consumer.py`](examples/example_consumer.py) for a
minimal reader that consumes the stream from a file or via stdin.

## Pose signal format (`mediapipe_pose_demo.py`)

```json
{"frame": 12, "t": 1699999999.1234,
 "poses": [{"id": 0, "landmarks": {"nose": [x, y, z, visibility], "left_shoulder": [x, y, z, visibility], ...}}],
 "hands": [{"handedness": "Left", "score": 0.95, "landmarks": {"wrist": [x, y, z], "thumb_tip": [x, y, z], ...}}]}
```

`poses[].landmarks` has all 33 BlazePose body points, each
`[x_pixels, y_pixels, z_relative, visibility]`. `hands[].landmarks` has all
21 hand points, each `[x_pixels, y_pixels, z_relative]`. `z` is relative
depth (roughly hip-depth-normalized, not metric), not directly comparable to
YOLO's pixel-space output. `handedness` is from the subject's own
perspective (mirrored relative to the camera), and hands are **not** linked
to a specific `poses[].id` -- matching a hand to a body is left to the
consumer if needed. `id` is a per-frame list index, not a stable track ID
across frames.
