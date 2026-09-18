# DanceDanceConvolution

Pose-detection demos: run pose estimation on a webcam, video, or image, show
an annotated preview, and stream per-frame keypoint signals as JSON Lines so
a separate process (e.g. a model consuming pose-sequence windows) can pick
them up. Three detectors are available in the web UI / `video_pipeline.py`:

| | Multi-person | Hands | Notes |
|---|---|---|---|
| **YOLO** | Yes, no cap | No | Fastest. Measured 100% frame coverage on a 476-frame test clip. |
| **RTMW** (recommended for hands) | Yes, no cap | Yes, linked to the right person | Slower than YOLO on CPU but far more reliable hands -- see below. |
| **MediaPipe** | Capped, needs a slider | Yes, but unreliable | Hands not linked to a body. Measured only 15% frame coverage on hands (0% with both hands at once) on the same test clip -- see [Known limitations](#known-limitations). |

- **`app.py`** + **`video_pipeline.py`** -- the main workflow: a local web UI
  to upload a video and get back an overlay video and a tracking JSONL. See
  [Web UI](#web-ui-video-in-tracking-json--overlay-video-out) below.
- **`pose_demo.py`** -- YOLO11 pose, live webcam demo. Fast, 17-keypoint COCO
  body pose only (no hands).
- **`mediapipe_pose_demo.py`** -- MediaPipe PoseLandmarker + HandLandmarker,
  live webcam demo. 33-keypoint body pose *plus* 21 landmarks per hand
  (fingers included), same hand-reliability caveat as above.

## Setup

Use a virtualenv for this project rather than your system/base Python --
`mediapipe`, `gradio`, and friends pull in a lot of transitive dependencies
that can clash with unrelated packages otherwise.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`pose_demo.py`'s weights (`yolo11n-pose.pt`) and the MediaPipe models
(`models/*.task`) download automatically on first run. `video_pipeline.py`'s
overlay export uses `ffmpeg` if it's on your `PATH` (`brew install ffmpeg`)
to produce a browser-playable H.264 file; without it, the overlay is still
written but in a codec most browsers won't preview inline.

## Web UI (video in, tracking JSON + overlay video out)

```bash
python app.py
```

Opens a local Gradio app at http://127.0.0.1:7860. Upload a video, pick a
model, click Analyze:

- **YOLO** (body only, fast) -- no configuration needed.
- **RTMW** (body + hands, multi-person) -- recommended when you need hands;
  no configuration needed, no person-count cap.
- **MediaPipe** (body + hands) -- set "Number of people in video" to at
  least how many dancers are in frame (defaults to 4); expect hands to drop
  out often (see the table above).

Each run writes to `runs/<timestamp>/`: `overlay.mp4` (annotated video) and
`tracking.jsonl` (one JSON record per frame -- formats documented below,
`t` is seconds into the video, i.e. `frame / fps`, not wall-clock time,
since this is for analyzing a recorded clip rather than a live feed).

`video_pipeline.py` has the underlying `analyze_video_yolo()`,
`analyze_video_rtmw()`, and `analyze_video_mediapipe()` functions if you want
to call them directly from a script instead of the UI.

## Run the YOLO demo (body only, fast, live webcam)

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

## Run the MediaPipe demo (body + hands, live webcam)

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

**Known limitation:** this script uses MediaPipe's `VIDEO` running mode for
live tracking continuity, but `PoseLandmarker` in `VIDEO`/`LIVE_STREAM` mode
caps at 1 detected person regardless of `--num-poses` (its cross-frame
tracking only follows a single ROI; tested against `mediapipe==0.10.31`).
`video_pipeline.py`/`app.py` avoid this by running each frame as an
independent `IMAGE`-mode detection instead, since they don't need live
temporal tracking. If reliable multi-person live tracking turns out to
matter, switching this script to the same `IMAGE`-mode-per-frame approach is
the fix -- at the cost of the smoothing `VIDEO` mode provides.

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

## Pose signal format (`analyze_video_rtmw` / RTMW)

Same `people` structure as above (and `examples/example_consumer.py` reads
it unmodified), extended with feet and per-person-linked hands:

```json
{"frame": 12, "t": 0.4, "people": [
  {"id": 0, "bbox": [x1, y1, x2, y2], "conf": 0.93,
   "keypoints": {"nose": [x, y, conf], "left_shoulder": [x, y, conf], ...,
                 "left_big_toe": [x, y, conf], ...,
                 "left_hand_wrist": [x, y, conf], "left_hand_thumb_tip": [x, y, conf], ...,
                 "right_hand_wrist": [x, y, conf], ...}}
]}
```

`keypoints` has the same 17 COCO body points as `pose_demo.py`, plus 6 feet
points (`left_big_toe`, `left_small_toe`, `left_heel`, and the `right_`
equivalents), plus 21 `left_hand_*`/21 `right_hand_*` points (only included
when that hand's detection confidence clears `hand_conf`, default 0.3) --
all `[x_pixels, y_pixels, confidence]`. Unlike MediaPipe, hands are always
keyed to the same person `id` they belong to. There's no cross-frame
tracking (no `--track` equivalent), so `id` is a per-frame detection index,
not a stable identity -- add a tracker (e.g. IoU or ByteTrack over the
`bbox`es) if you need persistent per-dancer IDs across frames. `mode` can be
`"performance"` (most accurate, slowest), `"balanced"`, or `"lightweight"`
(default -- ~15 FPS on CPU in testing, ~4x faster than `"balanced"` with a
modest accuracy cost).

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

## Known limitations

**MediaPipe's hand detection is unreliable on real footage.** Measured on a
476-frame (~16s) single-dancer test clip processed through the web UI:

| | YOLO (body) | MediaPipe (body) | MediaPipe (hands) |
|---|---|---|---|
| Frames with a detection | 476/476 (100%) | 469/476 (98.5%) | 73/476 (15.3%) |
| Frames with both instances at once | -- | -- | 0/476 (0%) |
| Longest dropout streak | -- | -- | 155 consecutive frames |

MediaPipe's body pose is solid (comparable to YOLO), but `HandLandmarker`
runs on the full downscaled frame, where a hand is a small, often
motion-blurred region -- it drops out constantly and never picked up both
hands simultaneously in this clip. This is why RTMW is the recommended
choice when hand data matters: as a top-down pipeline (person detector, then
pose *on a crop of that person*), it gives the hand detector far more
effective resolution on the hand itself. If you need MediaPipe specifically
(e.g. its face mesh), the fix would be to crop around each wrist (from body
keypoints) before running `HandLandmarker`, rather than run it on the whole
frame -- not currently implemented.
