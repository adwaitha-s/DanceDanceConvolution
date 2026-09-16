# DanceDanceConvolution

Pose-detection demo: runs YOLO11 pose estimation on a webcam, video, or image,
shows an annotated preview, and streams per-frame keypoint signals as JSON
Lines so a separate process (e.g. a model consuming pose-sequence windows)
can pick them up.

## Setup

```bash
pip install -r requirements.txt
```

Model weights (`yolo11n-pose.pt`) download automatically on first run.

## Run the demo

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

## Pose signal format

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
