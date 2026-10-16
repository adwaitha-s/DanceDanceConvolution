# DanceDanceConvolution

Pose-detection demos: run pose estimation on a webcam, video, or image, show
an annotated preview, and stream per-frame keypoint signals as JSON Lines so
a separate process (e.g. a model consuming pose-sequence windows) can pick
them up. Three detectors are available in the web UI / `video_pipeline.py`:

| | Multi-person | Hands | Notes |
|---|---|---|---|
| **YOLO** | Yes, no cap | No | Fastest. Measured 100% frame coverage on a 476-frame test clip. |
| **RTMW** (recommended for hands) | Yes, no cap | Yes, linked to the right person | Slower than YOLO on CPU. Measured 99.9% of person-detections had a hand, on a 2-dancer clip -- see [Known limitations](#known-limitations). |
| **MediaPipe** | Capped, needs a slider | Yes, but unreliable | Hands not linked to a body. Measured only 15% frame coverage on hands (0% with both hands at once) on a 1-dancer clip -- see [Known limitations](#known-limitations). |

- **`app.py`** + **`video_pipeline.py`** -- the main workflow: a local web UI
  to upload a video, pick the dancers and see who deviates and when. See
  [Web UI](#web-ui) below.
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

## Web UI

```bash
python app.py
```

Opens a local Gradio app at http://127.0.0.1:7860. One page, three steps; each
step opens the next by itself:

1. **Video** -- upload a clip and click *Analyze video*, or pick a clip you
   already analyzed under *Previous runs* and click *Open run*. The uploaded
   video is kept with the run, so it is never asked for again.
2. **Dancers** -- everyone found in the clip, with the predicted dancers
   already ticked. Fix the ticks if needed, choose what to compare against,
   click *Compare dancers*.
3. **Results** -- the deviation overlay video beside the timeline. A line on
   the timeline follows the video; clicking the timeline, dragging *Inspect
   time* or clicking a *Worst moments* row jumps the video there.

The pose model is under *Pose model (advanced)* in step 1:

- **YOLO** (body only, fast) -- no configuration needed.
- **RTMW** (body + hands, multi-person) -- recommended when you need hands;
  no configuration needed, no person-count cap.
- **MediaPipe** (body + hands) -- set "Number of people in video" to at
  least how many dancers are in frame (defaults to 4); expect hands to drop
  out often (see the table above).

MediaPipe runs are tracking-only (they can't be compared in steps 2-3).

Each run writes to `runs/<timestamp>/`: `overlay.mp4` (annotated video),
`source.<ext>` (the upload), `meta.json` (file name, model, length -- used to
name the run in *Previous runs*) and `tracking.jsonl` (one JSON record
per frame -- formats documented below,
`t` is seconds into the video, i.e. `frame / fps`, not wall-clock time,
since this is for analyzing a recorded clip rather than a live feed). The
overlay and JSONL can also be downloaded under *Raw tracking output* in step 1.
Set `DDC_RUNS_DIR` to keep runs somewhere other than `./runs`.

`video_pipeline.py` has the underlying `analyze_video_yolo()`,
`analyze_video_rtmw()`, and `analyze_video_mediapipe()` functions if you want
to call them directly from a script instead of the UI.

### One person, one track (step 2)

RTMW's overlay draws detections independently in each frame; its detection indices
are not persistent identities. `ddc.tracking.stable_track_ids` associates people
using predicted torso motion, body size and clothing-color histograms from the saved
source video. Arm movements have little influence on identity. Pose-only JSONL files
fall back to motion and geometry; explicitly marked YOLO tracker IDs are preserved.
Clothing descriptors are used only for association, never as dancer-classifier inputs.

After a detection dropout, compatible fragments up to 5 s apart are joined before
short tracks are discarded. Tracks with overlapping time spans are never stitched
together. Step 2 shows a visibility timeline and the number of joined fragments.
Track assignments are shared by the people list, features, previews and comparison.
Old cached results are invalidated when tracking changes; saved UI labels need a new
review because the person numbers may change.

People wearing similar clothes can still be mixed up or left split after prolonged
occlusion. Under "Same person listed twice? Merge" in
step 2, tick the duplicates and click Merge (Reset merges undoes it); on the
CLI use `--list-tracks` then `--merge 2,5` (repeatable).

### Deviation reference (step 2)

Step 3 scores each dancer's deviation against a reference pose. Pick one
under "Compare each dancer against":

- **Composite of all dancers** (default) -- per-frame consensus of everyone kept.
- **One of the dancers in this video** -- choose a dancer; everyone else is scored
  against that dancer. The reference dancer is drawn in white as `REF` and isn't scored.
- **Separate solo video** -- upload a solo clip (pose-tracked with RTMW, cached in
  `runs/<ts>/reference_solo/`). Both videos are assumed to start together; use
  "Solo offset (s)" to shift the solo's timeline if they don't.

CLI equivalents: `python -m ddc.analyze runs/<ts> --reference dancer --reference-dancer 0`
or `--reference solo --solo-video solo.mp4 --solo-offset 0.5`.

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

## Evaluate dancer vs non-dancer pose features

The detector deliberately labels every persistent person as a track; it does
not infer whether somebody is dancing.  `ddc.dancer_features` collects five
scale- and location-invariant track-level signals from a detector run:

1. wrist height relative to the hips;
2. wrist speed;
3. limb extension (shoulder-to-wrist and hip-to-ankle reach);
4. elbow/knee motion; and
5. whole-pose motion.

Create a small CSV which labels the tracks you have reviewed in a run:

```csv
track_id,label
0,dancer
1,dancer
2,non_dancer
3,non_dancer
```

Then run the evaluator (at least two tracks from each class are required):

```bash
python -m ddc.dancer_features runs/20261008-120000/tracking.jsonl labels.csv --out reports/my_run
```

It writes `track_features.csv`, `feature_selection_report.json`, and
`feature_selection_accuracy.svg`.  The chart uses leave-one-track-out
accuracy and compares each individual feature with three methods that select
the best three features inside each validation fold: a **filter** (Fisher
score), **wrapper** (greedy forward selection), and **embedded** (L1 logistic
selection).  This prevents the accuracy graph from selecting features using
the held-out track's label.

For an end-to-end installation check only, use the synthetic example below.
Its score is intentionally not a real-world model result.

```bash
python -m ddc.dancer_features --demo --out reports/synthetic_feature_demo
```

## Screen non-dancers before composite-pose analysis

### Next ten videos: supervised training cohort

The next-ten-video cohort is armed in this workspace. Restart the app to load
the current controls. Each successful new **Analyze video** run is retained
with its source, tracking data and a training-run marker. Reopening a run does
not consume another slot. Capture stops at ten runs and persists across restarts.
To arm this workflow in a fresh checkout, run:

```powershell
python -m ddc.training arm
```

`runs/training_registry.xlsx` contains exactly **40 candidate features**:
10 time, 8 frequency, 8 time-frequency, 9 biomechanics and 5 human-related
fields. Human-related fields describe external participant reports or the
reviewed target; demographics, health and exertion are not estimated from
video or used as skeleton predictors. The broader legacy feature registry is
retained separately.

Statistics update as each completed run enters the folder, up to ten runs.
Saving or confirming labels in Dancers updates both `training_registry.xlsx`
and `dancer_feature_registry.xlsx` in the same action. The Video tab has no
training dropdown or separate refresh button. CLI `refresh` also discovers
completed run folders and omits folders that have been removed. Older folders present when the cohort was
armed and incomplete analyses are excluded. Pose features and preprocessed
sequences are cached so a label change does not rerun pose detection.

In **Dancers**, **Save reviewed labels** writes the current checklist to the
registry's **Validation label** column and refits from the updated labels.
If **Compare dancers** is pressed before a review has been saved, it confirms
the original displayed predictions as all correct, with zero corrections.
Unsaved checklist edits are not stored by Compare. A saved review always
takes precedence, including after reopening the run. When no model prediction
is available, Save is required to supply manual labels; Compare does not
invent predictions.

The training workbook is saved in `runs/training_registry.xlsx`. Labels are
normally reviewed in Dancers; an edited workbook can also be imported with
the CLI command below. Blank labels remain unreviewed. The registry includes per-run feature
statistics, per-track selected predictor values and contributions, and
prediction scores. Predictions become supervised labels only through an
explicit Save or Compare confirmation, or an imported label review. Predictions
shown in Dancers are preserved beside their validation labels after refitting.

LASSO here means squared-error L1 regression with dancer=1 and non_dancer=0,
not the legacy L1 logistic screener. Median imputation and standardization are
fit within video-group cross-validation folds while tuning alpha. The five
largest nonzero standardized coefficients determine the selected predictors;
a five-input LASSO model is then fit. If the data cannot support five nonzero
predictors, the pipeline reports that rather than filling the list with zero
weights. Per-run predictions tune and train on **other enrolled runs only**.
Its linear score is not a calibrated dancer probability. Selection CV MSE is
used for alpha tuning and is not an independent performance estimate.

LASSO and supervised PyTorch LSTM training are **rolling**: as soon as enough
reviewed data supports the five predictors, the LSTM fits all currently reviewed,
usable sequences. Every newly reviewed video or changed validation label triggers
a new fit. Unreviewed clips still contribute feature statistics and await labels;
they do not stop training on usable reviewed clips. Once all ten videos are
reviewed and usable, all ten contribute to the model.
Its inputs are the clip's middle five seconds: 17 body joints, hip-centered
and torso-scaled xy, plus 17 visibility masks, resampled to 100 frames at
20 Hz. Pixels, backgrounds, audio, demographic fields and reviewed labels do
not enter the input tensor. The five LASSO features are recorded with the model
as predictor-selection results; the LSTM receives the skeleton sequence itself.
Interpolation is limited to 0.25-second gaps. Tracks with less than 50% visible
joint coverage in the center window and clips shorter than five seconds are
flagged and excluded from the rolling fit until replaced or made usable.
The workbook lists these outstanding inputs. YOLO, RTMW and MediaPipe body schemas are supported;
MediaPipe reviews use the workbook because its UI comparison remains unavailable.

Outputs under `runs/training/` are `lasso_model.json`, `skeleton_lstm.pt`,
`supervised_skeleton_sequences.npz` and `lstm_report.json`. Each enrolled run
also retains `training/middle_5s_skeleton.npz` and preprocessing metadata.
Changing labels or merges invalidates the previous fit. The checkpoint report
explicitly identifies the current training inputs; training loss is not test
accuracy. Ten clips may contain repeated people, and no independent test-set
accuracy is claimed.

CLI equivalents:

```powershell
python -m ddc.training status
python -m ddc.training import-review --workbook edited_registry.xlsx
python -m ddc.training refresh --no-train
python -m ddc.training refresh --epochs 60
python -m ddc.training replace-run --run RUN_NAME
```

`replace-run` retains the original files and frees a slot for the next successful
analysis, useful for a short or unusable clip. Review the regenerated workbook
after replacing a run or merging tracks. `--runs-dir` and `DDC_RUNS_DIR` select
the run directory.

Excel export uses Node.js with `@oai/artifact-tool`, automatically located in
the Codex bundled runtime on this machine. Elsewhere, supply `DDC_NODE` and
`DDC_ARTIFACT_MODULE_ROOT` (a node_modules directory containing that package).
Python dependencies are in `requirements.txt`. Workbook export failures leave
saved video data, labels and model outputs intact and are reported in the UI.
Method references: [scikit-learn LASSO](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html),
[grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html),
[PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).

In the web UI, this screening happens in step 2 (**Dancers**). Opening a run
there scores every saved track with the five standardized
predictors displayed in `runs/dancer_feature_registry.xlsx` and pre-selects only
the predicted dancers. It does
**not** run pose detection again. The action also adds a Labels-style worksheet
for the run. Use **Save reviewed labels** beside the checklist after changing
the review: it updates Reviewed label, Validation status, and Correction
recorded in that worksheet. Clicking **Compare** before saving a review
records the original predictor labels as validated correct, ignoring unsaved
checklist edits. Once a review is saved, Compare preserves its labels. Every saved
review rebuilds both registries. `runs/dancer_feature_registry.xlsx` retains
predictor summaries, the feature catalog and one labels sheet per run.
Individual-dancer statistics worksheets and their registry payloads are no
longer saved. Pose features and supervised skeleton sequences remain in the
training pipeline for model fitting.

**Render preview of the ticked dancers** draws a video of just the ticked people. It re-encodes the
whole clip, so it only runs when you click it, at up to 720p (using the hardware decoder and
encoder on macOS). Renders are cached under `runs/<run>/overlays/`, keyed on the run, merges,
ticked dancers and source video, so repeating a request is instant; the three newest are kept
per run.

Track ids come from `ddc.tracking.stable_track_ids` (automatic stitching plus any manual
merges), the same numbering the step 2 people list shows. Merging people re-scores the
renumbered list, and **Save reviewed labels** stores the run's merges in `merges.json` beside
`labels.csv` so labels keep pointing at the same people. `labels.csv` files written before
track stitching used the old numbering; delete them (and `prediction_validation.json` /
`feature_inventory.json`) and relabel those runs.

Predictions for a run come from a model trained on *other* runs only, and
camera-framing features (frame position, clip length, detected size) are never
used as predictors. The Excel workbook (`runs/dancer_feature_registry.xlsx`) is written
with `openpyxl`; if it cannot be built, comparing still works and shows a note.

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
run with `--track`, otherwise a per-frame detection index. The frame's `id_type`
field explicitly distinguishes `"track"` from `"detection"`; older files without
this field use geometric association rather than trusting ambiguous raw IDs.

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

**RTMW confirmed reliable on a real 2-dancer clip.** Measured on a 786-frame
test clip processed through the web UI:

| | Result |
|---|---|
| Frames with >=1 person | 667/786 (84.9%) |
| Person-instances with >=1 hand | 1340/1342 (99.9%) |
| Person-instances with both hands | 1335/1342 (99.5%) |
| Person-count distribution | 2 people: 659 frames, 3 people: 8 frames (brief false positive), 0 people: 119 frames |

The 119 zero-person frames are all one contiguous block at the very end of
the clip (frames 667-786) -- consistent with the dancers leaving frame
before the clip ends, not a detection dropout. In every frame where a person
*was* detected, hands came through essentially every time (99.9%), a sharp
contrast with MediaPipe's 15.3% hand coverage on comparable footage above.
