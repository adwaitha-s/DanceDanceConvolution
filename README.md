# DanceDanceConvolution

DanceDanceConvolution analyzes saved videos in a local Gradio app. It detects
people, lets you correct their tracks and dancer labels, predicts dancers with
five LASSO selected pose features and an LSTM, and compares selected dancers
against a pose reference. Training records live in SQLite; videos and pose
tracks remain in their run folders.

## Setup and launch

Create a virtual environment, install the dependencies, and launch the app:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

The app opens at `http://127.0.0.1:7860`. Set `DDC_RUNS_DIR` to store runs
outside `./runs`. `ffmpeg` on `PATH` enables browser compatible H.264 overlays.
The detector weights download on first use. Use one running app per run
directory, especially when the SQLite database is synced by OneDrive.

## Video and GUI workflow

1. **Video:** Upload a video and select YOLO, RTMW, or MediaPipe; click
   **Analyze video**. You can reopen an earlier run under **Previous runs**.
   Each run saves the source video, `tracking.jsonl`, `meta.json`, and an
   annotated overlay under `runs/<run>/`. Raw tracking files remain
   downloadable from the Video tab.
2. **Dancers:** The app lists persistent people with a visibility timeline and
   thumbnails. Review the predicted dancer checklist. Use **Render preview
   of the ticked dancers** to make a cached preview on demand. To combine
   fragments of one person, select them in **Same person listed twice?
   Merge**, then click **Save merges**. **Reset merges** and **Save merges**
   undo a saved correction. Review the checklist again after a merge changes
   person numbers. **Save reviewed labels** records a corrected checklist.
   **Compare dancers** confirms the originally displayed predictions when
   no review was saved; it does not record unsaved checklist edits.
3. **Results:** Compare the selected dancers with the group composite, one
   dancer from the clip, or a separate solo video. The deviation overlay,
   timeline, worst moments table, inspect time control, and video seeking are
   interactive. A solo video may need a time offset to align it with the
   analyzed clip.

YOLO detects COCO body keypoints quickly. RTMW detects multiple people and
links hands to each person's body; use it when hand tracking matters.
MediaPipe supplies body and hand landmarks, but its hands are not reliably
linked to a person. MediaPipe runs can be tracked and downloaded; the GUI's
dancer review and pose comparison use YOLO or RTMW runs. Detection remains separate
from classifying a tracked person as a dancer.

### Stable tracks and saved merges

`ddc.tracking.stable_track_ids` associates detections through motion, body
geometry, and saved source video clothing descriptors. It stitches compatible
fragments after brief dropouts and preserves explicit YOLO tracker IDs.
The same IDs feed the people list, feature extraction, preview, dancer
prediction, and comparison. Clothing descriptors help identity matching;
they are not dancer classifier inputs.

**Save merges** writes `merge_review.json` and teaches a separate identity
matching model from confirmed same person pairs and simultaneously visible
different person pairs. Its weights live in `runs/identity_model.json`. New
analyses save identity predictions before dancer screening, so reopening a
run retains its person numbers. Saved merges are also recorded in
`merges.json` beside reviewed `labels.csv`. A stale label review is excluded
when tracking inputs or person numbers change.

Identity matching can still split or mix people after long occlusions.
Review and save merges in the GUI when the people list is wrong. Saving a
merge does not label anyone as a dancer.

## Fifteen validated training videos

The training cohort accepts the first **15 completed runs with a saved,
current dancer/non_dancer label for every merged track**. Analyzed but
unreviewed runs wait for validation and do not count toward the 15. The
reviewed videos are used for rolling training: as enough labeled examples
become available, each newly validated run refits LASSO and the LSTM using
the eligible cohort. When the fifteenth validated run completes a successful
fit, the models are frozen. Later videos use the fitted models to predict
dancers from their saved merged tracks; they do not add training records or
change model weights. Saved identity merge reviews can continue to improve
track association independently.

Start or inspect the cohort through the app or CLI:

```powershell
python -m ddc.training arm
python -m ddc.training status
python -m ddc.training refresh
python -m ddc.training export
python -m ddc.training backup --output backups/lasso-lstm-training.sqlite3
```

`arm` excludes runs that already existed before the cohort was armed.
Reopening a run never consumes another slot. The CLI also supports
`import-review --workbook edited_registry.xlsx` for workbook label
reviews. `--runs-dir` and `DDC_RUNS_DIR` select the run directory.

### Feature labels and model inputs

The feature catalog has 40 named fields: 10 time, 8 frequency, 8 time
frequency, 9 biomechanics, and 5 human related fields. The human fields
describe external reports or the reviewed target. Age, health, gender,
fatigue, camera position, detected size, pixels, audio, and clothing are
not inferred from video or used as dancer predictors.

LASSO uses standardized pose feature values and human reviewed labels
(`dancer=1`, `non_dancer=0`). It chooses five nonzero coefficients by
video grouped cross validation. The registry labels each selected feature
and shows its coefficient and direction: positive means **more dancer-like**
as that feature rises; negative means **more non-dancer-like**. The LASSO
linear score is not a calibrated dancer probability.

The LSTM receives exactly those five selected feature measurements in the
same order, measured in nine overlapping windows of the middle five seconds
of each usable track. Its input shape is `(tracks, 9, 5)`. Missing value
medians and standardization come from reviewed training windows and are
reused at prediction. The LSTM outputs the dancer label and its probability.
Sequences with insufficient pose coverage or long tracking gaps are marked
unavailable. LASSO scores and coefficients do not enter the LSTM tensor.

The GUI predicts against the currently merged track IDs. A saved human
review provides training labels only while the 15 run cohort is still open.
Prediction alone never turns an unreviewed track into a training label.

### Training data and registry

`runs/lasso_lstm_training.sqlite3` is the authoritative training database.
It stores cohort membership, validated reviews, cached feature values,
model versions, selected LASSO coefficients, LSTM checkpoint metadata, and
prediction snapshots. `runs/lasso_lstm_registry.xlsx` is the inspection
workbook for the 15 enrolled, validated runs. It shows the labeled LASSO
top five, feature weights, reviewed labels, and LSTM dancer predictions and
probabilities. It is a generated export of SQLite and can be rebuilt with
`python -m ddc.training export`. There is no separate dancer feature registry.

Raw source video, `tracking.jsonl`, merge files, and feature sequence arrays
remain in their run folders. Model artifacts and sequence caches are stored
under `runs/training/` and `runs/<run>/training/`. Keep the run folders with
SQLite when moving the project. The SQLite database commits before the Excel
export. If Excel has the workbook open or the export runtime is unavailable,
the database remains current and `status` reports a pending export; close the
workbook and run `export` to retry. Excel export uses Node.js and
`@oai/artifact-tool`; outside the bundled Codex runtime, set `DDC_NODE` and
`DDC_ARTIFACT_MODULE_ROOT`.

The SQLite `current_weights` and `current_predictions` views show the
latest model snapshot. `model_versions`, `review_batches`, `snapshots`,
`feature_values`, and `predictions` retain history. The `backup` command
uses SQLite's backup API and requires a new destination file. Back up the
run folders separately to preserve videos and pose arrays.

## Direct analysis and data format

`video_pipeline.py` exposes `analyze_video_yolo`, `analyze_video_rtmw`, and
`analyze_video_mediapipe` to other Python code. `python -m ddc.analyze
runs/<run> --reference dancer --reference-dancer 0` compares against one
saved dancer; `--reference solo --solo-video solo.mp4 --solo-offset 0.5`
uses a separate solo video.

Every line of `tracking.jsonl` is a JSON frame with `frame`, `t` (seconds
from the start of the clip), and detected `people`. Each person has an ID,
bounding box, confidence, and named pose keypoints. RTMW extends the COCO
body points with feet and linked `left_hand_*` / `right_hand_*` points.
MediaPipe records poses and hands separately. RTMW detection indices are per
frame; `ddc.tracking` supplies persistent IDs for the GUI and models.
