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
Fix the people list under **Fix who's who** in step 2, then **Save
corrections**. Saving corrections does not label anyone as a dancer.

- **Merge**: tick the same person listed more than once.
- **Split from**: tick one person and enter *From (s)*; from that time on
  their detections become a new person (added at the end of the list).
- **Swap**: tick two people the tracker swapped, with *From* and *To*.
- **Remove**: tick one person to drop their detections between *From* and *To*.
- **Undo** steps back one change; **Reset all** clears them.

Corrections are an ordered edit log (`ddc/identity.py`): each edit refers to
people as numbered on screen when it was made, so replaying the log always
reproduces the same list. A log of only merges is saved in the original
merge-groups form, so `merges.json` from earlier versions still loads; runs
that use split/swap/remove save the edit list instead. Identity learning
skips any track an edit split across people, since it no longer belongs to
one person.

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

## Comparing dancers

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

### Timing tolerance (step 2)

By default each dancer frame is compared with the reference at the same instant,
so being a little early or late counts as a pose difference. "Timing tolerance (s)"
under Advanced settings (0–1 s, default 0) relaxes this: each dancer frame is scored
against whichever reference frame within ±tolerance it matches best, so the score
reflects shape rather than small timing slips. The overlay's reference ghost and the
per-joint feedback use that matched pose. 0 is strictest; around 0.1–0.25 s forgives
typical slips, while larger values can hide real mistakes in fast choreography.
Applies to every reference mode. CLI: `--time-tolerance 0.25`.

### Coaching dashboard (step 3)

After **Compare dancers**, Results shows an overall movement-similarity score,
per-dancer scores and comparable-frame coverage, the closest body region, and the
joint/region to review. **Personalized feedback** lists up to three separated
moments per dancer; click a row to seek the overlay. **Inspect time** gives joint
feedback at any frame. The existing deviation timeline, worst moments and overlay
remain available. The CSV retains its original columns (including `score`, the
raw deviation) and appends `movement_similarity`, a per-frame heuristic score;
blank similarity cells mean there was insufficient comparable data.

**Formula:** for each eligible frame, average the existing joint distances in
normalized torso lengths over comparable joints. Average those frame deviations
for each dancer, then use `100 / (1 + mean_deviation)`. For example, deviations
of 0, 0.25 and 1 torso length map to 100, 80 and 50. The overall score is the
unweighted average of available dancer scores, excluding a reference dancer.
It is not the average of per-frame scores because the mapping is nonlinear.

A frame needs at least 6 of 17 reliable, comparable body joints. Low-confidence,
nonfinite or missing joints are excluded. A composite joint needs a visible
peer at the same joint in that frame, so a dancer never earns a perfect score
by matching only themselves. A reference dancer is not scored and their
unobserved joints are excluded even if smoothing filled a gap. Solo references
use the existing interpolated/smoothed poses within their time overlap.
No eligible data yields **N/A**, not zero or 100. Coverage is eligible frames
out of all clip frames; low coverage may make scores misleadingly high.

Suggestions use the same eligible joints. A joint difference below 0.15 torso
lengths is described as close. Larger differences receive a review suggestion;
"higher/lower" is used only for a vertical difference of at least 0.15 torso
lengths that is at least 1.5 times its horizontal difference. Positive image y
means lower. This describes height relative to the hips in the normalized 2D
camera view, not depth or absolute screen position. Torso-rotation normalization
disables directional wording. Horizontal/ambiguous differences receive neutral
wording. Left/right joint names are the detector's anatomical labels. The 6-joint
minimum and 0.15 tolerance are design heuristics defined in `ddc/coaching.py`.

**Limitations:** this is a movement-similarity heuristic, not a scientifically
validated accuracy percentage. Reference choice, smoothing, occlusion, camera
perspective, tracking errors and video timing affect the result. Two dancers
compared with their shared composite get symmetric differences; the system
cannot tell which dancer is correct. The closest region is a relative strength,
not proof of good technique. Scores do not evaluate rhythm, artistry or 3D form.
No additional detector or language-model API calls are needed for coaching.

Run regression tests with `python -m pytest tests -q` in your virtualenv.

#### Key choreography moments to practice

Results also ranks up to five **sustained pairwise pose mismatches**. Each row
names the dancers, the peak timestamp to click, the time range to review, the
body region and joint differing most at the peak, the mean joint distance in
torso lengths, and the number of shared reliable joints. This helps the group
choose specific sections of choreography to practice. It complements the
per-dancer personalized feedback and existing Worst moments table.

The detector compares normalized poses directly between each pair of selected
dancers, independently of the composite/solo reference. Reference dancers can
participate in these comparisons without receiving a score. A frame requires
at least six jointly visible, confident joints and at least one distance of
0.15 torso lengths. A candidate must persist for at least 0.25 seconds; missing
comparisons and timestamp gaps larger than `max(0.25 seconds, 2 × median frame
interval)` break a candidate. Candidates are ranked by their peak mean distance
across shared joints, and reported peaks are at least one second apart. These
thresholds are review heuristics, not scientifically validated tolerances.

These events identify simultaneous **pose disagreement**, which may be useful
when reviewing synchronization. They do **not** prove a timing lag or identify
who is early/late: that would require temporal alignment analysis. Intentional
choreography differences, perspective and detector errors can also cause pose
disagreement. An empty table means no qualifying event was found with sufficient
data; it does not certify perfect synchronization. With only one selected dancer,
use the existing feedback against the solo reference instead of pairwise events.

#### A quieter Results view

Results opens with movement-similarity scores, comparable-frame coverage, the
video overlay, and every detected priority practice moment (up to five), presented
as replay cards with the dancers, joint, peak timestamp, and practice range. Click a suggested
moment to seek the video, or move **Inspect time**. **View detailed feedback**
expands the timeline, strengths and improvement areas, all priority moments,
worst moments, timestamped joint suggestions, the scoring explanation, and CSV
export. The detail panel closes whenever you compare again or open another run,
so each session starts with the short overview. No analysis data is discarded.

The interface uses local system fonts, ivory and graphite colors, muted indigo
accents, and a responsive layout with light and dark color modes. Scores,
comparison rules, and pose analysis are unchanged.

#### Individual dancer reports

Open **Results → View detailed feedback → Feedback report for** and select a
dancer. Their report shows movement similarity, pose coverage, closest region,
and area to review. The summary, worst moments, personalized suggestions, and
current-frame feedback show only that dancer; priority moments include only pairs
involving them. Reference dancers are explicitly not scored. Choose **All dancers**
to restore the group tables. Video, group timeline, top-level score overview, and
CSV export retain the full group context. The selector resets for every comparison
or new run. Missing scores and limited pose coverage are shown explicitly.

With a dancer selected, **Download report** saves their report as a standalone
HTML page (score, coverage, personalized feedback, practice moments involving
them, worst moments, and how scores work) to `runs/<run>/reports/`. It opens in
any browser and prints cleanly, so it can be shared with the dancer directly.

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
