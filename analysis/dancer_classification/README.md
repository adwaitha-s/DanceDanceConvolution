# Dancer vs. non-dancer: which pose features separate them?

Feature extraction and selection for telling dancers from bystanders using only pose
keypoints. It is a first step toward replacing the manual "pick the dancers" step in
`detect_tracks` (`ddc/analyze.py`) with an automatic one.

## Data and outcome
- **Run:** `runs/20261007-133746` (RTMW pose, 30 fps, 31 s). Three dancers stand in a row; a
  group of seated spectators is at the right edge. `stabilize()` finds 18 tracks.
- **Outcome:** `is_dancer`. Dancers = tracks 1, 2, 3 (recovered by reproducing that run's
  `deviation.csv`, and checked on the overlay video). Every other track is a spectator.
- **Samples:** one tracked person x one 1.5 s window (hop 0.5 s), 323 windows
  (173 dancer / 150 spectator; 3 dancer tracks, 9 spectator tracks).

## Features (`extract_features.py`, 21, pose signal only)
| Group | Features |
|---|---|
| Time | `wrist_speed_mean`, `wrist_jerk_rms`, `ankle_speed_mean`, `body_sway_std` |
| Frequency | `wrist_dom_freq_hz`, `beat_band_frac`, `wrist_spec_entropy` |
| Time-frequency | `stft_centroid_hz`, `stft_flux` (STFT of wrist height) |
| Domain knowledge | `leg_extension`, `knee_extension`, `arm_reach`, `wrist_height`, `lr_wrist_asym`, `torso_lean_deg`, `travel_speed` |
| Human / context | `visible_joint_frac`, `mean_kp_conf`, `torso_px`, `x_pos_px`, `y_pos_px` |

Positions are in torso lengths (hip-centred, via `ddc.normalize`) so body size and camera
distance cancel, except the context group. The context features are **scene shortcuts**
here (spectators sit on the right, farther from the camera: `x_pos_px` alone has AUC 0.98),
so the headline uses the 16 pose-only features.

## Method and result
`compare_methods.py` ranks features with four methods (mutual information, random forest,
LASSO / L1 logistic, RFE) **inside each training fold** and scores the top 5 on held-out
data. Two strict splits are used because overlapping windows leak under shuffled k-fold:
held-out 6 s time blocks (with overlapping training windows purged) and held-out people.
All four methods tie (AUC 0.98 to 0.99), so `final_selection.py` uses **LASSO**: the
features are highly redundant (e.g. `leg_extension` ~ `knee_extension`, r = 0.92), and L1
keeps one of a pair and gives signed weights.

Final set (LASSO top 5, minus `stft_flux` whose weight is about 0 once `stft_centroid_hz` is in):
`leg_extension` (+), `knee_extension` (+), `arm_reach` (-), `stft_centroid_hz` (+).

| Held out | AUC | All 16 features |
|---|---|---|
| People never seen in training | 0.984 | 0.988 |
| Unseen time blocks | 0.979 | 0.983 |

Leg extension alone reaches AUC about 0.975. See `results/` for figures, `ranking.csv`
and `summary.txt`.

## Caveats
- The non-dancers are **seated**, so this is largely a standing-vs-sitting detector. It is
  untested on standing non-dancers.
- One clip, one camera, three dancers. The 4th selected feature is unstable across folds;
  the three posture features are not.
- The 4-feature cut (dropping near-zero weights) was decided after seeing the full-data fit;
  its score is computed with that rule redone inside each fold.

## Reproduce
```bash
pip install -r analysis/dancer_classification/requirements.txt
# optional, needs the run's tracking.jsonl (runs/ is gitignored); features.csv is committed
python analysis/dancer_classification/extract_features.py --run-dir runs/20261007-133746
python analysis/dancer_classification/compare_methods.py
python analysis/dancer_classification/final_selection.py
```
Everything is seeded; repeated runs give identical output.
