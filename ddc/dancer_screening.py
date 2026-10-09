"""Screen persistent pose tracks before composite-pose analysis.

This module consumes existing ``tracking.jsonl`` output.  It does not invoke a
pose model.  It builds a 50-feature, per-track dataset, ranks features with an
embedded L1-logistic model, then produces a dancers-only JSONL and overlay.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import io as ddc_io
from .dancer_features import load_labels
from .skeleton import BODY, CONF_THR, EDGES, J
from .tracking import stabilize, to_dense


@dataclass(frozen=True)
class FeatureSpec:
    category: str
    name: str
    definition: str
    unit: str


# Human-related signals here are recording/visibility conditions only.  The
# screen intentionally does not infer demographic or physiological attributes.
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("Time", "track_duration_s", "Duration from first to last observed frame", "s"),
    FeatureSpec("Time", "visible_frame_fraction", "Share of source frames with a visible tracked pose", "fraction"),
    FeatureSpec("Time", "motion_onset_s", "Time to 10% of cumulative pose motion", "s"),
    FeatureSpec("Time", "motion_peak_time_s", "Time of maximum whole-pose speed", "s"),
    FeatureSpec("Time", "active_motion_fraction", "Share of visible frames above the track motion threshold", "fraction"),
    FeatureSpec("Time", "longest_still_period_s", "Longest run under the stillness speed threshold", "s"),
    FeatureSpec("Time", "motion_temporal_centroid_s", "Motion-energy weighted mean time", "s"),
    FeatureSpec("Time", "motion_time_iqr_s", "Interquartile span of motion-energy time", "s"),
    FeatureSpec("Time", "wrist_motion_onset_s", "Time to 10% of cumulative wrist motion", "s"),
    FeatureSpec("Time", "mean_pose_confidence", "Mean detector confidence across visible body joints", "confidence"),
    FeatureSpec("Frequency", "pose_dominant_frequency_hz", "Dominant non-DC frequency of whole-pose motion", "Hz"),
    FeatureSpec("Frequency", "wrist_dominant_frequency_hz", "Dominant non-DC frequency of wrist motion", "Hz"),
    FeatureSpec("Frequency", "limb_dominant_frequency_hz", "Dominant non-DC frequency of elbow and knee motion", "Hz"),
    FeatureSpec("Frequency", "pose_spectral_entropy", "Normalized spectral entropy of whole-pose motion", "fraction"),
    FeatureSpec("Frequency", "wrist_spectral_entropy", "Normalized spectral entropy of wrist motion", "fraction"),
    FeatureSpec("Frequency", "pose_low_band_power", "Motion energy from 0.1 to 1 Hz", "fraction"),
    FeatureSpec("Frequency", "pose_mid_band_power", "Motion energy from 1 to 3 Hz", "fraction"),
    FeatureSpec("Frequency", "pose_high_band_power", "Motion energy above 3 Hz", "fraction"),
    FeatureSpec("Frequency", "pose_harmonic_ratio", "Second-harmonic to dominant-frequency energy ratio", "ratio"),
    FeatureSpec("Frequency", "wrist_zero_cross_rate_hz", "Mean-centered wrist-motion zero crossings per second", "Hz"),
    FeatureSpec("Time-frequency", "pose_window_energy_mean", "Mean 1-second moving-window motion energy", "torso lengths²/s²"),
    FeatureSpec("Time-frequency", "pose_window_energy_std", "Standard deviation of 1-second motion energy", "torso lengths²/s²"),
    FeatureSpec("Time-frequency", "pose_window_energy_slope", "Slope of windowed motion energy over time", "torso lengths²/s³"),
    FeatureSpec("Time-frequency", "pose_spectral_flux", "Mean frame-to-frame spectral change across windows", "fraction"),
    FeatureSpec("Time-frequency", "wrist_spectral_flux", "Mean wrist-spectrum change across windows", "fraction"),
    FeatureSpec("Time-frequency", "motion_burst_count", "Count of motion-energy bursts", "count"),
    FeatureSpec("Time-frequency", "motion_burst_duration_s", "Mean duration of motion-energy bursts", "s"),
    FeatureSpec("Time-frequency", "high_freq_window_fraction", "Windows dominated by frequencies at or above 2 Hz", "fraction"),
    FeatureSpec("Time-frequency", "low_high_energy_ratio", "Low-band to high-band windowed motion-energy ratio", "ratio"),
    FeatureSpec("Time-frequency", "windowed_motion_cv", "Coefficient of variation of windowed motion energy", "ratio"),
    FeatureSpec("Physical motion", "wrist_height_mean_torso", "Mean wrist height above hip center", "torso lengths"),
    FeatureSpec("Physical motion", "wrist_height_std_torso", "Variation in wrist height above hip center", "torso lengths"),
    FeatureSpec("Physical motion", "wrist_speed_mean_torso_s", "Mean wrist speed", "torso lengths/s"),
    FeatureSpec("Physical motion", "limb_extension_mean_torso", "Mean shoulder-wrist and hip-ankle reach", "torso lengths"),
    FeatureSpec("Physical motion", "limb_extension_std_torso", "Variation in reach length", "torso lengths"),
    FeatureSpec("Physical motion", "pose_motion_mean_torso_s", "Mean whole-pose speed", "torso lengths/s"),
    FeatureSpec("Physical motion", "pose_motion_std_torso_s", "Variation in whole-pose speed", "torso lengths/s"),
    FeatureSpec("Physical motion", "arm_span_torso", "Median wrist-to-wrist span", "torso lengths"),
    FeatureSpec("Physical motion", "leg_stride_torso", "Median ankle-to-ankle span", "torso lengths"),
    FeatureSpec("Physical motion", "left_right_symmetry", "Mean normalized left/right limb-length difference", "ratio"),
    FeatureSpec("Physical motion", "torso_tilt_std_deg", "Standard deviation of shoulder-line tilt", "degrees"),
    FeatureSpec("Physical motion", "knee_flexion_mean_deg", "Mean left/right knee flexion angle", "degrees"),
    FeatureSpec("Physical motion", "elbow_flexion_mean_deg", "Mean left/right elbow flexion angle", "degrees"),
    FeatureSpec("Human / recording conditions", "person_height_median_px", "Median detected person-box height", "px"),
    FeatureSpec("Human / recording conditions", "person_height_cv", "Coefficient of variation of person-box height", "ratio"),
    FeatureSpec("Human / recording conditions", "bounding_box_area_fraction", "Median box area relative to the frame", "fraction"),
    FeatureSpec("Human / recording conditions", "bounding_box_area_cv", "Coefficient of variation of box area", "ratio"),
    FeatureSpec("Human / recording conditions", "visible_joint_fraction", "Share of all joints above confidence threshold", "fraction"),
    FeatureSpec("Human / recording conditions", "occlusion_proxy", "One minus the visible-joint fraction", "fraction"),
    FeatureSpec("Human / recording conditions", "centeredness", "Median distance from frame center, normalized by half-diagonal", "fraction"),
)
FEATURE_NAMES = tuple(spec.name for spec in FEATURE_SPECS)


def _safe_mean(values, default: float = 0.0) -> float:
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else default


def _safe_std(values, default: float = 0.0) -> float:
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return float(a.std()) if a.size else default


def _coefficient_of_variation(values) -> float:
    mean = _safe_mean(values)
    return _safe_std(values) / (abs(mean) + 1e-9)


def _interpolate(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    ix = np.arange(len(values))
    return np.interp(ix, ix[finite], values[finite])


def _torso_scale(pose: np.ndarray) -> float:
    needed = [J[n] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    if not np.all(pose[needed, 2] >= CONF_THR):
        return float("nan")
    shoulders = pose[[J["left_shoulder"], J["right_shoulder"]], :2].mean(0)
    hips = pose[[J["left_hip"], J["right_hip"]], :2].mean(0)
    return float(np.linalg.norm(shoulders - hips))


def _speed_signal(poses: np.ndarray, scales: np.ndarray, t: np.ndarray,
                  joint_ids: list[int]) -> np.ndarray:
    signal = np.zeros(len(poses), float)
    for f in range(1, len(poses)):
        dt = float(t[f] - t[f - 1])
        if dt <= 0 or not (np.isfinite(scales[f]) and np.isfinite(scales[f - 1])):
            continue
        before, after = poses[f - 1, joint_ids], poses[f, joint_ids]
        valid = (before[:, 2] >= CONF_THR) & (after[:, 2] >= CONF_THR)
        if valid.any():
            d = np.linalg.norm(after[valid, :2] - before[valid, :2], axis=1).mean()
            signal[f] = d / (dt * (scales[f] + scales[f - 1]) / 2)
    return signal


def _dominant_frequency(signal: np.ndarray, fps: float) -> tuple[float, np.ndarray, np.ndarray]:
    signal = _interpolate(signal)
    signal = signal - signal.mean()
    if len(signal) < 4 or np.allclose(signal, 0):
        return 0.0, np.array([0.0]), np.array([0.0])
    power = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1 / max(fps, 1e-9))
    if len(power) <= 1:
        return 0.0, freqs, power
    index = 1 + int(np.argmax(power[1:]))
    return float(freqs[index]), freqs, power


def _spectral_entropy(power: np.ndarray) -> float:
    p = np.asarray(power[1:], float)
    total = p.sum()
    if len(p) < 2 or total <= 1e-12:
        return 0.0
    p = p / total
    return float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))


def _band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    total = power[1:].sum()
    if total <= 1e-12:
        return 0.0
    return float(power[(freqs >= low) & (freqs < high)].sum() / total)


def _energy_windows(signal: np.ndarray, fps: float) -> tuple[np.ndarray, list[np.ndarray], int]:
    window = max(4, int(round(fps)))
    step = max(1, window // 2)
    signal = _interpolate(signal)
    if len(signal) < window:
        signal = np.pad(signal, (0, window - len(signal)), mode="edge")
    segments = [signal[start:start + window] for start in range(0, len(signal) - window + 1, step)]
    if not segments:
        segments = [signal]
    energy = np.array([float(np.mean(part ** 2)) for part in segments])
    return energy, segments, step


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    den = np.linalg.norm(u) * np.linalg.norm(v)
    if den <= 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / den, -1, 1))))


def _run_lengths(mask: np.ndarray) -> list[int]:
    lengths: list[int] = []
    current = 0
    for item in mask:
        if item:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def _feature_values(poses: np.ndarray, t: np.ndarray, bboxes: list[list[float]],
                    frame_width: int, frame_height: int) -> dict[str, float]:
    fps = 1 / max(float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30, 1e-6)
    scales = np.array([_torso_scale(p) for p in poses])
    pose_speed = _speed_signal(poses, scales, t, list(range(len(BODY))))
    wrist_speed = _speed_signal(poses, scales, t, [J["left_wrist"], J["right_wrist"]])
    limb_speed = _speed_signal(poses, scales, t, [J["left_elbow"], J["right_elbow"], J["left_knee"], J["right_knee"]])
    energy, windows, window_step = _energy_windows(pose_speed, fps)
    wrist_energy, wrist_windows, _ = _energy_windows(wrist_speed, fps)
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    visible = (poses[:, :, 2] >= CONF_THR) & np.isfinite(poses[:, :, 0])
    valid_frames = visible.any(axis=1)
    pose_power_f, pose_freqs, pose_power = _dominant_frequency(pose_speed, fps)
    wrist_power_f, wrist_freqs, wrist_power = _dominant_frequency(wrist_speed, fps)
    limb_power_f, _, _ = _dominant_frequency(limb_speed, fps)
    peak_index = int(np.argmax(pose_speed)) if len(pose_speed) else 0
    cumulative = np.cumsum(pose_speed)
    wrist_cumulative = np.cumsum(wrist_speed)
    onset = int(np.searchsorted(cumulative, cumulative[-1] * .10)) if cumulative.size and cumulative[-1] > 0 else 0
    wrist_onset = int(np.searchsorted(wrist_cumulative, wrist_cumulative[-1] * .10)) if wrist_cumulative.size and wrist_cumulative[-1] > 0 else 0
    if cumulative.size and cumulative[-1] > 0:
        q25 = int(np.searchsorted(cumulative, cumulative[-1] * .25))
        q75 = int(np.searchsorted(cumulative, cumulative[-1] * .75))
        centroid = float(np.average(t - t[0], weights=pose_speed + 1e-12))
        motion_iqr = float(t[q75] - t[q25])
    else:
        centroid = motion_iqr = 0.0
    still_runs = _run_lengths(pose_speed < 0.05)
    threshold = float(np.median(pose_speed) + 0.5 * np.std(pose_speed))
    burst_runs = _run_lengths(energy > (energy.mean() + 0.5 * energy.std()))
    spectra = []
    wrist_spectra = []
    dominant_windows = []
    for part, wrist_part in zip(windows, wrist_windows):
        _, freqs, power = _dominant_frequency(part, fps)
        _, _, wp = _dominant_frequency(wrist_part, fps)
        spectra.append(power / (power.sum() + 1e-12))
        wrist_spectra.append(wp / (wp.sum() + 1e-12))
        dominant_windows.append(_dominant_frequency(part, fps)[0])
    flux = _safe_mean([np.linalg.norm(spectra[i] - spectra[i - 1]) for i in range(1, len(spectra))])
    wrist_flux = _safe_mean([np.linalg.norm(wrist_spectra[i] - wrist_spectra[i - 1]) for i in range(1, len(wrist_spectra))])
    harmonic_index = int(np.argmax(pose_power[1:]) + 1) if len(pose_power) > 1 else 0
    harmonic = pose_power[min(harmonic_index * 2, len(pose_power) - 1)] if harmonic_index else 0
    fundamental = pose_power[harmonic_index] if harmonic_index else 0

    wrist_heights: list[float] = []
    extensions: list[float] = []
    arm_spans: list[float] = []
    leg_spans: list[float] = []
    symmetry: list[float] = []
    torso_tilts: list[float] = []
    knee_angles: list[float] = []
    elbow_angles: list[float] = []
    for p, scale in zip(poses, scales):
        if not np.isfinite(scale) or scale <= 0:
            continue
        hips = p[[J["left_hip"], J["right_hip"]], :2].mean(0)
        for wrist in (J["left_wrist"], J["right_wrist"]):
            if p[wrist, 2] >= CONF_THR:
                wrist_heights.append((hips[1] - p[wrist, 1]) / scale)
        lengths = []
        for parent, child in (("left_shoulder", "left_wrist"), ("right_shoulder", "right_wrist"),
                              ("left_hip", "left_ankle"), ("right_hip", "right_ankle")):
            a, b = J[parent], J[child]
            if p[a, 2] >= CONF_THR and p[b, 2] >= CONF_THR:
                lengths.append(float(np.linalg.norm(p[a, :2] - p[b, :2]) / scale))
        extensions.extend(lengths)
        if p[J["left_wrist"], 2] >= CONF_THR and p[J["right_wrist"], 2] >= CONF_THR:
            arm_spans.append(float(np.linalg.norm(p[J["left_wrist"], :2] - p[J["right_wrist"], :2]) / scale))
        if p[J["left_ankle"], 2] >= CONF_THR and p[J["right_ankle"], 2] >= CONF_THR:
            leg_spans.append(float(np.linalg.norm(p[J["left_ankle"], :2] - p[J["right_ankle"], :2]) / scale))
        if len(lengths) == 4:
            symmetry.append(float((abs(lengths[0] - lengths[1]) + abs(lengths[2] - lengths[3])) / (sum(lengths) + 1e-9)))
        if p[J["left_shoulder"], 2] >= CONF_THR and p[J["right_shoulder"], 2] >= CONF_THR:
            v = p[J["right_shoulder"], :2] - p[J["left_shoulder"], :2]
            torso_tilts.append(float(np.degrees(np.arctan2(v[1], v[0]))))
        for hip, knee, ankle in (("left_hip", "left_knee", "left_ankle"), ("right_hip", "right_knee", "right_ankle")):
            a, b, c = J[hip], J[knee], J[ankle]
            if min(p[a, 2], p[b, 2], p[c, 2]) >= CONF_THR:
                knee_angles.append(_angle(p[a, :2], p[b, :2], p[c, :2]))
        for shoulder, elbow, wrist in (("left_shoulder", "left_elbow", "left_wrist"), ("right_shoulder", "right_elbow", "right_wrist")):
            a, b, c = J[shoulder], J[elbow], J[wrist]
            if min(p[a, 2], p[b, 2], p[c, 2]) >= CONF_THR:
                elbow_angles.append(_angle(p[a, :2], p[b, :2], p[c, :2]))

    bbox_array = np.asarray(bboxes, float) if bboxes else np.empty((0, 4))
    widths = bbox_array[:, 2] - bbox_array[:, 0] if len(bbox_array) else np.array([])
    heights = bbox_array[:, 3] - bbox_array[:, 1] if len(bbox_array) else np.array([])
    areas = widths * heights if len(widths) else np.array([])
    centers = np.column_stack(((bbox_array[:, 0] + bbox_array[:, 2]) / 2,
                               (bbox_array[:, 1] + bbox_array[:, 3]) / 2)) if len(bbox_array) else np.empty((0, 2))
    half_diag = math.hypot(frame_width / 2, frame_height / 2)
    center_dist = np.linalg.norm(centers - [frame_width / 2, frame_height / 2], axis=1) / max(half_diag, 1) if len(centers) else np.array([])

    return {
        "track_duration_s": duration,
        "visible_frame_fraction": float(valid_frames.mean()),
        "motion_onset_s": float(t[min(onset, len(t) - 1)] - t[0]),
        "motion_peak_time_s": float(t[min(peak_index, len(t) - 1)] - t[0]),
        "active_motion_fraction": float((pose_speed > threshold).mean()),
        "longest_still_period_s": max(still_runs, default=0) / fps,
        "motion_temporal_centroid_s": centroid,
        "motion_time_iqr_s": motion_iqr,
        "wrist_motion_onset_s": float(t[min(wrist_onset, len(t) - 1)] - t[0]),
        "mean_pose_confidence": _safe_mean(poses[:, :, 2][visible]),
        "pose_dominant_frequency_hz": pose_power_f,
        "wrist_dominant_frequency_hz": wrist_power_f,
        "limb_dominant_frequency_hz": limb_power_f,
        "pose_spectral_entropy": _spectral_entropy(pose_power),
        "wrist_spectral_entropy": _spectral_entropy(wrist_power),
        "pose_low_band_power": _band_power(pose_freqs, pose_power, .1, 1),
        "pose_mid_band_power": _band_power(pose_freqs, pose_power, 1, 3),
        "pose_high_band_power": _band_power(pose_freqs, pose_power, 3, float("inf")),
        "pose_harmonic_ratio": float(harmonic / (fundamental + 1e-12)),
        "wrist_zero_cross_rate_hz": float(np.count_nonzero(np.diff(np.signbit(wrist_speed - wrist_speed.mean()))) / max(duration, 1e-9)),
        "pose_window_energy_mean": _safe_mean(energy),
        "pose_window_energy_std": _safe_std(energy),
        "pose_window_energy_slope": float(np.polyfit(np.arange(len(energy)) * window_step / fps, energy, 1)[0]) if len(energy) > 1 else 0.0,
        "pose_spectral_flux": flux,
        "wrist_spectral_flux": wrist_flux,
        "motion_burst_count": float(len(burst_runs)),
        "motion_burst_duration_s": _safe_mean(np.asarray(burst_runs) * window_step / fps),
        "high_freq_window_fraction": float(np.mean(np.asarray(dominant_windows) >= 2)) if dominant_windows else 0.0,
        "low_high_energy_ratio": float(_band_power(pose_freqs, pose_power, .1, 1) / (_band_power(pose_freqs, pose_power, 3, float("inf")) + 1e-9)),
        "windowed_motion_cv": _coefficient_of_variation(energy),
        "wrist_height_mean_torso": _safe_mean(wrist_heights),
        "wrist_height_std_torso": _safe_std(wrist_heights),
        "wrist_speed_mean_torso_s": _safe_mean(wrist_speed),
        "limb_extension_mean_torso": _safe_mean(extensions),
        "limb_extension_std_torso": _safe_std(extensions),
        "pose_motion_mean_torso_s": _safe_mean(pose_speed),
        "pose_motion_std_torso_s": _safe_std(pose_speed),
        "arm_span_torso": _safe_mean(arm_spans),
        "leg_stride_torso": _safe_mean(leg_spans),
        "left_right_symmetry": _safe_mean(symmetry),
        "torso_tilt_std_deg": _safe_std(torso_tilts),
        "knee_flexion_mean_deg": _safe_mean(knee_angles),
        "elbow_flexion_mean_deg": _safe_mean(elbow_angles),
        "person_height_median_px": float(np.median(heights)) if len(heights) else 0.0,
        "person_height_cv": _coefficient_of_variation(heights),
        "bounding_box_area_fraction": float(np.median(areas) / max(frame_width * frame_height, 1)) if len(areas) else 0.0,
        "bounding_box_area_cv": _coefficient_of_variation(areas),
        "visible_joint_fraction": float(visible.mean()),
        "occlusion_proxy": float(1 - visible.mean()),
        "centeredness": _safe_mean(center_dist),
    }


def collect_extended_features(tracking_jsonl: str | Path, frame_width: int = 1280,
                              frame_height: int = 720) -> list[dict]:
    """Return all 50 feature values for every persistent track in a detection run."""
    det = ddc_io.load_jsonl(tracking_jsonl)
    if det.n_frames == 0:
        raise ValueError("tracking.jsonl is empty")
    ids, n_tracks = stabilize(det.kps)
    if not n_tracks:
        raise ValueError("no persistent people found")
    dense = to_dense(det.kps, ids, n_tracks)
    bboxes = [[] for _ in range(n_tracks)]
    frame_mask = [np.zeros(det.n_frames, bool) for _ in range(n_tracks)]
    for frame, (people, frame_ids) in enumerate(zip(det.raw, ids)):
        for person, tid in zip(people, frame_ids):
            if tid >= 0:
                bbox = person.get("bbox")
                if bbox and len(bbox) == 4:
                    bboxes[tid].append(bbox)
                frame_mask[tid][frame] = True
    rows = []
    for tid in range(n_tracks):
        row = {"track_id": tid, "frames_present": int(frame_mask[tid].sum())}
        row.update(_feature_values(dense[:, tid], det.t, bboxes[tid], frame_width, frame_height))
        rows.append(row)
    return rows


def write_feature_catalog(path: str | Path) -> Path:
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "feature", "definition", "unit"])
        writer.writeheader()
        writer.writerows({"category": x.category, "feature": x.name, "definition": x.definition, "unit": x.unit}
                        for x in FEATURE_SPECS)
    return path


def write_track_features(rows: list[dict], path: str | Path) -> Path:
    path = Path(path)
    columns = ["track_id", "frames_present", *FEATURE_NAMES]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _fit_l1_logistic(x: np.ndarray, y: np.ndarray, l1: float = 0.08,
                     iterations: int = 3000) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    coef = np.zeros(z.shape[1])
    intercept = 0.0
    spectral = float(np.linalg.norm(z, ord=2) ** 2 / max(len(z), 1))
    step = 1.0 / max(.25 * spectral, 1e-3)
    for _ in range(iterations):
        logits = np.clip(z @ coef + intercept, -30, 30)
        residual = 1 / (1 + np.exp(-logits)) - y
        next_intercept = intercept - step * residual.mean()
        raw = coef - step * (z.T @ residual / len(z))
        next_coef = np.sign(raw) * np.maximum(np.abs(raw) - step * l1, 0)
        if np.max(np.abs(next_coef - coef)) < 1e-8 and abs(next_intercept - intercept) < 1e-8:
            coef, intercept = next_coef, next_intercept
            break
        coef, intercept = next_coef, next_intercept
    return mean, scale, coef, float(intercept)


def select_and_score(rows: list[dict], labels: dict[int, int], n_features: int = 3) -> dict:
    """Use embedded L1 logistic selection and score every persistent track."""
    train = [row for row in rows if row["track_id"] in labels]
    if len(train) < 4 or min(sum(labels[row["track_id"]] == c for row in train) for c in (0, 1)) < 2:
        raise ValueError("need at least two labeled dancers and two labeled non_dancers")
    x = np.asarray([[row[name] for name in FEATURE_NAMES] for row in train], float)
    y = np.asarray([labels[row["track_id"]] for row in train], float)
    mean, scale, coef, intercept = _fit_l1_logistic(x, y)
    # Preserve a complete top-three ranking if regularization zeros all but one.
    importance = np.abs(coef)
    if not np.any(importance):
        class_gap = np.abs(x[y == 1].mean(0) - x[y == 0].mean(0)) / (x.std(0) + 1e-9)
        importance = class_gap
    selected_idx = list(np.argsort(importance)[::-1][:n_features])
    # Classify only from the selected embedded features, retaining an interpretable
    # score.  Refit lets the selected set set the boundary without the other 47.
    selected_x = x[:, selected_idx]
    sm, ss, sc, si = _fit_l1_logistic(selected_x, y, l1=0.01)
    ranked = []
    dancer_ids = []
    for row in rows:
        values = np.asarray([row[FEATURE_NAMES[i]] for i in selected_idx], float)
        logit = float(np.clip(((values - sm) / ss) @ sc + si, -30, 30))
        probability = float(1 / (1 + np.exp(-logit)))
        predicted = int(probability >= .5)
        if predicted:
            dancer_ids.append(row["track_id"])
        ranked.append({
            "track_id": row["track_id"],
            "known_label": "dancer" if labels.get(row["track_id"]) == 1 else "non_dancer" if row["track_id"] in labels else "unlabeled",
            "dancer_probability": probability,
            "predicted_label": "dancer" if predicted else "non_dancer",
        })
    feature_rank = []
    for rank, index in enumerate(np.argsort(importance)[::-1], 1):
        spec = FEATURE_SPECS[index]
        feature_rank.append({
            "rank": rank, "feature": spec.name, "category": spec.category,
            "embedded_importance": float(importance[index]), "selected": rank <= n_features,
        })
    return {
        "method": "Embedded L1 logistic regression",
        "training_tracks": len(train),
        "dancers": int(y.sum()),
        "non_dancers": int((y == 0).sum()),
        "selected_features": [FEATURE_NAMES[i] for i in selected_idx],
        "feature_ranking": feature_rank,
        "track_predictions": ranked,
        "dancer_track_ids": sorted(dancer_ids),
    }


def filter_tracking_jsonl(tracking_jsonl: str | Path, output_jsonl: str | Path,
                          dancer_track_ids: set[int]) -> Path:
    """Write detector output with only the selected stable tracks retained."""
    det = ddc_io.load_jsonl(tracking_jsonl)
    ids, _ = stabilize(det.kps)
    output_jsonl = Path(output_jsonl)
    with output_jsonl.open("w") as f:
        for frame, (people, track_ids) in enumerate(zip(det.raw, ids)):
            retained = [person for person, tid in zip(people, track_ids) if tid in dancer_track_ids]
            f.write(json.dumps({"frame": frame, "t": float(det.t[frame]), "people": retained}) + "\n")
    return output_jsonl


def render_filtered_overlay(tracking_jsonl: str | Path, source_video: str | Path,
                            output_video: str | Path, dancer_track_ids: set[int]) -> Path:
    """Render only retained body skeletons over the original source video."""
    det = ddc_io.load_jsonl(tracking_jsonl)
    ids, _ = stabilize(det.kps)
    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise ValueError(f"could not open source video {source_video!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = Path(output_video)
    raw = output_video.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        for people, track_ids in zip(det.raw, ids):
            ok, frame = cap.read()
            if not ok:
                break
            for person, tid in zip(people, track_ids):
                if tid not in dancer_track_ids:
                    continue
                points = person.get("keypoints", {})
                for left, right in EDGES:
                    a, b = points.get(left), points.get(right)
                    if a and b and a[2] >= CONF_THR and b[2] >= CONF_THR:
                        cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 220, 70), 3, cv2.LINE_AA)
                for name in BODY:
                    p = points.get(name)
                    if p and p[2] >= CONF_THR:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 4, (255, 255, 255), -1, cv2.LINE_AA)
                bbox = person.get("bbox")
                if bbox:
                    cv2.putText(frame, f"Dancer {tid + 1}", (int(bbox[0]), max(22, int(bbox[1]) - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 70), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()
    # Keep the overlay web-friendly without running pose detection again.
    from video_pipeline import _reencode_h264
    _reencode_h264(raw, output_video)
    return output_video


def screen_and_filter_run(run_dir: str | Path, source_video: str | Path,
                          labels_csv: str | Path, frame_width: int = 1280,
                          frame_height: int = 720) -> dict:
    """The post-detection, pre-composite screening stage for one video run."""
    run_dir = Path(run_dir)
    tracking = run_dir / "tracking.jsonl"
    rows = collect_extended_features(tracking, frame_width, frame_height)
    catalog = write_feature_catalog(run_dir / "feature_catalog.csv")
    features = write_track_features(rows, run_dir / "extended_track_features.csv")
    result = select_and_score(rows, load_labels(labels_csv), n_features=3)
    retained = set(result["dancer_track_ids"])
    filtered_jsonl = filter_tracking_jsonl(tracking, run_dir / "tracking_dancers_only.jsonl", retained)
    filtered_overlay = render_filtered_overlay(tracking, source_video, run_dir / "dancers_only_overlay.mp4", retained)
    predictions_path = run_dir / "embedded_feature_screening.json"
    result.update({
        "feature_catalog_csv": str(catalog),
        "feature_values_csv": str(features),
        "filtered_tracking_jsonl": str(filtered_jsonl),
        "filtered_overlay": str(filtered_overlay),
    })
    predictions_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["screening_json"] = str(predictions_path)
    return result

