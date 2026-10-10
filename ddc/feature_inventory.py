"""Wide pose-feature inventory, embedded selection, and classifier overlays.

This module reuses an existing ``tracking.jsonl``.  It never invokes pose
detection.  The feature inventory is intentionally broad so the accompanying
workbook can document time, frequency, time-frequency, biomechanics, and
human/context candidates.  Sensitive demographics and health status are not
inferred from video; they are recorded as unavailable until voluntarily
supplied in an appropriate consented data source.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import io as ddc_io
from .dancer_features import CONF_THR, J, load_labels
from .skeleton import BODY, EDGES
from .tracking import stabilize, to_dense


@dataclass(frozen=True)
class FeatureSpec:
    id: str
    category: str
    name: str
    definition: str
    unit: str
    source_signal: str
    predictor_eligible: bool
    availability: str = "Derived from pose data"


def _specs() -> list[FeatureSpec]:
    specs: list[FeatureSpec] = []

    def add(ids, category, signal, unit="normalized units", eligible=True):
        for ident, name, definition in ids:
            specs.append(FeatureSpec(ident, category, name, definition, unit, signal, eligible))

    add([
        ("time_pose_motion_mean", "Mean pose motion", "Mean frame-to-frame all-joint speed."),
        ("time_pose_motion_std", "Pose motion variability", "Standard deviation of all-joint speed."),
        ("time_pose_motion_median", "Median pose motion", "Median all-joint speed."),
        ("time_pose_motion_iqr", "Pose motion IQR", "75th minus 25th percentile of all-joint speed."),
        ("time_pose_motion_p95", "Pose motion 95th percentile", "High-motion tail of all-joint speed."),
        ("time_pose_motion_peak", "Peak pose motion", "Largest all-joint speed."),
        ("time_pose_motion_active_fraction", "Active-motion fraction", "Fraction of frames above a fixed motion threshold."),
        ("time_pose_motion_trend", "Pose-motion trend", "Linear slope of all-joint speed over time."),
        ("time_pose_motion_autocorr_lag1", "Pose-motion lag-1 autocorrelation", "Frame-to-frame persistence of motion."),
        ("time_pose_motion_zero_cross_rate", "Pose-motion zero-crossing rate", "Rate of mean-centered speed direction changes."),
        ("time_wrist_speed_mean", "Mean wrist speed", "Mean speed of visible wrists."),
        ("time_wrist_speed_std", "Wrist-speed variability", "Standard deviation of visible wrist speed."),
        ("time_wrist_speed_p95", "Wrist-speed 95th percentile", "High-motion wrist-speed tail."),
        ("time_wrist_speed_active_fraction", "Active-wrist fraction", "Fraction of frames with actively moving wrists."),
        ("time_wrist_speed_trend", "Wrist-speed trend", "Linear slope of wrist speed over time."),
        ("time_limb_motion_mean", "Mean limb motion", "Mean elbow and knee speed."),
        ("time_limb_motion_std", "Limb-motion variability", "Standard deviation of elbow and knee speed."),
        ("time_limb_motion_p95", "Limb-motion 95th percentile", "High-motion limb-speed tail."),
    ], "Time-domain", "pose, wrist, and limb velocity signals")

    add([
        ("freq_pose_dominant_hz", "Dominant pose frequency", "Strongest non-DC frequency in pose motion."),
        ("freq_pose_centroid_hz", "Pose spectral centroid", "Power-weighted mean pose-motion frequency."),
        ("freq_pose_bandwidth_hz", "Pose spectral bandwidth", "Power-weighted frequency spread."),
        ("freq_pose_entropy", "Pose spectral entropy", "Distribution of pose-motion spectral power."),
        ("freq_pose_flatness", "Pose spectral flatness", "Geometric-to-arithmetic spectral power ratio."),
        ("freq_pose_low_band_ratio", "Pose low-band energy ratio", "Share of motion energy in 0.25–1.5 Hz."),
        ("freq_pose_high_band_ratio", "Pose high-band energy ratio", "Share of motion energy in 1.5–6 Hz."),
        ("freq_wrist_dominant_hz", "Dominant wrist frequency", "Strongest non-DC wrist-speed frequency."),
        ("freq_wrist_centroid_hz", "Wrist spectral centroid", "Power-weighted mean wrist frequency."),
        ("freq_wrist_entropy", "Wrist spectral entropy", "Distribution of wrist-motion spectral power."),
        ("freq_limb_dominant_hz", "Dominant limb frequency", "Strongest non-DC limb-speed frequency."),
        ("freq_limb_high_band_ratio", "Limb high-band energy ratio", "Share of limb energy in 1.5–6 Hz."),
    ], "Frequency-domain", "FFT of motion signals", "Hz or ratio")

    add([
        ("tf_pose_energy_mean", "Mean short-window pose energy", "Mean RMS pose motion across 2-second windows."),
        ("tf_pose_energy_std", "Pose-energy variability", "Standard deviation of short-window RMS pose motion."),
        ("tf_pose_centroid_mean_hz", "Mean short-window pose centroid", "Mean local spectral centroid of pose motion."),
        ("tf_pose_centroid_std_hz", "Pose-centroid variability", "Variation in local pose spectral centroid."),
        ("tf_pose_entropy_mean", "Mean short-window pose entropy", "Mean local spectral entropy of pose motion."),
        ("tf_pose_entropy_std", "Pose-entropy variability", "Variation in local pose spectral entropy."),
        ("tf_pose_spectral_flux", "Pose spectral flux", "Mean change between adjacent local pose spectra."),
        ("tf_wrist_energy_mean", "Mean short-window wrist energy", "Mean RMS wrist speed across 2-second windows."),
        ("tf_wrist_energy_std", "Wrist-energy variability", "Standard deviation of local wrist energy."),
        ("tf_wrist_spectral_flux", "Wrist spectral flux", "Mean change between adjacent local wrist spectra."),
    ], "Time-frequency", "windowed FFT of motion signals", "normalized units or Hz")

    add([
        ("physical_wrist_height_mean", "Mean wrist elevation", "Mean wrist height above hip midpoint in torso lengths."),
        ("physical_wrist_height_std", "Wrist-elevation variability", "Variation in wrist height above hip midpoint."),
        ("physical_limb_extension_mean", "Mean limb extension", "Mean shoulder-to-wrist and hip-to-ankle reach."),
        ("physical_limb_extension_std", "Limb-extension variability", "Variation in limb reach."),
        ("physical_limb_angle_range_rad", "Limb-angle range", "Range of unwrapped limb orientation angles."),
        ("physical_body_sway", "Horizontal body sway", "Standard deviation of hip-midpoint horizontal position."),
        ("physical_vertical_bounce", "Vertical body bounce", "Standard deviation of hip-midpoint vertical position."),
        ("physical_pose_compactness", "Pose compactness", "Mean joint distance from hip midpoint."),
        ("physical_wrist_height_asymmetry", "Wrist-height asymmetry", "Mean absolute left-right wrist-height difference."),
        ("physical_step_width", "Step width", "Mean ankle separation."),
        ("physical_shoulder_tilt_abs_deg", "Absolute shoulder tilt", "Mean absolute shoulder-line tilt."),
        ("physical_wrist_acceleration_rms", "Wrist acceleration RMS", "Root-mean-square change in wrist speed."),
    ], "Domain knowledge: biomechanics", "normalized pose geometry and velocity", "torso lengths, radians, degrees, or normalized units")

    add([
        ("context_frames_present", "Frames present", "Count of clip frames containing the stabilized person track."),
        ("context_track_duration_s", "Observed track duration", "Seconds the person was tracked."),
        ("context_visible_frame_fraction", "Visible-frame fraction", "Share of clip frames containing the stabilized track."),
        ("context_mean_pose_confidence", "Mean pose confidence", "Mean detector confidence across visible body points."),
        ("context_median_person_height_px", "Median detected height", "Median pose bounding-box height."),
    ], "Human and observation context", "tracking metadata", "seconds, fraction, confidence, or pixels",
        eligible=False)
    for ident, name, definition in [
        ("context_median_x_fraction", "Median horizontal location", "Median horizontal person location relative to frame width."),
        ("context_median_y_fraction", "Median vertical location", "Median vertical person location relative to frame height."),
        ("context_center_distance", "Median distance from frame centre", "Median normalized distance of the person from the frame centre."),
    ]:
        specs.append(FeatureSpec(ident, "Human and observation context", name, definition,
                                 "frame fraction", "tracking metadata", False,
                                 "Derived from pose overlay; excluded from predictors because it "
                                 "encodes camera framing, not dancing"))
    for ident, name, definition in [
        ("human_role_annotation", "Reviewed dancer role", "Human-reviewed dancer/non-dancer label; target only."),
        ("human_age_group", "Voluntarily reported age group", "Requires consented participant data; not inferred from video."),
        ("human_gender_identity", "Voluntarily reported gender identity", "Requires consented participant data; not inferred from video."),
        ("human_health_or_injury", "Voluntarily reported health or injury", "Requires consented participant data; not inferred from video."),
        ("human_fatigue_or_exertion", "Voluntarily reported fatigue or exertion", "Requires participant report or validated sensor data; not inferred from video."),
    ]:
        specs.append(FeatureSpec(ident, "Human-related", name, definition, "n.a.",
                                 "external consented data", False,
                                 "Not collected or intentionally not inferred"))
    return specs


FEATURE_SPECS = _specs()
SPEC_BY_ID = {spec.id: spec for spec in FEATURE_SPECS}
DERIVED_FEATURE_IDS = [spec.id for spec in FEATURE_SPECS if spec.availability.startswith("Derived")]
# Context features (clip length, frame position, detected size, confidence) describe
# how the camera saw a person, not how they move.  With few labelled tracks they
# are easy to overfit and do not transfer to a new clip, so they are never predictors.
ELIGIBLE_FEATURE_IDS = [spec.id for spec in FEATURE_SPECS if spec.predictor_eligible]


def _clean(signal: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, float).reshape(-1)
    good = np.isfinite(x)
    if good.sum() < 4:
        return np.array([], float)
    idx = np.arange(len(x))
    return np.interp(idx, idx[good], x[good])


def _finite_mean(signal: np.ndarray) -> float:
    values = np.asarray(signal, float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def _finite_std(signal: np.ndarray) -> float:
    values = np.asarray(signal, float)
    values = values[np.isfinite(values)]
    return float(values.std()) if len(values) else float("nan")


def _motion_signal(points: np.ndarray, scales: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.full(len(points), np.nan)
    for f in range(1, len(points)):
        dt = t[f] - t[f - 1]
        if not (dt > 0 and np.isfinite(scales[f]) and np.isfinite(scales[f - 1])):
            continue
        a, b = points[f - 1], points[f]
        valid = (a[:, 2] >= CONF_THR) & (b[:, 2] >= CONF_THR)
        if valid.any():
            out[f] = np.linalg.norm(b[valid, :2] - a[valid, :2], axis=1).mean() / (
                dt * (scales[f] + scales[f - 1]) / 2)
    return out


def _torso_scale(pose: np.ndarray) -> float:
    idx = [J[n] for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")]
    if not np.all(pose[idx, 2] >= CONF_THR):
        return float("nan")
    shoulders = pose[[J["left_shoulder"], J["right_shoulder"]], :2].mean(axis=0)
    hips = pose[[J["left_hip"], J["right_hip"]], :2].mean(axis=0)
    return float(np.linalg.norm(shoulders - hips))


def _time_features(prefix: str, signal: np.ndarray, threshold: float | None = None) -> dict[str, float]:
    x = _clean(signal)
    out: dict[str, float] = {}
    if not len(x):
        return out
    out[f"{prefix}_mean"] = float(np.mean(x))
    out[f"{prefix}_std"] = float(np.std(x))
    out[f"{prefix}_median"] = float(np.median(x))
    out[f"{prefix}_iqr"] = float(np.percentile(x, 75) - np.percentile(x, 25))
    out[f"{prefix}_p95"] = float(np.percentile(x, 95))
    out[f"{prefix}_peak"] = float(np.max(x))
    out[f"{prefix}_active_fraction"] = float(np.mean(x > (threshold if threshold is not None else np.median(x))))
    out[f"{prefix}_trend"] = float(np.polyfit(np.arange(len(x)), x, 1)[0])
    centered = x - x.mean()
    denom = np.dot(centered, centered)
    out[f"{prefix}_autocorr_lag1"] = float(np.dot(centered[:-1], centered[1:]) / denom) if denom else 0.0
    out[f"{prefix}_zero_cross_rate"] = float(np.mean(np.diff(np.signbit(centered)) != 0)) if len(x) > 1 else 0.0
    return out


def _spectrum(signal: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    x = _clean(signal)
    if len(x) < 8:
        return np.array([]), np.array([])
    x = x - x.mean()
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)
    keep = freqs > 0
    return freqs[keep], power[keep]


def _frequency_features(prefix: str, signal: np.ndarray, fs: float) -> dict[str, float]:
    f, power = _spectrum(signal, fs)
    if not len(power) or power.sum() <= 0:
        return {}
    p = power / power.sum()
    centroid = float(np.sum(f * p))
    out = {
        f"{prefix}_dominant_hz": float(f[np.argmax(power)]),
        f"{prefix}_centroid_hz": centroid,
        f"{prefix}_bandwidth_hz": float(np.sqrt(np.sum(((f - centroid) ** 2) * p))),
        f"{prefix}_entropy": float(-np.sum(p * np.log(p + 1e-12)) / math.log(len(p))),
        f"{prefix}_flatness": float(np.exp(np.mean(np.log(power + 1e-12))) / np.mean(power)),
        f"{prefix}_low_band_ratio": float(power[(f >= .25) & (f < 1.5)].sum() / power.sum()),
        f"{prefix}_high_band_ratio": float(power[(f >= 1.5) & (f <= 6)].sum() / power.sum()),
    }
    return out


def _time_frequency_features(prefix: str, signal: np.ndarray, fs: float) -> dict[str, float]:
    x = _clean(signal)
    if len(x) < 16:
        return {}
    window = min(len(x), max(16, int(round(fs * 2))))
    hop = max(4, window // 2)
    spectra, energy, centroid, entropy = [], [], [], []
    for start in range(0, len(x) - window + 1, hop):
        part = x[start:start + window]
        energy.append(float(np.sqrt(np.mean(part ** 2))))
        f, pwr = _spectrum(part, fs)
        if len(pwr) and pwr.sum() > 0:
            p = pwr / pwr.sum()
            spectra.append(p)
            centroid.append(float(np.sum(f * p)))
            entropy.append(float(-np.sum(p * np.log(p + 1e-12)) / math.log(len(p))))
    if not energy:
        return {}
    flux = []
    for before, after in zip(spectra, spectra[1:]):
        flux.append(float(np.linalg.norm(after - before)))
    return {
        f"{prefix}_energy_mean": float(np.mean(energy)),
        f"{prefix}_energy_std": float(np.std(energy)),
        f"{prefix}_centroid_mean_hz": float(np.mean(centroid)) if centroid else float("nan"),
        f"{prefix}_centroid_std_hz": float(np.std(centroid)) if centroid else float("nan"),
        f"{prefix}_entropy_mean": float(np.mean(entropy)) if entropy else float("nan"),
        f"{prefix}_entropy_std": float(np.std(entropy)) if entropy else float("nan"),
        f"{prefix}_spectral_flux": float(np.mean(flux)) if flux else 0.0,
    }


def _frame_values(poses: np.ndarray, scales: np.ndarray) -> dict[str, np.ndarray]:
    n = len(poses)
    values = {name: np.full(n, np.nan) for name in (
        "wrist_height", "limb_extension", "limb_angle", "body_x", "body_y",
        "compactness", "wrist_asymmetry", "step_width", "shoulder_tilt", "confidence",
    )}
    wrists = [J["left_wrist"], J["right_wrist"]]
    limbs = [(J["left_shoulder"], J["left_wrist"]), (J["right_shoulder"], J["right_wrist"]),
             (J["left_hip"], J["left_ankle"]), (J["right_hip"], J["right_ankle"])]
    for f, pose in enumerate(poses):
        scale = scales[f]
        visible = pose[:, 2] >= CONF_THR
        if visible.any():
            values["confidence"][f] = float(np.mean(pose[visible, 2]))
        if not (np.isfinite(scale) and scale > 0):
            continue
        hips = pose[[J["left_hip"], J["right_hip"]], :2]
        shoulders = pose[[J["left_shoulder"], J["right_shoulder"]], :2]
        if np.all(pose[[J["left_hip"], J["right_hip"]], 2] >= CONF_THR):
            hip_mid = hips.mean(axis=0)
            values["body_x"][f] = hip_mid[0] / scale
            values["body_y"][f] = hip_mid[1] / scale
            if visible.any():
                values["compactness"][f] = np.linalg.norm(pose[visible, :2] - hip_mid, axis=1).mean() / scale
            heights = [(hip_mid[1] - pose[w, 1]) / scale for w in wrists if pose[w, 2] >= CONF_THR]
            if heights:
                values["wrist_height"][f] = float(np.mean(heights))
            if len(heights) == 2:
                values["wrist_asymmetry"][f] = float(abs(heights[0] - heights[1]))
        extensions, angles = [], []
        for parent, child in limbs:
            if pose[parent, 2] >= CONF_THR and pose[child, 2] >= CONF_THR:
                vec = pose[child, :2] - pose[parent, :2]
                extensions.append(np.linalg.norm(vec) / scale)
                angles.append(math.atan2(vec[1], vec[0]))
        if extensions:
            values["limb_extension"][f] = float(np.mean(extensions))
            values["limb_angle"][f] = float(np.mean(angles))
        if pose[J["left_ankle"], 2] >= CONF_THR and pose[J["right_ankle"], 2] >= CONF_THR:
            values["step_width"][f] = float(np.linalg.norm(pose[J["left_ankle"], :2] - pose[J["right_ankle"], :2]) / scale)
        if np.all(pose[[J["left_shoulder"], J["right_shoulder"]], 2] >= CONF_THR):
            vec = shoulders[1] - shoulders[0]
            values["shoulder_tilt"][f] = abs(math.degrees(math.atan2(vec[1], vec[0])))
    return values


def collect_feature_inventory(tracking_jsonl: str | Path) -> list[dict]:
    """Return one wide feature record per persistent pose track."""
    det = ddc_io.load_jsonl(tracking_jsonl)
    if det.n_frames == 0:
        raise ValueError("tracking.jsonl is empty")
    ids, n_tracks = stabilize(det.kps)
    if not n_tracks:
        raise ValueError("no persistent people found")
    dense = to_dense(det.kps, ids, n_tracks)
    fs = 1 / float(np.median(np.diff(det.t))) if len(det.t) > 1 else 30.0
    frame_height = []
    frame_x = []
    frame_y = []
    for tid in range(n_tracks):
        h, x, y = [], [], []
        for people, assigned in zip(det.raw, ids):
            for person, candidate in zip(people, assigned):
                if candidate == tid and person.get("bbox"):
                    box = person["bbox"]
                    h.append(box[3] - box[1])
                    x.append((box[0] + box[2]) / 2)
                    y.append((box[1] + box[3]) / 2)
        frame_height.append(h)
        frame_x.append(x)
        frame_y.append(y)
    # Detector JSON does not carry image dimensions.  The largest observed
    # bounding-box edge is a stable, data-only proxy for the frame boundary.
    # This is preferable to normalizing each track against a different scale.
    frame_width = max((float(person["bbox"][2]) for people in det.raw for person in people
                       if person.get("bbox")), default=1.0)
    frame_height_px = max((float(person["bbox"][3]) for people in det.raw for person in people
                           if person.get("bbox")), default=1.0)

    rows = []
    for tid in range(n_tracks):
        poses = dense[:, tid]
        scales = np.array([_torso_scale(pose) for pose in poses])
        signals = _frame_values(poses, scales)
        pose_motion = _motion_signal(poses, scales, det.t)
        wrist_motion = _motion_signal(poses[:, [J["left_wrist"], J["right_wrist"]]], scales, det.t)
        limb_motion = _motion_signal(poses[:, [J["left_elbow"], J["right_elbow"], J["left_knee"], J["right_knee"]]], scales, det.t)
        values: dict[str, float | str | None] = {spec.id: None for spec in FEATURE_SPECS}
        values.update(_time_features("time_pose_motion", pose_motion, .15))
        values.update({key: val for key, val in _time_features("time_wrist_speed", wrist_motion, .15).items()
                       if key in SPEC_BY_ID})
        values.update({key: val for key, val in _time_features("time_limb_motion", limb_motion, .12).items()
                       if key in SPEC_BY_ID})
        values.update({key: val for key, val in _frequency_features("freq_pose", pose_motion, fs).items()
                       if key in SPEC_BY_ID})
        values.update({key: val for key, val in _frequency_features("freq_wrist", wrist_motion, fs).items()
                       if key in SPEC_BY_ID})
        values.update({key: val for key, val in _frequency_features("freq_limb", limb_motion, fs).items()
                       if key in SPEC_BY_ID})
        values.update({key: val for key, val in _time_frequency_features("tf_pose", pose_motion, fs).items()
                       if key in SPEC_BY_ID})
        values.update({key: val for key, val in _time_frequency_features("tf_wrist", wrist_motion, fs).items()
                       if key in SPEC_BY_ID})
        angle = _clean(signals["limb_angle"])
        wrist_speed = _clean(wrist_motion)
        dt = 1 / fs
        wrist_accel = np.diff(wrist_speed) / dt if len(wrist_speed) > 1 else np.array([])
        physical = {
            "physical_wrist_height_mean": _finite_mean(signals["wrist_height"]),
            "physical_wrist_height_std": _finite_std(signals["wrist_height"]),
            "physical_limb_extension_mean": _finite_mean(signals["limb_extension"]),
            "physical_limb_extension_std": _finite_std(signals["limb_extension"]),
            "physical_limb_angle_range_rad": np.ptp(np.unwrap(angle)) if len(angle) else np.nan,
            "physical_body_sway": _finite_std(signals["body_x"]),
            "physical_vertical_bounce": _finite_std(signals["body_y"]),
            "physical_pose_compactness": _finite_mean(signals["compactness"]),
            "physical_wrist_height_asymmetry": _finite_mean(signals["wrist_asymmetry"]),
            "physical_step_width": _finite_mean(signals["step_width"]),
            "physical_shoulder_tilt_abs_deg": _finite_mean(signals["shoulder_tilt"]),
            "physical_wrist_acceleration_rms": np.sqrt(np.mean(wrist_accel ** 2)) if len(wrist_accel) else np.nan,
        }
        values.update({key: float(val) if np.isfinite(val) else None for key, val in physical.items()})
        frames_present = int(np.isfinite(poses[:, :, 0]).any(axis=1).sum())
        values.update({
            "context_frames_present": float(frames_present),
            "context_track_duration_s": float(frames_present / fs),
            "context_visible_frame_fraction": float(frames_present / len(det.t)),
            "context_mean_pose_confidence": float(np.nanmean(signals["confidence"])),
            "context_median_person_height_px": float(np.median(frame_height[tid])) if frame_height[tid] else None,
            "context_median_x_fraction": float(np.median(frame_x[tid]) / frame_width) if frame_x[tid] else None,
            "context_median_y_fraction": float(np.median(frame_y[tid]) / frame_height_px) if frame_y[tid] else None,
            "context_center_distance": (
                float(math.hypot(np.median(frame_x[tid]) / frame_width - .5,
                                  np.median(frame_y[tid]) / frame_height_px - .5) / math.sqrt(.5))
                if frame_x[tid] else None
            ),
        })
        rows.append({"track_id": tid, "frames_present": frames_present, "values": values})
    return rows


def _fit_l1(z: np.ndarray, y: np.ndarray, l1: float = .025, iterations: int = 1800) -> tuple[np.ndarray, float]:
    coef = np.zeros(z.shape[1])
    intercept = 0.0
    spectral = float(np.linalg.norm(z, ord=2) ** 2 / max(1, len(z)))
    step = 1.0 / max(.25 * spectral, 1e-3)
    for _ in range(iterations):
        logits = np.clip(z @ coef + intercept, -30, 30)
        probability = 1 / (1 + np.exp(-logits))
        residual = probability - y
        new_intercept = intercept - step * float(residual.mean())
        raw = coef - step * (z.T @ residual / len(z))
        new_coef = np.sign(raw) * np.maximum(np.abs(raw) - step * l1, 0)
        if np.max(np.abs(new_coef - coef)) < 1e-8 and abs(new_intercept - intercept) < 1e-8:
            coef, intercept = new_coef, new_intercept
            break
        coef, intercept = new_coef, new_intercept
    return coef, float(intercept)


def embedded_selection(rows: list[dict], labels: dict[int, int], n_features: int = 3) -> tuple[dict, list[dict]]:
    """Fit embedded L1 logistic selection and return report plus track predictions."""
    eligible = list(ELIGIBLE_FEATURE_IDS)
    labelled = [row for row in rows if row["track_id"] in labels]
    if len(labelled) < 4:
        raise ValueError("need four labeled tracks for embedded feature selection")
    raw = np.asarray([[row["values"].get(feature, np.nan) for feature in eligible] for row in labelled], float)
    y = np.asarray([labels[row["track_id"]] for row in labelled], int)
    if set(y) != {0, 1} or min((y == 0).sum(), (y == 1).sum()) < 2:
        raise ValueError("need at least two dancers and two non_dancers")
    # Some physically valid stationary tracks have no non-DC spectrum.  Keep
    # them in the labelled sample and median-impute unavailable measurements;
    # excluding them would inadvertently discard non-dancer evidence.
    usable = np.isfinite(raw).sum(0) >= 2
    eligible = [feature for feature, keep in zip(eligible, usable) if keep]
    raw = raw[:, usable]
    medians = np.nanmedian(raw, axis=0)
    x = np.where(np.isfinite(raw), raw, medians)
    mean, std = x.mean(0), x.std(0)
    std[std < 1e-9] = 1.0
    coef, intercept = _fit_l1((x - mean) / std, y)
    order = np.argsort(np.abs(coef))[::-1]
    selected = [eligible[i] for i in order[:n_features]]
    # In the rare fully-shrunk case, retain the strongest standardized mean difference.
    if not np.any(np.abs(coef)):
        effect = np.abs(x[y == 1].mean(0) - x[y == 0].mean(0)) / std
        order = np.argsort(effect)[::-1]
        selected = [eligible[i] for i in order[:n_features]]
        coef = effect
    selections = []
    for rank, i in enumerate(order[:n_features], 1):
        selections.append({"rank": rank, "feature_id": eligible[i], "feature_name": SPEC_BY_ID[eligible[i]].name,
                           "category": SPEC_BY_ID[eligible[i]].category, "coefficient": float(coef[i]),
                           "absolute_influence": float(abs(coef[i]))})
    idx = [eligible.index(feature) for feature in selected]
    reduced_coef, reduced_intercept = _fit_l1((x[:, idx] - mean[idx]) / std[idx], y, l1=.01)
    predictions = []
    for row in rows:
        values = np.asarray([row["values"].get(feature, np.nan) for feature in selected], float)
        selected_medians = medians[idx]
        values = np.where(np.isfinite(values), values, selected_medians)
        score = float((values - mean[idx]) / std[idx] @ reduced_coef + reduced_intercept)
        probability = float(1 / (1 + math.exp(-max(-30, min(30, score)))))
        predicted = "dancer" if probability >= .5 else "non_dancer"
        predictions.append({"track_id": row["track_id"], "frames_present": row["frames_present"],
                            "reviewed_label": "dancer" if labels.get(row["track_id"]) == 1 else
                                              "non_dancer" if labels.get(row["track_id"]) == 0 else "",
                            "predicted_label": predicted, "dancer_probability": probability})
    training_hits = sum(pred["predicted_label"] == ("dancer" if labels[pred["track_id"]] else "non_dancer")
                        for pred in predictions if pred["track_id"] in labels)
    report = {
        "selection_method": "Embedded L1 logistic regression",
        "labeled_tracks": len(labelled), "dancers": int(y.sum()), "non_dancers": int((y == 0).sum()),
        "top_feature_count": len(selected), "selected_features": selections,
        "training_accuracy": training_hits / len(labelled),
        "excluded_human_features": [spec.id for spec in FEATURE_SPECS if spec.category == "Human-related"],
    }
    return report, predictions


def write_inventory_artifacts(tracking_jsonl: str | Path, labels_csv: str | Path, out_dir: str | Path) -> dict:
    """Create machine-readable wide/long inventory files without rerunning detection."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = collect_feature_inventory(tracking_jsonl)
    report, predictions = embedded_selection(rows, load_labels(labels_csv))
    catalog = [spec.__dict__ for spec in FEATURE_SPECS]
    payload = {"source_tracking_jsonl": str(tracking_jsonl), "catalog": catalog, "tracks": rows,
               "selection": report, "predictions": predictions}
    inventory_json = out_dir / "feature_inventory.json"
    inventory_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (out_dir / "feature_catalog.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(catalog[0]))
        writer.writeheader(); writer.writerows(catalog)
    headers = ["track_id", "frames_present", *[spec.id for spec in FEATURE_SPECS]]
    with (out_dir / "track_feature_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({"track_id": row["track_id"], "frames_present": row["frames_present"], **row["values"]})
    prediction_path = out_dir / "dancer_predictions.csv"
    with prediction_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    selection_path = out_dir / "embedded_feature_selection.json"
    selection_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"inventory": str(inventory_json), "catalog": str(out_dir / "feature_catalog.csv"),
            "matrix": str(out_dir / "track_feature_inventory.csv"), "predictions": str(prediction_path),
            "selection": str(selection_path), **report}


def _source_video(run_dir: Path, video_path: str | Path | None = None) -> Path:
    if video_path:
        return Path(video_path)
    saved = sorted(run_dir.glob("source.*"))
    if saved:
        return saved[0]
    raise ValueError("no source video saved for this run")


def render_classifier_overlay(run_dir: str | Path, show_non_dancers: bool = True,
                              video_path: str | Path | None = None,
                              predictions: list[dict] | None = None) -> dict:
    """Render an annotation-only dancer toggle from existing detections/predictions.

    The original video pixels are preserved.  Turning non-dancers off removes
    their skeleton, label, and JSON pose record; it does not inpaint people out
    of the camera footage.
    """
    import cv2
    from video_pipeline import _cfr_source, _reencode_h264

    run_dir = Path(run_dir)
    pred_path = run_dir / "dancer_predictions.csv"
    if predictions is None:
        if not pred_path.exists():
            raise ValueError("no dancer predictions for this run; validate another run first")
        with pred_path.open(newline="") as f:
            prediction_map = {int(row["track_id"]): row["predicted_label"] for row in csv.DictReader(f)}
    else:
        prediction_map = {int(row["track_id"]): row["predicted_label"] for row in predictions}
    det = ddc_io.load_jsonl(run_dir / "tracking.jsonl")
    ids, _ = stabilize(det.kps)
    source = _source_video(run_dir, video_path)
    # Match the exact CFR path used by analyze_video_rtmw/analyze_video_yolo;
    # direct OpenCV reads can stop early or drift on variable-frame-rate clips.
    source_context = _cfr_source(str(source))
    normalized_source = source_context.__enter__()
    cap = cv2.VideoCapture(str(normalized_source))
    if not cap.isOpened():
        source_context.__exit__(None, None, None)
        raise ValueError(f"could not open source video {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stem = "predicted_all_people" if show_non_dancers else "predicted_dancers_only"
    overlay = run_dir / f"{stem}_overlay.mp4"
    raw_overlay = overlay.with_suffix(".raw.mp4")
    json_out = run_dir / f"{stem}.jsonl"
    writer = cv2.VideoWriter(str(raw_overlay), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    colors = {"dancer": (45, 210, 80), "non_dancer": (70, 140, 255), "unclassified": (180, 180, 180)}
    try:
        with json_out.open("w") as output:
            for frame, (people, assigned) in enumerate(zip(det.raw, ids)):
                ok, image = cap.read()
                if not ok:
                    break
                kept = []
                for person, tid in zip(people, assigned):
                    label = prediction_map.get(int(tid), "unclassified") if tid >= 0 else "unclassified"
                    if not show_non_dancers and label != "dancer":
                        continue
                    clone = dict(person)
                    clone["dancer_prediction"] = label
                    clone["stable_track_id"] = int(tid)
                    kept.append(clone)
                    color = colors[label]
                    kp = person.get("keypoints", {})
                    for a, b in EDGES:
                        pa, pb = kp.get(a), kp.get(b)
                        if pa and pb and pa[2] >= CONF_THR and pb[2] >= CONF_THR:
                            cv2.line(image, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2)
                    if person.get("bbox"):
                        x0, y0, x1, y1 = (int(v) for v in person["bbox"])
                        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
                        cv2.putText(image, f"{label.replace('_', ' ')} #{tid}", (x0, max(20, y0 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2, cv2.LINE_AA)
                output.write(json.dumps({"frame": frame, "t": float(det.t[frame]), "people": kept}) + "\n")
                writer.write(image)
    finally:
        cap.release(); writer.release(); source_context.__exit__(None, None, None)
    _reencode_h264(raw_overlay, overlay)
    selected = []
    selection_path = run_dir / "embedded_feature_selection.json"
    if selection_path.exists():
        selected = json.loads(selection_path.read_text())["selected_features"]
    return {"overlay": str(overlay), "json": str(json_out), "show_non_dancers": show_non_dancers,
            "selected_features": selected}
