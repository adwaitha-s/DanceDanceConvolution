"""The fixed, auditable 40-candidate catalog for the 15-video cohort."""

from dataclasses import replace

from .feature_inventory import SPEC_BY_ID

# Thirty-five pose-derived candidates and five external human annotations.
FEATURE_IDS = [
    "time_pose_motion_mean", "time_pose_motion_std", "time_pose_motion_median",
    "time_pose_motion_iqr", "time_pose_motion_p95", "time_pose_motion_active_fraction",
    "time_pose_motion_autocorr_lag1", "time_wrist_speed_mean",
    "time_wrist_speed_std", "time_limb_motion_mean",
    "freq_pose_dominant_hz", "freq_pose_centroid_hz", "freq_pose_bandwidth_hz",
    "freq_pose_entropy", "freq_pose_flatness", "freq_pose_low_band_ratio",
    "freq_pose_high_band_ratio", "freq_wrist_dominant_hz",
    "tf_pose_energy_mean", "tf_pose_energy_std", "tf_pose_centroid_mean_hz",
    "tf_pose_centroid_std_hz", "tf_pose_entropy_mean", "tf_pose_entropy_std",
    "tf_pose_spectral_flux", "tf_wrist_spectral_flux",
    "physical_wrist_height_mean", "physical_limb_extension_mean",
    "physical_limb_angle_range_rad", "physical_body_sway", "physical_vertical_bounce",
    "physical_pose_compactness", "physical_wrist_height_asymmetry",
    "physical_step_width", "physical_wrist_acceleration_rms",
    "human_role_annotation", "human_age_group", "human_gender_identity",
    "human_health_or_injury", "human_fatigue_or_exertion",
]
def _unit(fid):
    if fid.startswith("human_"):
        return "external annotation"
    if fid.endswith("_hz"):
        return "Hz"
    if any(part in fid for part in ("_fraction", "_autocorr", "_entropy", "_flatness", "_ratio", "_flux")):
        return "dimensionless"
    if fid == "physical_limb_angle_range_rad":
        return "radians"
    if fid == "physical_wrist_acceleration_rms":
        return "torso lengths / s²"
    if fid.startswith("physical_"):
        return "torso lengths"
    return "torso lengths / s"


FEATURES = [replace(SPEC_BY_ID[fid], unit=_unit(fid)) for fid in FEATURE_IDS]
PREDICTOR_IDS = [s.id for s in FEATURES if s.predictor_eligible]
