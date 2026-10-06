"""Body-joint constants shared across ddc modules (COCO-17)."""

from __future__ import annotations

BODY = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
J = {n: i for i, n in enumerate(BODY)}

EDGES = [
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
]
EDGE_IDX = [(J[a], J[b]) for a, b in EDGES]

# Limb segments (parent, child) used for angle-based deviation.
LIMBS = [
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]
LIMB_IDX = [(J[a], J[b]) for a, b in LIMBS]

GROUPS = {
    "head": [J[n] for n in BODY[:5]],
    "arms": [J[n] for n in ("left_shoulder", "right_shoulder", "left_elbow",
                            "right_elbow", "left_wrist", "right_wrist")],
    "torso": [J["left_hip"], J["right_hip"]],
    "legs": [J[n] for n in ("left_knee", "right_knee", "left_ankle", "right_ankle")],
}

CONF_THR = 0.3
