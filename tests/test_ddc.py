import numpy as np
from pathlib import Path

from ddc.composite import build_composite, geometric_median
from ddc.deviation import compute_deviation
from ddc.normalize import normalize
from ddc.skeleton import BODY, J
from ddc.tracking import stabilize, to_dense


def base_pose():
    p = np.zeros((17, 3))
    p[:, 2] = 0.9
    coords = {
        "nose": (0, -2.4), "left_eye": (-.1, -2.5), "right_eye": (.1, -2.5),
        "left_ear": (-.2, -2.4), "right_ear": (.2, -2.4),
        "left_shoulder": (-.5, -1), "right_shoulder": (.5, -1),
        "left_elbow": (-.7, 0), "right_elbow": (.7, 0),
        "left_wrist": (-.7, 1), "right_wrist": (.7, 1),
        "left_hip": (-.3, 0), "right_hip": (.3, 0),
        "left_knee": (-.3, 1.5), "right_knee": (.3, 1.5),
        "left_ankle": (-.3, 3), "right_ankle": (.3, 3),
    }
    for n, xy in coords.items():
        p[J[n], :2] = xy
    return p


def place(p, cx, cy, s):
    q = p.copy()
    q[:, :2] = q[:, :2] * s + [cx, cy]
    return q


def neutral_pose():
    """A different pose shape (arms straight down) -- used as the "bystander":
    normalize() strips out screen position/scale, so a bystander test needs an
    actually different pose, not just a different spot on screen."""
    p = base_pose()
    p[J["left_elbow"], :2] = (-.5, -0.2)
    p[J["right_elbow"], :2] = (.5, -0.2)
    p[J["left_wrist"], :2] = (-.5, 0.6)
    p[J["right_wrist"], :2] = (.5, 0.6)
    return p


def test_geometric_median_robust_to_outlier():
    pts = np.array([[0, 0], [0.01, 0], [0, 0.01], [50, 50.0]])
    m = geometric_median(pts, np.ones(4))
    assert np.linalg.norm(m) < 0.1


def test_tracking_fixes_swapped_ids_and_duplicates():
    a, b = base_pose(), base_pose()
    frames = []
    for f in range(40):
        pa = place(a, 100 + f, 300, 40)
        pb = place(b, 500 - f, 300, 40)
        people = [pa, pb] if f % 2 else [pb, pa]          # swap order every frame
        if f == 20:
            people.append(pa + np.array([1, 1, 0]))        # duplicate detection
        if f == 30:
            people = people[:1]                            # dropped detection
        frames.append(np.stack(people))
    ids, n = stabilize(frames)
    assert n == 2
    dense = to_dense(frames, ids, n)
    # each track moves consistently: track x positions monotone in one direction
    xa = dense[:, :, J["nose"], 0]
    for t in range(2):
        v = xa[:, t][np.isfinite(xa[:, t])]
        assert abs(np.diff(v)).max() < 5


def _dense(offset_joint=None, n=3, F=10):
    d = np.full((F, n, 17, 3), np.nan)
    for t in range(n):
        for f in range(F):
            d[f, t] = place(base_pose(), 200 * (t + 1), 300, 30 + 5 * t)  # differing position/scale
    if offset_joint is not None:
        d[:, 0, J[offset_joint], 0] += 30 * 1.0   # 1 torso length at scale 30
    return d


def test_identical_dancers_have_zero_deviation_scale_translation_invariant():
    norm, *_ = normalize(_dense())
    dev = compute_deviation(norm)
    assert np.nanmax(dev.score) < 1e-6


def test_injected_offset_found_on_right_dancer_and_joint():
    norm, *_ = normalize(_dense("right_wrist"))
    dev = compute_deviation(norm, leave_one_out=True)
    assert np.nanargmax(np.nanmean(dev.score, 0)) == 0
    assert np.nanargmax(np.nanmean(dev.joint[:, 0], 0)) == J["right_wrist"]
    assert np.nanmean(dev.joint[:, 0, J["right_wrist"]]) > 0.5
    assert np.nanmax(np.nanmean(dev.score[:, 1:], 0)) < 0.05
    assert np.nanmax(dev.angle[:, 0]) > 10


def test_composite_ignores_outlier():
    norm, *_ = normalize(_dense("right_wrist"))
    comp = build_composite(norm)
    ref, *_ = normalize(_dense())
    j = J["right_wrist"]
    assert np.allclose(comp[:, j], ref[:, 1, j, :2], atol=1e-3)


def test_two_dancer_deviation_is_symmetric():
    norm, *_ = normalize(_dense("left_wrist", n=2))
    dev = compute_deviation(norm)
    assert np.allclose(dev.score[:, 0], dev.score[:, 1], atol=1e-6)


def test_timeline_figure_has_one_heatmap_row_per_dancer_group():
    from ddc.plots import timeline_figure
    from ddc.skeleton import GROUPS
    norm, *_ = normalize(_dense("right_wrist"))
    dev = compute_deviation(norm)
    fig = timeline_figure(dev, np.arange(norm.shape[0]) / 30.0)
    heat = [t for t in fig.data if t.type == "heatmap"][0]
    assert len(heat.z) == norm.shape[1] * len(GROUPS)
    assert len(heat.z[0]) == norm.shape[0]


def test_render_overlay_resamples_background_video_through_cfr_source(tmp_path, monkeypatch):
    """render_overlay must not read a background video frame-by-frame without the
    same VFR->CFR resampling analyze_video_* uses, or the drawn skeleton drifts
    out of sync with the background over time (the bug this test guards against)."""
    import cv2

    import video_pipeline
    from ddc.deviation import compute_deviation
    from ddc.render import render_overlay

    F = 5
    bg_path = tmp_path / "bg.mp4"
    vw = cv2.VideoWriter(str(bg_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    for _ in range(F):
        vw.write(np.zeros((64, 64, 3), np.uint8))
    vw.release()

    dense = np.stack([place(base_pose(), 30, 30, 10) for _ in range(F)])[:, None]
    norm, origin, scale, rot = normalize(dense)
    dev = compute_deviation(norm)
    t = np.arange(F) / 10.0

    calls = []
    real_cfr_source = video_pipeline._cfr_source

    def spy(path):
        calls.append(path)
        return real_cfr_source(path)

    monkeypatch.setattr(video_pipeline, "_cfr_source", spy)
    out = render_overlay(dense, dev, origin, scale, rot, t, tmp_path / "out.mp4", str(bg_path))

    assert calls == [str(bg_path)]
    assert Path(out).exists()


def _write_jsonl_run(tmp_path, people_per_frame, fps=30.0):
    """people_per_frame: list (per frame) of list of (17,3) keypoint arrays."""
    import json

    from ddc.skeleton import BODY as _BODY

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with open(run_dir / "tracking.jsonl", "w") as f:
        for frame_idx, people in enumerate(people_per_frame):
            rec = {"frame": frame_idx, "t": frame_idx / fps, "people": []}
            for pi, p in enumerate(people):
                xy = p[:, :2]
                x0, y0 = xy.min(0) - 10
                x1, y1 = xy.max(0) + 10
                rec["people"].append({
                    "id": pi, "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "conf": float(p[:, 2].mean()),
                    "keypoints": {n: p[j].tolist() for j, n in enumerate(_BODY)},
                })
            f.write(json.dumps(rec) + "\n")
    return run_dir


def test_detect_tracks_finds_dancers_and_a_bystander(tmp_path):
    from ddc.analyze import detect_tracks

    F = 40
    frames = []
    for f in range(F):
        dancer1 = place(base_pose(), 100 + f, 300, 40)
        dancer2 = place(base_pose(), 500 - f, 300, 40)
        bystander = place(base_pose(), 900, 900, 30)   # stands still, off to the side
        frames.append([dancer1, dancer2, bystander])
    run_dir = _write_jsonl_run(tmp_path, frames)

    tracks = detect_tracks(run_dir)
    assert len(tracks) == 3
    assert all(tr.frames_present == F for tr in tracks)
    assert all(tr.thumbnail.shape[2] == 3 for tr in tracks)


def test_excluding_a_bystander_track_removes_it_from_the_composite(tmp_path):
    """A track dropped via `selected_tracks` must not pollute the composite/deviation
    for the tracks that are kept -- this is the filter the Compare tab's people
    picker relies on."""
    from ddc.analyze import analyze_run

    F = 40
    frames = []
    for f in range(F):
        dancer1 = place(base_pose(), 200, 300, 40)
        dancer2 = place(base_pose(), 200, 300, 40)          # identical to dancer1
        bystander = place(neutral_pose(), 900, 900, 30)      # different pose, off to the side
        frames.append([dancer1, dancer2, bystander])
    run_dir = _write_jsonl_run(tmp_path, frames)

    r_all = analyze_run(run_dir, render=False, selected_tracks=[0, 1, 2])
    r_filtered = analyze_run(run_dir, render=False, selected_tracks=[0, 1])

    # With the bystander included, the two identical dancers still show deviation
    # (composite gets pulled toward the third, unrelated pose).
    assert np.nanmean(r_all.dev.score) > 0.05
    # With the bystander excluded, the two identical dancers deviate ~0 from each other.
    assert r_filtered.n_tracks == 2
    assert np.nanmean(r_filtered.dev.score) < 1e-6


def test_analyze_run_rejects_stale_track_selection(tmp_path):
    from ddc.analyze import analyze_run

    frames = [[place(base_pose(), 200, 300, 40)] for _ in range(20)]
    run_dir = _write_jsonl_run(tmp_path, frames)

    import pytest
    with pytest.raises(ValueError, match="don't match"):
        analyze_run(run_dir, render=False, selected_tracks=[0, 5])
    with pytest.raises(ValueError, match="no dancers selected"):
        analyze_run(run_dir, render=False, selected_tracks=[])


def test_source_video_persists_for_reuse_without_video_path(tmp_path):
    """Uploading a source video once should let later Compare runs on the same
    run reuse it (undimmed) without asking the user to re-upload it."""
    from ddc.analyze import _find_saved_source, _pick_background

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    src = tmp_path / "clip.mov"
    src.write_bytes(b"fake video bytes")

    assert _find_saved_source(run_dir) is None
    bg, dim = _pick_background(run_dir, str(src))
    assert bg == run_dir / "source.mov"
    assert bg.exists()
    assert dim == 1.0

    # A later call with no video_path reuses the saved copy, still undimmed.
    bg2, dim2 = _pick_background(run_dir, None)
    assert bg2 == run_dir / "source.mov"
    assert dim2 == 1.0

    # Re-uploading a video with a different extension replaces the saved copy.
    src2 = tmp_path / "clip2.mp4"
    src2.write_bytes(b"other bytes")
    bg3, dim3 = _pick_background(run_dir, str(src2))
    assert bg3 == run_dir / "source.mp4"
    assert not (run_dir / "source.mov").exists()


def test_pick_background_falls_back_to_dimmed_overlay(tmp_path):
    """With no uploaded/saved source video, the run's own overlay.mp4 (which
    already has a skeleton drawn on it) is used, dimmed so the new deviation
    skeleton drawn on top of it stays legible."""
    from ddc.analyze import _pick_background

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "overlay.mp4").write_bytes(b"fake overlay")

    bg, dim = _pick_background(run_dir, None)
    assert bg == run_dir / "overlay.mp4"
    assert dim == 0.3


def test_detect_tracks_persists_uploaded_source_video(tmp_path):
    from ddc.analyze import detect_tracks

    frames = [[place(base_pose(), 200, 300, 40)] for _ in range(20)]
    run_dir = _write_jsonl_run(tmp_path, frames)
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"fake video bytes")

    detect_tracks(run_dir, str(src))
    assert (run_dir / "source.mp4").exists()
