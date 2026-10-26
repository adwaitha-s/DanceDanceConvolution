"""Track stitching (one person lost and re-found -> one track) and manual merges."""

import numpy as np
import pytest

from ddc.tracking import merge_track_ids, stabilize, stitch_tracklets, to_dense
from test_ddc import _write_jsonl_run, base_pose, place


def _frames(spec, n_frames):
    """spec: list of (x0, vx, start, end) -- a person at x0 + vx*f for frames [start, end)."""
    out = []
    for f in range(n_frames):
        people = [place(base_pose(), x0 + vx * f, 300, 40)
                  for x0, vx, a, b in spec if a <= f < b]
        out.append(np.stack(people) if people else np.zeros((0, 17, 3)))
    return out


def _track_of(ids, f, det_idx):
    return int(ids[f][det_idx])


def test_person_lost_for_longer_than_max_gap_is_one_track():
    kps = _frames([(300, 0, 0, 60), (300, 0, 100, 160)], 160)   # 40-frame dropout
    _, n_raw = stabilize(kps, stitch=False)
    ids, n = stabilize(kps)
    assert n_raw == 2 and n == 1
    assert _track_of(ids, 10, 0) == _track_of(ids, 150, 0) == 0


def test_short_fragment_is_absorbed_not_dropped():
    # a 10-frame fragment (< min_track_len) between two long pieces of one person
    kps = _frames([(300, 0, 0, 50), (300, 0, 70, 80), (300, 0, 100, 150)], 150)
    _, n_raw = stabilize(kps, stitch=False)
    ids, n = stabilize(kps)
    assert n == 1
    assert all(ids[f][0] == 0 for f in range(150) if len(ids[f]))
    assert sum(len(a) and int(a[0] >= 0) for a in ids) == 50 + 10 + 50


def test_people_visible_at_the_same_time_are_never_merged():
    kps = _frames([(300, 0, 0, 100), (360, 0, 0, 100)], 100)   # side by side the whole time
    ids, n = stabilize(kps)
    assert n == 2
    assert all(len(set(a)) == 2 for a in ids)


def test_distant_reappearance_is_a_different_person():
    kps = _frames([(100, 0, 0, 60), (1800, 0, 100, 160)], 160)   # far across the frame
    ids, n = stabilize(kps)
    assert n == 2


def test_gap_longer_than_limit_is_not_stitched():
    kps = _frames([(300, 0, 0, 40), (300, 0, 300, 340)], 340)
    _, n = stabilize(kps, max_stitch_gap=150)
    assert n == 2


def test_stabilize_is_deterministic():
    kps = _frames([(300, 0, 0, 60), (300, 0, 100, 160), (900, 1, 0, 160)], 160)
    a, na = stabilize(kps)
    b, nb = stabilize(kps)
    assert na == nb and all((x == y).all() for x, y in zip(a, b))


def test_dropped_duplicate_detection_is_not_a_phantom_person():
    # the same person detected twice every frame: one skeleton is dropped as a duplicate,
    # and that must not turn into an extra "person"
    kps = [np.stack([place(base_pose(), 300, 300, 40)] * 2) for _ in range(40)]
    _, n = stabilize(kps)
    assert n == 1


def test_crossing_dancers_keep_identity_after_one_is_occluded():
    # Identical poses, changing detection order, and one indistinguishable
    # detection at the crossing: identity must follow motion, not screen side.
    kps = []
    for f in range(41):
        people = [place(base_pose(), 100 + 10 * f, 300, 40),
                  place(base_pose(), 500 - 10 * f, 300, 40)]
        kps.append(np.stack(people if f % 2 == 0 else people[::-1]))
    ids, n = stabilize(kps)
    assert n == 2
    dense = to_dense(kps, ids, n)
    for tid, direction in ((0, 1), (1, -1)):
        xs = dense[:, tid, 0, 0]
        visible = np.isfinite(xs)
        np.testing.assert_allclose(xs[visible], (100 if tid == 0 else 500)
                                   + direction * 10 * np.arange(41)[visible])


def test_moving_dancer_returns_to_own_track_after_dropout():
    kps = []
    for f in range(40):
        people = [place(base_pose(), 100 + 10 * f, 300, 40)] if not 17 <= f < 23 else []
        people.append(place(base_pose(), 300, 300, 40))
        kps.append(np.stack(people))
    ids, n = stabilize(kps)
    assert n == 2
    assert ids[0].tolist() == ids[-1].tolist() == [0, 1]
    assert all(ids[f].tolist() == [1] for f in range(17, 23))


def test_close_distinct_dancers_are_not_discarded_as_duplicates():
    kps = _frames([(300, 0, 0, 40), (310, 0, 0, 40)], 40)
    ids, n = stabilize(kps)
    assert n == 2
    assert all(a.tolist() == [0, 1] for a in ids)


@pytest.mark.parametrize("supplied_video", [False, True])
def test_saved_video_clothing_prevents_identity_swap_at_ambiguous_positions(tmp_path, supplied_video):
    import cv2
    from ddc.analyze import analyze_run, detect_tracks
    from ddc.io import load_jsonl
    from ddc.tracking import stable_track_ids

    frames = [[place(base_pose(), x, 200, 40) for x in
               ([100, 130] if f < 20 else [130, 100])] for f in range(40)]
    run = _write_jsonl_run(tmp_path, frames)
    video = tmp_path / "clip.mp4" if supplied_video else run / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 400))
    assert writer.isOpened()
    try:
        for poses in frames:
            image = np.zeros((400, 320, 3), np.uint8)
            for p, color in zip(poses, [(0, 0, 255), (255, 0, 0)]):
                x = int(p[0, 0])
                cv2.rectangle(image, (x - 14, 158), (x + 14, 201), color, -1)
                for offset in (-12, 12):
                    cv2.line(image, (x + offset, 200), (x + offset, 260), color, 10)
            writer.write(image)
    finally:
        writer.release()
    if supplied_video:
        assert len(detect_tracks(run, video_path=video)) == 2
        assert analyze_run(run, render=False).dense[-1, 0, 0, 0] == 130
    det = load_jsonl(run / "tracking.jsonl")
    ids, n = stable_track_ids(det)
    assert n == 2
    assert all(a.tolist() == [0, 1] for a in ids)
    # A merged view must not mutate the cached automatic assignments.
    merged, n = stable_track_ids(det, [[0, 1]])
    assert n == 1
    ids, n = stable_track_ids(det)
    assert n == 2 and ids[-1].tolist() == [0, 1]


def test_stitching_does_not_combine_people_seen_together_briefly():
    kps = _frames([(300, 0, 0, 21), (305, 0, 20, 40)], 40)
    first = [(f, 0) for f in range(21)]
    second = [(f, 1 if f == 20 else 0) for f in range(20, 40)]
    assert stitch_tracklets([first, second], kps) == [[0], [1]]


def test_floor_move_foreshortening_does_not_create_a_new_person():
    from ddc.skeleton import J
    standing = place(base_pose(), 300, 300, 40)
    floor = standing.copy()
    floor[:, :2] = (floor[:, :2] - [300, 300]) @ np.array([[0, 1], [-1, 0]]) + [325, 300]
    # Projected shoulders overlap the hips while the rest of the body remains visible.
    floor[[J["left_shoulder"], J["right_shoulder"]], :2] = floor[[J["left_hip"], J["right_hip"]], :2]
    kps = [standing[None]] * 40 + [np.zeros((0, 17, 3))] * 20 + [floor[None]] * 40
    boxes = [[[*p[:, :2].min(0), *p[:, :2].max(0)] for p in frame] for frame in kps]
    ids, n = stabilize(kps, boxes=boxes)
    assert n == 1
    assert ids[0].tolist() == ids[-1].tolist() == [0]


@pytest.mark.parametrize("persistent", [False, True])
def test_loaded_ids_are_trusted_only_when_explicitly_persistent(tmp_path, persistent):
    import json
    from ddc.analyze import analyze_run, detect_tracks
    from ddc.io import load_jsonl
    from ddc.skeleton import BODY
    from ddc.tracking import stable_track_ids

    records = []
    for f in range(40):
        # Persistent ids survive a long absence and a large move. Detection
        # indices at the same location still need frame-to-frame association.
        xs = [100, 500] if f < 20 or not persistent else [500, 100]
        people = [{"id": tid, "keypoints": dict(zip(BODY, place(base_pose(), x, 300, 40).tolist()))}
                  for tid, x in zip([23, 71], xs)]
        if f % 2:
            people.reverse()
        if persistent and 5 <= f < 20:
            people = []
        record = {"t": f / 30, "people": people}
        if persistent:
            record["id_type"] = "track"
        records.append(record)
    (tmp_path / "tracking.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    det = load_jsonl(tmp_path / "tracking.jsonl")
    assert det.ids_persistent == persistent
    ids, n = stable_track_ids(det)
    assert n == 2
    assert ids[0].tolist() == [0, 1]
    assert ids[-1].tolist() == [1, 0]
    assert len(detect_tracks(tmp_path)) == 2
    result = analyze_run(tmp_path, render=False)
    assert result.dense[-1, 0, 0, 0] == (500 if persistent else 100)


def test_merge_track_ids_combines_and_renumbers():
    # person A early, person B (far away, so not auto-stitched) late, person C throughout
    kps = _frames([(100, 0, 0, 40), (1500, 0, 50, 90), (900, 0, 0, 90)], 90)
    ids, n = stabilize(kps)
    assert n == 3
    # tracks number by first appearance: A=0, C=1, B=2 (frame 60 detections are [B, C])
    merged, m = merge_track_ids(kps, ids, [[0, 2]])
    assert m == 2
    assert _track_of(merged, 0, 0) == _track_of(merged, 60, 0) == 0   # merged keeps lowest id
    assert _track_of(merged, 0, 1) == _track_of(merged, 60, 1) == 1   # C renumbered, unchanged


def test_merge_track_ids_keeps_higher_confidence_on_same_frame_conflict():
    kps = _frames([(100, 0, 0, 40), (500, 0, 0, 40)], 40)
    for f in range(40):
        kps[f][1][:, 2] = 0.95                                        # second person more confident
    ids, _ = stabilize(kps)
    merged, m = merge_track_ids(kps, ids, [[0, 1]])
    assert m == 1
    assert all((a == np.array([-1, 0])).all() for a in merged)


def test_merge_track_ids_validates_input():
    kps = _frames([(100, 0, 0, 40), (500, 0, 0, 40)], 40)
    ids, _ = stabilize(kps)
    with pytest.raises(ValueError, match="don't match"):
        merge_track_ids(kps, ids, [[0, 7]])
    with pytest.raises(ValueError, match="only be in one"):
        merge_track_ids(kps, ids, [[0, 1], [1, 0]])
