"""Track stitching (one person lost and re-found -> one track) and manual merges."""

import numpy as np
import pytest

from ddc.tracking import merge_track_ids, stabilize
from test_ddc import base_pose, place


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
