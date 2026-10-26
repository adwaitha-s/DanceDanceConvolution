"""Identity edit log: merge / split / swap / remove replayed on per-frame track ids."""

import json

import numpy as np
import pytest

from ddc import identity
from ddc.feature_inventory import normalize_merges
from ddc.identity_learning import _examples
from ddc.tracking import merge_track_ids


def _run(n_people=4, n_frames=10):
    """Every person in every frame, detection i = person i, confidence rising with i."""
    kps = [np.stack([np.full((17, 3), 0.5 + 0.01 * i) for i in range(n_people)])
           for _ in range(n_frames)]
    ids = [np.arange(n_people) for _ in range(n_frames)]
    return kps, ids


def _person_of(ids, f, det):
    return int(ids[f][det])


def test_merges_only_log_matches_legacy_merge_groups():
    kps, ids = _run(5)
    # On screen: merge people 1 and 3; the list renumbers (3 -> gone, 4 -> 3); then merge 0 and 3.
    edits = [{"op": "merge", "groups": [[1, 3]]}, {"op": "merge", "groups": [[0, 3]]}]
    groups = identity.legacy_groups(edits, 5)
    assert groups == [[0, 4], [1, 3]]
    assert identity.canonical(edits, 5) == groups
    got, n = identity.apply_edits(kps, ids, edits)
    want, m = merge_track_ids(kps, ids, groups)
    assert n == m == 3 and all((a == b).all() for a, b in zip(got, want))


def test_split_appends_new_person_and_leaves_others_alone():
    kps, ids = _run(3)
    got, n = identity.apply_edits(kps, ids, [{"op": "split", "person": 1, "frame": 6}])
    assert n == 4
    assert [_person_of(got, f, 1) for f in (0, 5, 6, 9)] == [1, 1, 3, 3]
    assert all(_person_of(got, f, 0) == 0 and _person_of(got, f, 2) == 2 for f in range(10))


def test_swap_exchanges_people_only_inside_the_range():
    kps, ids = _run(3)
    got, n = identity.apply_edits(kps, ids, [{"op": "swap", "people": [2, 0], "frames": [3, 5]}])
    assert n == 3
    assert [_person_of(got, f, 0) for f in range(10)] == [0, 0, 0, 2, 2, 2, 0, 0, 0, 0]
    assert [_person_of(got, f, 2) for f in range(10)] == [2, 2, 2, 0, 0, 0, 2, 2, 2, 2]


def test_remove_drops_detections_and_renumbers_a_vanished_person():
    kps, ids = _run(3)
    got, n = identity.apply_edits(kps, ids, [{"op": "remove", "person": 1, "frames": [2, 4]}])
    assert n == 3 and [_person_of(got, f, 1) for f in (1, 2, 4, 5)] == [1, -1, -1, 1]
    got, n = identity.apply_edits(kps, ids, [{"op": "remove", "person": 1, "frames": [0, 9]}])
    assert n == 2 and _person_of(got, 0, 2) == 1      # person 3 moves up into the gap


def test_replay_is_deterministic_and_undo_is_dropping_the_last_edit():
    kps, ids = _run(4)
    history = [{"op": "merge", "groups": [[0, 1]]},
               {"op": "split", "person": 2, "frame": 4},
               {"op": "swap", "people": [0, 3], "frames": [0, 2]}]
    a, na = identity.apply_edits(kps, ids, history)
    b, nb = identity.apply_edits(kps, ids, json.loads(json.dumps(history)))   # saved and reloaded
    assert na == nb and all((x == y).all() for x, y in zip(a, b))
    before, nbefore = identity.apply_edits(kps, ids, history[:-1])
    undone, nundone = identity.apply_edits(kps, ids, history[:-1])
    assert nbefore == nundone and all((x == y).all() for x, y in zip(before, undone))
    assert identity.canonical(history, 4) == [identity._clean(e) for e in history]   # not merges-only


def test_merged_duplicates_in_one_frame_keep_the_more_confident_detection():
    kps, ids = _run(3)
    got, n = identity.apply_edits(kps, ids, [{"op": "merge", "groups": [[0, 2]]}])
    assert n == 2 and _person_of(got, 0, 0) == -1 and _person_of(got, 0, 2) == 0


def test_bad_edits_are_rejected():
    kps, ids = _run(3)
    with pytest.raises(ValueError, match="only 3 people"):
        identity.apply_edits(kps, ids, [{"op": "split", "person": 5, "frame": 0}])
    with pytest.raises(ValueError, match="two different"):
        identity.apply_edits(kps, ids, [{"op": "swap", "people": [1, 1], "frames": [0, 2]}])
    with pytest.raises(ValueError, match="unknown"):
        identity.apply_edits(kps, ids, [{"op": "rename"}])


def test_saved_forms_round_trip():
    assert identity.from_saved([[3, 1], [2, 4]]) == [{"op": "merge", "groups": [[1, 3], [2, 4]]}]
    log = [{"op": "swap", "people": [4, 1], "frames": [9, 2]}]
    assert normalize_merges(log) == [{"op": "swap", "people": [1, 4], "frames": [2, 9]}]
    assert normalize_merges([[2, 1], [5]]) == [[1, 2]]                     # legacy unchanged
    assert not identity.is_edit_log([[1, 2]]) and identity.is_edit_log(log)


def test_track_identity_marks_split_tracks_and_learning_skips_them():
    kps, ids = _run(3)
    final, _ = identity.apply_edits(kps, ids, [{"op": "split", "person": 2, "frame": 5}])
    who = identity.track_identity(ids, final)
    assert who == {0: 0, 1: 1, 2: None}                 # track 2 was split across two people
    groups = [{"frames": np.arange(10)} for _ in range(3)]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("ddc.identity_learning.pair_features", lambda a, b, fps: [0.0] * 6)
        pairs = [s["pair"] for s in _examples(groups, [], 30.0, who)]
    assert pairs == [[0, 1]]


def test_stable_track_ids_applies_an_edit_log(tmp_path):
    from ddc import io as ddc_io
    from ddc.tracking import stable_track_ids
    from test_scale import _write_run

    jl = tmp_path / "tracking.jsonl"
    _write_run(jl, n_frames=60)
    det = ddc_io.load_jsonl(jl)
    ids, n, info = stable_track_ids(det, [{"op": "split", "person": 0, "frame": 30}], return_info=True)
    assert n == 3 and info["fragments"] == [1, 1, 1]
    assert {int(t) for a in ids[:30] for t in a} == {0, 1}
    assert {int(t) for a in ids[30:] for t in a} == {1, 2}
    assert all((a == b).all() for a, b in zip(info["base"], stable_track_ids(det)[0]))
