"""Explicit merge learning, persistence, and stable downstream person identities."""

import json

import numpy as np
import pytest

from ddc.feature_registry import load_merges, review_is_current
from ddc.identity_learning import (MODEL_FILE, PREDICTION_FILE, REVIEW_FILE,
                                   initialize_predictions, predict_groups,
                                   save_merge_review, train_model)
from ddc.io import load_jsonl
from ddc.tracking import stable_track_ids, tracking_stamp
from test_ddc import _write_jsonl_run, base_pose, place


def _run(root, name, fragments=2, spectator=True):
    # Gaps exceed the heuristic stitcher's 5s limit. All fragments are one person.
    frames = []
    for f in range((fragments - 1) * 210 + 20):
        people = [place(base_pose(), 100, 300, 40)] if f % 210 < 20 else []
        if spectator:
            people.append(place(base_pose(), 1500, 300, 60))
        frames.append(people)
    staging = root / f"stage-{name}"
    staging.mkdir()
    run = _write_jsonl_run(staging, frames)
    run.rename(root / name)
    staging.rmdir()
    return root / name


def test_saves_merge_examples_and_replaces_instead_of_duplicating_them(tmp_path):
    run = _run(tmp_path, "review", fragments=4)
    assert stable_track_ids(load_jsonl(run / "tracking.jsonl"))[1] == 5
    model = save_merge_review(run, [[0, 2, 3, 4]])
    assert load_merges(run) == [[0, 2, 3, 4]]
    assert model["status"] == "ready"
    assert model["positive_pairs"] == 6
    assert model["negative_pairs"] == 4
    again = save_merge_review(run, [[0, 2, 3, 4]])
    assert model == again
    reset = save_merge_review(run, [])
    assert load_merges(run) == []
    assert reset["positive_pairs"] == 0 and reset["status"] == "collecting"
    assert json.loads((tmp_path / MODEL_FILE).read_text())["status"] == "collecting"


def test_merge_training_works_without_scikit_learn(tmp_path, monkeypatch):
    import builtins
    original_import = builtins.__import__

    def without_sklearn(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ModuleNotFoundError("No module named 'sklearn'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sklearn)
    reviewed = _run(tmp_path, "reviewed", fragments=4)
    report = save_merge_review(reviewed, [[0, 2, 3, 4]])
    assert report["status"] == "ready"
    future = _run(tmp_path, "future")
    assert initialize_predictions(future) == [[0, 2]]
    assert stable_track_ids(load_jsonl(future / "tracking.jsonl"))[1] == 2


def test_saved_merges_improve_a_future_run_without_changing_older_ids(tmp_path):
    reviewed = _run(tmp_path, "reviewed", fragments=4)
    save_merge_review(reviewed, [[0, 2, 3, 4]])
    future = _run(tmp_path, "future")
    det = load_jsonl(future / "tracking.jsonl")
    before = tracking_stamp(det.path)
    assert stable_track_ids(det)[1] == 3
    assert initialize_predictions(future) == [[0, 2]]
    assert tracking_stamp(det.path) != before
    ids, count, info = stable_track_ids(det, return_info=True)
    assert count == 2
    assert ids[0].tolist() == ids[-1].tolist() == [0, 1]
    assert info["fragments"] == [2, 1]
    # Downstream detection, features and comparison share the learned identities.
    from ddc.analyze import analyze_run, detect_tracks
    from ddc.feature_inventory import collect_feature_inventory
    assert len(detect_tracks(future)) == 2
    assert len(collect_feature_inventory(det.path)) == 2
    assert analyze_run(future, render=False).n_tracks == 2
    # Saving a new review changes the global model, never this frozen assignment.
    snapshot = (future / PREDICTION_FILE).read_bytes()
    save_merge_review(reviewed, [])
    assert initialize_predictions(future) == []
    assert (future / PREDICTION_FILE).read_bytes() == snapshot
    assert stable_track_ids(load_jsonl(det.path))[1] == 2
    assert stable_track_ids(load_jsonl(reviewed / "tracking.jsonl"))[1] == 5


def test_positive_only_reviews_wait_for_valid_negative_examples(tmp_path):
    run = _run(tmp_path, "only-person", spectator=False)
    report = save_merge_review(run, [[0, 1]])
    assert report["positive_pairs"] == 1 and report["negative_pairs"] == 0
    assert report["status"] == "collecting"
    future = _run(tmp_path, "future", spectator=False)
    assert initialize_predictions(future) == []
    assert stable_track_ids(load_jsonl(future / "tracking.jsonl"))[1] == 2


def test_unknown_disjoint_pairs_are_not_negative_labels(tmp_path):
    run = _run(tmp_path, "unreviewed-fragment", fragments=3, spectator=False)
    report = save_merge_review(run, [[0, 1]])
    assert report["positive_pairs"] == 1
    assert report["negative_pairs"] == 0


def test_invalid_merges_do_not_overwrite_a_saved_review(tmp_path):
    run = _run(tmp_path, "review")
    save_merge_review(run, [[0, 2]])
    before = (run / REVIEW_FILE).read_bytes()
    with pytest.raises(ValueError, match="don't match"):
        save_merge_review(run, [[0, 99]])
    with pytest.raises(ValueError, match="only be in one"):
        save_merge_review(run, [[0, 2], [1, 2]])
    assert (run / REVIEW_FILE).read_bytes() == before


def test_changed_tracking_discards_stale_training_examples_and_saved_merges(tmp_path):
    run = _run(tmp_path, "review")
    save_merge_review(run, [[0, 2]])
    path = run / "tracking.jsonl"
    path.write_text(path.read_text() + "\n")
    report = train_model(tmp_path)
    assert report["positive_pairs"] == report["negative_pairs"] == 0
    # An obsolete explicit merge review must not fall back to bare integer ids.
    assert load_merges(run) == []


def test_merges_are_saved_even_when_model_training_fails(tmp_path, monkeypatch):
    import ddc.identity_learning as learning
    run = _run(tmp_path, "review")
    def fail(root):
        raise RuntimeError("fitting failed")
    monkeypatch.setattr(learning, "train_model", fail)
    assert save_merge_review(run, [[0, 2]])["status"] == "training_failed"
    assert load_merges(run) == [[0, 2]]


def test_saved_merge_changes_invalidate_previous_dancer_labels(tmp_path):
    from ddc.training import signature
    run = _run(tmp_path, "review")
    (run / "merges.json").write_text("[]")
    (run / "training_review.json").write_text(json.dumps({"signature": signature(run)}))
    assert review_is_current(run)
    save_merge_review(run, [[0, 2]])
    assert not review_is_current(run)


def test_learned_model_cannot_join_overlapping_or_implausible_tracks(tmp_path):
    from ddc.identity_learning import _track_groups
    run = _run(tmp_path, "tracks", fragments=3)
    det = load_jsonl(run / "tracking.jsonl")
    ids, count = stable_track_ids(det)
    groups = _track_groups(det, ids, count)
    # Force the model to say yes to everything; structural constraints still win.
    model = {"mean": [0.] * 6, "scale": [1.] * 6, "coefficients": [0.] * 6,
             "intercept": 30., "threshold": .85}
    assert predict_groups(groups, model, 30.) == [[0, 2, 3]]
    # A chain may not connect first to last across a gap beyond the 15s limit.
    far = dict(groups[3], frames=groups[3]["frames"] + 600)
    assert predict_groups([groups[0], groups[2], far], model, 30.) == [[0, 1]]


def test_existing_reviewed_runs_are_not_renumbered_by_new_model(tmp_path):
    run = _run(tmp_path, "training", fragments=4)
    save_merge_review(run, [[0, 2, 3, 4]])
    existing = _run(tmp_path, "existing")
    (existing / "labels.csv").write_text("track_id,label\n0,dancer\n")
    assert initialize_predictions(existing) == []
    assert not (existing / PREDICTION_FILE).exists()


def test_prediction_snapshot_invalidates_training_and_overlay_caches(tmp_path):
    from ddc.training import signature
    from ddc.overlay_cache import overlay_key
    run = _run(tmp_path, "training", fragments=4)
    save_merge_review(run, [[0, 2, 3, 4]])
    future = _run(tmp_path, "future")
    # Cache key construction does not decode the source.
    source = tmp_path / "cache-source.mp4"
    source.write_bytes(b"source stamp")
    before_signature = signature(future)
    before_overlay = overlay_key(future, source, [], [0], 720)
    assert initialize_predictions(future) == [[0, 2]]
    assert signature(future) != before_signature
    assert overlay_key(future, source, [], [0], 720) != before_overlay
