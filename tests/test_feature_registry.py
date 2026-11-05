import json

import pytest

from ddc.dancer_features import load_labels
from ddc.feature_registry import load_merges, review_is_current, validate_checklist
from test_data import write_synthetic_demo


def _make_run(root, name):
    run = root / name
    run.mkdir()
    write_synthetic_demo(run / "tracking.jsonl", root / f"{name}-source-labels.csv", frames=40)
    return run


def test_legacy_track_reviews_are_not_attached_to_new_identities(tmp_path):
    from ddc.tracking import TRACKING_VERSION, tracking_stamp

    run = _make_run(tmp_path, "reviewed")
    validation = run / "prediction_validation.json"
    validation.write_text(json.dumps({"status": "validated_correct"}))
    (run / "merges.json").write_text(json.dumps([[0, 1]]))
    assert not review_is_current(run)
    assert load_merges(run) == []
    (run / "merges.json").write_text("[]")
    validation.write_text(json.dumps({"tracking_version": TRACKING_VERSION,
                                      "tracking_stamp": tracking_stamp(run / "tracking.jsonl")}))
    assert review_is_current(run)


def test_review_records_corrections_and_preserves_original_predictions(tmp_path):
    run = _make_run(tmp_path, "candidate")
    predictions = [{"track_id": i, "predicted_label": "dancer" if i < 4 else "non_dancer",
                    "dancer_probability": .8 if i < 4 else .2} for i in range(8)]

    unchanged = validate_checklist(tmp_path, "candidate", [0, 1, 2, 3], predictions)
    assert unchanged["validation"]["status"] == "validated_correct"
    assert unchanged["validation"]["corrected_track_ids"] == []

    corrected = validate_checklist(tmp_path, "candidate", [1, 2, 3, 4], predictions)
    assert corrected["validation"]["status"] == "corrected"
    assert corrected["validation"]["corrected_track_ids"] == [0, 4]
    assert load_labels(run / "labels.csv") == {i: int(i in (1, 2, 3, 4)) for i in range(8)}
    validation = json.loads((run / "prediction_validation.json").read_text())
    assert validation["predictor_labels"] == {
        str(item["track_id"]): item["predicted_label"] for item in predictions}
    assert review_is_current(run)
    assert not (tmp_path / "dancer_feature_registry.xlsx").exists()


def test_unknown_track_cannot_change_saved_review(tmp_path):
    run = _make_run(tmp_path, "candidate")
    with pytest.raises(ValueError, match="unknown track"):
        validate_checklist(tmp_path, "candidate", [99], [])
    assert not (run / "labels.csv").exists()
    assert not (run / "merges.json").exists()


def test_feature_rows_use_the_same_track_ids_as_find_people(tmp_path):
    from ddc.analyze import detect_tracks
    from ddc.feature_inventory import collect_feature_inventory

    run = _make_run(tmp_path, "candidate")
    for groups in (None, [[0, 1]], [[0, 1], [4, 5]]):
        shown = [t.track_id for t in detect_tracks(run, merge_groups=groups)]
        rows = [r["track_id"] for r in collect_feature_inventory(run / "tracking.jsonl", groups)]
        assert rows == shown


def test_saved_merges_travel_with_the_labels(tmp_path):
    run = _make_run(tmp_path, "candidate")
    predictions = [{"track_id": i, "predicted_label": "dancer" if i < 3 else "non_dancer",
                    "dancer_probability": .8 if i < 3 else .2} for i in range(7)]
    validate_checklist(tmp_path, "candidate", [0, 1, 2], predictions, merge_groups=[[0, 1]])
    assert load_merges(run) == [[0, 1]]
    assert len(load_labels(run / "labels.csv")) == 7
    assert review_is_current(run)
