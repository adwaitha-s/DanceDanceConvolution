import json

import numpy as np

from ddc.deviation import compute_deviation
from ddc.plots import timeline_figure


def test_app_builds_and_lists_runs_with_readable_labels(tmp_path, monkeypatch):
    import app

    assert app.demo is not None
    assert not any(getattr(block, "label", None) == "Ten-video supervised training"
                   for block in app.demo.blocks.values())

    new = tmp_path / "20261010-144020"
    old = tmp_path / "20260929-161605"
    empty = tmp_path / "20260101-000000"
    mediapipe = tmp_path / "20260102-000000"
    for d in (new, old, empty, mediapipe):
        d.mkdir()
        (d / "tracking.jsonl").write_text("" if d is empty else "{}\n")
    (new / "meta.json").write_text(json.dumps(
        {"video": "IMG_2326.mov", "model": app.MODEL_RTMW, "duration_s": 49.36}))
    (mediapipe / "meta.json").write_text(json.dumps({"model": app.MODEL_MEDIAPIPE}))
    (tmp_path / "feature_model_registry.json").write_text("{}")   # not a run
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)

    assert app.list_runs() == [
        ("Oct 10 14:40 · IMG_2326.mov · 49 s", "20261010-144020"),
        ("Sep 29 16:16", "20260929-161605"),
    ]


def test_review_status_tracks_saved_and_predicted_selection(monkeypatch):
    import app

    monkeypatch.setattr(app, "_LAST", {
        "track_ids": {"Person 1": 0, "Person 2": 1},
        "predictions": [{"track_id": 0}], "predicted_selected_ids": frozenset({0}),
    })
    assert "automatic selection" in app._review_status(["Person 1"])
    assert "Unsaved" in app._review_status(["Person 1", "Person 2"])
    app._LAST["saved_selection"] = frozenset({0, 1})
    assert "saved" in app._review_status(["Person 1", "Person 2"])


def test_timeline_height_grows_with_dancers_and_stays_bounded():
    def fig(n_dancers):
        rng = np.random.default_rng(0)
        norm = rng.normal(size=(20, n_dancers, 17, 3))
        norm[..., 2] = 0.9
        return timeline_figure(compute_deviation(norm), np.arange(20) / 30.0)

    small, large = fig(2), fig(12)
    assert 420 <= small.layout.height < large.layout.height <= 720
    assert timeline_figure(compute_deviation(np.ones((5, 2, 17, 3))), np.arange(5.0),
                           height=300).layout.height == 300


def test_compare_before_save_confirms_original_predictions_despite_unsaved_edits(monkeypatch):
    import app
    monkeypatch.setattr(app, "_LAST", {
        "predictions": [{"track_id": 0, "predicted_label": "dancer"},
                        {"track_id": 1, "predicted_label": "non_dancer"}], "merges": []})
    calls = []
    def validate(root, name, selected, predicted, **kwargs):
        calls.append((selected, kwargs["review_action"]))
        return {"validation": {"status": "validated_correct", "corrected_track_ids": []}}
    monkeypatch.setattr(app, "validate_checklist", validate)
    monkeypatch.setattr(app, "rebuild_registry_workbook", lambda root: None)
    note = app._validate_comparison("run", [1])
    assert calls == [([0], "compare_confirmed")]
    assert "all correct" in note
    assert "Unsaved" in note
    assert app._LAST["saved_selection"] == frozenset({0})


def test_compare_keeps_saved_corrections_even_when_checklist_returns_to_predictions(monkeypatch):
    import app
    monkeypatch.setattr(app, "_LAST", {"saved_selection": frozenset({1}),
        "saved_validation": {"corrected_track_ids": [0,1]},
        "predictions": [{"track_id": 0, "predicted_label": "dancer"}]})
    def unexpected(*args, **kwargs):
        raise AssertionError("Compare must not rewrite an existing saved review")
    monkeypatch.setattr(app, "validate_checklist", unexpected)
    assert "retained" in app._validate_comparison("run", [0])
    assert app._LAST["saved_selection"] == frozenset({1})


def test_compare_without_predictions_does_not_invent_validation_labels(monkeypatch):
    import app
    monkeypatch.setattr(app, "_LAST", {"predictions": []})
    assert "Save reviewed labels" in app._validate_comparison("run", [0])
