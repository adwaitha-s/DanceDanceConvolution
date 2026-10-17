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


def test_save_merges_button_and_reopening_use_original_automatic_ids(tmp_path, monkeypatch):
    import app
    from test_identity_learning import _run
    from ddc.identity_learning import save_merge_review

    run = _run(tmp_path, "review")
    save_merge_review(run, [[0, 2]])
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(app, "_LAST", {})
    monkeypatch.setattr(app, "_predict_people",
                        lambda name, groups, tracks: ([tr.label for tr in tracks], "test"))
    out = app.run_detect(run.name)
    assert len(out) == app.N_PEOPLE_OUTPUTS
    assert len(out[0]) == 2
    assert app._LAST["n_auto"] == 3  # further merges must use original ids
    assert app._LAST["merges"] == [[0, 2]]
    assert any(getattr(block, "value", None) == "Save merges" for block in app.demo.blocks.values())
    assert "Merges saved" in app.run_save_merges(run.name)

    reset = app.run_reset_merges(run.name)
    assert len(reset[0]) == 3
    assert "Unsaved" in reset[-1]
    assert "Merges saved" in app.run_save_merges(run.name)
    assert len(app.run_detect(run.name)[0]) == 3


def test_cumulative_merges_remain_correct_after_reopening(tmp_path, monkeypatch):
    import app
    from test_identity_learning import _run
    from ddc.identity_learning import save_merge_review

    run = _run(tmp_path, "review", fragments=3)
    save_merge_review(run, [[0, 2]])
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(app, "_LAST", {})
    monkeypatch.setattr(app, "_predict_people",
                        lambda name, groups, tracks: ([tr.label for tr in tracks], "test"))
    app.run_detect(run.name)
    labels = app._LAST["track_labels"]
    app.run_merge(run.name, [labels[0], labels[2]])
    assert app._LAST["merges"] == [[0, 2, 3]]
    app.run_save_merges(run.name)
    assert len(app.run_detect(run.name)[0]) == 2
    assert app._LAST["merges"] == [[0, 2, 3]]


def test_saving_merges_refreshes_predictions_from_persisted_groups(tmp_path, monkeypatch):
    import app
    from test_identity_learning import _run

    run = _run(tmp_path, "review")
    monkeypatch.setattr(app, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(app, "_LAST", {})
    scored_groups = []

    def predict(name, groups, tracks):
        scored_groups.append(groups)
        return [tr.label for tr in tracks], "test"

    monkeypatch.setattr(app, "_predict_people", predict)
    app.run_detect(run.name)
    labels = app._LAST["track_labels"]
    app.run_merge(run.name, [labels[0], labels[2]])
    message = app.run_save_merges(run.name)

    scored_groups.clear()
    shown = app.refresh_saved_merges(run.name, message)
    assert scored_groups == [[[0, 2]]]
    assert app._LAST["merges"] == [[0, 2]]
    assert shown[-1] == message
