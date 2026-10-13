import json

import numpy as np

from ddc.deviation import compute_deviation
from ddc.plots import timeline_figure


def test_app_builds_and_lists_runs_with_readable_labels(tmp_path, monkeypatch):
    import app

    assert app.demo is not None

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


def test_coaching_results_callbacks_and_reset(tmp_path, monkeypatch):
    import pandas as pd
    import app
    from ddc.analyze import analyze_run
    from test_ddc import _write_jsonl_run, base_pose, place

    reference, altered = base_pose(), base_pose()
    altered[app.BODY.index('left_wrist'), 1] += .6
    run = _write_jsonl_run(tmp_path, [[place(reference, 200, 300, 40),
                                     place(altered, 500, 300, 40)] for _ in range(40)])
    result = analyze_run(run, render=False, reference_mode='dancer', reference_dancer=0)
    monkeypatch.setattr(app, 'RUNS_DIR', tmp_path)
    monkeypatch.setattr(app, '_LAST', {'track_run': run.name, 'track_labels': ['Person 2', 'Person 4'],
                                     'track_ids': {'Person 2': 0, 'Person 4': 1}})
    monkeypatch.setattr(app, 'analyze_run', lambda *a, **kw: result)
    outputs = app.run_compare(run.name, ['Person 2', 'Person 4'], 'Auto (on for 3+ dancers)',
                              False, 5, app.REF_DANCER, 'Person 2', None, 0)
    assert len(outputs) == 12
    rows, feedback = outputs[4], outputs[-1]
    assert len(rows) == 1 and rows[0][0] == 'Dancer 2 (Person 4)'
    assert rows[0][1] < 100 and '40/40' in rows[0][2]
    assert 'Overall movement similarity:' in outputs[6]
    assert 'not scored' in outputs[6]
    assert feedback and 'lower' in feedback[0][2]
    assert feedback[0][0] == rows[0][0]
    assert 'left wrist' in app.frame_detail(feedback[0][1])
    event = type('Event', (), {'index': (0, 0)})()
    assert app.jump_to_moment(pd.DataFrame(feedback), event) == feedback[0][1]
    assert len(app.begin_review(None)) == len(app.review_reset_outputs)
    assert len(app.begin_review(run.name)) == len(app.review_reset_outputs)
    assert app.begin_review(run.name)[-1] == []


def test_frame_detail_explains_unavailable_comparison(tmp_path, monkeypatch):
    import app
    from ddc.analyze import analyze_run
    from test_ddc import _write_jsonl_run, base_pose, place

    run = _write_jsonl_run(tmp_path, [[place(base_pose(), 200, 300, 40)] for _ in range(40)])
    result = analyze_run(run, render=False)
    monkeypatch.setattr(app, '_LAST', {'res': result})
    assert 'insufficient comparable pose data' in app.frame_detail(0)
    assert result.coaching.overall is None
