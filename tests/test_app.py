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
    assert len(outputs) == 18
    rows, feedback = outputs[4], outputs[11]
    assert len(rows) == 1 and rows[0][0] == 'Dancer 2 (Person 4)'
    assert rows[0][1] < 100 and '40/40' in rows[0][2]
    assert 'Overall movement similarity:' in outputs[6]
    assert 'not scored' in outputs[6]
    assert feedback and 'lower' in feedback[0][2]
    assert feedback[0][0] == rows[0][0]
    key_rows = outputs[12]
    assert len(key_rows) == 1
    assert key_rows[0][0] == 'Dancer 1 (Person 2) & Dancer 2 (Person 4)'
    assert key_rows[0][4:6] == ['arms', 'left wrist']
    assert key_rows[0][7] == 17
    assert app.jump_to_moment(pd.DataFrame(key_rows), type('Event', (), {'index': (0, 0)})()) == key_rows[0][1]
    assert 'left wrist' in app.frame_detail(feedback[0][1])
    own = app.filter_dancer_report(rows[0][0], 0)
    assert own[0] == rows and own[2] == feedback
    assert 'Dancer 1 (Person 2): reference' not in own[4]
    assert 'Area to review' in own[5] and '40/40' in own[5]
    ref = app.filter_dancer_report('Dancer 1 (Person 2)', 0)
    assert ref[0] == [] and ref[1] == [] and ref[2] == []
    assert ref[3] == key_rows
    assert 'not scored' in ref[4] and 'not scored' in ref[5]
    assert rows[0][0] not in ref[4]
    assert outputs[16]['choices'] == ['All dancers', 'Dancer 1 (Person 2)', rows[0][0]]
    event = type('Event', (), {'index': (0, 0)})()
    assert app.jump_to_moment(pd.DataFrame(feedback), event) == feedback[0][1]
    assert len(app.begin_review(None)) == len(app.review_reset_outputs)
    assert len(app.begin_review(run.name)) == len(app.review_reset_outputs)
    reset = app.begin_review(run.name)
    assert reset[-7:-5] == ([], [])
    assert reset[-5] == '' and reset[-4] == ''
    assert reset[-3]['open'] is False
    assert outputs[-3]['open'] is False
    assert 'Movement similarity' in outputs[13]
    assert 'Review left wrist' in outputs[14]
    assert len(outputs[12]) == len(key_rows)  # detailed rows remain available
    assert f'data-ddc-time="{key_rows[0][1]:.6f}"' in outputs[14]


def test_frame_detail_explains_unavailable_comparison(tmp_path, monkeypatch):
    import app
    from ddc.analyze import analyze_run
    from test_ddc import _write_jsonl_run, base_pose, place

    run = _write_jsonl_run(tmp_path, [[place(base_pose(), 200, 300, 40)] for _ in range(40)])
    result = analyze_run(run, render=False)
    monkeypatch.setattr(app, '_LAST', {'res': result})
    assert 'insufficient comparable pose data' in app.frame_detail(0)
    assert result.coaching.overall is None


def test_overview_handles_missing_scores_and_escapes_dancer_names():
    from types import SimpleNamespace
    import app

    res = SimpleNamespace(
        dev=SimpleNamespace(reference_track=0, mode='dancer'),
        coaching=SimpleNamespace(overall=None, dancers=[
            {'reference': True},
            {'reference': False, 'score': None, 'valid_frames': 0,
             'total_frames': 100, 'coverage': 0.0},
        ]),
    )
    html = app.results_overview(res, ['Reference <1>', 'Dancer <script>'], [])
    assert 'N/A' in html and '0/100 comparable frames' in html
    assert 'Reference &lt;1&gt;' in html and 'Dancer &lt;script&gt;' in html
    assert '<script>' not in html
    assert 'Limited pose coverage' in html
    assert 'No practice moments to show' in html
    assert html.count('class="ddc-dancer"') == 1  # exclude the reference score


def test_practice_cards_include_every_moment_and_escape_content():
    from html.parser import HTMLParser
    import app

    rows = [['Dancer <1> & Dancer 2', 1.234567 + i, 1., 2., 'arms', 'left wrist', .2, 17]
            for i in range(5)]
    html = app.practice_preview(rows)
    class Buttons(HTMLParser):
        def __init__(self):
            super().__init__()
            self.buttons = []
        def handle_starttag(self, tag, attrs):
            if tag == 'button':
                self.buttons.append(dict(attrs))
    parser = Buttons()
    parser.feed(html)
    assert len(parser.buttons) == len(rows) == 5
    assert [float(b['data-ddc-time']) for b in parser.buttons] == [r[1] for r in rows]
    assert all(b['type'] == 'button' and 'Replay moment' in b['aria-label'] for b in parser.buttons)
    assert '1.00–2.00 s' in html
    assert 'Dancer &lt;1&gt; &amp; Dancer 2' in html
    assert '<1>' not in html
    assert 'No qualifying practice moments' in app.practice_preview([])
    assert '<button' not in app.practice_preview([])


def test_practice_cards_escape_joint_and_region_names():
    import app
    html = app.practice_preview([['Dancer 1', 2., 1., 3., '<img>', '" onclick="bad()', .2, 17]])
    assert '<img>' not in html and '&lt;img&gt;' in html
    assert '&quot; onclick=&quot;bad()' in html
    assert 'aria-label="Replay moment 1: &quot;' in html


def test_individual_reports_filter_exact_identity_and_restore_group(monkeypatch):
    import app
    names = ['Dancer 1', 'Dancer 10', 'Dancer 3']
    summary = [[name, 80] for name in names]
    moments = [[name, i, 'left_wrist', .3] for i, name in enumerate(names)]
    feedback = [[name, i, 'Review wrist'] for i, name in enumerate(names)]
    priorities = [['pair 1', 1], ['pair 2', 2], ['pair 3', 3]]
    monkeypatch.setattr(app, '_LAST', {
        'names': names, 'report_rows': (summary, moments, feedback, priorities),
        'report_pairs': [('Dancer 1', 'Dancer 3'), ('Dancer 10', 'Dancer 3'),
                         ('Dancer 1', 'Dancer 10')],
    })
    selected = app.filter_dancer_report('Dancer 1', 2)
    assert selected[:3] == ([summary[0]], [moments[0]], [feedback[0]])
    assert selected[3] == [priorities[0], priorities[2]]
    assert app.filter_dancer_report('All dancers', 2)[:4] == (summary, moments, feedback, priorities)
    assert app.filter_dancer_report('unknown dancer', 2)[:4] == ([], [], [], [])
    assert len(app._LAST['report_rows'][0]) == 3


def test_individual_report_missing_score_reference_and_escaping(monkeypatch):
    from types import SimpleNamespace
    import app
    monkeypatch.setattr(app, '_LAST', {
        'names': ['Reference <1>', 'Dancer <2>'],
        'res': SimpleNamespace(coaching=SimpleNamespace(dancers=[
            {'reference': True},
            {'reference': False, 'score': None, 'valid_frames': 0, 'total_frames': 20,
             'coverage': 0., 'strength': '<script>bad</script>', 'improvement': 'No data'},
        ])),
    })
    ref = app.dancer_report('Reference <1>')
    assert 'not scored' in ref and 'Reference &lt;1&gt;' in ref
    html = app.dancer_report('Dancer <2>')
    assert 'N/A' in html and '0/20' in html and 'Limited pose coverage' in html
    assert 'Dancer &lt;2&gt;' in html and '<script>' not in html
    assert 'Choose a dancer' in app.dancer_report('All dancers')
