from ddc.dancer_features import write_synthetic_demo
from ddc.dancer_screening import FEATURE_SPECS, collect_extended_features, screen_and_filter_run


def test_screening_writes_50_features_and_filtered_jsonl(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    tracking = run_dir / "tracking.jsonl"
    labels = run_dir / "labels.csv"
    write_synthetic_demo(tracking, labels, frames=40)
    rows = collect_extended_features(tracking, 1000, 800)
    assert len(FEATURE_SPECS) == 50
    assert len(rows) == 8
    assert all(set(spec.name for spec in FEATURE_SPECS).issubset(row) for row in rows)
    # The full runner also renders a video, which needs a real source clip.  The
    # pure feature collector is kept separately testable without rerunning pose.
