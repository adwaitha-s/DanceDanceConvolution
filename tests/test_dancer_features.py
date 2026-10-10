import json

from ddc.dancer_features import (
    FEATURE_NAMES,
    collect_track_features,
    evaluate_run,
    write_synthetic_demo,
)


def test_synthetic_pose_features_and_selection_evaluation(tmp_path):
    tracking = tmp_path / "tracking.jsonl"
    labels = tmp_path / "labels.csv"
    out = tmp_path / "out"
    write_synthetic_demo(tracking, labels, frames=50)

    rows = collect_track_features(tracking)
    assert len(rows) == 8
    assert all(len(row.values) == len(FEATURE_NAMES) for row in rows)
    # The first four demo people dance and visibly move their wrists.
    assert min(row.values[1] for row in rows[:4]) > max(row.values[1] for row in rows[4:])

    report = evaluate_run(tracking, labels, out)
    assert report["samples"] == 8
    assert set(report["selected_features"]) == {"Filter", "Wrapper", "Embedded"}
    assert (out / "track_features.csv").exists()
    graph = out / "feature_selection_accuracy.svg"
    assert graph.exists()
    assert "Dancer vs non-dancer accuracy" in graph.read_text(encoding="utf-8")
    assert json.loads((out / "feature_selection_report.json").read_text())["samples"] == 8
