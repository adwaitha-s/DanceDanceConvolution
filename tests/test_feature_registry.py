import json

from ddc.dancer_features import write_synthetic_demo
from ddc.feature_registry import build_registry, predict_run, validate_checklist


def _make_run(root, name, *, labels):
    run = root / name
    run.mkdir()
    label_path = run / "labels.csv" if labels else root / f"{name}-source-labels.csv"
    write_synthetic_demo(run / "tracking.jsonl", label_path, frames=40)
    return run


def test_registry_scores_with_only_the_workbook_top_five_features(tmp_path):
    _make_run(tmp_path, "training", labels=True)
    _make_run(tmp_path, "candidate", labels=False)

    model, predictions = predict_run(tmp_path, "candidate")

    assert len(model["feature_ids"]) == 5
    assert model["feature_ids"] == [item["feature_id"] for item in model["selected_features"]]
    assert all(len(item["top_predictors"]) == 5 for item in predictions)
    assert all(
        [feature["feature_id"] for feature in item["top_predictors"]] == model["feature_ids"]
        for item in predictions
    )


def test_review_validation_records_corrections_and_preserves_original_prediction(tmp_path):
    _make_run(tmp_path, "training", labels=True)
    candidate = _make_run(tmp_path, "candidate", labels=False)
    _, predictions = predict_run(tmp_path, "candidate")
    predicted_dancers = [item["track_id"] for item in predictions
                         if item["predicted_label"] == "dancer"]

    unchanged = validate_checklist(tmp_path, "candidate", predicted_dancers, predictions)
    assert unchanged["validation"]["status"] == "validated_correct"
    assert unchanged["validation"]["corrected_track_ids"] == []

    changed_selection = [track for track in predicted_dancers if track != 0]
    if 0 not in predicted_dancers:
        changed_selection.append(0)
    corrected = validate_checklist(tmp_path, "candidate", changed_selection, predictions)
    assert corrected["validation"]["status"] == "corrected"
    assert 0 in corrected["validation"]["corrected_track_ids"]

    validation = json.loads((candidate / "prediction_validation.json").read_text())
    assert validation["predictor_labels"] == {
        str(item["track_id"]): item["predicted_label"] for item in predictions
    }
    rebuilt = build_registry(tmp_path)
    candidate_detail = next(run for run in rebuilt["runs_detail"] if run["run"] == "candidate")
    assert {str(track["track_id"]): track["predicted_label"] for track in candidate_detail["tracks"]} == validation["predictor_labels"]
