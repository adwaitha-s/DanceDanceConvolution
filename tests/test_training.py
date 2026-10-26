import json

import numpy as np
import pytest

from test_data import write_synthetic_demo
from ddc.lasso_training import fit_lasso, score
from ddc.skeleton_lstm import make_model, middle_sequences, train_lstm, predict_lstm, select_sequences, STEPS, INPUT_VERSION
from ddc.training_store import TrainingStore
from ddc.training import arm, refresh, register_run, signature
from ddc.training_features import FEATURES, PREDICTOR_IDS


def run(root, name, frames=180):
    folder = root / name
    folder.mkdir()
    write_synthetic_demo(folder / "tracking.jsonl", folder / "labels.csv", frames=frames)
    (folder / "meta.json").write_text(json.dumps({"frames": frames, "fps": 30, "duration_s": frames/30}))
    (folder / "training_review.json").write_text(json.dumps({"signature": signature(folder)}))
    return folder


def test_catalog_has_exactly_40_across_requested_categories():
    assert len(FEATURES) == len({s.id for s in FEATURES}) == 40
    assert len({s.category for s in FEATURES}) == 5
    assert len(PREDICTOR_IDS) == 35
    assert not any(s.predictor_eligible for s in FEATURES if s.id.startswith("human_"))


def test_middle_crop_contains_temporal_measurements_of_exact_selected_features(tmp_path):
    folder = run(tmp_path, "clip", frames=240)
    chosen = ["physical_step_width", "physical_wrist_acceleration_rms", "time_pose_motion_iqr",
              "physical_limb_angle_range_rad", "tf_pose_energy_std"]
    tracks, crop = middle_sequences(folder, feature_ids=chosen)
    assert crop["start_s"] == pytest.approx(1.5)
    assert crop["end_s"] == pytest.approx(6.5)
    assert len(tracks) > 0
    for track in tracks.values():
        assert track["sequence"].shape == (STEPS, 5)
        assert track["feature_ids"] == chosen
        assert np.isfinite(track["sequence"]).all()
    assert any(np.std(track["sequence"], axis=0).max() > 1e-4 for track in tracks.values())
    assert crop["window_seconds"] == 3 and crop["hop_seconds"] == .25


def test_short_clip_is_flagged_not_padded(tmp_path):
    folder = run(tmp_path, "short", frames=60)
    with pytest.raises(ValueError, match="at least 5s"):
        middle_sequences(folder)
    arm(tmp_path)  # Existing clip cannot accidentally enter the future cohort.
    folder = run(tmp_path, "newshort", frames=60)
    register_run(tmp_path, folder.name)
    state = refresh(tmp_path, train=False, export=False)
    assert not state["lstm_current"]
    assert any("at least 5s" in p for p in state["problems"])
    assert (folder / "tracking.jsonl").exists()


def records():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(100, 35))
    y = (x[:, :8].sum(1) > 0).astype(int)
    return [{"run": f"r{i//10}", "label": int(y[i]), "values": dict(zip(PREDICTOR_IDS, x[i]))}
            for i in range(len(x))]


def test_lasso_returns_five_nonzero_ranked_weights_and_finite_scores():
    rows = records()
    model = fit_lasso(rows)
    assert len(model["feature_ids"]) == 5
    assert np.all(np.abs(model["ranking_coefficients"]) > 0)
    assert sorted(np.abs(model["ranking_coefficients"]), reverse=True) == list(np.abs(model["ranking_coefficients"]))
    assert set(model["feature_ids"]) <= set(PREDICTOR_IDS)
    result = score({"values": {}}, model)
    assert np.isfinite(result["score"])
    assert len(result["contributions"]) == 5


def test_lasso_refuses_to_invent_five_features():
    rows = records()
    for row in rows:
        row["values"] = {fid: 1.0 for fid in PREDICTOR_IDS}
    with pytest.raises(ValueError, match="fewer than five nonzero"):
        fit_lasso(rows)


def test_lasso_fits_and_scores_without_scikit_learn(monkeypatch):
    import builtins
    original_import = builtins.__import__

    def without_sklearn(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ModuleNotFoundError("No module named 'sklearn'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sklearn)
    model = fit_lasso(records())
    assert len(model["feature_ids"]) == 5
    assert np.isfinite(model["selection_cv_mse"])
    assert np.isfinite(score({"values": {}}, model)["score"])


def test_saved_label_refresh_trains_without_scikit_learn(tmp_path, monkeypatch):
    import builtins
    import ddc.training as training
    from ddc.feature_registry import validate_checklist

    arm(tmp_path)
    for name in ("first", "second"):
        folder = run(tmp_path, name)
        register_run(tmp_path, folder.name)
    monkeypatch.setattr(training, "export_workbook", lambda root, payload: None)
    original_import = builtins.__import__

    def without_sklearn(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ModuleNotFoundError("No module named 'sklearn'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sklearn)
    outcome = validate_checklist(tmp_path, "second", [0, 1, 2, 3], [])
    assert "Training refresh needs attention" not in outcome["training_note"]
    assert "rolling_trained" in outcome["training_note"]
    state = TrainingStore(tmp_path).state()
    assert state["selected_features"] and state["lstm_current"]


def test_torch_trains_and_checkpoint_reloads(tmp_path):
    import torch
    rng = np.random.default_rng(2)
    chosen = PREDICTOR_IDS[:5]
    samples = [{"sequence": rng.normal(size=(STEPS,5)).astype("float32"), "label": i%2,
                "feature_ids": chosen,
                "track_id": i, "run": f"r{i}"} for i in range(10)]
    report = train_lstm(samples, tmp_path, chosen, "test", epochs=2)
    assert len(report["runs"]) == 10
    assert np.isfinite(report["final_training_loss"])
    checkpoint = torch.load(tmp_path / "feature_lstm.pt", weights_only=True)
    model = make_model()
    model.load_state_dict(checkpoint["state_dict"])
    assert model.lstm.input_size == 5
    assert checkpoint["input_size"] == 5 and checkpoint["selected_lasso_features"] == chosen
    assert checkpoint["input_version"] == INPUT_VERSION
    assert model(torch.zeros((2,STEPS,5))).shape == (2,)
    data = np.load(tmp_path / "supervised_feature_sequences.npz")
    assert data["x"].shape == (10,STEPS,5)
    np.testing.assert_array_equal(data["feature_ids"], chosen)
    predictions = predict_lstm(samples, (tmp_path / "feature_lstm.pt").read_bytes(), "test")
    assert len(predictions) == 10
    assert all(0 <= row["probability"] <= 1 for row in predictions.values())
    with pytest.raises(ValueError, match="does not match"):
        predict_lstm(samples, (tmp_path / "feature_lstm.pt").read_bytes(), "stale")


def test_lstm_column_selection_excludes_every_unselected_feature_and_score():
    ids = list(PREDICTOR_IDS)
    x = np.tile(np.arange(len(ids), dtype=np.float32), (STEPS, 1))
    chosen = [ids[i] for i in [7, 2, 30, 1, 19]]
    sample = {"sequence": x, "feature_ids": ids, "prediction": {"score": 999}, "label": 1}
    selected = select_sequences([sample], chosen)[0]
    np.testing.assert_array_equal(selected["sequence"], x[:, [7, 2, 30, 1, 19]])
    assert selected["feature_ids"] == chosen
    changed = x.copy()
    changed[:, [i for i in range(len(ids)) if ids[i] not in chosen]] = -999
    np.testing.assert_array_equal(select_sequences([{**sample, "sequence": changed,
        "prediction": {"score": -1}, "label": 0}], chosen)[0]["sequence"], selected["sequence"])
    with pytest.raises(ValueError, match="exactly five distinct"):
        select_sequences([sample], chosen[:-1])


def test_lstm_preprocessing_is_fitted_on_reviewed_samples_and_reused_at_prediction(tmp_path):
    import torch
    chosen = PREDICTOR_IDS[:5]
    samples = [{"sequence": np.full((STEPS, 5), float(i)), "feature_ids": chosen,
                "label": i % 2, "run": str(i), "track_id": i} for i in range(4)]
    samples[0]["sequence"][0, 0] = np.nan
    train_lstm(samples, tmp_path, chosen, "five-features", epochs=1)
    path = tmp_path / "feature_lstm.pt"
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["preprocessing"]["mean"][1:] == [1.5]*4
    before = checkpoint["preprocessing"].copy()
    novel = np.full((STEPS, 5), 4.25, dtype=np.float32)
    result = predict_lstm([{**samples[0], "sequence": novel}], path.read_bytes(), "five-features")
    model = make_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    expected_x = ((novel - np.asarray(before["mean"], np.float32)) /
                  np.asarray(before["std"], np.float32))
    with torch.no_grad():
        expected_probability = float(torch.sigmoid(model(torch.tensor(expected_x[None])))[0])
    assert result[("0", 0)]["probability"] == pytest.approx(expected_probability)
    assert torch.load(path, weights_only=True)["preprocessing"] == before
    with pytest.raises(ValueError, match="feature order"):
        predict_lstm([{**samples[0], "feature_ids": chosen[::-1]}], path.read_bytes(), "five-features")
    with pytest.raises(ValueError, match="nine timesteps"):
        train_lstm([{**s, "sequence": np.zeros((100, 51))} for s in samples],
                   tmp_path, chosen, "legacy", epochs=1)


def test_feature_sequence_does_not_use_frames_outside_middle_crop_or_bridge_long_gaps(tmp_path, monkeypatch):
    from ddc.io import load_jsonl
    import ddc.skeleton_lstm as lstm
    folder = run(tmp_path, "clip", frames=240)
    det = load_jsonl(folder / "tracking.jsonl")
    # Isolate input extraction from cross-frame identity association.
    ids = [np.arange(len(points)) for points in det.kps]
    monkeypatch.setattr(lstm, "stable_track_ids", lambda *a: (ids, 8))
    monkeypatch.setattr(lstm, "load_jsonl", lambda *a: det)
    chosen = PREDICTOR_IDS[:5]
    original, crop = middle_sequences(folder, feature_ids=chosen)
    for i, t in enumerate(det.t):
        if not crop["start_s"] <= t < crop["end_s"]:
            det.kps[i][:, :, :2] = -99999
    changed, _ = middle_sequences(folder, feature_ids=chosen)
    for tid in original:
        np.testing.assert_array_equal(original[tid]["sequence"], changed[tid]["sequence"])
    for i, t in enumerate(det.t):
        if 3 <= t <= 4:
            det.kps[i][:] = np.nan
    unsupported, _ = middle_sequences(folder, feature_ids=chosen)
    assert unsupported == {}


def test_saved_review_updates_workbook_validation_labels_and_keeps_original_predictions(tmp_path, monkeypatch):
    import ddc.training as training
    from ddc.feature_registry import validate_checklist
    arm(tmp_path)
    folder = run(tmp_path, "clip")
    register_run(tmp_path, folder.name)
    predicted = [{"track_id": i, "predicted_label": "dancer" if i < 4 else "non_dancer",
                  "dancer_probability": .8 if i < 4 else .2} for i in range(8)]
    exports = []
    monkeypatch.setattr(training, "export_workbook", lambda root, payload: exports.append(payload))
    outcome = validate_checklist(tmp_path, "clip", [1,2,3,4], predicted)
    assert outcome["validation"]["corrected_track_ids"] == [0,4]
    rows = exports[-1]["runs"][0]["tracks"]
    assert rows[0]["manual_label"] == "non_dancer"
    assert rows[0]["prediction"]["predicted_label"] == "dancer"
    assert rows[0]["validation_status"] == "Corrected"
    assert rows[4]["manual_label"] == "dancer"
    assert rows[4]["review_action"] == "saved_review"
    # Refits cannot change the predictions against which those labels were reviewed.
    state = refresh(tmp_path, train=False, import_excel=False, export=False)
    assert state["reviewed_tracks"] == 8


def test_removed_run_does_not_block_training_registry_refresh(tmp_path):
    arm(tmp_path)
    folder = run(tmp_path, "removed", frames=30)
    register_run(tmp_path, folder.name)
    # Simulate tracking removed from its run without deleting any test files.
    (folder / "tracking.jsonl").rename(tmp_path / "retained_tracking.jsonl")
    state = refresh(tmp_path, train=False, import_excel=False, export=False)
    assert state["runs"] == []
    assert state["missing_runs"] == ["removed"]
