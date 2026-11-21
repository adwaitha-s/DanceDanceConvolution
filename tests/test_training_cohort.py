"""The dancer model trains on exactly the first fifteen validated videos."""

import json

import numpy as np

from ddc.training_features import PREDICTOR_IDS
from ddc.training_store import TrainingStore


def _validated_run(training, root, name, label):
    run = root / name
    run.mkdir()
    (run / "meta.json").write_text(json.dumps({"frames": 180, "fps": 30}))
    (run / "tracking.jsonl").write_text('{"frame": 0, "t": 0, "people": []}\n')
    (run / "labels.csv").write_text(
        "track_id,label\n0," + ("dancer" if label else "non_dancer") + "\n")
    (run / "training_review.json").write_text(json.dumps({"signature": training.signature(run)}))
    return run


def test_fifteen_validated_runs_freeze_training_and_later_predictions_are_read_only(tmp_path, monkeypatch):
    import ddc.training as training

    training.arm(tmp_path)
    assert TrainingStore(tmp_path).state()["target"] == 15
    values = {fid: float(index + 1) for index, fid in enumerate(PREDICTOR_IDS)}
    monkeypatch.setattr(training, "collect_feature_inventory", lambda *args: [
        {"track_id": 0, "frames_present": 180, "values": values.copy()}])
    monkeypatch.setattr(training, "_feature_rows", lambda run, merges, sig: [
        {"track_id": 0, "frames_present": 180, "values": values.copy()}])
    sequence = np.ones((9, len(PREDICTOR_IDS)), np.float32)
    monkeypatch.setattr(training, "_sequences", lambda *args: (
        {0: {"sequence": sequence, "coverage": 1., "feature_ids": list(PREDICTOR_IDS)}},
        {"start_s": 0., "end_s": 5.}))
    monkeypatch.setattr(training, "middle_sequences", lambda *args: (
        {0: {"sequence": sequence, "coverage": 1., "feature_ids": list(PREDICTOR_IDS)}},
        {"start_s": 0., "end_s": 5.}))

    fits = []
    def fit(records):
        fits.append(tuple(sorted({record["run"] for record in records})))
        return {"feature_ids": PREDICTOR_IDS[:5], "ranking_coefficients": [.2] * 5,
                "coefficient": [.2] * 5, "intercept": 0., "medians": [0.] * 5,
                "mean": [0.] * 5, "std": [1.] * 5,
                "runs": sorted({record["run"] for record in records}),
                "reviewed_tracks": len(records)}
    monkeypatch.setattr(training, "fit_lasso", fit)
    trained = []
    def train(samples, output, features, fingerprint, epochs):
        trained.append(len(samples))
        output.mkdir(exist_ok=True)
        (output / "feature_lstm.pt").write_bytes(b"checkpoint")
        return {"fingerprint": fingerprint, "runs": sorted({s["run"] for s in samples}),
                "samples": len(samples)}
    monkeypatch.setattr(training, "train_lstm", train)
    monkeypatch.setattr(training, "predict_lstm", lambda samples, checkpoint, fingerprint: {
        (sample["run"], sample["track_id"]): {"predicted_label": "dancer", "probability": .8}
        for sample in samples})

    for index in range(2):
        run = _validated_run(training, tmp_path, f"clip{index:02d}", index % 2)
        training.register_run(tmp_path, run.name)
    pending = tmp_path / "pending"
    pending.mkdir()
    (pending / "meta.json").write_text('{"frames": 180}')
    (pending / "tracking.jsonl").write_text('{"frame": 0, "t": 0, "people": []}\n')
    assert training.register_run(tmp_path, "pending")["runs"] == ["clip00", "clip01"]
    rolling = training.refresh(tmp_path, import_excel=False, export=False)
    assert rolling["status"] == "rolling_trained"
    assert rolling["runs"] == ["clip00", "clip01"]
    assert not rolling["frozen"]
    assert trained == [2]

    for index in range(2, 16):
        run = _validated_run(training, tmp_path, f"clip{index:02d}", index % 2)
        training.register_run(tmp_path, run.name)
    assert training.register_run(tmp_path, "clip15")["runs"] == [f"clip{i:02d}" for i in range(15)]
    final = training.refresh(tmp_path, import_excel=False, export=False)
    assert final["status"] == "trained"
    assert final["membership_closed"] and final["frozen"]
    assert trained == [2, 15]
    fits_before = len(fits)
    store = TrainingStore(tmp_path)
    with store.connect() as db:
        snapshot_before = db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
        reviews_before = db.execute("SELECT COUNT(*) FROM review_batches").fetchone()[0]

    # The sixteenth reviewed video may be classified, but cannot update the cohort.
    training.register_run(tmp_path, "clip15")
    assert training.refresh(tmp_path, import_excel=False, export=False)["runs"] == final["runs"]
    model, predictions = training.predict_run(tmp_path, "clip15")
    assert model["feature_ids"] == PREDICTOR_IDS[:5]
    assert predictions[0]["dancer_probability"] == .8
    assert predictions[0]["method"] == "LSTM probability"
    assert len(fits) == fits_before and trained == [2, 15]
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == snapshot_before
        assert db.execute("SELECT COUNT(*) FROM review_batches").fetchone()[0] == reviews_before
