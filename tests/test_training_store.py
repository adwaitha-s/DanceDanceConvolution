import copy
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from ddc.training_store import DATABASE, LEGACY_DATABASE, SCHEMA_VERSION, TrainingStore


def payload(coefficient=.2, label="dancer"):
    return {
        "state": {"runs": ["clip"], "target": 10, "updated_at": "2026-10-16T12:00:00Z"},
        "features": [],
        "model": {"feature_ids": ["motion"], "ranking_coefficients": [.3],
                  "coefficient": [coefficient], "medians": [0.], "mean": [1.], "std": [2.]},
        "lstm": {"fingerprint": "v1", "samples": 1},
        "runs": [{"run": "clip", "signature": "pose-v1", "tracks": [{
            "track_id": 0, "frames_present": 180, "values": {"motion": 3., "missing": None},
            "manual_label": label, "prediction": {"predicted_label": "non_dancer", "score": .2},
            "predictor_method": "Saved screening", "lstm_status": "Training fit",
            "lstm_prediction": {"predicted_label": "dancer", "probability": .8,
                                "fingerprint": "v1", "in_training": True}}]}]}


def test_migration_uses_json_once_and_keeps_older_model_history(tmp_path):
    state = {"runs": ["clip", "new"], "target": 10}
    (tmp_path / "training_cohort.json").write_text(json.dumps(state))
    (tmp_path / "training_workbook_payload.json").write_text(json.dumps(payload()))
    store = TrainingStore(tmp_path)
    assert store.state() == state  # Migration must not undo a later enrollment.
    assert store.model("v1")[0]["samples"] == 1
    (tmp_path / "training_cohort.json").write_text("not valid JSON")
    assert TrainingStore(tmp_path).state() == state
    with pytest.raises(ValueError, match="membership changed"):
        store.payload()


def test_existing_sqlite_registry_is_renamed_without_losing_reviews(tmp_path):
    store = TrainingStore(tmp_path)
    store.save_state({"runs": ["clip"], "target": 15})
    store.reviews("clip", "pose-v1", {0: 1})
    store.path.rename(tmp_path / LEGACY_DATABASE)

    migrated = TrainingStore(tmp_path)
    assert migrated.path.name == DATABASE
    assert migrated.state()["runs"] == ["clip"]
    assert migrated.reviews("clip", "pose-v1") == {0: 1}
    assert not (tmp_path / LEGACY_DATABASE).exists()


def test_legacy_snapshot_import_uses_current_lstm_checkpoint_name(tmp_path):
    (tmp_path / "training_cohort.json").write_text(json.dumps({"runs": ["clip"], "target": 15}))
    (tmp_path / "training_workbook_payload.json").write_text(json.dumps(payload()))
    checkpoint = tmp_path / "training/feature_lstm.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"saved-lstm")

    report, saved = TrainingStore(tmp_path).model("v1")
    assert report["fingerprint"] == "v1"
    assert saved == b"saved-lstm"


def test_snapshot_is_atomic_and_keeps_versioned_weights_predictions_and_checkpoint(tmp_path):
    store = TrainingStore(tmp_path)
    store.save_snapshot(payload(), "v1", b"checkpoint-1")
    changed = payload(.4, "non_dancer")
    changed["lstm"]["fingerprint"] = "v2"
    changed["runs"][0]["tracks"][0]["lstm_prediction"]["fingerprint"] = "v2"
    store.save_snapshot(changed, "v2", b"checkpoint-2")
    assert store.model("v1")[1] == b"checkpoint-1"
    assert len(store.payload()["weight_history"]) == 2
    with store.connect() as db:
        assert db.execute("SELECT coefficient FROM current_weights").fetchone()[0] == .4
        predictions = db.execute("SELECT kind,score FROM current_predictions ORDER BY kind").fetchall()
        assert [tuple(row) for row in predictions] == [("lstm", .8), ("reviewed_screening", .2)]
        assert db.execute("SELECT COUNT(*) FROM feature_values WHERE value IS NULL").fetchone()[0] == 2
    invalid = copy.deepcopy(changed)
    invalid["runs"][0]["tracks"] *= 2  # Duplicate identity must roll the entire commit back.
    with pytest.raises(sqlite3.IntegrityError):
        store.save_snapshot(invalid, "v3", b"bad")
    assert store.model("v3") == (None, None)
    assert store.payload()["model"]["coefficient"] == [.4]


def test_unchanged_refresh_deduplicates_history_but_tracks_pending_export(tmp_path):
    store = TrainingStore(tmp_path)
    first = store.save_snapshot(payload(), "v1")
    same = payload()
    same["state"]["updated_at"] = "2026-10-16T13:00:00Z"
    assert store.save_snapshot(same, "v1") == first
    assert store.sync_status()["status"] == "pending"
    store.sync_status("workbook is open")
    assert store.sync_status()["error"] == "workbook is open"
    store.sync_status("")
    assert store.sync_status()["status"] == "synced"


def test_review_history_is_identity_safe_and_empty_labels_clear_reviews(tmp_path):
    store = TrainingStore(tmp_path)
    assert store.reviews("clip", "pose-v1", {0: 1, 1: 0}) == {0: 1, 1: 0}
    assert store.reviews("clip", "pose-v2") == {}
    assert store.reviews("clip", "pose-v1") == {0: 1, 1: 0}
    assert store.reviews("clip", "pose-v1", {}) == {}
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM review_batches").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError):
        store.reviews("clip", "pose-v1", {0: 2})
    assert store.reviews("clip", "pose-v1") == {}


def test_workbook_review_transaction_survives_failed_csv_mirror(tmp_path):
    store = TrainingStore(tmp_path)
    store.ingest_reviews("clip", "pose", {0: 0}, "original-file")
    store.import_reviews([("clip", "pose", {0: 1}, "original-file")])
    # A failed CSV mirror leaves the old file untouched. It must not undo the import.
    assert store.ingest_reviews("clip", "pose", {0: 0}, "original-file") == {0: 1}
    # A subsequent explicit file edit is still accepted as a new review.
    assert store.ingest_reviews("clip", "pose", {0: 0}, "edited-file") == {0: 0}
    with pytest.raises(sqlite3.IntegrityError):
        store.import_reviews([("clip", "pose", {0: 1}, "edited-file"),
                              ("other", "pose", {0: 7}, None)])
    assert store.reviews("clip", "pose") == {0: 0}


def test_backup_is_consistent_and_does_not_overwrite_files(tmp_path):
    store = TrainingStore(tmp_path)
    store.save_snapshot(payload(), "v1", b"checkpoint")
    path = store.backup(tmp_path / "backup.sqlite3")
    with sqlite3.connect(path) as backup:
        assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert backup.execute("SELECT checkpoint FROM model_versions").fetchone()[0] == b"checkpoint"
    with pytest.raises(ValueError, match="new backup"):
        store.backup(path)
    with pytest.raises(ValueError, match="new backup"):
        store.backup(store.path)


def test_concurrent_initialization_and_future_schema_guard(tmp_path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(lambda _: TrainingStore(tmp_path).state(), range(4))) == [None]*4
    with TrainingStore(tmp_path).connect() as db:
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION+1}")
    with pytest.raises(RuntimeError, match="newer"):
        TrainingStore(tmp_path)


def test_export_failure_is_pending_and_retry_reads_db_without_training(tmp_path, monkeypatch):
    import ddc.training as training
    store = TrainingStore(tmp_path)
    store.save_snapshot(payload(), "v1")
    def locked(*args):
        raise PermissionError("workbook is open")
    monkeypatch.setattr(training, "export_workbook", locked)
    with pytest.raises(RuntimeError, match="Excel update pending"):
        training.export_database(tmp_path)
    assert store.sync_status()["status"] == "pending"
    exported = []
    monkeypatch.setattr(training, "export_workbook", lambda root, data: exported.append(data))
    monkeypatch.setattr(training, "fit_lasso", lambda *a: pytest.fail("export must not fit"))
    monkeypatch.setattr(training, "train_lstm", lambda *a: pytest.fail("export must not train"))
    training.export_database(tmp_path)
    assert exported[0]["model"]["coefficient"] == [.2]
    assert store.sync_status()["status"] == "synced"


def test_refresh_recovers_state_reviews_and_features_without_json_or_label_csv(tmp_path, monkeypatch):
    import ddc.training as training
    training.arm(tmp_path)
    folder = tmp_path / "clip"
    folder.mkdir()
    (folder / "meta.json").write_text('{"frames": 1}')
    (folder / "tracking.jsonl").write_text('{}\n')
    (folder / "labels.csv").write_text('track_id,label\n0,dancer\n')
    (folder / "training_review.json").write_text(json.dumps({"signature": training.signature(folder)}))
    monkeypatch.setattr(training, "collect_feature_inventory", lambda *args: [
        {"track_id": 0, "frames_present": 1, "values": {}}])
    assert training.register_run(tmp_path, "clip")["runs"] == ["clip"]
    store = TrainingStore(tmp_path)
    store.features("clip", training.signature(folder), [{"track_id": 0, "frames_present": 1, "values": {}}])
    def short(*args):
        raise ValueError("short clip")
    monkeypatch.setattr(training, "_sequences", short)
    assert training.refresh(tmp_path, train=False, import_excel=False, export=False)["reviewed_tracks"] == 1
    (folder / "labels.csv").unlink()
    (tmp_path / "training_cohort.json").write_text('corrupt compatibility export')
    assert training.refresh(tmp_path, train=False, import_excel=False, export=False)["reviewed_tracks"] == 1
    assert store.payload()["runs"][0]["tracks"][0]["manual_label"] == "dancer"
